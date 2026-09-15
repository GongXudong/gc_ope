"""P0.5：加权 MLP 密度估计器 + Flow Matching 密度估计器（plan.md P0.5）。

两个估计器都是**学习型**（需要训练），与 KDE/GMM/Histogram/Gaussian 的
"一次性拟合"不同：``fit_evaluator`` 内执行完整的"数据准备 → 训练"循环，
训练目标就是带时间衰减权重的历史目标数据（协议与现有 4 估计器完全
一致，AC-2.1/2.2）。每个 checkpoint 触发一次重训（师兄明确确认的协议）。

实现说明：
- 用 torch（conda 环境 gc_ope 自带 torch 2.9.0），固定 ``random_state``
  保证重训确定性。
- 两个类都继承 ``EvaluatorBase``，暴露 ``fit_evaluator`` / ``evaluate`` /
  ``evaluate_grid`` 三个接口，与 Histogram/Gaussian 估计器同风格。
- ``fit_diagnostics_`` 记录：网络参数量、loss 曲线末值（NN）/ 采样数
  与参数量（FM），供 run_config 可追溯。
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluator_replacements import _positive_samples_and_weights


def _torch_device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# 方法 1：加权 MLP 密度估计
# ---------------------------------------------------------------------------

class WeightedMLPDensityEvaluator(EvaluatorBase):
    """加权 MLP 密度估计器（P0.5 方法 1）。

    网络结构：``dim → 16 → 16 → 1``，输出**单个 log-density 标量**
    （选 log-density 而非 density 的原因：密度值在极端区域可小到 1e-20，
    MSE 对大值样本不公平；log 空间误差对数量级变化均匀，且数值稳定）。

    训练目标 ``y_i``：leave-one-out 核密度目标值
    ``y_i = log( (1/(N-1)) * Σ_{j≠i} w_j K_h(g_i - g_j) )``，
    其中 ``K_h`` 为与 KDE 估计器一致的高斯核（bandwidth 相同），``w_j``
    为时间衰减权重。选 leave-one-out 而非 1/NK 的原因：避免样本对自身
    核贡献的过拟合（每个点都在自己核峰处，密度被系统性高估），
    且目标值完全由"其它样本 + 权重"决定，可复现、无随机采样噪声。

    损失函数（师兄原话"最后算 loss 的时候根据权重加权"）::

        loss = Σ_i w_i * (f_θ(g_i) - y_i)²

    即每个样本误差平方乘其时间权重再求和（非 importance-weighted，
    非 reweight-batch，就是字面意义的加权求和）。
    """

    n_epochs: int
    lr: float
    weight_decay: float
    hidden_width: int
    kde_bandwidth: float
    _net: Any
    fit_diagnostics_: dict[str, Any]

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] = {},
        n_epochs: int = 200,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_width: int = 16,
        kde_bandwidth: float = 0.2,
        random_state: int = 0,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_width = int(hidden_width)
        self.kde_bandwidth = float(kde_bandwidth)
        self._random_state = int(random_state)
        self._net = None
        self._loss_curve: list[float] = []
        self.fit_diagnostics_ = {}

    def _build_targets(self, positive: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """leave-one-out 加权核密度目标值（log 空间）。"""
        dim = positive.shape[1]
        h = self.kde_bandwidth
        log_norm = -0.5 * dim * math.log(2.0 * math.pi * h ** 2)
        w_norm = weights / weights.sum()
        targets = np.empty(positive.shape[0], dtype=float)
        for i in range(positive.shape[0]):
            diff = positive - positive[i]
            sq = (diff ** 2).sum(axis=1)
            # 核值 K_h(g_i - g_j)，i 自身贡献（j=i）置 0（leave-one-out）
            kernel = np.exp(-0.5 * sq / h ** 2)
            kernel[i] = 0.0
            loo = float(np.sum(w_norm * kernel))
            targets[i] = log_norm + math.log(max(loo, 1e-300))
        return targets

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch
        import torch.nn as nn

        positive, weights = _positive_samples_and_weights(self.eval_res_container)
        scaled = self.scaler.fit_transform(positive)
        targets = self._build_targets(positive, weights)

        # 标准化空间训练：输入 scaled，目标值对应 raw-space log-density
        # （Jacobian 项已含在 targets 的 log_norm 里，与 KDE 估计器同协议）
        torch.manual_seed(self._random_state)
        np.random.seed(self._random_state)
        dev = _torch_device()

        dim = scaled.shape[1]
        w = int(math.ceil(scaled.shape[0] * 0.8))  # train/val 80/20
        x_train = torch.tensor(scaled[:w], dtype=torch.float32).to(dev)
        y_train = torch.tensor(targets[:w], dtype=torch.float32).to(dev)
        wt_train = torch.tensor(weights[:w], dtype=torch.float32).to(dev)
        x_all = torch.tensor(scaled, dtype=torch.float32).to(dev)
        y_all = torch.tensor(targets, dtype=torch.float32).to(dev)

        net = nn.Sequential(
            nn.Linear(dim, self.hidden_width), nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width), nn.ReLU(),
            nn.Linear(self.hidden_width, 1),
        ).to(dev)
        n_params = int(sum(p.numel() for p in net.parameters()))

        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self._loss_curve = []
        for _ in range(self.n_epochs):
            opt.zero_grad()
            pred = net(x_train).squeeze(-1)
            err = (pred - y_train) ** 2
            loss_tensor = torch.sum(wt_train * err)
            loss = float(loss_tensor.item())
            loss_tensor.backward()
            opt.step()
            self._loss_curve.append(loss)

        with torch.no_grad():
            net.eval()
            scaled_pred = net(x_all).squeeze(-1).cpu().numpy()

        self._net = net
        self._n_params = n_params
        self.fit_diagnostics_ = {
            "n_positive_samples": int(positive.shape[0]),
            "n_dim": int(dim),
            "n_params": n_params,
            "hidden_width": self.hidden_width,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "kde_bandwidth_targets": self.kde_bandwidth,
            "loss_final": self._loss_curve[-1] if self._loss_curve else float("nan"),
            "loss_first": self._loss_curve[0] if self._loss_curve else float("nan"),
            "random_state": self._random_state,
            "target_scheme": "leave-one-out weighted KDE log-density",
        }
        densities = np.exp(np.clip(scaled_pred, -745.0, 709.0))
        return positive, scaled, weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        import torch
        if self._net is None:
            raise RuntimeError("WeightedMLPDensityEvaluator.evaluate() 需要先调用 fit_evaluator()")
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        scaled = self.scaler.transform(goals) if scale else goals
        with torch.no_grad():
            x = torch.tensor(scaled, dtype=torch.float32).to(next(self._net.parameters()).device)
            log_d = self._net(x).squeeze(-1).cpu().numpy()
        if return_density:
            return scaled, np.exp(np.clip(log_d, -745.0, 709.0))
        return scaled, log_d

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        import torch
        if self._net is None:
            raise RuntimeError("WeightedMLPDensityEvaluator.evaluate_grid() 需要先调用 fit_evaluator()")
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2:
            raise ValueError("grid 必须是 2 维数组")
        scaled = self.scaler.transform(grid)
        with torch.no_grad():
            x = torch.tensor(scaled, dtype=torch.float32).to(next(self._net.parameters()).device)
            log_d = self._net(x).squeeze(-1).cpu().numpy()
        # Jacobian 校正：网络在标准化空间训练，输出对应 raw-space log-density
        scale = np.asarray(self.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("拟合的 scaler 存在非正或非有限的 scale")
        log_d -= np.log(scale).sum()
        if return_log_density:
            return log_d
        return np.exp(np.clip(log_d, -745.0, 709.0))

    def kl_divergence_uniform_to_kde_integrate(
        self, samples: Union[list, np.ndarray], dV: float, u_density: float
    ) -> float:
        goals = np.asarray(samples, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        _, densities = self.evaluate(goals, scale=True, return_density=True)
        densities = np.maximum(densities, np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)


# ---------------------------------------------------------------------------
# 方法 2：Flow Matching 密度估计
# ---------------------------------------------------------------------------

class FlowMatchingDensityEvaluator(EvaluatorBase):
    """Flow Matching 密度估计器（P0.5 方法 2，最小可用实现）。

    采用**条件直线 flow**（rectified flow，linear path）::

        给定条件源 z~N(0,I)，条件目标 x（加权采样的历史成功目标），
        条件向量场 u_t(z|x) = (x - z) / (1 - t)，  路径 γ_t = (1-t)·z + t·x

    网络：2 层 MLP（``dim → 16 → 16 → dim``，输出与输入同维向量场），
    训练损失 ``E[t, z, x] || u_θ(γ_t, t) - u_t(z|x) ||²``，
    其中 ``x`` 按时间衰减权重采样（**权重注入方案：按权重采样条件目标**，
    即高权重样本被选作条件的频率更高，网络学到的向量场隐式偏向高权重
    区域；比"损失按权重加权求和"更自然——FM 的损失是无条件期望，
    直接加权会偏向"高权重且大误差"样本，扭曲向量场几何）。

    密度评估：**采样近似**（生成 N 个样本 → 经验密度，用与 KDE 估计器
    一致的带宽 0.2 算核密度）。选采样而非 reverse 积分的原因：
    3D 下 reverse ODE 积分（RK4, 100 步）成本远高于采样一次；采样只需
    前向跑 N 个随机起点，实现简单，正确性 sanity check 通过
    "样本数 vs 密度形状是否合理"目检。
    """

    n_epochs: int
    lr: float
    weight_decay: float
    hidden_width: int
    n_flow_samples: int
    n_integration_steps: int
    n_bins_sampled: int
    _net: Any
    _x_weighted: np.ndarray | None
    _x_weighted_weights: np.ndarray | None
    fit_diagnostics_: dict[str, Any]

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] = {},
        n_epochs: int = 150,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        hidden_width: int = 16,
        n_flow_samples: int = 5000,
        random_state: int = 0,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_width = int(hidden_width)
        self.n_flow_samples = int(n_flow_samples)
        self._random_state = int(random_state)
        self._net = None
        self._x_weighted = None
        self._x_weighted_weights = None
        self.fit_diagnostics_ = {}

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        import torch
        import torch.nn as nn

        positive, weights = _positive_samples_and_weights(self.eval_res_container)
        scaled = self.scaler.fit_transform(positive)
        self._x_weighted = scaled
        self._x_weighted_weights = weights / weights.sum()

        torch.manual_seed(self._random_state)
        np.random.seed(self._random_state)
        dev = _torch_device()

        dim = scaled.shape[1]
        x_all = torch.tensor(scaled, dtype=torch.float32).to(dev)
        w_all = torch.tensor(self._x_weighted_weights, dtype=torch.float32).to(dev)
        n_cond = int(min(512, x_all.shape[0]))  # 每 epoch 采 512 个条件目标

        net = nn.Sequential(
            nn.Linear(dim + 1, self.hidden_width), nn.ReLU(),
            nn.Linear(self.hidden_width, self.hidden_width), nn.ReLU(),
            nn.Linear(self.hidden_width, dim),
        ).to(dev)
        n_params = int(sum(p.numel() for p in net.parameters()))
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        gen = torch.Generator(device=dev).manual_seed(self._random_state)
        self._loss_curve = []

        for _ in range(self.n_epochs):
            idx = torch.multinomial(w_all, n_cond, replacement=True, generator=gen)
            x_cond = x_all[idx]
            z = torch.randn(n_cond, dim, device=dev, generator=gen)
            t = torch.rand(n_cond, 1, device=dev, generator=gen)
            gamma = (1 - t) * z + t * x_cond
            target_vec = (x_cond - z) / (1 - t + 1e-5)
            inp = torch.cat([gamma, t], dim=1)
            pred = net(inp)
            loss_tensor = torch.mean((pred - target_vec) ** 2)
            loss = float(loss_tensor.item())
            opt.zero_grad()
            loss_tensor.backward()
            opt.step()
            self._loss_curve.append(loss)

        self._net = net
        self._n_params = n_params
        # 采样用于密度估计（forward ODE 用 50 步 RK4 近似 100 步会太慢）
        self._n_flow_samples_final = self.n_flow_samples
        self.fit_diagnostics_ = {
            "n_positive_samples": int(positive.shape[0]),
            "n_dim": int(dim),
            "n_params": n_params,
            "hidden_width": self.hidden_width,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "n_cond_per_epoch": int(n_cond),
            "n_flow_samples": self.n_flow_samples,
            "loss_final": self._loss_curve[-1] if self._loss_curve else float("nan"),
            "loss_first": self._loss_curve[0] if self._loss_curve else float("nan"),
            "random_state": self._random_state,
            "weight_injection": "weighted conditional target sampling (multinomial over w_i)",
            "density_scheme": "empirical KDE over n_flow_samples forward samples, bandwidth=0.2",
        }
        # 返回的 densities：在正样本上的经验密度（sanity check 用，非主输出）
        samples_std = self._sample_std(dev, seed_offset=1)
        from sklearn.neighbors import KernelDensity
        kde = KernelDensity(bandwidth=0.2, kernel="gaussian").fit(samples_std)
        densities = np.exp(np.clip(kde.score_samples(scaled), -745.0, 709.0))
        return positive, scaled, weights, densities

    def _sample_std(self, dev: str, seed_offset: int = 1, n: int | None = None) -> np.ndarray:
        """前向跑 n 个随机起点（z~N(0,I)），返回**标准化空间**样本。"""
        import torch
        net = self._net
        if net is None:
            raise RuntimeError("FlowMatchingDensityEvaluator._sample_std() 需要先调用 fit_evaluator()")
        n = n if n is not None else self.n_flow_samples
        dim = self._x_weighted.shape[1]
        gen = torch.Generator(device=dev).manual_seed(self._random_state + seed_offset)
        z = torch.randn(n, dim, device=dev, generator=gen)
        net.eval()
        n_steps = 50
        dt = 1.0 / n_steps
        x = z
        for step in range(n_steps):
            t_now = torch.full((n, 1), step * dt, device=dev)
            with torch.no_grad():
                v = net(torch.cat([x, t_now], dim=1))
            x = x + v * dt
        return x.cpu().numpy()

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        """用 forward 采样 + KDE 近似 raw-space 密度（采样近似方案）。

        网格点变换到标准化空间，与标准化空间采样样本同域算 KDE，
        最后用 Jacobian（1/prod(scale)）校正回 raw-space 密度。
        """
        if self._net is None:
            raise RuntimeError("FlowMatchingDensityEvaluator.evaluate_grid() 需要先调用 fit_evaluator()")
        dev = _torch_device()
        import torch
        samples_std = self._sample_std(dev, seed_offset=2)
        from sklearn.neighbors import KernelDensity
        s_std = self.scaler.transform(grid)
        kde = KernelDensity(bandwidth=0.2, kernel="gaussian").fit(samples_std)
        log_d = kde.score_samples(s_std)
        scale = np.asarray(self.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("拟合的 scaler 存在非正或非有限的 scale")
        log_d = log_d - np.log(scale).sum()
        if return_log_density:
            return log_d
        return np.exp(np.clip(log_d, -745.0, 709.0))

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        """接口兼容：直接走 evaluate_grid（FM 密度是采样近似，无解析网格接口）。"""
        if self._net is None:
            raise RuntimeError("FlowMatchingDensityEvaluator.evaluate() 需要先调用 fit_evaluator()")
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        scaled = self.scaler.transform(goals) if scale else goals
        log_d = self.evaluate_grid(goals, return_log_density=True)
        if return_density:
            return scaled, np.exp(np.clip(log_d, -745.0, 709.0))
        return scaled, log_d

    def kl_divergence_uniform_to_kde_integrate(
        self, samples: Union[list, np.ndarray], dV: float, u_density: float
    ) -> float:
        goals = np.asarray(samples, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        _, densities = self.evaluate(goals, scale=True, return_density=True)
        densities = np.maximum(densities, np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)
