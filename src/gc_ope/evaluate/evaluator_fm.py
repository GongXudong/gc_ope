"""二维 Flow Matching 密度估计器。

协议：
- 只使用历史 fixed-eval 中的成功目标 (x, y)；
- 时间折扣通过每个训练 epoch 的加权目标抽样注入；不使用 validation/early stopping；
- 在 StandardScaler 标准化空间训练 CondOT/线性概率路径的速度场；
- 通过固定步长 RK4 反向积分，并用二维 exact divergence 计算 continuous-flow log density；
- ``evaluate`` 返回标准化空间密度，``evaluate_grid`` 返回 raw goal 空间密度。

这是项目自己的最小实现，不依赖外部 flow-matching 包。二维 exact divergence
使实现可审计，但 likelihood 评估会比单纯采样昂贵；正式 sweep 前应先测量耗时。
一个 epoch 表示一次加权抽取 samples_per_epoch 个目标及一次优化器更新，
不是遍历全部历史样本。独立配对 x0/x1，不进行 minibatch OT 匹配，不加 KDE。
离散历史目标只提供训练样本；有限步训练的连续流并不等于精确的离散经验分布。
"""

from __future__ import annotations

import math
import time
from typing import Any, Union

import numpy as np

from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights


class _VelocityMLP:
    """延迟定义的 torch MLP 工厂，避免模块导入时强制加载 torch。"""

    @staticmethod
    def build(hidden_features: int):
        import torch.nn as nn

        return nn.Sequential(
            nn.Linear(3, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, 2),
        )


class FlowMatchingDensityEvaluator(EvaluatorBase):
    """二维 CondOT Flow Matching 估计器，CPU-only、无 validation。"""

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] | None = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        hidden_features: int = 32,
        samples_per_epoch: int = 2000,
        ode_steps: int = 32,
        likelihood_batch_size: int = 1024,
        weight_decay: float = 0.0,
        random_state: int = 0,
        device: str = "cpu",
    ):
        super().__init__(
            evaluation_result_container_class,
            evaluation_result_container_kwargs or {},
        )
        for name, value in {"n_epochs": n_epochs, "samples_per_epoch": samples_per_epoch,
                            "ode_steps": ode_steps, "hidden_features": hidden_features,
                            "likelihood_batch_size": likelihood_batch_size}.items():
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
                raise ValueError(f"{name} 必须是正整数")
        if not isinstance(random_state, (int, np.integer)) or random_state < 0:
            raise ValueError("random_state 必须是非负整数")
        if lr <= 0 or not np.isfinite(lr):
            raise ValueError("lr must be positive and finite")
        if weight_decay < 0 or not np.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if device != "cpu":
            raise ValueError("FlowMatchingDensityEvaluator currently supports device='cpu' only")

        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.hidden_features = int(hidden_features)
        self.samples_per_epoch = int(samples_per_epoch)
        self.ode_steps = int(ode_steps)
        self.likelihood_batch_size = int(likelihood_batch_size)
        self.weight_decay = float(weight_decay)
        self.random_state = int(random_state)
        self.device = device
        self.model = None
        self._fitted = False
        self.fit_diagnostics_: dict[str, Any] = {}
        self._loss_curve: list[float] = []

    @staticmethod
    def _validate_goals(values: np.ndarray, name: str) -> np.ndarray:
        goals = np.asarray(values, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        if goals.ndim != 2 or goals.shape[1] != 2 or len(goals) == 0:
            raise ValueError(f"{name} must have shape (n, 2) and be non-empty")
        if not np.all(np.isfinite(goals)):
            raise ValueError(f"{name} contains non-finite values")
        return goals

    @staticmethod
    def _validate_weights(values: np.ndarray, n: int) -> np.ndarray:
        weights = np.asarray(values, dtype=float).reshape(-1)
        if len(weights) != n or not np.all(np.isfinite(weights)) or np.any(weights <= 0):
            raise ValueError("sample weights must be positive and finite")
        return weights

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """按加权历史成功目标训练速度场。"""
        import torch

        self._fitted = False
        positive, weights = positive_samples_and_weights(self.eval_res_container)
        positive = self._validate_goals(positive, "positive samples")
        weights = self._validate_weights(weights, len(positive))
        if len(positive) < 2:
            raise ValueError("Flow Matching requires at least two positive samples")

        torch.set_num_threads(1)
        device = torch.device(self.device)
        scaled = self.scaler.fit_transform(positive).astype(np.float32)
        x1_all = torch.from_numpy(scaled).to(device)
        probabilities = torch.from_numpy((weights / weights.sum()).astype(np.float32)).to(device)

        # 只固定本模型初始化，不改变调用方的全局随机数状态。
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.random_state)
            model = _VelocityMLP.build(self.hidden_features).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        generator = torch.Generator(device=device).manual_seed(self.random_state + 17)
        self._loss_curve = []
        t0 = time.perf_counter()
        model.train()
        for _ in range(self.n_epochs):
            indices = torch.multinomial(
                probabilities,
                self.samples_per_epoch,
                replacement=True,
                generator=generator,
            )
            x1 = x1_all[indices]
            x0 = torch.randn(
                (self.samples_per_epoch, 2), device=device, generator=generator
            )
            t = torch.rand(
                (self.samples_per_epoch, 1), device=device, generator=generator
            )
            xt = (1.0 - t) * x0 + t * x1
            target_velocity = x1 - x0
            input_tensor = torch.cat([xt, t], dim=1)
            predicted_velocity = model(input_tensor)
            loss = torch.mean((predicted_velocity - target_velocity) ** 2)
            if not torch.isfinite(loss):
                raise FloatingPointError("FM 训练 loss 非有限，拒绝输出结果")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            self._loss_curve.append(float(loss.detach().cpu()))

        model.eval()
        self.model = model
        self._fitted = True
        train_time = time.perf_counter() - t0
        density_t0 = time.perf_counter()
        try:
            scaled_log_density = self._log_density_scaled(torch.from_numpy(scaled))
        except Exception:
            self._fitted = False
            raise
        densities = np.exp(np.clip(scaled_log_density, -745.0, 709.0))
        self.fit_diagnostics_ = {
            "n_positive_samples": int(len(positive)),
            "n_dim": 2,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "hidden_features": self.hidden_features,
            "samples_per_epoch": self.samples_per_epoch,
            "epoch_definition": "一次加权有放回抽样及一次优化器更新，非全数据遍历",
            "early_stopping": False,
            "validation_fraction": 0.0,
            "likelihood_batch_size": self.likelihood_batch_size,
            "fit_density_time_s": time.perf_counter() - density_t0,
            "constant_features": np.flatnonzero(np.ptp(positive, axis=0) == 0).tolist(),
            "ode_steps": self.ode_steps,
            "random_state": self.random_state,
            "device": self.device,
            "n_params": int(sum(p.numel() for p in model.parameters())),
            "loss_first": self._loss_curve[0],
            "loss_final": self._loss_curve[-1],
            "training_time_s": float(train_time),
            "density_scheme": "CondOT FM + RK4 + exact 2D divergence",
            "training_scheme": "weighted target resampling, no validation",
        }
        return positive, scaled, weights, densities

    def _require_fitted(self) -> None:
        if not self._fitted or self.model is None:
            raise RuntimeError("FlowMatchingDensityEvaluator requires fit_evaluator() first")

    def _velocity(self, x, t):
        import torch
        t_column = torch.full((x.shape[0], 1), float(t), dtype=x.dtype, device=x.device)
        return self.model(torch.cat([x, t_column], dim=1))

    def sample(self, n_samples: int, random_state: int = 0) -> np.ndarray:
        """从标准高斯出发，沿同一个速度场正向 RK4 积分到 t=1。"""
        import torch
        self._require_fitted()
        generator = torch.Generator(device=self.device).manual_seed(random_state)
        x = torch.randn((n_samples, 2), generator=generator, device=self.device)
        dt = 1.0 / self.ode_steps
        with torch.no_grad():
            for step in range(self.ode_steps):
                t = step * dt
                k1 = self._velocity(x, t)
                k2 = self._velocity(x + dt * k1 / 2, t + dt / 2)
                k3 = self._velocity(x + dt * k2 / 2, t + dt / 2)
                k4 = self._velocity(x + dt * k3, t + dt)
                x = x + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if not torch.isfinite(x).all():
            raise FloatingPointError("FM 正向采样产生非有限坐标")
        return self.scaler.inverse_transform(x.cpu().numpy())

    def _velocity_and_divergence(self, x, t):
        import torch
        # 外层即使处于 no_grad，exact divergence 仍然需要输入梯度。
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            velocity = self._velocity(x, t)
            divergence = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
            for coordinate in range(2):
                gradient = torch.autograd.grad(
                    velocity[:, coordinate].sum(), x,
                    create_graph=False, retain_graph=coordinate == 0,
                )[0]
                divergence = divergence + gradient[:, coordinate]
        return velocity.detach(), divergence.detach()

    def _log_density_scaled(self, scaled_goals):
        """分块查询，避免 MC 大批量目标占用过多内存。"""
        self._require_fitted()
        if scaled_goals.ndim != 2 or scaled_goals.shape[1] != 2 or len(scaled_goals) == 0:
            raise ValueError("标准化查询必须是非空的 (n, 2) 数组")
        result = np.concatenate([
            self._integrate_log_density(chunk)
            for chunk in scaled_goals.split(self.likelihood_batch_size)
        ]).astype(float)
        if not np.all(np.isfinite(result)):
            raise FloatingPointError("FM log density 非有限，请检查训练及 ODE 步数")
        return result

    def _integrate_log_density(self, scaled_goals):
        """用 t=1 -> 0 的 RK4 ODE 反向积分计算标准化空间 log density。"""
        import torch

        self._require_fitted()
        x = scaled_goals.to(dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        log_det = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
        dt = -1.0 / self.ode_steps
        for step in range(self.ode_steps):
            t = 1.0 - step / self.ode_steps
            k1_x, k1_d = self._velocity_and_divergence(x, t)
            k2_x, k2_d = self._velocity_and_divergence(x + 0.5 * dt * k1_x, t + 0.5 * dt)
            k3_x, k3_d = self._velocity_and_divergence(x + 0.5 * dt * k2_x, t + 0.5 * dt)
            k4_x, k4_d = self._velocity_and_divergence(x + dt * k3_x, t + dt)
            x = x + (dt / 6.0) * (k1_x + 2.0 * k2_x + 2.0 * k3_x + k4_x)
            log_det = log_det + (dt / 6.0) * (k1_d + 2.0 * k2_d + 2.0 * k3_d + k4_d)

        source_log_density = -0.5 * (x.square().sum(dim=1) + 2.0 * math.log(2.0 * math.pi))
        return (source_log_density + log_det).detach().cpu().numpy()

    def _scaled_log_density(self, goals: np.ndarray) -> np.ndarray:
        self._require_fitted()
        query = self._validate_goals(goals, "desired_goals")
        scaled = self.scaler.transform(query).astype(np.float32)
        import torch
        return self._log_density_scaled(torch.from_numpy(scaled))

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        import torch
        self._require_fitted()
        goals = self._validate_goals(desired_goals, "desired_goals")
        transformed = self.scaler.transform(goals) if scale else goals
        log_density = self._log_density_scaled(torch.from_numpy(transformed.astype(np.float32)))
        if return_density:
            return transformed, np.exp(np.clip(log_density, -745.0, 709.0))
        return transformed, log_density

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        goals = self._validate_goals(grid, "grid")
        scaled_log_density = self._scaled_log_density(goals)
        scale = np.asarray(self.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("fitted scaler has a non-positive or non-finite scale")
        raw_log_density = scaled_log_density - np.log(scale).sum()
        if return_log_density:
            return raw_log_density
        return np.exp(np.clip(raw_log_density, -745.0, 709.0))

    def kl_divergence_uniform_to_kde_integrate(
        self,
        samples: Union[list, np.ndarray],
        dV: float,
        u_density: float,
    ) -> float:
        if not np.isfinite(dV) or dV <= 0 or not np.isfinite(u_density) or u_density <= 0:
            raise ValueError("dV 和 u_density 必须有限且为正")
        goals = self._validate_goals(np.asarray(samples, dtype=float), "samples")
        densities = np.maximum(self.evaluate_grid(goals), np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)
