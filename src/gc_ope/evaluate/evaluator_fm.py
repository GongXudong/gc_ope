"""FM：三个独立初始化的加噪 Flow Matching 密度等权混合。

每个成员用加权成功目标训练 500 次，再以 RK4 和精确二维散度计算密度。
混合的是概率密度；采样先等概率选择成员。全程 CPU-only。
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np

from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights, uniform_grid_kl, validate_hidden_layers
from gc_ope.evaluate.flow_training import FlowTraining
from scipy.special import logsumexp


class _VelocityMLP:
    """延迟定义的 torch MLP 工厂，避免模块导入时强制加载 torch。"""

    @staticmethod
    def build(hidden_layer_sizes):
        import torch.nn as nn

        # 输入为 (x,y,t)，最后一层输出二维速度；列表中的每一项对应一个隐藏层。
        layers, width = [], 3
        for hidden in hidden_layer_sizes:
            layers.extend([nn.Linear(width, hidden), nn.SiLU()])
            width = hidden
        layers.append(nn.Linear(width, 2))
        return nn.Sequential(*layers)


class _FMDensityInterface(EvaluatorBase):
    """二维 CondOT Flow Matching 估计器，CPU-only、无 validation。"""

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] | None = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        hidden_layer_sizes: tuple[int, ...] = (32, 32, 32),
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
                            "ode_steps": ode_steps,
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
        self.hidden_layer_sizes = validate_hidden_layers(hidden_layer_sizes)
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


class _FMMember(FlowTraining, _FMDensityInterface):
    """FM 集成的内部成员：目标端点加噪，固定预算拟合连续速度场。"""

    def __init__(self, noise_std=.1, **kwargs):
        kwargs.setdefault("n_epochs", 500)
        super().__init__(**kwargs)
        self._configure_regularization(noise_std, False, .15, 100, 100, 20, 1e-4, 300)

    def _new_model(self):
        return _VelocityMLP.build(self.hidden_layer_sizes).to("cpu")

    def _training_loss(self, model, x, w, generator):
        import torch
        indices = torch.multinomial(w, self.samples_per_epoch, replacement=True, generator=generator)
        target = x[indices]
        if self.noise_std:
            target = target + self.noise_std * torch.randn(target.shape, generator=generator)
        source = torch.randn(target.shape, generator=generator)
        t = torch.rand((len(target), 1), generator=generator)
        xt = (1 - t) * source + t * target
        return ((model(torch.cat([xt, t], 1)) - (target - source))**2).mean()

    def _install_model(self, model):
        self.model = model


class FlowMatchingDensityEvaluator(EvaluatorBase):
    """少数独立 FM 的等权混合；每个成员使用同一批带权历史成功目标。"""

    def __init__(self, evaluation_result_container_class=EvaluationResultContainer,
                 evaluation_result_container_kwargs=None, n_members=3, random_state=0,
                 member_parameters=None):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs or {})
        if isinstance(n_members, bool) or not isinstance(n_members, int) or n_members < 1:
            raise ValueError("集成成员数必须为正整数")
        if not isinstance(random_state, int) or random_state < 0:
            raise ValueError("初始化种子必须为非负整数")
        self.n_members, self.random_state = n_members, random_state
        self.member_parameters = dict(n_epochs=500, hidden_layer_sizes=(32, 32, 32), samples_per_epoch=2000,
                                      noise_std=.1)
        self.member_parameters.update(member_parameters or {})
        if "random_state" in self.member_parameters:
            raise ValueError("请在集成顶层指定 random_state，以保证成员独立初始化")
        # 启动前检查成员参数；此时不创建网络、不拟合数据。
        _FMMember(**self.member_parameters)
        self.members_ = []

    def fit_evaluator(self):
        self.members_ = []
        positive, weights = positive_samples_and_weights(self.eval_res_container)
        self.scaler.fit(positive)
        members = []
        for index in range(self.n_members):
            model = _FMMember(random_state=self.random_state + index, **self.member_parameters)
            # 成员只读取容器；其网络和 scaler 独立。参考侧有自己的集成与容器。
            model.eval_res_container = self.eval_res_container
            model.fit_evaluator()
            members.append(model)
        self.members_ = members
        self.fit_diagnostics_ = dict(n_members=self.n_members,
            member_seeds=[self.random_state+i for i in range(self.n_members)],
            members=[m.fit_diagnostics_ for m in members],
            quality_warnings=[f"成员 {i}：{message}" for i, m in enumerate(members)
                for message in m.fit_diagnostics_.get("quality_warnings", [])],
            density_scheme="equal-weight arithmetic mixture of independent FM densities")
        scaled, density = self.evaluate(positive)
        return positive, scaled, weights, density

    def _require_fitted(self):
        if len(self.members_) != self.n_members:
            raise RuntimeError("请先拟合 FM 集成")

    def evaluate(self, desired_goals, scale=True, return_density=True):
        self._require_fitted()
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        transformed = self.scaler.transform(goals) if scale else goals
        raw = goals if scale else self.scaler.inverse_transform(goals)
        logs = np.stack([member.log_density(raw) for member in self.members_])
        # 先在共同 raw 空间平均密度，再转回旧 evaluate 约定的标准化空间。
        mixed = logsumexp(logs, axis=0) - np.log(self.n_members) + np.log(self.scaler.scale_).sum()
        return transformed, np.exp(mixed) if return_density else mixed

    def sample(self, n_samples, random_state=0):
        self._require_fitted()
        if self.n_members == 1:
            return self.members_[0].sample(n_samples, random_state)
        rng = np.random.default_rng(random_state)
        assignment = rng.integers(self.n_members, size=n_samples)
        samples = np.empty((n_samples, 2))
        for index, member in enumerate(self.members_):
            mask = assignment == index
            if mask.any():
                samples[mask] = member.sample(int(mask.sum()), int(rng.integers(0, 2**32)))
        return samples

    def evaluate_grid(self, grid, return_log_density=False):
        values = self.log_density(grid)
        return values if return_log_density else np.exp(values)

    def kl_divergence_uniform_to_kde_integrate(self, samples, dV, u_density):
        return uniform_grid_kl(self.log_density(samples), dV, u_density)
