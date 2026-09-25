"""基于 Zuko Neural Spline Flow 的二维 Normalizing Flow 估计器。

协议：
- 输入只使用历史 fixed-eval 中的成功目标 (x, y)；
- 历史 checkpoint 的时间折扣由 ``desired_goal_weights`` 提供；
- 在 StandardScaler 标准化空间训练加权负对数似然；
- ``evaluate(..., return_density=False)`` 返回标准化坐标空间 log density，
  由现有 KL 工具统一减去 Jacobian 转回 raw goal 空间；
- ``evaluate_grid`` 直接返回 raw goal 空间密度。

当前实现固定使用 CPU。这样 NF 不会因为 PyTorch 自动使用 CUDA 而与项目中
其他实验争用 GPU；并且与 GMM/NN 的离线 CPU 评估边界一致。
"""

from __future__ import annotations

import math
from typing import Any, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights


class NormalizingFlowDensityEvaluator(EvaluatorBase):
    """二维 Zuko NSF 密度估计器。"""

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] | None = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        hidden_features: int = 32,
        transforms: int = 4,
        bins: int = 8,
        hidden_layer_sizes: tuple[int, ...] | list[int] | None = None,
        noise_std: float = 0.0,
        patience: int | None = None,
        random_state: int = 0,
        device: str = "cpu",
    ):
        super().__init__(
            evaluation_result_container_class,
            evaluation_result_container_kwargs or {},
        )
        if n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if lr <= 0 or not np.isfinite(lr):
            raise ValueError("lr must be positive and finite")
        if weight_decay < 0 or not np.isfinite(weight_decay):
            raise ValueError("weight_decay must be finite and non-negative")
        if hidden_features <= 0 or transforms <= 0 or bins < 2:
            raise ValueError("hidden_features/transforms must be positive and bins >= 2")
        if hidden_layer_sizes is None:
            layer_sizes = (int(hidden_features), int(hidden_features))
        else:
            layer_sizes = tuple(hidden_layer_sizes)
            if len(layer_sizes) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
                for value in layer_sizes
            ):
                raise ValueError("hidden_layer_sizes must contain exactly two positive integers")
            hidden_features = int(layer_sizes[0])
        if noise_std < 0 or not np.isfinite(noise_std):
            raise ValueError("noise_std must be finite and non-negative")
        if patience is not None and (
            isinstance(patience, bool) or not isinstance(patience, (int, np.integer)) or patience <= 0
        ):
            raise ValueError("patience must be a positive integer when provided")
        if device != "cpu":
            raise ValueError("NormalizingFlowDensityEvaluator currently supports device='cpu' only")

        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_features = int(hidden_features)
        self.hidden_layer_sizes = tuple(int(value) for value in layer_sizes)
        self.transforms = int(transforms)
        self.bins = int(bins)
        self.noise_std = float(noise_std)
        self.patience = None if patience is None else int(patience)
        self.random_state = int(random_state)
        self.device = device
        self.flow = None
        self._distribution = None
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
        """拟合加权 NSF，并返回与其他 evaluator 兼容的四元组。"""
        try:
            import torch
            import zuko
        except ImportError as exc:  # pragma: no cover - 环境缺依赖时的明确错误
            raise RuntimeError("NormalizingFlowDensityEvaluator requires torch and zuko") from exc

        positive, weights = positive_samples_and_weights(self.eval_res_container)
        positive = self._validate_goals(positive, "positive samples")
        weights = self._validate_weights(weights, len(positive))
        if len(positive) < 2:
            raise ValueError("Normalizing Flow requires at least two positive samples")

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        torch.set_num_threads(1)

        scaled = self.scaler.fit_transform(positive).astype(np.float32)
        x = torch.from_numpy(scaled)
        w = torch.from_numpy((weights / weights.sum()).astype(np.float32))

        flow = zuko.flows.NSF(
            features=2,
            transforms=self.transforms,
            bins=self.bins,
            hidden_features=self.hidden_layer_sizes,
        ).to(self.device)
        distribution = flow()
        optimizer = torch.optim.Adam(
            flow.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self._loss_curve = []
        generator = torch.Generator(device=self.device).manual_seed(self.random_state + 17)
        best_loss = float("inf")
        best_state = None
        stale_epochs = 0
        flow.train()
        for _ in range(self.n_epochs):
            train_x = x
            if self.noise_std > 0:
                train_x = x + self.noise_std * torch.randn(
                    x.shape, generator=generator, dtype=x.dtype, device=x.device
                )
            log_prob = distribution.log_prob(train_x)
            loss = -(w * log_prob).sum()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # Zuko's lazy distribution is backed by the current flow parameters;
            # reacquire it after each optimizer step to avoid stale transforms.
            distribution = flow()
            self._loss_curve.append(float(loss.detach().cpu()))
            loss_value = self._loss_curve[-1]
            if loss_value < best_loss - 1e-12:
                best_loss = loss_value
                stale_epochs = 0
                if self.patience is not None:
                    best_state = {key: value.detach().clone() for key, value in flow.state_dict().items()}
            elif self.patience is not None:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        if best_state is not None:
            flow.load_state_dict(best_state)

        flow.eval()
        self.flow = flow
        self._distribution = flow()
        self._fitted = True
        with torch.no_grad():
            scaled_log_density = self._distribution.log_prob(x).cpu().numpy()
        densities = np.exp(np.clip(scaled_log_density, -745.0, 709.0))
        self.fit_diagnostics_ = {
            "n_positive_samples": int(len(positive)),
            "n_dim": 2,
            "n_epochs": self.n_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "hidden_features": self.hidden_features,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "transforms": self.transforms,
            "bins": self.bins,
            "noise_std": self.noise_std,
            "patience": self.patience,
            "random_state": self.random_state,
            "device": self.device,
            "n_params": int(sum(parameter.numel() for parameter in flow.parameters())),
            "loss_first": self._loss_curve[0],
            "loss_final": self._loss_curve[-1],
            "density_scheme": "Zuko NSF in standardized 2D space",
            "training_scheme": "weighted negative log likelihood",
        }
        return positive, scaled, weights, densities

    def _require_fitted(self) -> None:
        if not self._fitted or self._distribution is None:
            raise RuntimeError("NormalizingFlowDensityEvaluator requires fit_evaluator() first")

    def _scaled_log_density(self, goals: np.ndarray) -> np.ndarray:
        import torch

        self._require_fitted()
        query = self._validate_goals(goals, "desired_goals")
        scaled = self.scaler.transform(query).astype(np.float32)
        with torch.no_grad():
            values = self._distribution.log_prob(torch.from_numpy(scaled)).cpu().numpy()
        return np.asarray(values, dtype=float)

    def evaluate(
        self,
        desired_goals: np.ndarray,
        scale: bool = True,
        return_density: bool = True,
    ):
        """评估密度；scale=True 时返回标准化空间的密度。"""
        goals = self._validate_goals(desired_goals, "desired_goals")
        log_density = self._scaled_log_density(goals)
        transformed = self.scaler.transform(goals) if scale else goals
        if return_density:
            return transformed, np.exp(np.clip(log_density, -745.0, 709.0))
        return transformed, log_density

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        """在 raw (x,y) 网格上返回 Jacobian 校正后的密度。"""
        goals = self._validate_goals(grid, "grid")
        log_density = self._scaled_log_density(goals)
        scale = np.asarray(self.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("fitted scaler has a non-positive or non-finite scale")
        raw_log_density = log_density - np.log(scale).sum()
        if return_log_density:
            return raw_log_density
        return np.exp(np.clip(raw_log_density, -745.0, 709.0))

    def sample(self, n_samples: int, random_state: int = 0) -> np.ndarray:
        """从 NSF 采样并还原到原始目标坐标。"""
        import torch
        self._require_fitted()
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(int(random_state))
            scaled = self._distribution.sample((int(n_samples),)).cpu().numpy()
        return self.scaler.inverse_transform(scaled)

    def kl_divergence_uniform_to_kde_integrate(
        self,
        samples: Union[list, np.ndarray],
        dV: float,
        u_density: float,
    ) -> float:
        goals = self._validate_goals(np.asarray(samples, dtype=float), "samples")
        densities = np.maximum(self.evaluate_grid(goals), np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)
