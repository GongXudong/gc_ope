"""NF：加噪加权似然、空间分组验证选轮数、全数据重新拟合；CPU-only。"""

from __future__ import annotations

from typing import Any, Union

import numpy as np

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluator_common import validate_hidden_layers
from gc_ope.evaluate.flow_training import FlowTraining


class _NFDensityInterface(EvaluatorBase):
    """二维 Zuko NSF 密度估计器。"""

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] | None = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        hidden_layer_sizes: tuple[int, ...] = (16, 16),
        transforms: int = 4,
        bins: int = 8,
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
        if transforms <= 0 or bins < 2:
            raise ValueError("transforms must be positive and bins >= 2")
        if device != "cpu":
            raise ValueError("NormalizingFlowDensityEvaluator currently supports device='cpu' only")

        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.hidden_layer_sizes = validate_hidden_layers(hidden_layer_sizes)
        self.transforms = int(transforms)
        self.bins = int(bins)
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
        self._require_fitted()
        goals = self._validate_goals(desired_goals, "desired_goals")
        transformed = self.scaler.transform(goals) if scale else goals
        # 与旧 KDE 一致：scale=False 表示输入已经标准化。
        import torch
        with torch.no_grad():
            log_density = self._distribution.log_prob(
                torch.from_numpy(transformed.astype(np.float32))
            ).cpu().numpy().astype(float)
        if return_density:
            return transformed, np.exp(np.clip(log_density, -745.0, 709.0))
        return transformed, log_density

    def sample(self, n_samples: int, random_state: int = 0) -> np.ndarray:
        """通过 NSF 的逆变换采样，并还原到原始目标坐标。"""
        import torch
        self._require_fitted()
        with torch.random.fork_rng(devices=[]), torch.no_grad():
            torch.manual_seed(random_state)
            scaled = self._distribution.sample((n_samples,)).cpu().numpy()
        return self.scaler.inverse_transform(scaled)

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


class NormalizingFlowDensityEvaluator(FlowTraining, _NFDensityInterface):
    """加噪加权似然的 NF；原 NF 的标准化、采样与密度接口保持一致。"""

    def __init__(self, noise_std=.2, early_stopping=True, validation_fraction=.15,
                 min_epochs=30, patience=50, validation_interval=10, tol=1e-4,
                 fallback_epochs=50, **kwargs):
        kwargs.setdefault("n_epochs", 300)
        kwargs.setdefault("hidden_layer_sizes", (16, 16))
        kwargs.setdefault("transforms", 2)
        super().__init__(**kwargs)
        self._configure_regularization(noise_std, early_stopping, validation_fraction,
                                       min_epochs, patience, validation_interval, tol, fallback_epochs)

    def _new_model(self):
        import zuko
        return zuko.flows.NSF(features=2, transforms=self.transforms, bins=self.bins,
                              hidden_features=self.hidden_layer_sizes).to("cpu")

    def _training_loss(self, model, x, w, generator):
        import torch
        noisy = x + self.noise_std * torch.randn(x.shape, generator=generator) if self.noise_std else x
        return -(w * model().log_prob(noisy)).sum()

    def _validation_log_density(self, model, x):
        import torch
        with torch.no_grad():
            return model().log_prob(torch.as_tensor(x, dtype=torch.float32)).numpy()

    def _install_model(self, model):
        self.flow, self._distribution = model, model()
