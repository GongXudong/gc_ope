"""基于 sklearn MLPClassifier 的离线目标成功率分布估计器。

协议：
1. 历史 fixed-eval 数据的全部目标作为输入，``termination == reach target``
   作为二分类标签；
2. checkpoint 时间折扣通过 ``MLPClassifier.fit(sample_weight=...)`` 注入；
3. 当前 checkpoint fixed-eval CSV 中去重后的 (x, y) 是离散支持点；
4. ``P(success | goal)`` 与均匀离散先验相乘并归一化，得到目标分布。

该类故意不复用旧的 PyTorch ``WeightedMLPDensityEvaluator``：旧类拟合的是
leave-one-out KDE 的 log-density 回归，不是用户确认的成功率分类器协议。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


class GoalSuccessMLPClassifierEvaluator:
    """goal -> P(success) MLP，并在当前 checkpoint 的离散支持上形成密度。"""

    def __init__(
        self,
        *,
        hidden_width: int = 16,
        hidden_layer_sizes: tuple[int, ...] | list[int] | None = None,
        n_epochs: int = 100,
        lr: float = 1e-3,
        alpha: float = 1e-4,
        bandwidth: float | None = None,
        random_state: int = 0,
        probability_floor: float = 1e-12,
        early_stopping: bool = True,
        validation_fraction: float = 0.1,
        n_iter_no_change: int = 10,
    ) -> None:
        if hidden_width <= 0:
            raise ValueError("hidden_width must be positive")
        if hidden_layer_sizes is None:
            layer_sizes = (int(hidden_width), int(hidden_width))
        else:
            layer_sizes = tuple(hidden_layer_sizes)
            if len(layer_sizes) != 2 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0
                for value in layer_sizes
            ):
                raise ValueError("hidden_layer_sizes must contain exactly two positive integers")
            hidden_width = int(layer_sizes[0])
        if n_epochs <= 0:
            raise ValueError("n_epochs must be positive")
        if lr <= 0 or not np.isfinite(lr):
            raise ValueError("lr must be positive and finite")
        if alpha < 0 or not np.isfinite(alpha):
            raise ValueError("alpha must be non-negative and finite")
        if bandwidth is not None and (bandwidth <= 0 or not np.isfinite(bandwidth)):
            raise ValueError("bandwidth must be positive and finite when provided")
        if probability_floor <= 0 or not np.isfinite(probability_floor):
            raise ValueError("probability_floor must be positive and finite")
        if not 0.0 < validation_fraction < 1.0 or not np.isfinite(validation_fraction):
            raise ValueError("validation_fraction must be between 0 and 1")
        if n_iter_no_change <= 0:
            raise ValueError("n_iter_no_change must be positive")

        self.hidden_width = int(hidden_width)
        self.hidden_layer_sizes = tuple(int(value) for value in layer_sizes)
        self.n_epochs = int(n_epochs)
        self.lr = float(lr)
        self.alpha = float(alpha)
        self.bandwidth = None if bandwidth is None else float(bandwidth)
        self.random_state = int(random_state)
        self.probability_floor = float(probability_floor)
        self.early_stopping = bool(early_stopping)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.scaler = StandardScaler()
        self.classifier: MLPClassifier | None = None
        self.support_goals_: np.ndarray | None = None
        self.support_cell_area_: float | None = None
        self.fit_diagnostics_: dict[str, Any] = {}

    @staticmethod
    def _validate_xy(values: np.ndarray, name: str, *, allow_empty: bool = False) -> np.ndarray:
        arr = np.asarray(values, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2 or (not allow_empty and len(arr) == 0):
            raise ValueError(f"{name} must have shape (n, 2) and be non-empty")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains non-finite values")
        return arr

    @staticmethod
    def _grid_cell_area(support_goals: np.ndarray) -> float:
        xs = np.unique(support_goals[:, 0])
        ys = np.unique(support_goals[:, 1])
        if len(xs) < 2 or len(ys) < 2:
            raise ValueError("support_goals must contain at least two x and y coordinates")
        dx = np.diff(xs)
        dy = np.diff(ys)
        if not np.allclose(dx, dx[0]) or not np.allclose(dy, dy[0]):
            raise ValueError("support_goals must form an equally spaced rectangular grid")
        area = float(dx[0] * dy[0])
        if not np.isfinite(area) or area <= 0:
            raise ValueError("support grid cell area must be positive and finite")
        return area

    def fit_classifier(
        self,
        historical_goals: np.ndarray,
        success_labels: np.ndarray,
        sample_weights: np.ndarray,
        support_goals: np.ndarray,
    ) -> "GoalSuccessMLPClassifierEvaluator":
        """按历史成功/失败数据拟合分类器并固定当前 checkpoint 的支持网格。"""
        x = self._validate_xy(historical_goals, "historical_goals")
        y = np.asarray(success_labels, dtype=int).reshape(-1)
        w = np.asarray(sample_weights, dtype=float).reshape(-1)
        support = self._validate_xy(support_goals, "support_goals")
        if len(y) != len(x) or len(w) != len(x):
            raise ValueError("success_labels and sample_weights must match historical_goals")
        if not np.all(np.isin(y, [0, 1])):
            raise ValueError("success_labels must contain only 0/1")
        if np.unique(y).size < 2:
            raise ValueError("MLPClassifier requires both success and failure labels")
        if not np.all(np.isfinite(w)) or np.any(w <= 0):
            raise ValueError("sample_weights must be positive and finite")

        self.scaler.fit(x)
        scaled_x = self.scaler.transform(x)
        self.classifier = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation="relu",
            solver="adam",
            alpha=self.alpha,
            batch_size="auto",
            learning_rate_init=self.lr,
            max_iter=self.n_epochs,
            shuffle=True,
            random_state=self.random_state,
            tol=1e-4,
            n_iter_no_change=(self.n_iter_no_change if self.early_stopping else self.n_epochs + 1),
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
        )
        # sklearn 1.7.2 的 MLPClassifier.fit 支持 sample_weight；这里显式传入
        # 每个历史评估样本的 checkpoint 时间折扣，不把折扣改成重复采样。
        self.classifier.fit(scaled_x, y, sample_weight=w)
        self.support_goals_ = np.unique(support, axis=0)
        self.support_cell_area_ = self._grid_cell_area(self.support_goals_)
        self.fit_diagnostics_ = {
            "n_historical_samples": int(len(x)),
            "n_success": int(np.sum(y == 1)),
            "n_failure": int(np.sum(y == 0)),
            "n_support_goals": int(len(self.support_goals_)),
            "hidden_width": self.hidden_width,
            "hidden_layer_sizes": list(self.hidden_layer_sizes),
            "n_epochs": self.n_epochs,
            "n_iter": int(self.classifier.n_iter_),
            "early_stopping": self.early_stopping,
            "validation_fraction": self.validation_fraction if self.early_stopping else 0.0,
            "n_iter_no_change": self.n_iter_no_change if self.early_stopping else self.n_epochs + 1,
            "n_validation_samples": int(len(x) * self.validation_fraction) if self.early_stopping else 0,
            "best_validation_score": (float(self.classifier.best_validation_score_) if self.early_stopping else None),
            "lr": self.lr,
            "alpha": self.alpha,
            "bandwidth": self.bandwidth,
            "random_state": self.random_state,
            "target_scheme": "goal_to_P(success)_with_sample_weight",
            "support_scheme": "current_checkpoint_unique_fixed_eval_goals",
            "cell_area": self.support_cell_area_,
            "loss_final": float(self.classifier.loss_curve_[-1]) if self.classifier.loss_curve_ else float("nan"),
        }
        return self

    def _require_fitted(self) -> None:
        if self.classifier is None or self.support_goals_ is None or self.support_cell_area_ is None:
            raise RuntimeError("GoalSuccessMLPClassifierEvaluator requires fit_classifier() first")

    def predict_success_probability(self, goals: np.ndarray) -> np.ndarray:
        self._require_fitted()
        query = self._validate_xy(goals, "goals")
        assert self.classifier is not None
        probabilities = self.classifier.predict_proba(self.scaler.transform(query))[:, 1]
        return np.maximum(np.asarray(probabilities, dtype=float), self.probability_floor)

    def _raw_density(self, goals: np.ndarray) -> np.ndarray:
        self._require_fitted()
        assert self.support_goals_ is not None
        assert self.support_cell_area_ is not None
        support_probability = self.predict_success_probability(self.support_goals_)
        normalizer = float(np.sum(support_probability))
        if not np.isfinite(normalizer) or normalizer <= 0:
            raise ValueError("NN success probabilities have an invalid support normalizer")
        if self.bandwidth is None:
            query_probability = self.predict_success_probability(goals)
            return (query_probability / normalizer) / self.support_cell_area_

        # 搜索协议中的 bandwidth 将分类器的成功率平滑为连续密度。
        distances = (goals[:, None, :] - self.support_goals_[None, :, :]) / self.bandwidth
        kernels = np.exp(-0.5 * np.sum(distances * distances, axis=2))
        kernel_normalizer = 2.0 * np.pi * self.bandwidth**2
        return (kernels @ support_probability) / (normalizer * kernel_normalizer)

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        """返回查询点密度；raw density 由当前 checkpoint 支持网格归一化得到。"""
        self._require_fitted()
        query = self._validate_xy(desired_goals, "desired_goals")
        raw_density = self._raw_density(query)
        if scale:
            scale_jacobian = float(np.prod(self.scaler.scale_))
            if not np.isfinite(scale_jacobian) or scale_jacobian <= 0:
                raise ValueError("fitted scaler has an invalid Jacobian")
            values = raw_density * scale_jacobian
            transformed = self.scaler.transform(query)
        else:
            values = raw_density
            transformed = query
        if return_density:
            return transformed, values
        return transformed, np.log(np.maximum(values, np.finfo(float).tiny))

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        raw_density = self._raw_density(self._validate_xy(grid, "grid"))
        if return_log_density:
            return np.log(np.maximum(raw_density, np.finfo(float).tiny))
        return raw_density

    def sample(self, n_samples: int, random_state: int = 0) -> np.ndarray:
        """从成功率加权的高斯核混合采样。"""
        self._require_fitted()
        if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer")
        rng = np.random.default_rng(random_state)
        support = np.asarray(self.support_goals_, dtype=float)
        probability = self.predict_success_probability(support)
        probability /= probability.sum()
        centers = support[rng.choice(len(support), size=int(n_samples), replace=True, p=probability)]
        if self.bandwidth is None:
            return centers
        return centers + rng.normal(0.0, self.bandwidth, size=centers.shape)
