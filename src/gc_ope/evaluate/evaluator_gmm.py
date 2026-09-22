"""对历史成功目标按时间权重重采样，再拟合高斯混合分布。"""

from numbers import Integral, Real
from typing import Any, Literal, Union

import numpy as np
from sklearn.mixture import GaussianMixture

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights


def weighted_resample(
    samples: np.ndarray,
    weights: np.ndarray,
    size: int,
    random_state: int | np.random.RandomState | np.random.Generator | None = None,
) -> np.ndarray:
    """按非负权重有放回重采样。

    sklearn GMM 不支持 sample_weight，因此用固定数量的重采样近似加权拟合。
    """
    data = np.asarray(samples, dtype=float)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("samples must be a non-empty 2-D array")
    if not np.all(np.isfinite(data)):
        raise ValueError("samples must be finite")
    if isinstance(size, bool) or not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError("size must be a positive integer")
    probability = np.asarray(weights, dtype=float)
    if probability.ndim != 1 or probability.shape[0] != data.shape[0]:
        raise ValueError("weights must be a vector aligned with samples")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0):
        raise ValueError("weights must be finite and non-negative")
    total = probability.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("at least one weight must be positive")
    probability = probability / total
    if isinstance(random_state, (np.random.Generator, np.random.RandomState)):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)
    indices = rng.choice(data.shape[0], size=int(size), replace=True, p=probability)
    return data[indices]


class GMMEvaluator(EvaluatorBase):
    """拟合成功目标的混合高斯分布，接口与旧 KDEEvaluator 保持一致。"""

    gmm: GaussianMixture

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] | None = None,
        n_components: int = 2,
        resample_size: int = 1000,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        n_init: int = 1,
        max_iter: int = 200,
        tol: float = 1e-3,
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
        reg_covar: float = 1e-6,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs or {})
        self._validate_integer("n_components", n_components, minimum=1)
        self._validate_integer("resample_size", resample_size, minimum=2)
        self._validate_integer("n_init", n_init, minimum=1)
        self._validate_integer("max_iter", max_iter, minimum=1)
        if covariance_type not in {"full", "tied", "diag", "spherical"}:
            raise ValueError(
                "covariance_type must be one of 'full', 'tied', 'diag', or 'spherical'"
            )
        if not isinstance(tol, Real) or not np.isfinite(tol) or tol <= 0:
            raise ValueError("tol must be a finite positive number")
        if not isinstance(reg_covar, Real) or not np.isfinite(reg_covar) or reg_covar < 0:
            raise ValueError("reg_covar must be a finite non-negative number")
        self.n_components = int(n_components)
        self.resample_size = int(resample_size)
        self.random_state = random_state

        # sklearn GMM 不接受 Generator；为模型生成固定种子，重采样仍用原随机流。
        gmm_random_state = random_state
        if isinstance(random_state, np.random.Generator):
            gmm_random_state = int(random_state.integers(0, np.iinfo(np.int32).max))

        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=gmm_random_state,
            reg_covar=reg_covar,
        )
        self._fitted = False
        self.constant_features_: np.ndarray | None = None
        self.fit_diagnostics_: dict[str, Any] = {}

    @staticmethod
    def _validate_integer(name: str, value: Any, minimum: int) -> None:
        """拒绝会被 int() 静默截断的非整数参数。"""
        if isinstance(value, bool) or not isinstance(value, Integral) or int(value) != value:
            raise ValueError(f"{name} must be an integer")
        if int(value) < minimum:
            raise ValueError(f"{name} must be at least {minimum}")

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # 统一获取成功目标及其时间权重；分类器的全标签处理独立保留。
        positive_samples, sample_weights = positive_samples_and_weights(self.eval_res_container)
        scaled_positive_samples = self.scaler.fit_transform(positive_samples)
        # 常量坐标会被标准化为零；记录它，便于检查近退化的协方差。
        self.constant_features_ = np.isclose(
            np.ptp(positive_samples, axis=0), 0.0, rtol=0.0, atol=1e-12
        )
        # 混合分量数不能超过实际成功样本数和重采样数量。
        effective_components = min(
            self.n_components,
            scaled_positive_samples.shape[0],
            self.resample_size,
        )
        if effective_components != self.gmm.n_components:
            self.gmm.set_params(n_components=effective_components)
        fitted_samples = weighted_resample(
            scaled_positive_samples, sample_weights, self.resample_size, self.random_state
        )
        self.gmm.fit(fitted_samples)
        self._fitted = True
        log_densities = self.gmm.score_samples(scaled_positive_samples)
        densities = np.exp(np.clip(log_densities, -745.0, 709.0))
        # 先缩放权重再计算有效样本数，避免平方溢出。
        max_weight = float(np.max(sample_weights))
        normalized_weights = sample_weights / max_weight
        self.fit_diagnostics_ = {
            "n_positive_samples": int(positive_samples.shape[0]),
            "n_positive_weighted_samples": int(np.count_nonzero(sample_weights > 0)),
            "effective_sample_size": float(
                np.sum(normalized_weights) ** 2 / np.sum(normalized_weights**2)
            ),
            "constant_features": np.flatnonzero(self.constant_features_).tolist(),
            "effective_components": int(effective_components),
            "resample_size": int(self.resample_size),
            "converged": bool(self.gmm.converged_),
            "n_iter": int(self.gmm.n_iter_),
            "lower_bound": float(self.gmm.lower_bound_),
        }
        return positive_samples, scaled_positive_samples, sample_weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        if not self._fitted:
            raise RuntimeError("GMMEvaluator.evaluate() 需要先调用 fit_evaluator() 拟合 GMM")
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        scaled_goals = self.scaler.transform(goals) if scale else goals
        log_densities = self.gmm.score_samples(scaled_goals)
        if return_density:
            return scaled_goals, np.exp(np.clip(log_densities, -745.0, 709.0))
        return scaled_goals, log_densities

    def sample(self, n_samples: int, random_state: int = 0) -> np.ndarray:
        """用独立随机流采样，不修改拟合模型持有的随机数状态。"""
        from copy import copy
        if not self._fitted:
            raise RuntimeError("请先拟合 GMM")
        sampler = copy(self.gmm)
        sampler.random_state = random_state
        scaled, _ = sampler.sample(n_samples)
        return self.scaler.inverse_transform(scaled)

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        """在规则网格上评估 GMM 密度，用于热力图渲染。

        与 evaluate() 的差别：evaluate() 在 StandardScaler 标准化空间上计算密度
        （单位是 1/标准化坐标，无法直接在原始目标坐标画密度场）；
        本方法把网格点标准化后求密度，再用 Jacobian 校正（密度按 1/prod(scale)
        缩放），使返回的密度单位对应原始目标坐标，可直接用于
        matplotlib 的 imshow/pcolormesh。
        """
        if not self._fitted:
            raise RuntimeError("GMMEvaluator.evaluate_grid() 需要先调用 fit_evaluator() 拟合 GMM")
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2 or grid.shape[1] != 2:
            raise ValueError("grid 必须是 2 维目标空间的 (n, 2) 数组")
        log_density = self.gmm.score_samples(self.scaler.transform(grid))
        # 标准化变换的 Jacobian 是 diag(1/scale)，密度按 1/prod(scale) 缩放
        scale = np.asarray(self.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("拟合的 scaler 存在非正或非有限的 scale，无法转换到原始坐标密度")
        log_density -= np.log(scale).sum()
        if return_log_density:
            return log_density
        return np.exp(np.clip(log_density, -745.0, 709.0))

    def kl_divergence_uniform_to_kde_integrate(
        self, samples: Union[list, np.ndarray], dV: float, u_density: float
    ) -> float:
        """沿用旧 evaluator 的均匀分布网格积分接口。"""
        if dV <= 0 or u_density <= 0:
            raise ValueError("dV and u_density must be positive")
        _, densities = self.evaluate(np.asarray(samples), return_density=True)
        densities = np.maximum(densities, np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)
