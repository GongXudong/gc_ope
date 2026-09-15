"""P0 replacement estimators: weighted histogram and weighted multivariate Gaussian."""

from typing import Any, Union

import numpy as np
from sklearn.preprocessing import StandardScaler

from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer, WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_base import EvaluatorBase


def _positive_samples_and_weights(
    container: EvaluationResultContainer,
) -> tuple[np.ndarray, np.ndarray]:
    """从评估结果容器中取成功目标的原始坐标与权重。

    权重来源与 KDE/GMM 估计器完全一致（plan.md AC-0.2）：
    - WeightedEvaluationResultContainer → 外部写入的绝对时间权重；
    - 普通容器 → 全 1（无时间衰减）。
    """
    all_samples = np.asarray(container.desired_goal_list, dtype=float)
    flags = np.asarray(container.success_list, dtype=bool)
    if all_samples.ndim != 2 or all_samples.shape[0] == 0:
        raise ValueError("no evaluation samples are available")
    if flags.ndim != 1 or flags.shape[0] != all_samples.shape[0]:
        raise ValueError("success_list must align with desired_goal_list")
    if not np.all(np.isfinite(all_samples)):
        raise ValueError("evaluation samples must be finite")
    positive = all_samples[flags]
    if positive.shape[0] == 0:
        raise ValueError("no successful evaluation samples are available")
    if isinstance(container, WeightedEvaluationResultContainer):
        all_weights = np.asarray(container.desired_goal_weights, dtype=float)
        if all_weights.ndim != 1 or all_weights.shape[0] != all_samples.shape[0]:
            raise ValueError("desired_goal_weights must align with desired_goal_list")
        weights = all_weights[flags]
    elif isinstance(container, EvaluationResultContainer):
        weights = np.ones(positive.shape[0], dtype=float)
    else:
        raise ValueError(f"Can not process EvaluationResultContainer type: {type(container)}!")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample weights must be finite and non-negative")
    if not np.any(weights > 0):
        raise ValueError("at least one successful sample weight must be positive")
    return positive, weights


def gaussian_mean_cov(
    samples: np.ndarray,
    weights: np.ndarray,
    reg_covar: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """加权均值与协方差（公共实现，AC-0.2：各估计器共享同一份 weighting 来源）。

    对角加 ``reg_covar`` 防近奇异（固定维目标或早期数据不足时）。
    """
    samples = np.asarray(samples, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if samples.ndim != 2 or samples.shape[0] != weights.shape[0]:
        raise ValueError("samples and weights must align")
    total = float(weights.sum())
    if total <= 0:
        raise ValueError("at least one weight must be positive")
    mean = np.average(samples, axis=0, weights=weights)
    diff = samples - mean
    cov = (diff * weights[:, None]).T @ diff / total
    cov = 0.5 * (cov + cov.T)  # 数值对称
    if reg_covar > 0:
        cov = cov + reg_covar * np.eye(cov.shape[0])
    return mean, cov


def _gaussian_log_density(
    x: np.ndarray, mean: np.ndarray, inv_cov: np.ndarray, logdet: float, dim: int
) -> np.ndarray:
    d = x - mean
    maha = np.einsum("ij,jk,ik->i", d, inv_cov, d)
    log_norm = -0.5 * (dim * np.log(2.0 * np.pi) + logdet)
    return log_norm - 0.5 * maha


class WeightedHistogramEvaluator(EvaluatorBase):
    """时间衰减加权直方图（plan.md 4.1）。

    把目标空间（标准化后）划分为 n_bins^d 规则网格，bin 概率为
    ``P(B_j) = Σ_{i∈B_j} w_i / Σ_i w_i``，密度 = 概率 / bin 体积。

    与 KDE 的区别：完全没有 kernel smoothing——样本落不进任何邻域的
    区域密度严格为 0，因此可作"KDE 的 kernel 是否影响结论"的对照。

    维度自适应：估计器不感知 2D/3D，直接对目标数据的列维度做估计。
    固定 z 维不应传入（AC-0.3，列选择由调用方负责）。
    """

    n_bins: int
    bin_probs: np.ndarray | None
    bin_volume: float
    edges_: list[np.ndarray] | None
    fit_diagnostics_: dict[str, Any]

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] = {},
        n_bins: int = 10,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs)
        if isinstance(n_bins, bool) or not isinstance(n_bins, int) or n_bins < 2:
            raise ValueError("n_bins must be an integer >= 2")
        self.n_bins = int(n_bins)
        self.bin_probs = None
        self.bin_volume = float("nan")
        self.edges_ = None
        self.fit_diagnostics_ = {}

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """在标准化正样本上建加权直方图（标准化空间估计，与 KDE 的
        固定网格 KL 诊断同格点可比）。"""
        positive, weights = _positive_samples_and_weights(self.eval_res_container)
        scaled = self.scaler.fit_transform(positive)
        counts, edges = _weighted_histogram(scaled, weights, self.n_bins)
        total = float(weights.sum())
        probs = counts / total
        self.bin_volume = float(np.prod(np.diff(edges[0])))
        self.bin_probs = probs
        self.edges_ = edges
        densities = probs / self.bin_volume  # 标准化空间密度（单位：1/标准化体积）
        self.fit_diagnostics_ = {
            "n_positive_samples": int(positive.shape[0]),
            "n_dim": int(positive.shape[1]),
            "n_bins": int(self.n_bins),
            "n_nonempty_bins": int(np.count_nonzero(probs > 0)),
            "probability_sum": float(probs.sum()),
            "bin_volume": self.bin_volume,
        }
        return positive, scaled, weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        """查询目标落在哪个 bin（盒函数取值，bin 外为 0）。"""
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        if self.bin_probs is None:
            raise RuntimeError("WeightedHistogramEvaluator.evaluate() 需要先调用 fit_evaluator()")
        scaled = self.scaler.transform(goals) if scale else goals
        idx = self._bin_flat_index(scaled)
        probs = self.bin_probs[idx]

        if return_density:
            return scaled, probs / self.bin_volume
        log_density = np.full(probs.shape, -np.inf, dtype=float)
        np.log(probs, out=log_density, where=probs > 0)
        return scaled, log_density

    def _bin_flat_index(self, scaled: np.ndarray) -> np.ndarray:
        """把 (N, d) 的每维 bin 索引合成 flat bin index（行主序，0..n_bins^d-1）。"""
        idx = np.empty((scaled.shape[0], len(self.edges_)), dtype=int)
        for j, edge in enumerate(self.edges_):
            idx[:, j] = np.clip(np.searchsorted(edge, scaled[:, j], side="right") - 1, 0, self.n_bins - 1)
        flat = np.zeros(scaled.shape[0], dtype=int)
        for j in range(len(self.edges_)):
            flat = flat * self.n_bins + idx[:, j]
        return flat

    def kl_divergence_uniform_to_kde_integrate(
        self, samples: Union[list, np.ndarray], dV: float, u_density: float
    ) -> float:
        """与 GMM 估计器一致的离散格点近似（诊断用途，非连续 KL）。

        直方图密度是盒函数，格点处取值；概率为 0 的 bin 用 tiny 截断。
        """
        goals = np.asarray(samples, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        _, densities = self.evaluate(goals, scale=True, return_density=True)
        densities = np.maximum(densities, np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        """在规则网格上评估盒函数密度（Jacobian 校正到原始目标坐标）。

        与 GMM/Gaussian 的 evaluate_grid 同协议，供 2D 热图渲染。
        """
        if self.bin_probs is None:
            raise RuntimeError("WeightedHistogramEvaluator.evaluate_grid() 需要先调用 fit_evaluator()")
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2:
            raise ValueError("grid 必须是 2 维数组")
        idx = self._bin_flat_index(self.scaler.transform(grid))
        probs = self.bin_probs[idx]
        scale = np.asarray(self.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("拟合的 scaler 存在非正或非有限的 scale，无法转换到原始坐标密度")
        density = probs / self.bin_volume * np.prod(1.0 / scale)
        if return_log_density:
            log_density = np.full(probs.shape, -np.inf, dtype=float)
            np.log(density, out=log_density, where=density > 0)
            return log_density
        return density


def _weighted_histogram(
    scaled: np.ndarray,
    weights: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """对 [0,1]^d 内标准化样本做 n_bins^d 加权直方图。

    标准化后样本可能略超出 [0,1]（scaler 对极值敏感），超出部分
    按最近边界 bin 计入，保证 Σcount = Σw。
    """
    d = scaled.shape[1]
    edges = [np.linspace(0.0, 1.0, n_bins + 1)] * d
    counts = np.zeros(n_bins ** d, dtype=float)
    # 每个样本落在唯一的 bin：按 (idx_0, ..., idx_{d-1}) 的行主序索引累加权重
    idx = np.empty((scaled.shape[0], d), dtype=int)
    for j, edge in enumerate(edges):
        idx[:, j] = np.clip(np.searchsorted(edge, scaled[:, j], side="right") - 1, 0, n_bins - 1)
    flat = np.zeros(scaled.shape[0], dtype=int)
    for j in range(d):
        flat = flat * n_bins + idx[:, j]
    np.add.at(counts, flat, weights)
    return counts, edges


class WeightedGaussianEvaluator(EvaluatorBase):
    """时间衰减加权多元 Gaussian（plan.md 4.2）。

    均值与协方差都由同一组时间权重计算（AC-1.2）：

    .. math::
        \\mu = \\frac{\\sum_i w_i g_i}{\\sum_i w_i}, \\qquad
        \\Sigma = \\frac{\\sum_i w_i (g_i-\\mu)(g_i-\\mu)^T}{\\sum_i w_i}

    协方差加 ``reg_covar`` 的小对角项防止近奇异。在标准化空间拟合，
    ``evaluate`` 与 KDE/GMM 接口一致（标准化空间密度，固定网格诊断
    可比）；``evaluate_raw_space`` / ``evaluate_grid`` 做 Jacobian
    校正，用于 MC-KL 与 2D 热图渲染。
    """

    reg_covar: float
    _mean_scaled: np.ndarray | None
    _inv: np.ndarray | None
    _logdet: float
    fit_diagnostics_: dict[str, Any]

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] = {},
        reg_covar: float = 1e-6,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs)
        if not np.isfinite(reg_covar) or reg_covar < 0:
            raise ValueError("reg_covar must be a finite non-negative number")
        self.reg_covar = float(reg_covar)
        self._mean_scaled = None
        self._inv = None
        self._logdet = 0.0
        self.fit_diagnostics_ = {}

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        positive, weights = _positive_samples_and_weights(self.eval_res_container)
        scaled = self.scaler.fit_transform(positive)
        mean, cov = gaussian_mean_cov(scaled, weights, self.reg_covar)
        sign, logabsdet = np.linalg.slogdet(cov)
        if sign <= 0:
            raise ValueError("加权协方差非正定，无法拟合 Gaussian")
        inv = np.linalg.inv(cov)
        scale = np.asarray(self.scaler.scale_, dtype=float)
        # 原始坐标下的等价参数（AC-1.2 手算对照用）
        self._mean_scaled = mean
        self._inv = inv
        self._logdet = float(logabsdet)
        self.fit_diagnostics_ = {
            "n_positive_samples": int(positive.shape[0]),
            "n_dim": int(positive.shape[1]),
            "mean_raw": (mean / scale).tolist(),
            "cov_raw": (cov / np.outer(scale, scale)).tolist(),
            "cov_symmetric": bool(np.allclose(cov, cov.T)),
            "eigen_min": float(np.linalg.eigvalsh(cov)[0]),
            "reg_covar": float(self.reg_covar),
        }
        densities = np.exp(_gaussian_log_density(scaled, mean, inv, self._logdet, scaled.shape[1]))
        return positive, scaled, weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        """标准化空间密度（接口与 KDE/GMM 的 evaluate 一致）。"""
        if self._mean_scaled is None:
            raise RuntimeError("WeightedGaussianEvaluator.evaluate() 需要先调用 fit_evaluator()")
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        scaled = self.scaler.transform(goals) if scale else goals
        log_density = _gaussian_log_density(scaled, self._mean_scaled, self._inv, self._logdet, scaled.shape[1])
        if return_density:
            return scaled, np.exp(np.clip(log_density, -745.0, 709.0))
        return scaled, log_density

    def evaluate_grid(self, grid: np.ndarray, return_log_density: bool = False) -> np.ndarray:
        """在规则网格上评估 Gaussian 密度（Jacobian 校正到原始目标坐标）。"""
        if self._mean_scaled is None:
            raise RuntimeError("WeightedGaussianEvaluator.evaluate_grid() 需要先调用 fit_evaluator()")
        grid = np.asarray(grid, dtype=float)
        if grid.ndim != 2:
            raise ValueError("grid 必须是 2 维数组")
        scaled = self.scaler.transform(grid)
        log_density = _gaussian_log_density(scaled, self._mean_scaled, self._inv, self._logdet, grid.shape[1])
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
        """与 GMM 估计器一致的离散格点近似（诊断用途）。"""
        goals = np.asarray(samples, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        _, densities = self.evaluate(goals, scale=True, return_density=True)
        densities = np.maximum(densities, np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)
