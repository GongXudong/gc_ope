"""直接最大化加权对数似然的高斯混合；不通过重采样近似权重。"""

from copy import deepcopy
from numbers import Integral
import warnings

import numpy as np
from scipy.linalg import solve_triangular
from scipy.special import logsumexp
from sklearn.base import BaseEstimator
from sklearn.cluster import KMeans
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils.validation import check_is_fitted


class WeightedGaussianMixture(BaseEstimator):
    """支持 full / diag 协方差以及 fit(X, sample_weight=...)。

    最大化 sum_i w_i log(sum_k pi_k N(x_i | mu_k, Sigma_k)) / sum_i w_i。
    E-step 的条件责任度仍为 r_ik；M-step 使用 w_i*r_ik 计算充分统计量。
    权重整体缩放不改变拟合或停止条件，零权重记录完全不参与初始化与拟合。
    """

    def __init__(self, n_components=5, covariance_type="full", n_init=1,
                 max_iter=200, tol=1e-3, reg_covar=1e-6, random_state=0):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.reg_covar = reg_covar
        self.random_state = random_state

    def _validate_parameters(self):
        for name in ["n_components", "n_init", "max_iter"]:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} 必须是正整数")
        if self.covariance_type not in {"full", "diag"}:
            raise ValueError("covariance_type 只支持 full 或 diag")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol 必须有限且为正")
        if not np.isfinite(self.reg_covar) or self.reg_covar < 0:
            raise ValueError("reg_covar 必须有限且非负")

    @staticmethod
    def _data(X):
        data = np.asarray(X, dtype=float)
        if data.ndim != 2 or min(data.shape) == 0 or not np.isfinite(data).all():
            raise ValueError("X 必须是非空、有限的二维数组")
        return data

    def _m_step(self, X, weights, responsibilities):
        # N_k = sum_i w_i*r_ik；不能先乘权重再按行归一化，否则权重会被消掉。
        weighted = weights[:, None] * responsibilities
        mass = weighted.sum(axis=0)
        if not np.isfinite(mass).all() or np.any(mass <= 0):
            raise FloatingPointError("混合分量有效权重为零，请减少分量数或调整正则")
        self.weights_ = mass / mass.sum()
        self.means_ = weighted.T @ X / mass[:, None]

        # 用中心化残差计算协方差，避免 E[x²]-E[x]² 的数值相消。
        covariances = []
        for k in range(self.n_components):
            residual = X - self.means_[k]
            if self.covariance_type == "full":
                covariance = (residual.T * weighted[:, k]) @ residual / mass[k]
                covariance = (covariance + covariance.T) / 2
                covariance.flat[::X.shape[1] + 1] += self.reg_covar
            else:
                covariance = (weighted[:, k, None] * residual**2).sum(axis=0) / mass[k]
                covariance += self.reg_covar
            covariances.append(covariance)
        self.covariances_ = np.asarray(covariances)

    def _log_joint(self, X):
        # log(pi_k N_k)，在 log 空间计算 E-step，避免小密度下溢。
        values = np.empty((len(X), self.n_components))
        for k in range(self.n_components):
            residual = X - self.means_[k]
            covariance = self.covariances_[k]
            if self.covariance_type == "full":
                try:
                    cholesky = np.linalg.cholesky(covariance)
                except np.linalg.LinAlgError as exc:
                    raise ValueError("协方差非正定，请增大 reg_covar") from exc
                standardized = solve_triangular(cholesky, residual.T, lower=True).T
                quadratic = np.sum(standardized**2, axis=1)
                log_determinant = 2 * np.log(np.diag(cholesky)).sum()
            else:
                if np.any(covariance <= 0):
                    raise ValueError("方差非正，请增大 reg_covar")
                quadratic = np.sum(residual**2 / covariance, axis=1)
                log_determinant = np.log(covariance).sum()
            values[:, k] = np.log(self.weights_[k]) - .5 * (
                X.shape[1] * np.log(2 * np.pi) + log_determinant + quadratic)
        if not np.isfinite(values).all():
            raise FloatingPointError("GMM 对数密度非有限")
        return values

    def fit(self, X, y=None, sample_weight=None):
        self._validate_parameters()
        # 重拟合失败后不能把上次模型误认为本次有效结果。
        self.__dict__.pop("n_features_in_", None)
        X = self._data(X)
        weights = np.ones(len(X)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if weights.shape != (len(X),) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("sample_weight 必须与样本对齐、有限且非负")
        active = weights > 0
        if not active.any():
            raise ValueError("至少一个样本权重必须为正")
        X, weights = X[active], weights[active]
        if len(np.unique(X, axis=0)) < self.n_components:
            raise ValueError("分量数不能超过正权重样本的不同坐标数")
        # 先除最大值，避免大权重求和溢出。归一化使收敛阈值不依赖样本总权重。
        weights = weights / weights.max()
        weights /= weights.sum()
        rng = np.random.default_rng(self.random_state)
        best_score, best = -np.inf, None
        for _ in range(self.n_init):
            # KMeans 仅用于初始化；它支持 sample_weight，未复制或重采样数据。
            labels = KMeans(n_clusters=self.n_components, n_init=1,
                            random_state=int(rng.integers(2**31 - 1))).fit_predict(X, sample_weight=weights)
            self._m_step(X, weights, np.eye(self.n_components)[labels])
            curve = [float(weights @ logsumexp(self._log_joint(X), axis=1))]
            converged = False
            for iteration in range(1, self.max_iter + 1):
                joint = self._log_joint(X)
                responsibilities = np.exp(joint - logsumexp(joint, axis=1)[:, None])
                self._m_step(X, weights, responsibilities)
                score = float(weights @ logsumexp(self._log_joint(X), axis=1))
                curve.append(score)
                if abs(curve[-1] - curve[-2]) < self.tol:
                    converged = True
                    break
            if score > best_score:
                best_score = score
                best = deepcopy((self.weights_, self.means_, self.covariances_,
                                 converged, iteration, curve))
        (self.weights_, self.means_, self.covariances_, self.converged_,
         self.n_iter_, self.lower_bounds_) = best
        self.lower_bound_ = best_score
        self.n_features_in_ = X.shape[1]
        if not self.converged_:
            warnings.warn("加权 GMM 达到迭代上限，尚未满足收敛条件", ConvergenceWarning)
        return self

    def _query(self, X):
        check_is_fitted(self, "n_features_in_")
        X = self._data(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("查询目标维数与拟合数据不同")
        return X

    def score_samples(self, X):
        return logsumexp(self._log_joint(self._query(X)), axis=1)

    def predict_proba(self, X):
        joint = self._log_joint(self._query(X))
        return np.exp(joint - logsumexp(joint, axis=1)[:, None])

    def sample(self, n_samples=1, random_state=None):
        """返回样本与分量标签；独立随机流不改变拟合过程和全局 numpy 状态。"""
        check_is_fitted(self, "n_features_in_")
        if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples <= 0:
            raise ValueError("n_samples 必须是正整数")
        rng = np.random.default_rng(self.random_state if random_state is None else random_state)
        labels = rng.choice(self.n_components, size=n_samples, p=self.weights_)
        samples = np.empty((n_samples, self.n_features_in_))
        for k in range(self.n_components):
            selected = labels == k
            noise = rng.normal(size=(selected.sum(), self.n_features_in_))
            covariance = self.covariances_[k]
            if self.covariance_type == "full":
                samples[selected] = noise @ np.linalg.cholesky(covariance).T + self.means_[k]
            else:
                samples[selected] = noise * np.sqrt(covariance) + self.means_[k]
        return samples, labels
