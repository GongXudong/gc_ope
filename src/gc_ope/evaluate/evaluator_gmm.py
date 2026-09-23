"""GMM：直接优化加权负对数似然，协方差加入 0.05 的正则项。"""

import numpy as np
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights, uniform_grid_kl
from gc_ope.evaluate.utils.weighted_gmm import WeightedGaussianMixture


class GMMEvaluator(EvaluatorBase):
    def __init__(self, evaluation_result_container_class=EvaluationResultContainer,
                 evaluation_result_container_kwargs=None, n_components=5,
                 covariance_type="full", n_init=1, max_iter=200, tol=1e-3,
                 reg_covar=0.05, random_state=0):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs or {})
        self.n_components = n_components
        self.gmm = WeightedGaussianMixture(n_components, covariance_type, n_init,
                                           max_iter, tol, reg_covar, random_state)
        self.gmm._validate_parameters()
        self._fitted = False

    def fit_evaluator(self):
        self._fitted = False
        positive, weights = positive_samples_and_weights(self.eval_res_container)
        # 零权重记录没有统计贡献；离线时间权重本身均为正。
        positive, weights = positive[weights > 0], weights[weights > 0]
        # 保留旧 GMM 的标准化规则：在成功目标上做不加权 StandardScaler。
        scaled = self.scaler.fit_transform(positive)
        self.gmm.n_components = min(self.n_components, len(np.unique(scaled, axis=0)))
        self.gmm.fit(scaled, sample_weight=weights)
        self._fitted = True
        normalized = weights / weights.max()
        self.fit_diagnostics_ = dict(
            fitting_scheme="direct_weighted_em", covariance_type=self.gmm.covariance_type,
            n_positive_samples=len(positive), effective_components=self.gmm.n_components,
            effective_sample_size=float(normalized.sum()**2 / (normalized**2).sum()),
            converged=bool(self.gmm.converged_), n_iter=int(self.gmm.n_iter_),
            lower_bound=float(self.gmm.lower_bound_), lower_bounds=self.gmm.lower_bounds_,
            weighted_nll=-float(self.gmm.lower_bound_),
            quality_warnings=[] if self.gmm.converged_ else ["加权 EM 达到迭代上限，尚未收敛"],
        )
        return positive, scaled, weights, np.exp(self.gmm.score_samples(scaled))

    def evaluate(self, desired_goals, scale=True, return_density=True):
        if not self._fitted:
            raise RuntimeError("请先拟合加权 GMM")
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        transformed = self.scaler.transform(goals) if scale else goals
        log_density = self.gmm.score_samples(transformed)
        return transformed, np.exp(log_density) if return_density else log_density

    def sample(self, n_samples, random_state=0):
        if not self._fitted:
            raise RuntimeError("请先拟合加权 GMM")
        samples, _ = self.gmm.sample(n_samples, random_state)
        return self.scaler.inverse_transform(samples)

    def evaluate_grid(self, grid, return_log_density=False):
        values = self.log_density(grid)
        return values if return_log_density else np.exp(values)

    def kl_divergence_uniform_to_kde_integrate(self, samples, dV, u_density):
        return uniform_grid_kl(self.log_density(samples), dV, u_density)
