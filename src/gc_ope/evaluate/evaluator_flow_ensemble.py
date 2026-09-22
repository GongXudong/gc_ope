"""FM 密度集成：独立初始化，平均概率密度而不是平均 log density。"""

import numpy as np
from scipy.special import logsumexp
from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
from gc_ope.evaluate.evaluator_regularized_flow import RegularizedFMEvaluator
from gc_ope.evaluate.evaluator_common import positive_samples_and_weights, uniform_grid_kl


class FlowEnsembleEvaluator(EvaluatorBase):
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
        self.member_parameters = dict(n_epochs=500, hidden_features=32, samples_per_epoch=2000,
                                      noise_std=0., early_stopping=False)
        self.member_parameters.update(member_parameters or {})
        if "random_state" in self.member_parameters:
            raise ValueError("请在集成顶层指定 random_state，以保证成员独立初始化")
        self.members_ = []

    def fit_evaluator(self):
        self.members_ = []
        positive, weights = positive_samples_and_weights(self.eval_res_container)
        self.scaler.fit(positive)
        members = []
        for index in range(self.n_members):
            model = RegularizedFMEvaluator(random_state=self.random_state + index, **self.member_parameters)
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
