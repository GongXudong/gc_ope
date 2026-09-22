"""用成功率分类器及高斯核混合定义连续的能力分布。"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KernelDensity

from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
from gc_ope.evaluate.evaluator_common import uniform_grid_kl, InsufficientSamples


class GoalSuccessMLPClassifierEvaluator(EvaluatorBase):
    """先学习 P(成功|目标)，再以预测概率为权重平滑几何支持网格。

    支持网格只提供目标坐标，不提供额外的成功标签。密度为归一化的高斯
    混合，积分为 1；不能把网格上的概率除以面积后直接延伸到整个平面。
    """

    def __init__(
        self,
        evaluation_result_container_class=EvaluationResultContainer,
        evaluation_result_container_kwargs=None,
        support_goals=None,
        bandwidth=0.2,
        hidden_width=16,
        n_epochs=100,
        lr=1e-3,
        alpha=1e-4,
        random_state=0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs or {})
        if bandwidth <= 0 or not np.isfinite(bandwidth):
            raise ValueError("核带宽必须有限且为正")
        self.support_goals = support_goals
        self.bandwidth = bandwidth
        self.classifier = MLPClassifier(
            hidden_layer_sizes=(hidden_width, hidden_width),
            activation="relu", solver="adam", alpha=alpha,
            learning_rate_init=lr, max_iter=n_epochs, random_state=random_state,
            early_stopping=early_stopping, validation_fraction=validation_fraction,
            n_iter_no_change=n_iter_no_change if early_stopping else n_epochs + 1,
        )
        self.density = KernelDensity(bandwidth=bandwidth, kernel="gaussian")

    def fit_evaluator(self):
        """分类器使用全部标签；返回值继续遵守旧 evaluator 的四元组接口。"""
        container = self.eval_res_container
        goals = np.asarray(container.desired_goal_list, dtype=float)
        labels = np.asarray(container.success_list, dtype=bool)
        weights = np.asarray(getattr(container, "desired_goal_weights", np.ones(len(labels))))
        if goals.ndim != 2 or goals.shape[1] != 2 or len(goals) != len(labels):
            raise ValueError("NN 需要与标签对齐的二维目标")
        if not np.isfinite(goals).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
            raise ValueError("目标和权重必须有限，且权重为正")
        counts = np.bincount(labels.astype(int), minlength=2)
        if counts.min() == 0:
            raise InsufficientSamples("NN 需要成功和失败两类样本")
        if self.classifier.early_stopping:
            n_validation = int(np.ceil(len(labels) * self.classifier.validation_fraction))
            if counts.min() < 2 or min(n_validation, len(labels) - n_validation) < 2:
                raise InsufficientSamples("NN 样本不足以划分分层验证集")

        # 与原分类器一致：标准化使用所有训练目标，不只使用成功目标。
        scaled = self.scaler.fit_transform(goals)
        self.classifier.fit(scaled, labels.astype(int), sample_weight=weights)

        # 两侧使用同一几何网格；预测概率各自来自本侧独立拟合的分类器。
        support = np.asarray(self.support_goals, dtype=float)
        if support.ndim != 2 or support.shape[1] != 2 or not len(support) or not np.isfinite(support).all():
            raise ValueError("NN 需要有限的二维几何支持网格")
        self.support_goals_ = np.unique(support, axis=0)
        probability = self.predict_success_probability(self.support_goals_)
        self.support_weights_ = probability / probability.sum()
        self.density.fit(self.scaler.transform(self.support_goals_), sample_weight=self.support_weights_)
        positive = goals[labels]
        _, density = self.evaluate(positive)
        self.fit_diagnostics_ = {
            "n_samples": len(goals), "n_success": int(labels.sum()),
            "n_iter": self.classifier.n_iter_, "bandwidth": self.bandwidth,
            "density_scheme": "预测成功率加权的归一化高斯核混合",
        }
        return positive, scaled[labels], weights[labels], density

    def predict_success_probability(self, goals):
        """仅此接口返回成功率；evaluate 返回连续概率密度。"""
        probability = self.classifier.predict_proba(self.scaler.transform(goals))[:, 1]
        return np.maximum(probability, 1e-12)

    def evaluate(self, desired_goals, scale=True, return_density=True):
        goals = np.asarray(desired_goals, dtype=float)
        transformed = self.scaler.transform(goals) if scale else goals
        values = self.density.score_samples(transformed)
        return transformed, np.exp(values) if return_density else values

    def sample(self, n_samples, random_state=0):
        """先按预测权重选择核中心，再从对应高斯核采样。"""
        return self.scaler.inverse_transform(self.density.sample(n_samples, random_state))

    def kl_divergence_uniform_to_kde_integrate(self, samples, dV, u_density):
        return uniform_grid_kl(self.log_density(samples), dV, u_density)
