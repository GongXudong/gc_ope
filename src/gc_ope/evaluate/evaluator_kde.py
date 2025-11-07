from typing import Any, Literal
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer, WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_base import EvaluatorBase


class KDEEvaluator(EvaluatorBase):

    kde: KernelDensity

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] = {},
        kde_bandwidth: float = 1.0,
        kde_kernel: Literal['gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'] = "gaussian",
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs)
        self.kde = KernelDensity(
            bandwidth=kde_bandwidth,
            kernel=kde_kernel,
        )

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """使用KDE拟合evaluation_result_container中正样本的分布

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: 正样本，正样本对应的权重，以及KDE拟合之后正样本对应的概率密度
        """

        # 获取正样本
        all_samples = np.array(self.eval_res_container.desired_goal_list)
        sample_flags = np.array(self.eval_res_container.success_list)
        positive_samples = all_samples[sample_flags]

        # 数据标准化
        scaled_positive_samples = self.scaler.fit_transform(positive_samples)

        # 获取样本权重
        if isinstance(self.eval_res_container, WeightedEvaluationResultContainer):
            sample_weights = self.eval_res_container.desired_goal_weights[sample_flags]
        elif isinstance(self.eval_res_container, EvaluationResultContainer):
            sample_weights = np.ones((scaled_positive_samples.shape[0],))   
        else:
            raise ValueError(f"Can not process EvaluationResultContainer type: {type(self.eval_res_container)}!")

        # 拟合KDE，带样本权重
        self.kde.fit(
            X=scaled_positive_samples,
            sample_weight=sample_weights,
        )

        # 计算数据点的概率密度
        log_densities = self.kde.score_samples(scaled_positive_samples)
        densities = np.exp(log_densities)

        return scaled_positive_samples, sample_weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool=True) -> tuple[np.ndarray, np.ndarray]:
        # 数据标准化
        if scale:
            scaled_desired_goals = self.scaler.transform(desired_goals)
        else:
            scaled_desired_goals = desired_goals

        # 计算数据点的概率密度
        log_densities = self.kde.score_samples(scaled_desired_goals)
        densities = np.exp(log_densities)

        return scaled_desired_goals, densities
