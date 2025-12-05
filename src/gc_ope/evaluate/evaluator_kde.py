from typing import Any, Literal, Callable, Union
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

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """使用KDE拟合evaluation_result_container中正样本的分布。

        注意：每一次调用 KernelDensity.fit() 方法，都是一次全新的、独立的训练过程。它会完全覆盖模型之前学到的所有信息，只基于当前传入的数据集来重新构建模型。

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 正样本，缩放后的正样本，正样本对应的权重，以及KDE拟合之后正样本对应的概率密度
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

        return positive_samples, scaled_positive_samples, sample_weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool=True, return_density: bool=True) -> tuple[np.ndarray, np.ndarray]:
        # 数据标准化
        if scale:
            scaled_desired_goals = self.scaler.transform(desired_goals)
        else:
            scaled_desired_goals = desired_goals

        # 计算数据点的概率密度
        log_densities = self.kde.score_samples(scaled_desired_goals)

        if return_density:
            densities = np.exp(log_densities)
            return scaled_desired_goals, densities
        else:
            return scaled_desired_goals, log_densities

    def kl_divergence_uniform_to_kde_mc(
        self,
        sample_uniform_func: Callable[[], Union[list, np.ndarray]],
        u_density: float,
        n_samples: int=10000,
    ) -> float:
        """使用蒙特卡洛方法计算均匀分布u与KernelDensity p之间的KL距离，KL(u || p)

        Args:
            sample_uniform_func (Callable[[], Union[list, np.ndarray]]): 从均匀分布u中采样样本的函数
            u_density (float): 均匀分布u的概率密度
            n_samples (int, optional): 使用蒙特卡洛估计KL使用的样本数. Defaults to 10000.

        Returns:
            float: KL(u || p)
        """

        # 采样
        samples_u = np.array([sample_uniform_func() for _ in range(n_samples)])

        # print(samples_u)

        # 计算均匀分布的对数概率密度
        log_u = np.log(u_density)

        # print(f"log_u: {log_u}")

        # 计算p分布的对数概率密度
        scaled_samples, log_p = self.evaluate(samples_u, scale=True, return_density=False)

        # 数值稳定性处理
        mask = np.exp(log_p) > 1e-10
        # mask = np.exp(log_p) > -np.inf
        if np.sum(mask) == 0:
            return np.inf

        # KL散度计算
        # kl_value = log_u - np.mean(log_p[mask])
        kl_value = log_u - np.mean(log_p)

        print(u_density, np.exp(log_p))
        print(log_u, np.mean(log_p))

        return kl_value

    def kl_divergence_uniform_to_kde_integrate(
        self,
        samples: Union[list, np.ndarray],
        dV: float,
        totalV: float,
        u_density: float,
    ) -> float:
        """使用积分法计算均匀分布u与KernelDensity p之间的KL距离，KL(u || p)

        Args:
            sample_uniform_func (Callable[[], Union[list, np.ndarray]]): 从均匀分布u中采样样本的函数
            u_density (float): 均匀分布u的概率密度
            n_samples (int, optional): 使用蒙特卡洛估计KL使用的样本数. Defaults to 10000.

        Returns:
            float: KL(u || p)
        """

        # print(samples_u)

        # 计算均匀分布的对数概率密度
        log_u = np.log(u_density)

        # print(f"log_u: {log_u}")

        # 计算p分布的对数概率密度
        scaled_samples, p = self.evaluate(samples, scale=True, return_density=True)

        # normalized_p = (p * dV / np.sum(p * dV)) / dV
        normalized_p = (p / np.sum(p)) / dV

        log_p = np.log(normalized_p)

        # KL散度计算
        # kl_value = log_u - np.mean(log_p[mask])
        kl_value = u_density * np.sum(log_u - log_p) * dV

        # print(u_density, np.exp(log_p))
        # print(log_u, np.mean(log_p))

        return kl_value
