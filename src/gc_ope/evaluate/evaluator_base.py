from abc import ABC, abstractmethod
import numpy as np
from typing import Any, ClassVar, Optional, TypeVar, Union
from sklearn.preprocessing import StandardScaler

from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer


class EvaluatorBase(ABC):

    eval_res_container: EvaluationResultContainer
    scaler: StandardScaler

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] = {},
    ):
        
        self.eval_res_container = evaluation_result_container_class(**evaluation_result_container_kwargs)
        
        # 数据标准化
        self.scaler = StandardScaler()

    @abstractmethod
    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """拟合evaluation_result_container中正样本的分布    

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: 正样本，缩放后的正样本，正样本对应的权重，以及KDE拟合之后正样本对应的概率密度
        """
        pass

    @abstractmethod
    def evaluate(self, desired_goals: np.ndarray, scale: bool=True, return_density: bool=True) -> tuple[np.ndarray, np.ndarray]:
        """计算样本对应的概率密度

        Args:
            desired_goals (np.ndarray): 目标
            scale (bool): 是否对desired_goals进行标准化
            return_density (bool): True，返回概率密度；False，返回概率密度的log值

        Returns:
            tuple[np.ndarray, np.ndarray]: 标准化后的目标，目标对应的概率密度或者概率密度的log值
        """

    @abstractmethod
    def kl_divergence_uniform_to_kde_integrate(
        self,
        samples: Union[list, np.ndarray],
        dV: float,
        u_density: float,
    ) -> float:
        """使用积分法计算均匀分布u与KernelDensity p之间的KL距离，KL(u || p)

        Args:
            samples (Union[list, np.ndarray]): 在目标空间中按固定间隔均匀采样出的目标集合
            dV (float): 采样间隔的体积
            u_density (float): 均匀分布的概率密度

        Returns:
            float: KL(u || p)
        """
        pass
