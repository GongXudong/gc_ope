from typing import Union, Literal
from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces, Wrapper
from gymnasium.core import Env
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from typing import Any, TypeVar, Union, Dict, List
import numpy as np
from copy import deepcopy

from gc_ope.evaluate.evaluator_base import EvaluatorBase
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.env.utils import desired_goal_utils


class SyncEvaluationResultWrapper(Wrapper):
    """接收评估数据并存储的Wrapper
    """

    def __init__(
        self,
        env: Env,
    ):
        super().__init__(env)

        self.estimator: EvaluatorBase = None

    def after_sync_evaluation_stat_hook(self):
        pass

    def sync_evaluation_stat(self, evaluation_stat: List[Dict]):
        """接收最新评估数据，加入到kde_estimator的eval_res_container中

        Args:
            evaluation_stat (List[Dict]): 其中的item为dict，形如{"desired_goal": xx, "success": True, "cumulative_reward": 3.0}
        """

        # print(f"sync evaluation statistic: {evaluation_stat}")

        # 向self.estimator中加入评估数据
        self.estimator.eval_res_container.add_batch(
            desired_goal_batch=[item["desired_goal"] for item in evaluation_stat],
            success_batch=[item["success"] for item in evaluation_stat],
            cumulative_reward_batch=[item["cumulative_reward"] for item in evaluation_stat],
            discounted_cumulative_reward_batch=[0.0] * len(evaluation_stat),
            desired_goal_weight_batch = [1.0] * len(evaluation_stat),
        )

        self.after_sync_evaluation_stat_hook()


    def reset_evaluation_result_container(self):
        """重置评估数据容器(清空容器内的数据)
        """
        self.estimator.eval_res_container.reset()


    def get_info_to_log(self) -> dict:
        """从环境中获得关键数据，记录在sb3的logger中
        """
        return {}
