from gymnasium import ObservationWrapper, ActionWrapper, Env, spaces, Wrapper
from gymnasium.core import Env
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from typing import Any, TypeVar, Union, Dict, List
import numpy as np
from copy import deepcopy

from gc_ope.evaluate.evaluator_kde import KDEEvaluator
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.env.utils import desired_goal_utils


class MEGAWrapper(Wrapper):

    def __init__(
        self,
        env: Env,
        sample_N: int,
        kde_kernel: str="gaussian",
        kde_bandwidth: float=0.2,
        kde_data_discounted_factor: float=0.9,
    ):
        super().__init__(env)

        self.kde_estimator = KDEEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=dict(
                discounted_factor=kde_data_discounted_factor,
            ),
            kde_bandwidth=kde_bandwidth,
            kde_kernel=kde_kernel,
        )

        self.env_id = env.unwrapped.spec.id

        # 根据KDE采样时使用的超参数
        self.sample_N = sample_N
        self.need_re_estimate_kde_flag: bool = True
        self.kde_score_threshold: float = 0.0

    def sync_evaluation_stat(self, evaluation_stat: List[Dict]):
        """接收最新评估数据，加入到kde_estimator的eval_res_container中

        Args:
            evaluation_stat (List[Dict]): 其中的item为dict，形如{"desired_goal": xx, "success": True, "cumulative_reward": 3.0}
        """

        # print(f"sync evaluation statistic: {evaluation_stat}")

        # 向self.kde_estimator中加入评估数据
        self.kde_estimator.eval_res_container.add_batch(
            desired_goal_batch=[item["desired_goal"] for item in evaluation_stat],
            success_batch=[item["success"] for item in evaluation_stat],
            cumulative_reward_batch=[item["cumulative_reward"] for item in evaluation_stat],
            discounted_cumulative_reward_batch=[0.0] * len(evaluation_stat),
            desired_goal_weight_batch = [1.0] * len(evaluation_stat),
        )

        self.set_re_estimate()

    def set_re_estimate(self):
        self.need_re_estimate_kde_flag = True
    
    def estimate_kde(self):
        if self.need_re_estimate_kde_flag:
            # fit KDE
            dgs, scaled_dgs, dg_weights, dg_densities = self.kde_estimator.fit_evaluator()

            # get threshold for sampling
            self.kde_score_threshold = np.min(dg_densities)
            self.need_re_estimate_kde_flag = False
    
    def sample_goal(self):
        
        if sum(self.kde_estimator.eval_res_container.success_list) == 0:
            # 如果测试数据中还没有成功的目标，意味着没有拟合KDE的数据，此时使用环境原本的方法采样desired_goal

            desired_goal = desired_goal_utils.sample_a_desired_goal(self.env)

            print(f"sample from random: {desired_goal}")
            return desired_goal
        else:

            self.estimate_kde()

            # sample N candidate goals
            candidate_goals = [desired_goal_utils.sample_a_desired_goal(self.env) for _ in range(self.sample_N)]

            # compute candidate goals' KDE score
            scaled_candidate_goals, candidate_goal_scores = self.kde_estimator.evaluate(candidate_goals)

            # print(candidate_goal_scores)

            # kde_score低于self.kde_score_threshold的goal，认为是无法完成的goal，不采样这些目标
            candidate_goal_scores_backup = candidate_goal_scores.copy()
            candidate_goal_scores[candidate_goal_scores < self.kde_score_threshold] = np.inf

            if np.all(candidate_goal_scores==np.inf):
                # 处理所有采样点的score都小于threshold的情况！！！！
                candidate_goal_index = np.argmax(candidate_goal_scores_backup)
                print(f"\033[31m find max: point {candidate_goals[candidate_goal_index]} with score {candidate_goal_scores_backup[candidate_goal_index]}\033[0m from {candidate_goal_scores_backup}, score_threshold: {self.kde_score_threshold}")
            else:
                candidate_goal_index = np.argmin(candidate_goal_scores)
                print(f"\033[32m find min: point {candidate_goals[candidate_goal_index]} with score {candidate_goal_scores[candidate_goal_index]}\033[0m from {candidate_goal_scores}, score_threshold: {self.kde_score_threshold}")

            return candidate_goals[candidate_goal_index]


    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        # sample mega goal
        mega_goal = self.sample_goal()

        # modify obs and info
        return desired_goal_utils.reset_env_with_desired_goal(self.env, mega_goal, seed=seed, options=options)
