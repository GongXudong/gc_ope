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

    def __init__(
        self,
        env: Env,
        sample_n: int,
        sample_dg_method: Literal["rig", "discern", "mega"] = "mega",
    ):
        super().__init__(env)

        self.estimator: EvaluatorBase = None

        # 根据p_ag采样时使用的超参数
        self.sample_N = sample_n
        self.sample_dg_method: Literal["rig", "discern", "mega"] = sample_dg_method

        self.need_re_estimate_p_ag_flag: bool = True

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

        self.set_re_estimate()

    def set_re_estimate(self):
        self.need_re_estimate_p_ag_flag = True
    
    def estimate_p_ag(self):
        if self.need_re_estimate_p_ag_flag:
            
            print(f"\033[33mCheck in callback: {sum(self.estimator.eval_res_container.success_list)} successful dgs in eval.\033[0m")
            
            # fit KDE
            dgs, scaled_dgs, dg_weights, dg_densities = self.estimator.fit_evaluator()

            # get threshold for sampling
            self.p_ag_density_threshold = np.min(dg_densities)
            self.need_re_estimate_p_ag_flag = False

    def sample_goal(self):
        
        if sum(self.estimator.eval_res_container.success_list) == 0:
            # 如果测试数据中还没有成功的目标，意味着没有拟合KDE的数据，此时使用环境原本的方法采样desired_goal

            desired_goal = desired_goal_utils.sample_a_desired_goal(self.env)

            print(f"sample from random: {desired_goal}")
            return desired_goal
        else:

            self.estimate_p_ag()

            # sample N candidate goals
            candidate_goals = [desired_goal_utils.sample_a_desired_goal(self.env) for _ in range(self.sample_N)]

            # compute candidate goals' KDE score
            scaled_candidate_goals, candidate_goal_scores = self.estimator.evaluate(candidate_goals)

            # print(candidate_goal_scores)

            if self.sample_dg_method == "rig":
                return self._sample_goal_rig(candidate_goals, candidate_goal_scores)
            elif self.sample_dg_method == "discern":
                return self._sample_goal_descern(candidate_goals, candidate_goal_scores)
            elif self.sample_dg_method == "mega":
                return self._sample_goal_mega(candidate_goals, candidate_goal_scores)
            else:
                raise ValueError(f"Can not process sample_dg_method: {self.sample_dg_method}!")

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[Any, dict[str, Any]]:

        # sample desired goal
        desired_goal = self.sample_goal()

        # modify obs and info
        return desired_goal_utils.reset_env_with_desired_goal(self.env, desired_goal, seed=seed, options=options)

    def _sample_goal_rig(self, candidate_goals: list, candidate_goal_scores: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """按p_ag采样目标

        Args:
            candidate_goals (list): 候选目标集合
            candidate_goal_scores (Union[list, np.ndarray]): 候选目标权重集合

        Returns:
            Union[list, np.ndarray]: desired goal
        """

        # kde_score低于self.p_ag_density_threshold的goal，认为是无法完成的goal，不采样这些目标
        candidate_goal_scores_backup = candidate_goal_scores.copy()
        candidate_goal_scores[candidate_goal_scores < self.p_ag_density_threshold] = np.inf

        # 找到权重不是inf的索引
        valid_mask = ~np.isinf(candidate_goal_scores)
        valid_indices = np.where(valid_mask)[0]
        
        # 获取有效的目标和权重
        valid_goals = candidate_goals[valid_mask]
        valid_scores = candidate_goal_scores[valid_mask]

        if len(valid_goals) == 0:
            candidate_goal_index = np.random.choice(len(candidate_goals))
            print(f"\033[31m sample randomly from candidate goals: point {candidate_goals[candidate_goal_index]} with score {candidate_goal_scores_backup[candidate_goal_index]}\033[0m from {candidate_goal_scores_backup}, score_threshold: {self.p_ag_density_threshold}")

            return candidate_goals[candidate_goal_index]
        else:
            # 归一化权重
            assert np.all(valid_scores >= 0)
            probabilities = valid_scores / np.sum(valid_scores)

            # 按权重采样
            sampled_idx = np.random.choice(len(valid_goals), p=probabilities)
            original_idx = valid_indices[sampled_idx]
            print(f"\033[32m sample from p_ag: point {valid_goals[sampled_idx]} with score {valid_scores[sampled_idx]}\033[0m from {candidate_goal_scores}, score_threshold: {self.p_ag_density_threshold}")

            return valid_goals[sampled_idx]

    def _sample_goal_descern(self, candidate_goals: list, candidate_goal_scores: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """在p_ag的支撑集上均匀采样目标

        Args:
            candidate_goals (list): 候选目标集合
            candidate_goal_scores (Union[list, np.ndarray]): 候选目标权重集合

        Returns:
            Union[list, np.ndarray]: desired goal
        """

        # kde_score低于self.p_ag_density_threshold的goal，认为是无法完成的goal，不采样这些目标
        candidate_goal_scores_backup = candidate_goal_scores.copy()
        candidate_goal_scores[candidate_goal_scores < self.p_ag_density_threshold] = np.inf

        # 找到权重不是inf的索引
        valid_mask = ~np.isinf(candidate_goal_scores)
        valid_indices = np.where(valid_mask)[0]
        
        # 获取有效的目标和权重
        valid_goals = candidate_goals[valid_mask]
        valid_scores = candidate_goal_scores[valid_mask]

        if len(valid_goals) == 0:
            candidate_goal_index = np.random.choice(len(candidate_goals))
            print(f"\033[31m sample randomly from candidate goals: point {candidate_goals[candidate_goal_index]} with score {candidate_goal_scores_backup[candidate_goal_index]}\033[0m from {candidate_goal_scores_backup}, score_threshold: {self.p_ag_density_threshold}")

            return candidate_goals[candidate_goal_index]
        else:
            # 均匀分布采样
            sampled_idx = np.random.choice(len(valid_goals))
            original_idx = valid_indices[sampled_idx]
            print(f"\033[32m samle uniformly from valid goals: point {valid_goals[sampled_idx]} with score {valid_scores[sampled_idx]}\033[0m from {candidate_goal_scores}, score_threshold: {self.p_ag_density_threshold}")

            return valid_goals[sampled_idx]



    def _sample_goal_mega(self, candidate_goals: list, candidate_goal_scores: Union[list, np.ndarray]) -> Union[list, np.ndarray]:
        """以更高的概率采样p_ag小的目标

        Args:
            candidate_goals (list): 候选目标集合
            candidate_goal_scores (Union[list, np.ndarray]): 候选目标权重集合

        Returns:
            Union[list, np.ndarray]: desired goal
        """

        # kde_score低于self.p_ag_density_threshold的goal，认为是无法完成的goal，不采样这些目标
        candidate_goal_scores_backup = candidate_goal_scores.copy()
        candidate_goal_scores[candidate_goal_scores < self.p_ag_density_threshold] = np.inf

        if np.all(candidate_goal_scores==np.inf):
            # 处理所有采样点的score都小于threshold的情况！！！！
            candidate_goal_index = np.argmax(candidate_goal_scores_backup)
            print(f"\033[31m find max: point {candidate_goals[candidate_goal_index]} with score {candidate_goal_scores_backup[candidate_goal_index]}\033[0m from {candidate_goal_scores_backup}, score_threshold: {self.p_ag_density_threshold}")
        else:
            candidate_goal_index = np.argmin(candidate_goal_scores)
            print(f"\033[32m find min: point {candidate_goals[candidate_goal_index]} with score {candidate_goal_scores[candidate_goal_index]}\033[0m from {candidate_goal_scores}, score_threshold: {self.p_ag_density_threshold}")

        return candidate_goals[candidate_goal_index]
