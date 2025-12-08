from typing import Literal
import numpy as np
import gymnasium as gym
from gc_ope.algorithm.curriculum.mega_wrapper import MEGAWrapper
from gc_ope.env.utils import desired_goal_utils


class OMEGAWrapper(MEGAWrapper):

    def __init__(
        self,
        env: gym.Env,
        sample_n: int,
        kde_kernel: str="gaussian",
        kde_bandwidth: float=0.2,
        kde_data_discounted_factor: float=0.9,
        sample_dg_method: Literal["rig", "discern", "mega"] = "mega",
        b_used_in_omega: float=-3.0,  # Hyper-parameter used in OMEGA, refer to the Algorithm 2 in "Maximum Entropy Gain Exploration for Long Horizon Multi-Goal Reinforcement Learning"
        eval_kl_u_p_args: dict={},
    ):
        super().__init__(
            env=env,
            sample_n=sample_n,
            kde_kernel=kde_kernel,
            kde_bandwidth=kde_bandwidth,
            kde_data_discounted_factor=kde_data_discounted_factor,
            sample_dg_method=sample_dg_method,
        )

        self.b_used_in_omega = b_used_in_omega
        self.eval_kl_u_p_args = eval_kl_u_p_args

        self.kl_value = 1e8  # 初始化为一个大数
        self.alpha = 0.0

    def estimate_p_ag(self):
        if self.need_re_estimate_p_ag_flag:
            super().estimate_p_ag()
    
            # 计算KL(p_dg | p_ag)
            all_dgs, dV = desired_goal_utils.get_all_possible_dgs_and_dV(
                env=self.env,
                step_list=self.eval_kl_u_p_args.get("step_list", None),
            )

            self.kl_value = self.estimator.kl_divergence_uniform_to_kde_integrate(
                samples=all_dgs,
                dV=dV,
                u_density=1.0 / desired_goal_utils.get_desired_goal_space_volumn(env=self.env),
            )

            # 计算alpha
            self.alpha = 1 / np.max([self.b_used_in_omega + self.kl_value, 1])

            print(f"\033[34m calc in env: KL(p_dg, p_ag)={self.kl_value}, alpha={self.alpha}\033[0m")

    def get_info_to_log(self):
        
        self.estimate_p_ag()
        
        print(f"\033[34m retrieving info from env: KL(p_dg, p_ag)={self.kl_value}, alpha={self.alpha}\033[0m")
        
        return {
            "KL[p_dg|p_ag]": self.kl_value,
            "alpha": self.alpha,
        }

    def sample_goal(self):
        """先估计p_{ag}，然后根据p_{ag}使用MEGA/RIG/DISCERN中的一种方法采样desired goal

        Returns:
            Union[list, np.ndarray]: desired goal
        """

        if sum(self.estimator.eval_res_container.success_list) == 0:
            # 如果测试数据中还没有成功的目标，意味着没有拟合KDE的数据，此时使用环境原本的方法采样desired_goal

            desired_goal = desired_goal_utils.sample_a_desired_goal(self.env)

            print(f"sample from random: {desired_goal}")
            return desired_goal
        else:

            self.estimate_p_ag()

            # 根据alpha判断如何采样desired goal
            tmp = np.random.rand()

            if tmp < self.alpha:
                desired_goal = desired_goal_utils.sample_a_desired_goal(self.env)
                print(f"\033[34m sample from omega random: {desired_goal}\033[0m, with KL(p_dg, p_ag)={self.kl_value}, alpha={self.alpha}")

                return desired_goal
            else:
                # sample N candidate goals
                candidate_goals = np.array([desired_goal_utils.sample_a_desired_goal(self.env) for _ in range(self.sample_N)])

                # compute candidate goals' KDE score
                scaled_candidate_goals, candidate_goal_scores = self.estimator.evaluate(candidate_goals, return_density=True)

                # print(candidate_goal_scores)

                return self._sample_goal_with_help_of_p_ag(candidate_goals, candidate_goal_scores)
