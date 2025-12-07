
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np
import gymnasium as gym
from gc_ope.env.get_env import get_env
from gc_ope.env.utils import desired_goal_utils
from gc_ope.env.utils.my_reach import desired_goal_utils as reach_desired_goal_utils
from gc_ope.evaluate.utils.get_kde_estimator import get_kde_estimator_for_eval_res
from gc_ope.utils.load_config_with_hydra import load_config


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def test_kl_divergence_uniform_to_kde_integrate():
    print("In test kl divergence uniform to kde integrate:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )
    cfg.env.env_id = "MyReachSparse-v0"

    cfg.env.train_env.num_process = 2
    cfg.env.evaluation_env.num_process = 1
    cfg.env.callback_env.num_process = 1

    cfg.env.train_env.seed = 123
    
    env = get_env(env_cfg=cfg.env)
    print(env)

    for idx in range(10000, 1000001, 10000):
        # 加载评估数据集
        eval_res_file = f"checkpoints/myreach/easy/sac/seed_1/rl_model_{idx}_steps_eval_res_on_fixed.csv"

        # 初始化评估器
        kde_estimator = get_kde_estimator_for_eval_res(
            res_dir=eval_res_file,
            goal_keys_in_csv=["x", "y", "z"],
        )

        # for i in range(len(kde_estimator.eval_res_container.success_list)):
        #     kde_estimator.eval_res_container.success_list[i] = True

        eval_res_flag = any(kde_estimator.eval_res_container.success_list)
        if eval_res_flag:
            print(f"success num: {np.sum(kde_estimator.eval_res_container.success_list)}")
            dgs_for_eval_res, scaled_dgs_for_eval_res, dg_weights_for_eval_res, dg_densities_for_eval_res = kde_estimator.fit_evaluator()

            print(
                kde_estimator.kl_divergence_uniform_to_kde_integrate(
                    samples=desired_goal_utils.get_all_possible_dgs(env, step_x=0.02, step_y=0.02, step_z=0.02),
                    dV=0.02 * 0.02 * 0.02,
                    u_density=1.0 / desired_goal_utils.get_desired_goal_space_volumn(env),
                )
            )

            # a, b = kde_estimator.evaluate([np.array([100, 0, 0])], scale=True, return_density=True)
            # print(b)


if __name__ == "__main__":
    test_kl_divergence_uniform_to_kde_integrate()
