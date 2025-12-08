
from pathlib import Path
from functools import partial
import pandas as pd
import numpy as np
import gymnasium as gym
from gc_ope.env.get_env import get_env
from gc_ope.env.utils import desired_goal_utils
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_kde import KDEEvaluator
from gc_ope.evaluate.utils.get_kde_estimator import get_kde_estimator_for_eval_res
from gc_ope.utils.load_config_with_hydra import load_config


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def test_kl_divergence_uniform_to_kde_integrate():
    print("In test kl divergence uniform to kde integrate:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )
    cfg.env = dict(
        env_id = "MyPointMaze_Large_Diverse_G-v3",
        continuing_task = False,
        reward_type = "sparse",
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", "g", "g", "g", 1, "g", "g", "g", "g", "g", 1],
            [1, "g", 1, 1, "g", 1, "g", 1, "g", 1, "g", 1],
            [1, "g", "g", "g", "g", "g", "g", 1, "g", "g", "g", 1],
            [1, "g", 1, 1, 1, 1, "g", 1, 1, 1, "g", 1],
            [1, "g", "g", 1, "g", 1, "g", "g", "g", "g", "g", 1],
            [1, 1, "g", 1, "g", 1, "g", 1, "g", 1, 1, 1],
            [1, "g", "g", 1, "g", "g", "g", 1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        train_env = dict(
            num_process=2,
            seed=3,
        ),
        evaluation_env = dict(
            num_process=2,
            seed=3,
        ),
        callback_env = dict(
            num_process=2,
            seed=3,
        ),
        env_str = "pointmaze",
    )
    
    env = get_env(env_cfg=cfg.env)
    print(env)

    for idx in range(10000, 1000001, 10000):
        # 加载评估数据集
        eval_res_file = f"checkpoints/my_pointmaze/sac/seed_1/rl_model_{idx}_steps_eval_res_on_fixed.csv"

        # 初始化评估器
        kde_estimator = get_kde_estimator_for_eval_res(
            res_dir=eval_res_file,
            goal_keys_in_csv=["x", "y"],
        )

        # for i in range(len(kde_estimator.eval_res_container.success_list)):
        #     kde_estimator.eval_res_container.success_list[i] = True

        eval_res_flag = any(kde_estimator.eval_res_container.success_list)
        if eval_res_flag:
            print(f"success num: {np.sum(kde_estimator.eval_res_container.success_list)}")
            dgs_for_eval_res, scaled_dgs_for_eval_res, dg_weights_for_eval_res, dg_densities_for_eval_res = kde_estimator.fit_evaluator()

            print(
                kde_estimator.kl_divergence_uniform_to_kde_integrate(
                    samples=desired_goal_utils.get_all_possible_dgs(env, n=5),
                    dV=np.pow(2 * env.unwrapped.position_noise_range * env.unwrapped.maze.maze_size_scaling / 5, 2),
                    u_density=1.0 / desired_goal_utils.get_desired_goal_space_volumn(env),
                )
            )

            # a, b = kde_estimator.evaluate([np.array([100, 0, 0])], scale=True, return_density=True)
            # print(b)


def test_kl_divergence_uniform_to_kde_integrate_with_mock_uniform_evaluation_data():
    """生成mock测试数据，desired goal按固定间隔在整个目标空间采样，假设全部是成功的（模拟p_ag是在目标空间上的均匀分布的情况）
    此时，检测KL(u, p)的值是否接近于0
    """
    
    print("In test kl divergence uniform to kde integrate with mock uniform evaluation data:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env = dict(
        env_id = "MyPointMaze_Large_Diverse_G-v3",
        continuing_task = False,
        reward_type = "sparse",
        maze_map = [
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", "g", "g", "g", 1, "g", "g", "g", "g", "g", 1],
            [1, "g", 1, 1, "g", 1, "g", 1, "g", 1, "g", 1],
            [1, "g", "g", "g", "g", "g", "g", 1, "g", "g", "g", 1],
            [1, "g", 1, 1, 1, 1, "g", 1, 1, 1, "g", 1],
            [1, "g", "g", 1, "g", 1, "g", "g", "g", "g", "g", 1],
            [1, 1, "g", 1, "g", 1, "g", 1, "g", 1, 1, 1],
            [1, "g", "g", 1, "g", "g", "g", 1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
        train_env = dict(
            num_process=2,
            seed=3,
        ),
        evaluation_env = dict(
            num_process=2,
            seed=3,
        ),
        callback_env = dict(
            num_process=2,
            seed=3,
        ),
        env_str = "pointmaze",
    )
    
    env = get_env(env_cfg=cfg.env)
    print(env)

    all_dgs = desired_goal_utils.get_all_possible_dgs(env, n=5)
    print(f"desired goal num: {len(all_dgs)}")

    kde_estimator = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs=dict(
            discounted_factor=0.9,
        ),
        kde_bandwidth=1.0,
        kde_kernel="gaussian",
    )

    all_dgs = all_dgs[:]

    kde_estimator.eval_res_container.add_batch(
        desired_goal_batch=all_dgs,
        success_batch=[True] * len(all_dgs),
        cumulative_reward_batch=[1.0] * len(all_dgs),
        discounted_cumulative_reward_batch=[1.0] * len(all_dgs),
        desired_goal_weight_batch=[1.0] * len(all_dgs),
    )

    dgs_for_eval_res, scaled_dgs_for_eval_res, dg_weights_for_eval_res, dg_densities_for_eval_res = kde_estimator.fit_evaluator()

    print("KL(u, p) =",
        kde_estimator.kl_divergence_uniform_to_kde_integrate(
            samples=desired_goal_utils.get_all_possible_dgs(env, n=5),
            dV=np.pow(2 * env.unwrapped.position_noise_range * env.unwrapped.maze.maze_size_scaling / 5, 2),
            u_density=1.0 / desired_goal_utils.get_desired_goal_space_volumn(env),
        )
    )

    # 前300个dg -> KL(u, p) = 14.77
    # 前500个dg -> KL(u, p) = 2.44
    # 前600个dg -> KL(u, p) = 0.38
    # 所有dg     -> KL(u, p) = 0.02


if __name__ == "__main__":
    test_kl_divergence_uniform_to_kde_integrate()
    # test_kl_divergence_uniform_to_kde_integrate_with_mock_uniform_evaluation_data()
