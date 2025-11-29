from pathlib import Path
from stable_baselines3 import PPO, SAC
from omegaconf import DictConfig
from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.algorithm.utils.my_evaluate_policy import evaluate_policy_with_stat


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def test_my_evaluate_policy_stat_with_reach():

    algo = SAC.load(PROJECT_ROOT_DIR / "checkpoints/myreach/easy/sac/seed_1/best_model.zip")

    cfg = DictConfig(content={
        "env": dict(
            env_id = "MyReachSparse-v0",
            train_env = dict(
                num_process = 2,
                seed = 2,
            ),
            evaluation_env = dict(
                num_process = 1,
                seed = 5,
            ),
            callback_env = dict(
                num_process = 1,
                seed = 8,
            ),
        )
    })

    vec_env, _, _ = get_vec_env(cfg.env)

    mean_return, std_return, mean_success, stat_dict_arr = evaluate_policy_with_stat(
        model=algo.policy,
        env=vec_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(mean_return, std_return, mean_success, stat_dict_arr)


def test_my_evaluate_policy_stat_with_push():

    algo = PPO.load(PROJECT_ROOT_DIR / "checkpoints/my_push/ppo/seed_1/best_model.zip")

    cfg = DictConfig(content={
        "env": dict(
            env_id = "MyPushSparse-v0",
            train_env = dict(
                num_process = 2,
                seed = 2,
            ),
            evaluation_env = dict(
                num_process = 1,
                seed = 5,
            ),
            callback_env = dict(
                num_process = 1,
                seed = 8,
            ),
        )
    })

    vec_env, _, _ = get_vec_env(cfg.env)

    mean_return, std_return, mean_success, stat_dict_arr = evaluate_policy_with_stat(
        model=algo.policy,
        env=vec_env,
        n_eval_episodes=10,
        deterministic=True,
    )

    print(mean_return, std_return, mean_success, stat_dict_arr)


def test_my_evaluate_policy_stat_with_pointmaze():

    algo = SAC.load(PROJECT_ROOT_DIR / "checkpoints/my_pointmaze/sac/seed_1/best_model.zip")

    customized_map = [
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, "r", "g", "g", "g", 1, "g", "g", "g", "g", "g", 1],
        [1, "g", 1, 1, "g", 1, "g", 1, "g", 1, "g", 1],
        [1, "g", "g", "g", "g", "g", "g", 1, "g", "g", "g", 1],
        [1, "g", 1, 1, 1, 1, "g", 1, 1, 1, "g", 1],
        [1, "g", "g", 1, "g", 1, "g", "g", "g", "g", "g", 1],
        [1, 1, "g", 1, "g", 1, "g", 1, "g", 1, 1, 1],
        [1, "g", "g", 1, "g", "g", "g", 1, "g", "g", "g", 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    ]

    cfg = DictConfig(content={
        "env": dict(
            env_id = "MyPointMaze_Large_Diverse_G-v3",
            maze_map = customized_map,
            reward_type = "sparse",
            continuing_task = False,
            train_env = dict(
                num_process = 2,
                seed = 2,
            ),
            evaluation_env = dict(
                num_process = 1,
                seed = 5,
            ),
            callback_env = dict(
                num_process = 1,
                seed = 8,
            ),
        ),
    })

    vec_env, _, _ = get_vec_env(cfg.env)

    mean_return, std_return, mean_success, stat_dict_arr = evaluate_policy_with_stat(
        model=algo.policy,
        env=vec_env,
        n_eval_episodes=10,
        deterministic=True,
        success_key_in_info="success",
    )

    print(mean_return, std_return, mean_success, stat_dict_arr)


def test_my_evaluate_policy_stat_with_flycraft():

    algo = SAC.load(PROJECT_ROOT_DIR / "checkpoints/flycraft/easy/sac/seed_1/best_model.zip")

    cfg = DictConfig(content={
        "env": dict(
            env_id = "FlyCraft-v0",
            config_file = "configs/env_configs/flycraft/env_config_for_sac_easy.json",
            train_env = dict(
                num_process = 2,
                seed = 2,
                custom_config = dict(
                    debug_mode = False,
                    flag_str = "test",
                ),
            ),
            evaluation_env = dict(
                num_process = 1,
                seed = 5,
                custom_config = dict(
                    debug_mode = False,
                    flag_str = "test",
                ),
            ),
            callback_env = dict(
                num_process = 1,
                seed = 8,
                custom_config = dict(
                    debug_mode = False,
                    flag_str = "test",
                ),
            ),
        )
    })

    vec_env, _, _ = get_vec_env(cfg.env)

    mean_return, std_return, mean_success, stat_dict_arr = evaluate_policy_with_stat(
        model=algo.policy,
        env=vec_env,
        n_eval_episodes=10,
        deterministic=True,
        success_key_in_info="is_success",
    )

    print(mean_return, std_return, mean_success, stat_dict_arr)


if __name__ == "__main__":
    # test_my_evaluate_policy_stat_with_reach()
    # test_my_evaluate_policy_stat_with_push()
    # test_my_evaluate_policy_stat_with_pointmaze()
    test_my_evaluate_policy_stat_with_flycraft()
