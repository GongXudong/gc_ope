"""
指定一个文件夹，对文件夹内的*.zip进行测试，将结果保存成与zip文件同名的csv
"""

from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import itertools
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from ray.util.multiprocessing import Pool

from stable_baselines3 import PPO, SAC

from gc_ope.env.get_env import get_pointmaze_env
from gc_ope.env.utils.my_maze.get_grid_inner_points import generate_all_possible_dgs


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def rollout(
    policy_dir_str: str, 
    algo_type: str,
    env_id: str,
    env_params: dict,
    target_goals_x: list,
    target_goals_y: list,
    gamma: float=0.995,
    seed: int=42,
):
    target_goals = pd.DataFrame({
        "x": target_goals_x,
        "y": target_goals_y,
    })

    env = get_pointmaze_env(
        env_id=env_id,
        seed=seed,
        **env_params
    )

    MAX_EPISODE_STEPS = env.get_wrapper_attr("_max_episode_steps")

    if algo_type == "ppo":
        algo_class = PPO
    elif algo_type =="sac":
        algo_class = SAC
    else:
        raise ValueError(f"Config Value Error: the value of 'algo_type' must be one of: ppo, sac!")

    algo = algo_class.load(
        policy_dir_str,
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }
    )
    algo.policy.set_training_mode(False)

    res_dict = {
        "x": [],
        "y": [],
        "length": [],
        "termination": [],
        "achieved x": [],
        "achieved y": [],
        "cumulative_rewards": [],
        "discounted_cumulative_rewards": [],
    }

    # 枚举任务
    for index, target in tqdm(target_goals.iterrows(), total=target_goals.shape[0]):
        # 为环境设置任务
        obs, info = env.reset()
        
        env.unwrapped.goal = np.array([target[0], target[1]])

        tmp_obs, tmp_info = env.unwrapped.point_env.reset()
        obs_dict = env.unwrapped._get_obs(tmp_obs)
        tmp_info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - env.unwrapped.goal) <= 0.45
        )

        obs, info = obs_dict, tmp_info

        # 
        terminate, truncated = False, False
        s_index = 0
        reward_list = []

        while not (terminate or truncated):
            action, _ = algo.predict(observation=obs, deterministic=True)
            obs, reward, terminate, truncated, info = env.step(action=action)

            reward_list.append(reward)
            s_index += 1
        
        reward_arr = np.array(reward_list)
        cumulative_rewards = reward_arr.sum()

        gammas = np.power(gamma, np.arange(len(reward_arr)))
        discounted_cumulative_rewards = np.sum(reward_arr * gammas)

        # 记录该episode信息
        res_dict["x"].append(target["x"])
        res_dict["y"].append(target["y"])
        res_dict["achieved x"].append(obs["achieved_goal"][0])
        res_dict["achieved y"].append(obs["achieved_goal"][1])
        res_dict["length"].append(s_index)
        res_dict["termination"].append("reach target" if s_index < MAX_EPISODE_STEPS else "timeout")
        res_dict["cumulative_rewards"].append(cumulative_rewards)
        res_dict["discounted_cumulative_rewards"].append(discounted_cumulative_rewards)

    return res_dict


@hydra.main(version_base=None, config_path="../configs/evaluate", config_name="config")
def evaluate_agent(cfg: DictConfig) -> None:
    
    # 1.确定需要测试的desired_goal集合
    # 1.1 从环境中读取desired_goal的范围
    env = get_pointmaze_env(
        env_id=cfg.env.env_id,
        maze_map=cfg.env.maze_map,
        reward_type=cfg.env.reward_type,
        continuing_task=cfg.env.continuing_task,
    )

    EPS = 1e-6
    
    # 1.2 根据配置文件指定的方法生成desired_goal集合
    if cfg.eval_cfg.dg_gen_method == "random":

        # 在desired_goal space内随机生成要测试的目标
        x_list = []
        y_list = []
        for i in range(cfg.eval_cfg.eval_dg_num):
            obs, info = env.reset()
            x_list.append(obs["desired_goal"][0])
            y_list.append(obs["desired_goal"][1])
        
        evaluation_goals = pd.DataFrame({
            "x": x_list,
            "y": y_list,
        })
    elif cfg.eval_cfg.dg_gen_method == "fixed":

        # 在desired_goal space内以固定间隔生成要测试的目标
        all_dgs = np.array(generate_all_possible_dgs(env=env.unwrapped, n=cfg.eval_cfg.n))
        
        evaluation_goals = pd.DataFrame({
            "x": all_dgs[:, 0],
            "y": all_dgs[:, 1],
        })
    else:
        raise ValueError(f"Config Value Error: the value of 'dg_gen_method' must be one of: randomm, fixed!")

    # 2.遍历文件夹中的所有ckpt
    folder = Path(PROJECT_ROOT_DIR / cfg.ckpt_dir)
    
    # 2.1 检查路径是否存在且为文件夹
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder}")
    
    # 2.2 遍历文件夹内所有以.zip结尾的文件，获取文件名
    with Pool(processes=cfg.process_num) as pool:

        for file in folder.glob("*.zip"):
            print(f"Begin to process: {str(file)}")

            # 设置要分成的份数
            n = cfg.process_num
            # 计算每份的行数
            chunk_size = len(evaluation_goals) // n
            # 分割DataFrame
            chunks = [evaluation_goals.iloc[i*chunk_size:(i+1)*chunk_size] for i in range(n)]
            # 如果不能完全均分，处理剩余的数据
            if len(evaluation_goals) % n != 0:
                # 将剩余的数据分配到最后一个chunk
                last_chunk = evaluation_goals.iloc[n*chunk_size:]
                chunks[-1] = pd.concat([chunks[-1], last_chunk])

            res = pool.starmap(
                rollout,
                [[
                    str(file),
                    cfg.algo_type,
                    cfg.env.env_id,
                    dict(
                        maze_map=cfg.env.maze_map,
                        reward_type=cfg.env.reward_type,
                        continuing_task=cfg.env.continuing_task,
                    ),
                    list(target.x),
                    list(target.y),
                    cfg.gamma,
                    cfg.seed,
                ] for target in chunks]
            )

            res_df = pd.concat([pd.DataFrame(tmp) for tmp in res])

            csv_res_name = file.parent / f"{file.stem}_{cfg.eval_res_csv_file_suffix}.csv"
            res_df.to_csv(csv_res_name, index=False)
            print(f"Finish processing {str(file)}, save res to {csv_res_name}")

    exit()

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo ppo --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name ppo_easy_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_end2end_1.csv


if __name__ == "__main__":
    evaluate_agent()
