"""
指定评估策略（eval_pi），在行为策略采样数据（data_behavior）中各episode的目标（desired_goal）上进行rollout，
记录每个episode的结果，并输出平均回报（return）。
"""

from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import itertools
import logging
from tqdm import tqdm
import hydra
from omegaconf import OmegaConf, DictConfig
from ray.util.multiprocessing import Pool

from stable_baselines3 import PPO, SAC
from stable_baselines3.common import type_aliases

from gc_ope.env.get_env import get_gym_env
from gc_ope.env.utils.my_reach.desired_goal_utils import get_all_possible_dgs, get_random_dgs
from gc_ope.algorithm.ope.logged_dataset import LoggedDataset


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent

def rollout(
    policy_path_str: str, 
    algo_type: str,
    env_id: str,
    target_goals_x: list,
    target_goals_y: list,
    target_goals_z: list,
    gamma: float=0.995,
    seed: int=42,
):
    """
    Runs the policy on target goals and outputs the results per episode.
    """
    ## 准备数据
    target_goals = pd.DataFrame({
        "x": target_goals_x,
        "y": target_goals_y,
        "z": target_goals_z,
    })
    
    res_dict = {
        "x": [],
        "y": [],
        "z": [],
        "length": [],
        "termination": [],
        "achieved x": [],
        "achieved y": [],
        "achieved z": [],
        "cumulative_rewards": [],
        "discounted_cumulative_rewards": [],
    }

    ## 创建环境
    env = get_gym_env(
        env_id=env_id,
    )
    env.reset(seed=seed)
    MAX_EPISODE_STEPS = env.get_wrapper_attr("_max_episode_steps")

    ## 创建策略
    if algo_type == "ppo":
        algo_class = PPO
    elif algo_type =="sac" or "her" or "omega":
        algo_class = SAC
    else:
        raise ValueError(f"Config Value Error: the value of 'algo_type' must be one of: ppo, sac, her!")

    algo = algo_class.load(
        policy_path_str, 
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
    )
    algo.policy.set_training_mode(False)

    ## 枚举任务
    for index, target in tqdm(target_goals.iterrows(), total=target_goals.shape[0]):
        # 为环境设置任务
        # target_x, target_y, target_z = target["x"], target["y"], target["z"]
        obs, info = env.reset()
        tmp_episode_goal = np.array([target["x"], target["y"], target["z"]])
        env.unwrapped.task.goal = tmp_episode_goal
        env.unwrapped.task.sim.set_base_pose("target", tmp_episode_goal, np.array([0.0, 0.0, 0.0, 1.0]))

        obs = {
            "achieved_goal": env.unwrapped.task.get_achieved_goal().astype(np.float32),
            "desired_goal": env.unwrapped.task.get_goal().astype(np.float32),
            "observation": env.unwrapped.robot.get_obs().astype(np.float32)
        }

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
        res_dict["z"].append(target["z"])
        res_dict["achieved x"].append(obs["achieved_goal"][0])
        res_dict["achieved y"].append(obs["achieved_goal"][1])
        res_dict["achieved z"].append(obs["achieved_goal"][2])
        res_dict["length"].append(s_index)
        res_dict["termination"].append("reach target" if s_index < MAX_EPISODE_STEPS else "timeout")
        res_dict["cumulative_rewards"].append(cumulative_rewards)
        res_dict["discounted_cumulative_rewards"].append(discounted_cumulative_rewards)

    return res_dict

def evaluate_agent(eval_pi_ckpt_path: "type_aliases.PolicyPredictor",
                   data_behavior: LoggedDataset, 
                   behavior_pi_name: str,
                   algo_type: str, env_id: str,
                   process_num: int=4, 
                   gamma: float=0.995, seed: int=42,
                   logger=None) -> None:
    """
    output the average return per episode (sum of undiscounted rewards)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    # 1. 从data_behavior中获取所有的desired_goal，每个episode只取第一个desired_goal（因为一个episode只有一个goal）
    # 找到所有step_index == 0的索引，这些是每个episode的第一步
    first_step_indices = np.where(data_behavior.step_index == 0)[0]
    evaluation_goals = pd.DataFrame({
        "x": [data_behavior.obs_dict[idx]["desired_goal"][0] for idx in first_step_indices],
        "y": [data_behavior.obs_dict[idx]["desired_goal"][1] for idx in first_step_indices],
        "z": [data_behavior.obs_dict[idx]["desired_goal"][2] for idx in first_step_indices],
    })

    # 2. 在特定goal上进行rollout
    with Pool(processes=process_num) as pool:

        logger.info(f"Begin to process: {eval_pi_ckpt_path}")

        # 设置要分成的份数
        n = process_num
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
                eval_pi_ckpt_path,
                algo_type,
                env_id,
                list(target.x),
                list(target.y),
                list(target.z),
                gamma,
                seed,
            ] for target in chunks]
        )

        res_df = pd.concat([pd.DataFrame(tmp) for tmp in res])

        csv_res_name = eval_pi_ckpt_path.parent / f"{eval_pi_ckpt_path.stem}_eval-on_{behavior_pi_name}.csv"
        res_df.to_csv(csv_res_name, index=False)
        logger.info(f"Finish processing {eval_pi_ckpt_path}, save res to {csv_res_name}")

    mean_return = res_df["cumulative_rewards"].mean()
    std_return = res_df["cumulative_rewards"].std()
    logger.info(f"Eval on {behavior_pi_name}'s goals: return = {mean_return} ± {std_return}")
    return mean_return, std_return

if __name__ == "__main__":
    evaluate_agent()
