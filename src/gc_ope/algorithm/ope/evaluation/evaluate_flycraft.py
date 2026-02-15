"""
指定评估策略（eval_pi），在行为策略采样数据（data_behavior）中各episode的目标（desired_goal）上进行rollout，
记录每个episode的结果，并输出平均回报（return）。
"""

from pathlib import Path
from copy import deepcopy
import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from ray.util.multiprocessing import Pool

from omegaconf import DictConfig, OmegaConf
from typing import Optional, Union

from stable_baselines3 import PPO, SAC
from stable_baselines3.common import type_aliases

from gc_ope.env.get_env import get_env
from gc_ope.env.get_env import get_flycraft_env
from gc_ope.algorithm.ope.logged_dataset import LoggedDataset


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent.parent


def termination_shotcut(termination_str: str):
    if termination_str.find("reach_target") != -1:
        return "reach target"
    if termination_str.find("timeout") != -1:
        return "timeout"
    if termination_str.find("move_away") != -1:
        return "continuous move away"
    if termination_str.find("roll") != -1:
        return "continuous roll"
    if termination_str.find("crash") != -1:
        return "crash"
    if termination_str.find("extreme") != -1:
        return "extreme state"
    if termination_str.find("negative") != -1:
        return "negative overload"

def rollout(
    policy_path_str: str,
    algo_type: str,
    env_cfg: Optional[DictConfig],
    target_goals_v: list,
    target_goals_mu: list,
    target_goals_chi: list,
    gamma: float=0.995,
    seed: int=42,
):
    """
    Runs the policy on target goals and outputs the results per episode.
    """
    target_goals = pd.DataFrame({
        "v": target_goals_v,
        "mu": target_goals_mu,
        "chi": target_goals_chi,
    })

    env = get_env(env_cfg)
    env.reset(seed=seed)

    env.unwrapped.task.goal_sampler.use_fixed_goal = True

    if algo_type == "ppo":
        algo_class = PPO
    elif algo_type =="sac" or "her" or "omega":
        algo_class = SAC
    else:
        raise ValueError(f"Config Value Error: the value of 'algo_type' must be one of: ppo, sac!")

    algo = algo_class.load(
        policy_path_str, 
        env=env,
        custom_objects={
            "observation_space": env.observation_space,
            "action_space": env.action_space
        }
    )
    algo.policy.set_training_mode(False)

    res_dict = {
        "v": [],
        "mu": [],
        "chi": [],
        "length": [],
        "termination": [],
        "achieved v": [],
        "achieved mu": [],
        "achieved chi": [],
        "cumulative_rewards": [],
        "discounted_cumulative_rewards": [],
    }

    # 枚举任务
    for index, target in tqdm(target_goals.iterrows(), total=target_goals.shape[0]):
        # 为环境设置任务
        # target_v, target_mu, target_chi, expert_length = target["v"], target["mu"], target["chi"], target["length"]
        env.unwrapped.task.goal_sampler.goal_v = target["v"]
        env.unwrapped.task.goal_sampler.goal_mu = target["mu"]
        env.unwrapped.task.goal_sampler.goal_chi = target["chi"]

        # 采样一个episode
        obs, info = env.reset()

        terminate = False
        s_index = 0
        reward_list = []

        while not terminate:
            action, _ = algo.predict(observation=obs, deterministic=True)
            obs, reward, terminate, truncated, info = env.step(action=action)

            reward_list.append(reward)
            s_index += 1
        
        reward_arr = np.array(reward_list)
        cumulative_rewards = reward_arr.sum()

        gammas = np.power(gamma, np.arange(len(reward_arr)))
        discounted_cumulative_rewards = np.sum(reward_arr * gammas)

        # 记录该episode信息
        res_dict["v"].append(target["v"])
        res_dict["mu"].append(target["mu"])
        res_dict["chi"].append(target["chi"])
        res_dict["achieved v"].append(deepcopy(info["plane_next_state"]["v"]))
        res_dict["achieved mu"].append(deepcopy(info["plane_next_state"]["mu"]))
        res_dict["achieved chi"].append(deepcopy(info["plane_next_state"]["chi"]))
        res_dict["length"].append(s_index)
        res_dict["termination"].append(termination_shotcut(info["termination"]))
        res_dict["cumulative_rewards"].append(cumulative_rewards)
        res_dict["discounted_cumulative_rewards"].append(discounted_cumulative_rewards)
        
    return res_dict


def evaluate_agent(eval_pi_ckpt_path: "type_aliases.PolicyPredictor",
                   data_behavior: LoggedDataset, 
                   behavior_pi_name: str,
                   algo_type: str, 
                   env_cfg: Optional[DictConfig] = None,
                   process_num: int=4, 
                   gamma: float=0.995, 
                   seed: int=42,
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
        "v": [data_behavior.obs_dict[idx]["desired_goal"][0] for idx in first_step_indices],
        "mu": [data_behavior.obs_dict[idx]["desired_goal"][1] for idx in first_step_indices],
        "chi": [data_behavior.obs_dict[idx]["desired_goal"][2] for idx in first_step_indices],
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
                env_cfg,
                list(target.v),
                list(target.mu),
                list(target.chi),
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
