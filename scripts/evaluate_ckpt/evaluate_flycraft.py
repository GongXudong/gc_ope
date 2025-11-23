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

from gc_ope.env.get_env import get_flycraft_env


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


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
    policy_dir_str: str, 
    algo_type: str,
    env_config_path: str,
    env_custom_config: dict,
    target_goals_v: list,
    target_goals_mu: list,
    target_goals_chi: list,
    gamma: float=0.995,
    seed: int=42,
):
    target_goals = pd.DataFrame({
        "v": target_goals_v,
        "mu": target_goals_mu,
        "chi": target_goals_chi,
    })

    env = get_flycraft_env(
        seed=seed,
        config_file=env_config_path,
        custom_config=env_custom_config,
    )
    env.reset(seed=seed)

    env.unwrapped.task.goal_sampler.use_fixed_goal = True

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


@hydra.main(version_base=None, config_path="../../configs/evaluate", config_name="config")
def evaluate_agent(cfg: DictConfig) -> None:
    
    # 1.确定需要测试的desired_goal集合
    # 1.1 从环境中读取desired_goal的范围
    env = get_flycraft_env(
        seed=0,
        config_file=str(PROJECT_ROOT_DIR / cfg.env.config_file),
        custom_config=OmegaConf.to_container(cfg.env.custom_config),
    )
    v_min = env.unwrapped.task.config["goal"]["v_min"]
    v_max = env.unwrapped.task.config["goal"]["v_max"]
    mu_min = env.unwrapped.task.config["goal"]["mu_min"]
    mu_max = env.unwrapped.task.config["goal"]["mu_max"]
    chi_min = env.unwrapped.task.config["goal"]["chi_min"]
    chi_max = env.unwrapped.task.config["goal"]["chi_max"]

    EPS = 1e-6
    
    # 1.2 根据配置文件指定的方法生成desired_goal集合
    if cfg.eval_cfg.dg_gen_method == "random":

        # 在desired_goal space内随机生成要测试的目标
        evaluation_goals = pd.DataFrame({
            "v": [np.random.random() * (v_max - v_min) + v_min for i in range(cfg.eval_cfg.eval_dg_num)],
            "mu": [np.random.random() * (mu_max - mu_min) + mu_min for i in range(cfg.eval_cfg.eval_dg_num)],
            "chi": [np.random.random() * (chi_max - chi_min) + chi_min for i in range(cfg.eval_cfg.eval_dg_num)],
        })
    elif cfg.eval_cfg.dg_gen_method == "fixed":

        # 在desired_goal space内以固定间隔生成要测试的目标
        v_list = np.arange(start=v_min, stop=v_max+EPS, step=cfg.eval_cfg.v_interval)
        mu_list = np.arange(start=mu_min, stop=mu_max+EPS, step=cfg.eval_cfg.mu_interval)
        chi_list = np.arange(start=chi_min, stop=chi_max+EPS, step=cfg.eval_cfg.chi_interval)

        combinations = np.array(list(itertools.product(v_list, mu_list, chi_list)))
        
        evaluation_goals = pd.DataFrame({
            "v": combinations[:, 0],
            "mu": combinations[:, 1],
            "chi": combinations[:, 2],
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
                    str(PROJECT_ROOT_DIR / cfg.env.config_file),
                    {},
                    list(target.v),
                    list(target.mu),
                    list(target.chi),
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
