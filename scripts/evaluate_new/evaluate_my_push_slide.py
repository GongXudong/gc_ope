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

from gc_ope.env.get_env import get_gym_env
from gc_ope.env.utils.my_push.desired_goal_utils import get_all_possible_dgs, get_random_dgs


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def rollout(
    policy_dir_str: str, 
    algo_type: str,
    env_id: str,
    target_goals_x: list,
    target_goals_y: list,
    target_goals_z: list,
    gamma: float=0.995,
    seed: int=42,
):
    target_goals = pd.DataFrame({
        "x": target_goals_x,
        "y": target_goals_y,
        "z": target_goals_z,
    })

    env = get_gym_env(
        env_id=env_id,
    )
    env.reset(seed=seed)

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
            "action_space": env.action_space
        }
    )
    algo.policy.set_training_mode(False)

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

    # 枚举任务
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
            "observation": np.concatenate([
                env.unwrapped.robot.get_obs().astype(np.float32),
                env.unwrapped.task.get_obs().astype(np.float32),
            ])
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


@hydra.main(version_base=None, config_path="../../configs/evaluate", config_name="config")
def evaluate_agent(cfg: DictConfig) -> None:
    
    # 1.遍历每个ckpt和对应的RB
    folder = Path(PROJECT_ROOT_DIR / cfg.ckpt_dir)
    
    # 1.1 检查路径是否存在且为文件夹
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder}")

    # 1.2 遍历
    for ckpt_file in folder.glob("*.zip"):
        # ckpt
        print(f"Begin to process: {str(ckpt_file)}")
        # DONE: 找这个ckpt对应的RB，用正则表达式找出ckpt_idx, 构造RB的文件名，rb_file
        epoch_idx = ckpt_file.stem.split("_")[2]
        # 检测epoch_idx是否为整数
        assert epoch_idx.isdigit(), f"提取的索引 '{epoch_idx}' 不是整数！请检查ckpt文件名格式，当前文件：{ckpt_file.name}"
        rb_file = ckpt_file.parent / f"rl_model_replay_buffer_{epoch_idx}_steps.pkl"
        if not rb_file.exists():
            raise FileNotFoundError(f"文件不存在: {rb_file}")

        # DONE: 加载RB中的数据，构造evaluation_goals，
        import pickle
        # 核心：读取pkl文件并加载RB数据
        try:
            # 使用with语句自动管理文件句柄，避免资源泄露
            with open(rb_file, 'rb') as f:
                # 加载pkl中的数据（rb_data即为整个回放缓冲区的原始数据）
                rb_data = pickle.load(f)

            # 1. 先获取achieved_goal
            observations = rb_data.observations
            achieved_goals = observations['achieved_goal']
            # 2. 整理为DataFrame格式
            achieved_goals = achieved_goals.squeeze(axis=1) if achieved_goals.ndim == 3 else achieved_goals
            evaluation_goals = pd.DataFrame({
                "x": achieved_goals[:, 0],
                "y": achieved_goals[:, 1],
                "z": achieved_goals[:, 2],
            })
            # 3. 随机取1000条数据
            if len(evaluation_goals) <= 1000:
                print(f"\n⚠️  数据量不足1000条（仅{len(evaluation_goals)}条），已取全部数据")
            else:
                evaluation_goals = evaluation_goals.sample(n=1000, random_state=42).reset_index(drop=True)
            
            print(evaluation_goals.head())
            input()

        except pickle.UnpicklingError as e:
            # 处理pickle加载错误（比如文件损坏、格式不兼容）
            raise ValueError(f"RB文件格式错误/损坏，无法解析pickle数据：{e}")
        except Exception as e:
            # 捕获其他意外错误（比如权限不足、文件读取失败）
            raise RuntimeError(f"加载RB文件时发生未知错误：{e}")

        # DONE: 在evaluation_goals集合上评估
        with Pool(processes=cfg.process_num) as pool:
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
                    str(ckpt_file),
                    cfg.algo_type,
                    cfg.env.env_id,
                    list(target.x),
                    list(target.y),
                    list(target.z),
                    cfg.gamma,
                    cfg.seed,
                ] for target in chunks]
            )

            res_df = pd.concat([pd.DataFrame(tmp) for tmp in res])

            csv_res_name = ckpt_file.parent / f"{ckpt_file.stem}_{cfg.eval_res_csv_file_suffix}.csv"
            res_df.to_csv(csv_res_name, index=False)
            print(f"Finish processing {str(ckpt_file)}, save res to {csv_res_name}")

    exit()

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo ppo --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name ppo_easy_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_end2end_1.csv


if __name__ == "__main__":
    evaluate_agent()
