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
    
    import logging
    logger = logging.getLogger(__name__)

    # 0.实例化缩放器用于缩放状态
    env = get_flycraft_env(
        seed=0,
        config_file=str(PROJECT_ROOT_DIR / cfg.env.config_file),
        custom_config=OmegaConf.to_container(cfg.env.custom_config),
    )
    from gc_ope.env.utils.flycraft.my_wrappers import ScaledObservationWrapper
    scaled_obs_wrapper = ScaledObservationWrapper(env)

    # 1.遍历每个ckpt和对应的RB
    folder = Path(PROJECT_ROOT_DIR / cfg.ckpt_dir)
    logger.info(f"开始评估，ckpt文件夹: {folder}")
    
    # 1.1 检查路径是否存在且为文件夹
    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder}")

    # 1.2 遍历
    for ckpt_file in folder.glob("*.zip"):
        # ckpt
        # print(str(ckpt_file.stem))
        if "best_model" in str(ckpt_file.stem):
            logger.info(f"跳过文件（'best_model'）: {str(ckpt_file)}")
            continue
        logger.info(f"Begin to process: {str(ckpt_file)}")
        # DONE: 找这个ckpt对应的RB，用正则表达式找出ckpt_idx, 构造RB的文件名，rb_file
        ckpt_idx = ckpt_file.stem.split("_")[2]
        # 检测epoch_idx是否为整数
        assert ckpt_idx.isdigit(), f"提取的索引 '{ckpt_idx}' 不是整数！请检查ckpt文件名格式，当前文件：{ckpt_file.name}"
        rb_file = ckpt_file.parent / f"rl_model_replay_buffer_{ckpt_idx}_steps.pkl"
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

            # 1. 先获取achieved_goal并做逆缩放（从[0,1]恢复到物理量）
            observations = rb_data.observations
            achieved_goals = observations["achieved_goal"]

            # 回放缓冲中 achieved_goal 可能的维度： (step, env, goal_dim) 或 (n, goal_dim)
            # 将其展平为二维 (N, goal_dim) 再做 inverse_transform
            if achieved_goals.ndim == 3:
                achieved_flat = achieved_goals.reshape(-1, achieved_goals.shape[-1])
            elif achieved_goals.ndim == 2:
                achieved_flat = achieved_goals.reshape(-1, achieved_goals.shape[-1])
            else:
                # 兜底，尽量把最后一维作为 feature 维度
                achieved_flat = achieved_goals.reshape(-1, achieved_goals.shape[-1])
            # print(f"achieved_flat shape: {achieved_flat.shape}")
            # print(f"achieved_flat sample: {achieved_flat[:5]}")

            try:
                achieved_original = scaled_obs_wrapper.goal_scalar.inverse_transform(achieved_flat)
            except Exception as e:
                logger.warning(f"goal inverse_transform failed, using raw values: {e}")
                achieved_original = achieved_flat
            # print(f"achieved_original shape: {achieved_original.shape}")
            # print(f"achieved_original sample: {achieved_original[:5]}")
            # input("Press Enter to continue...")

            # 2. 整理为DataFrame格式
            evaluation_goals = pd.DataFrame({
                "v": achieved_original[:, 0],
                "mu": achieved_original[:, 1],
                "chi": achieved_original[:, 2],
            })
            # 3. 随机取1000条数据
            if len(evaluation_goals) <= 1000:
                logger.info(f"\n⚠️  数据量不足1000条（仅{len(evaluation_goals)}条），已取全部数据")
            else:
                evaluation_goals = evaluation_goals.sample(n=1000, random_state=42).reset_index(drop=True)
            
            logger.info(f"评估目标数量: {len(evaluation_goals)}")

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

            csv_res_name = ckpt_file.parent / f"{ckpt_file.stem}_{cfg.eval_res_csv_file_suffix}.csv"
            res_df.to_csv(csv_res_name, index=False)
            term_dict = res_df['termination'].value_counts().to_dict()
            reach_count = term_dict.get("reach target", 0)
            reach_ratio = reach_count / len(res_df)
            logger.info(f"Finish. reach_ratio: {reach_ratio}, ckpt_idx: {ckpt_idx}, save res to {csv_res_name}")

    exit()

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/ppo/easy/ppo_bc_config_10hz_128_128_easy_1.json --algo ppo --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name ppo_easy_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_1.csv

# python evaluate/sb3_rollout_parallel_for_checking_precision_termination_smooth.py --config-file-name configs/train/sac/easy_her_end_to_end_mode/sac_config_10hz_128_128_1.json --algo sac --eval-traj-num 1000 --process-num 10 --seed 123 --save-file-name sac_easy_her_end2end_1.csv


if __name__ == "__main__":
    evaluate_agent()
