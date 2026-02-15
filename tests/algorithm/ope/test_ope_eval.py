from pathlib import Path
import numpy as np
import torch as th
import os
import pickle

PROJECT_ROOT_DIR = Path(__file__).absolute().parent.parent.parent.parent
print(PROJECT_ROOT_DIR)

import logging
import time
from datetime import datetime

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy

import hydra
from omegaconf import DictConfig

from gc_ope.env.get_env import get_env
from gc_ope.utils.load_config_with_hydra import load_config

from gc_ope.algorithm.ope.logged_dataset import collect_logged_dataset, compute_eval_policy_cache
from gc_ope.algorithm.ope.fqe import FQETrainer
from gc_ope.algorithm.ope.ope_input import build_ope_inputs
from gc_ope.algorithm.ope.estimators import (
    DMEstimator,
    TISEstimator,
    PDISEstimator,
    DREstimator,
    SelfNormalizedTIS,
    SelfNormalizedPDIS,
    SelfNormalizedDR,
)
from gc_ope.algorithm.ope.evaluation.evaluate_on_bgoal import evaluate_agent

@hydra.main(version_base=None, config_path=str("../../../configs/ope"), config_name="config")
def test_ope(ope_cfg: DictConfig) -> None:
    """
    OPE 测试主函数
    使用 Hydra 加载 ope 配置，并在函数内部加载 train 配置作为 env_cfg
    """
    
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    env_cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    # DONE: 改为从"../../../configs/ope/config.yaml"中读取OPE评估的环境和策略参数、行为策略采样参数、在线评估参数、FQE参数
    ckpt_path_1 = PROJECT_ROOT_DIR / f"checkpoints/{ope_cfg.env}/{ope_cfg.algo}/seed_1/best_model"
    ckpt_path_2 = PROJECT_ROOT_DIR / f"checkpoints/{ope_cfg.env}/{ope_cfg.algo}/seed_2/best_model"

    env_cfg.env.env_id = ope_cfg.env_id

    data_save_path = PROJECT_ROOT_DIR / f"{ope_cfg.data_collection.save_root}/{ope_cfg.env}_{ope_cfg.algo}_{ope_cfg.data_collection.num_episodes}eps.pkl"
    if not os.path.exists(data_save_path.parent):
        os.makedirs(data_save_path.parent)

    # 配置基础日志
    logging.basicConfig(
        level=logging.INFO,  # 设置日志级别
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=Path(__file__).absolute().parent / f"{datetime.now().strftime('%Y-%m-%d_%H-%M')}_{ope_cfg.env}_{ope_cfg.algo}_{ope_cfg.data_collection.num_episodes}eps.log"
    )
    # 获取logger
    logger = logging.getLogger(__name__)
    logger.info(f"Using device: {device}")

    #================
    #STEP1： 准备环境与策略
    env = get_env(env_cfg.env)

    # 根据checkpoint路径确定策略类型
    # 注意：HER 实际上是 SAC + HerReplayBuffer，所以使用 SAC.load() 来加载
    # HER 模型需要传递 env 参数，因为 HerReplayBuffer 需要环境来初始化
    if "sac" in str(ckpt_path_1) or "her" in str(ckpt_path_1):
        # 对于 HER 模型，需要传递 env 参数；对于普通 SAC，传递 env 也是安全的
        behavior_algo = SAC.load(ckpt_path_1, env=env)
        eval_algo = SAC.load(ckpt_path_2, env=env)
    elif "ppo" in str(ckpt_path_1):
        behavior_algo = PPO.load(ckpt_path_1, env=env)
        eval_algo = PPO.load(ckpt_path_2, env=env)
    else:
        raise ValueError(f"Unsupported algorithm: {ckpt_path_1}")

    gamma = float(getattr(eval_algo, "gamma", 0.99))
    logger.info(f"gamma= {gamma}")

    #STEP2： 采样行为数据（已实现：数据采样和评价策略缓存分离）
    # 现在 collect_logged_dataset 只进行数据采样，评价策略缓存会在 build_ope_inputs 中自动计算
    # TODO：做日志存档
    n_episodes = ope_cfg.data_collection.num_episodes
    max_steps = ope_cfg.data_collection.max_steps

    if not os.path.exists(data_save_path):
        logger.info("Collecting dataset")
        dataset = collect_logged_dataset(
            env=env,
            behavior_algo=behavior_algo,
            n_episodes=n_episodes,
            max_steps=max_steps,
        )
        with open(data_save_path, "wb") as f:
            pickle.dump(dataset, f)
        logger.info(f"Dataset collected & saved to {data_save_path} successfully")
    else:
        logger.info("Loading dataset from file")
        with open(data_save_path, "rb") as f:    
            dataset = pickle.load(f)
        logger.info("Dataset loaded successfully")
    
    #STEP: 在行为策略的目标上进行在线评估
    evaluate_agent(
        eval_pi_ckpt_path=ckpt_path_2,
        data_behavior=dataset,
        behavior_pi_name=f"{ope_cfg.algo}_seed_1",
        algo_type=ope_cfg.algo,
        env_cfg=env_cfg.env,  # 传递环境配置，用于flycraft环境自动提取config_file和custom_config
        process_num=ope_cfg.online_evaluation.process_num,
        gamma=gamma,
        seed=ope_cfg.online_evaluation.seed,
        logger=logger,
    )

if __name__ == "__main__":
    test_ope()
