ONLINE_EVAL = True  # 设置为 True 则只对评价策略 pi_e 进行在线评估，不进行 OPE 测试

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

    # 根据checkpoint路径确定环境
    # if "flycraft" in str(ckpt_path_1):
    #     env_cfg.env.env_id = "FlyCraft-v0" #NOTE: 只有env_id需要修改
    # elif "my_reach" in str(ckpt_path_1):
    #     env_cfg.env.env_id = "MyReachSparse-v0" #sac，sparse
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
    if "sac" in str(ckpt_path_1) or "her" in str(ckpt_path_1) or "omega" in str(ckpt_path_1):
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


    
    #STEP2：[重点耗时] 采样行为数据（已实现：数据采样和评价策略缓存分离）
    # 现在 collect_logged_dataset 只进行数据采样，评价策略缓存会在 build_ope_inputs 中自动计算
    # DONE：做日志存档
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
    if ONLINE_EVAL:
        online_mean, online_std = evaluate_agent(
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
        
    #STEP3： 构造 OPE 输入（已实现：FQE 训练和预测集成到 build_ope_inputs）
    # FQE 训练和预测过程已集成，支持 q_function_method 参数（当前仅支持 "fqe"）
    # 可以通过 fqe_train_kwargs 自定义训练参数，通过 fqe_kwargs 自定义 FQE 初始化参数
    loss_log = []
    def _logger(epoch: int, loss: float):
        if epoch % 1 == 0 or epoch == 1:
            logger.info(f"Epoch {epoch:04d} | FQE loss={loss:.3f}")
        loss_log.append((epoch, loss))

    inputs = build_ope_inputs(
        dataset=dataset,
        eval_algo=eval_algo,
        gamma=gamma,
        logger=logger,
        # fqe=None,  # 如果为 None，会自动创建并训练
        # q_function_method="fqe",  # 默认 "fqe"
        fqe_train_kwargs={
            "batch_size": 512,  # 增大batch size加速训练
            "n_epochs": 50,
            "shuffle": True,
            "logger": _logger,
            "gradient_clip": 1.0,  # 梯度裁剪防止梯度爆炸
            "target_update_freq": 4,  # 每4个batch更新一次目标网络，提高稳定性
            "num_workers": 0,  # CUDA环境下多进程可能导致初始化错误，设为0
            "pin_memory": True,  # 加速GPU数据传输
            "lr_schedule": "step",  # 使用学习率衰减
            "lr_decay_factor": 0.5,  # 学习率衰减因子
            "lr_decay_epochs": [100, 200],  # 在第100和200个epoch衰减学习率
        },
        fqe_kwargs={
            "lr": 1e-4,  # 降低初始学习率，提高稳定性
            "tau": 0.01,  # 增大tau，使目标网络更新更平滑
            "device": device,
            # TODO：改为从config中设置是否使用goal-conditioned mode
            # 设置以下两个参数：obs_state_dim, goal_dim，会启动goal-conditioned mode
            # "obs_state_dim": dataset.obs_dict[0]['observation'].shape[0],
            # "goal_dim": dataset.obs_dict[0]['desired_goal'].shape[0],
        },
    )
    
    #STEP4： 计算 OPE 估计值
    # 使用新的类API，支持kernel和self-normalize
    dm_estimator = DMEstimator(gamma=gamma)
    dm_res = dm_estimator.estimate(inputs, ci_method="bootstrap")

    # Kernel版本（使用纯相似度核函数，解决权重过小问题）
    # 
    for kernel_type in ["gaussian", "epanechnikov", "triangular", "cosine", "uniform"]:
        logger.info(f"Testing OPE estimators with kernel: {kernel_type}=========")
        tis_kernel = TISEstimator(gamma=gamma, use_kernel=True, kernel=kernel_type, bandwidth="auto")
        pdis_kernel = PDISEstimator(gamma=gamma, use_kernel=True, kernel=kernel_type, bandwidth="auto")
        dr_kernel = DREstimator(gamma=gamma, use_kernel=True, kernel=kernel_type, bandwidth="auto")

        # 自归一化版本（进一步稳定数值）
        sn_tis = SelfNormalizedTIS(gamma=gamma, use_kernel=True, kernel=kernel_type, bandwidth="auto")
        sn_pdis = SelfNormalizedPDIS(gamma=gamma, use_kernel=True, kernel=kernel_type, bandwidth="auto")
        sn_dr = SelfNormalizedDR(gamma=gamma, use_kernel=True, kernel=kernel_type, bandwidth="auto")

        tis_kernel_res = tis_kernel.estimate(inputs, ci_method="bootstrap")
        pdis_kernel_res = pdis_kernel.estimate(inputs, ci_method="bootstrap")
        dr_kernel_res = dr_kernel.estimate(inputs, ci_method="bootstrap")

        sn_tis_res = sn_tis.estimate(inputs, ci_method="bootstrap")
        sn_pdis_res = sn_pdis.estimate(inputs, ci_method="bootstrap")
        sn_dr_res = sn_dr.estimate(inputs, ci_method="bootstrap")

        logger.info(f"ONLINE (≈ground truth): {online_mean} ± {online_std}")
        logger.info(f"DM: {dm_res}")
        logger.info(f"TIS: {tis_kernel_res}")
        logger.info(f"PDIS: {pdis_kernel_res}")
        logger.info(f"DR: {dr_kernel_res}")
        logger.info(f"SN-TIS: {sn_tis_res}")
        logger.info(f"SN-PDIS: {sn_pdis_res}")
        logger.info(f"SN-DR: {sn_dr_res}")
    '''
    /home/maxine/ai4robot/gc_ope/src/gc_ope/algorithm/ope/estimators.py:184: RuntimeWarning: overflow encountered in exp
      weight = np.exp((logp_e - logp_b).sum())
    /home/maxine/ai4robot/gc_ope/.venv/lib/python3.12/site-packages/numpy/lib/_function_base_impl.py:4671: RuntimeWarning: invalid value encountered in subtract
      diff_b_a = subtract(b, a)
    /home/maxine/ai4robot/gc_ope/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:57: RuntimeWarning: overflow encountered in accumulate
      return bound(*args, **kwds)
    /home/maxine/ai4robot/gc_ope/src/gc_ope/algorithm/ope/estimators.py:258: RuntimeWarning: invalid value encountered in add
      term = w_step * (r - q_sa) + w_prev * v_eval
    /home/maxine/ai4robot/gc_ope/src/gc_ope/algorithm/ope/estimators.py:258: RuntimeWarning: overflow encountered in multiply
      term = w_step * (r - q_sa) + w_prev * v_eval
    /home/maxine/ai4robot/gc_ope/src/gc_ope/algorithm/ope/estimators.py:253: RuntimeWarning: overflow encountered in exp
      ratios = np.exp(logp_e - logp_b)
    /home/maxine/ai4robot/gc_ope/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:57: RuntimeWarning: invalid value encountered in reduce
      return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
    DM (step-wise): EstimateResult(mean=-55.72665023803711, ci_lower=-55.892295837402344, ci_upper=-55.5826530456543)
    TIS: EstimateResult(mean=-inf, ci_lower=nan, ci_upper=nan)
    DR: EstimateResult(mean=nan, ci_lower=nan, ci_upper=nan)
    '''


if __name__ == "__main__":
    test_ope()
