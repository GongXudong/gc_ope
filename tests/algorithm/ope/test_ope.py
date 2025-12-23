from pathlib import Path
import numpy as np
import torch as th
import os
import pickle

from stable_baselines3 import SAC, PPO
from stable_baselines3.common.evaluation import evaluate_policy

from gc_ope.env.get_env import get_env
from gc_ope.utils.load_config_with_hydra import load_config

from gc_ope.algorithm.ope.logged_dataset import collect_logged_dataset, compute_eval_policy_cache
from gc_ope.algorithm.ope.fqe import FQETrainer
from gc_ope.algorithm.ope.ope_input import build_ope_inputs
from gc_ope.algorithm.ope.estimators import (
    dm_estimate, tis_estimate, dr_estimate,
    dm_compute_trajectory_values, tis_compute_trajectory_values, dr_compute_trajectory_values
)

PROJECT_ROOT_DIR = Path().absolute().parent.parent.parent
print(PROJECT_ROOT_DIR)

device = 'cuda' if th.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

#================
#STEP1： 准备环境与策略
env_cfg = load_config(
    config_path="../../../configs/train",
    config_name="config",
)

# TODO: 改为从"../../../configs/ope/config.yaml"中读取OPE评估的环境和策略参数、行为策略采样参数、在线评估参数、FQE参数
# TODO: 需要对应修改env_cfg.algo，env_cfg.env
ckpt_path_1 = PROJECT_ROOT_DIR / "checkpoints/flycraft/sac/seed_1/best_model"
ckpt_path_2 = PROJECT_ROOT_DIR / "checkpoints/flycraft/sac/seed_2/best_model"

# 根据checkpoint路径确定环境
if "flycraft" in str(ckpt_path_1):
    env_cfg.env.env_id = "FlyCraft-v0"
elif "flycraft" in str(ckpt_path_2):
    env_cfg.env.env_id = "FlyCraft-v0"

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
print("gamma=", gamma)

#STEP2： 采样行为数据（已实现：数据采样和评价策略缓存分离）
# 现在 collect_logged_dataset 只进行数据采样，评价策略缓存会在 build_ope_inputs 中自动计算
# TODO：做日志存档
n_episodes = 10
max_steps = 400

if not os.path.exists("dataset.pkl"):
    print("Collecting dataset")
    dataset = collect_logged_dataset(
        env=env,
        behavior_algo=behavior_algo,
        n_episodes=n_episodes,
        max_steps=max_steps,
    )
    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    print("Dataset collected & saved successfully")
else:
    print("Loading dataset from file")
    with open("dataset.pkl", "rb") as f:    
        dataset = pickle.load(f)
    print("Dataset loaded successfully")

#STEP3： 构造 OPE 输入（已实现：FQE 训练和预测集成到 build_ope_inputs）
# FQE 训练和预测过程已集成，支持 q_function_method 参数（当前仅支持 "fqe"）
# 可以通过 fqe_train_kwargs 自定义训练参数，通过 fqe_kwargs 自定义 FQE 初始化参数
loss_log = []
def _logger(epoch: int, loss: float):
    if epoch % 1 == 0 or epoch == 1:
        print(f"Epoch {epoch:04d} | FQE loss={loss:.3f}")
    loss_log.append((epoch, loss))

inputs = build_ope_inputs(
    dataset=dataset,
    eval_algo=eval_algo,
    gamma=gamma,
    # fqe=None,  # 如果为 None，会自动创建并训练
    # q_function_method="fqe",  # 默认 "fqe"
    fqe_train_kwargs={
        "batch_size": 512,  # 增大batch size加速训练
        "n_epochs": 100,
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
        "obs_state_dim": dataset.obs_dict[0]['observation'].shape[0],
        "goal_dim": dataset.obs_dict[0]['desired_goal'].shape[0],
    },
)
'''
Epoch 0001 | FQE loss=0.874
Epoch 0002 | FQE loss=0.869
Epoch 0003 | FQE loss=0.882
Epoch 0004 | FQE loss=0.945
Epoch 0005 | FQE loss=1.088
Epoch 0006 | FQE loss=1.336
Epoch 0007 | FQE loss=1.747
Epoch 0008 | FQE loss=2.365
Epoch 0009 | FQE loss=3.259
Epoch 0010 | FQE loss=4.380 DM (step-wise): EstimateResult(mean=-14.848934173583984
Epoch 0011 | FQE loss=5.747
Epoch 0012 | FQE loss=7.545
Epoch 0013 | FQE loss=9.698
Epoch 0014 | FQE loss=12.217
Epoch 0015 | FQE loss=16.638
Epoch 0016 | FQE loss=20.176
Epoch 0017 | FQE loss=23.389
Epoch 0018 | FQE loss=28.612
Epoch 0019 | FQE loss=37.594
Epoch 0020 | FQE loss=42.861 DM (step-wise): EstimateResult(mean=-55.72665023803711
Epoch 0021 | FQE loss=51.796
Epoch 0022 | FQE loss=59.927
Epoch 0023 | FQE loss=69.440
Epoch 0024 | FQE loss=86.010
Epoch 0025 | FQE loss=95.738
Epoch 0026 | FQE loss=106.316
Epoch 0027 | FQE loss=123.766
Epoch 0028 | FQE loss=138.375
Epoch 0029 | FQE loss=157.954
Epoch 0030 | FQE loss=188.884 DM (step-wise): EstimateResult(mean=-120.50849914550781
'''

#STEP4： 计算 OPE 估计值
# 支持 ci_method 参数："bootstrap"（默认）、"normal"、"t_test"
dm_all = dm_estimate(inputs, initial_only=False, ci_method="bootstrap")
# dm_init = dm_estimate(inputs, initial_only=True, ci_method="bootstrap")
tis_res = tis_estimate(inputs, ci_method="bootstrap")
dr_res = dr_estimate(inputs, ci_method="bootstrap")

print("DM (step-wise):", dm_all)
# print("DM (initial-state):", dm_init)
print("TIS:", tis_res) #TODO: 解决TIS计算报错的问题
print("DR:", dr_res)
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
/home/maxine/ai4robot/gc_ope/.venv/lib/python3.12/site-packages/numpy/_core/fromnumeric.py:86: RuntimeWarning: invalid value encountered in reduce
  return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
DM (step-wise): EstimateResult(mean=-55.72665023803711, ci_lower=-55.892295837402344, ci_upper=-55.5826530456543)
TIS: EstimateResult(mean=-inf, ci_lower=nan, ci_upper=nan)
DR: EstimateResult(mean=nan, ci_lower=nan, ci_upper=nan)
'''

# 在线评估真实回报（可选，耗时）
if False:
    # mean_r_b, std_r_b = evaluate_policy(behavior_algo, env, n_eval_episodes=5, deterministic=True)
    mean_r_e, std_r_e = evaluate_policy(eval_algo, env, n_eval_episodes=1000, deterministic=True)
    # print(f"behavior return: {mean_r_b:.2f} ± {std_r_b:.2f}")
    print(f"eval return:     {mean_r_e:.2f} ± {std_r_e:.2f}")
    '''
    eval return:     -87.16 ± 50.82
    '''