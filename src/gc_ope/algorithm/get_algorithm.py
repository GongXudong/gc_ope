
from copy import deepcopy
from omegaconf import DictConfig, OmegaConf
import torch as th
from stable_baselines3 import PPO, SAC, HerReplayBuffer
from stable_baselines3.common.buffers import DictReplayBuffer
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from gc_ope.algorithm.utils.my_schedule import linear_schedule


def get_algo(algo_cfg: DictConfig, env: VecEnv) -> BaseAlgorithm:
    
    if algo_cfg.type == "ppo":
        return get_ppo(algo_cfg, env)
    elif algo_cfg.type in ["sac", "her"]:
        return get_sac(algo_cfg, env)
    else:
        raise ValueError(f"Can not get algorithm for {algo_cfg.type}!")

def get_ppo(algo_cfg: DictConfig, env: VecEnv) -> OnPolicyAlgorithm:
    return PPO(
        policy=algo_cfg.policy,
        env=env,
        seed=algo_cfg.seed,
        batch_size=int(algo_cfg.batch_size),
        gamma=algo_cfg.gamma,
        n_steps=algo_cfg.n_steps,  # 采样时每个环境采样的step数
        n_epochs=algo_cfg.n_epochs,  # 采样的数据在训练中重复使用的次数
        ent_coef=algo_cfg.ent_coef,
        policy_kwargs={
            "net_arch": {
                "pi": OmegaConf.to_container(algo_cfg.net_arch),
                "vf": deepcopy(OmegaConf.to_container(algo_cfg.net_arch)),
            }
        },
        use_sde=algo_cfg.use_sde,  # 使用state dependant exploration,
        normalize_advantage=algo_cfg.normalize_advantage,
        learning_rate=linear_schedule(algo_cfg.learning_rate) if algo_cfg.learning_rate_decay else algo_cfg.learning_rate,
        device=algo_cfg.device,
    )

def get_sac(algo_cfg: DictConfig, env: VecEnv) -> OffPolicyAlgorithm:

    return SAC(
        policy=algo_cfg.policy,
        env=env,
        seed=algo_cfg.seed,
        replay_buffer_class=HerReplayBuffer if algo_cfg.use_her else DictReplayBuffer,
        replay_buffer_kwargs=dict(
            n_sampled_goal=algo_cfg.her_n_sampled_goal,
            goal_selection_strategy=algo_cfg.her_goal_selection_strategy,
        ) if algo_cfg.use_her else None,
        verbose=1,
        buffer_size=int(algo_cfg.buffer_size),
        learning_starts=int(algo_cfg.learning_starts),
        gradient_steps=int(algo_cfg.gradient_steps),
        learning_rate=linear_schedule(algo_cfg.learning_rate) if algo_cfg.learning_rate_decay else algo_cfg.learning_rate,
        gamma=algo_cfg.gamma,
        batch_size=int(algo_cfg.batch_size),
        policy_kwargs=dict(
            net_arch=OmegaConf.to_container(algo_cfg.net_arch),
            activation_fn=th.nn.Tanh
        ),
        device=algo_cfg.device,
    )

