from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv
import flycraft
from flycraft.utils_common.dict_utils import update_nested_dict

from gc_ope.env.utils.flycraft.vec_env_helper import get_vec_env as get_flycraft_vec_env


gym.register_envs(flycraft)
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def get_vec_env(env_cfg: DictConfig) -> tuple[VecEnv, VecEnv, VecEnv]:
    """读取配置文件，根据env_id自动返回对应环境的用于训练的vec_env，用于评估的vec_env，用于callback的vec_env

    Args:
        env_cfg (DictConfig): 环境配置文件

    Raises:
        ValueError: 当env_id无法识别时，抛出异常

    Returns:
        tuple[VecEnv, VecEnv, VecEnv]: 用于训练的vec_env，用于评估的vec_env，用于callback的vec_env
    """
    if env_cfg.env_id.startswith("FlyCraft"):
        return get_flycraft_envs(env_cfg)
    else:
        raise ValueError(f"Can not get vec_env for env: {env_cfg.env_id}!")


def get_flycraft_envs(env_cfg: DictConfig) -> tuple[VecEnv, VecEnv, VecEnv]:
    """读取配置文件，返回用于训练的env，用于评估的env，用于callback的env

    Args:
        env_cfg (DictConfig): 环境配置文件

    Returns:
        tuple[VecEnv, VecEnv, VecEnv]: 用于训练的vec_env，用于评估的vec_env，用于callback的vec_env
    """
    env_config_dict_in_training = {
        "num_process": env_cfg.train_env.num_process, 
        "seed": env_cfg.train_env.seed,
        "config_file": str(PROJECT_ROOT_DIR / env_cfg.config_file),
        "custom_config": OmegaConf.to_container(env_cfg.train_env.custom_config),  # use OmegaConf.to_container to safe convert DictConfig to dict
    }

    env_config_dict_in_eval = deepcopy(env_config_dict_in_training)
    update_nested_dict(env_config_dict_in_eval, {
        "num_process": env_cfg.evaluation_env.num_process,
        "seed": env_cfg.evaluation_env.seed,
        "custom_config": OmegaConf.to_container(env_cfg.evaluation_env.custom_config),
    })

    env_config_dict_in_callback = deepcopy(env_config_dict_in_training)
    update_nested_dict(env_config_dict_in_callback, {
        "num_process": env_cfg.callback_env.num_process,
        "seed": env_cfg.callback_env.seed,
        "custom_config": OmegaConf.to_container(env_cfg.callback_env.custom_config),
    })

    vec_env = get_flycraft_vec_env(
        **env_config_dict_in_training
    )
    # evaluate_policy使用的测试环境
    eval_env = get_flycraft_vec_env(
        **env_config_dict_in_eval
    )
    # 回调函数中使用的测试环境
    eval_env_in_callback = get_flycraft_vec_env(
        **env_config_dict_in_callback
    )

    return vec_env, eval_env, eval_env_in_callback
