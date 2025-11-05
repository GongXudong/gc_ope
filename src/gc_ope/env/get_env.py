from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig

import gymnasium as gym
import flycraft
from flycraft.utils_common.dict_utils import update_nested_dict

from gc_ope.env.utils.flycraft.vec_env_helper import make_env as make_flycraft_env


gym.register_envs(flycraft)
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def get_env(env_cfg: DictConfig) -> gym.Env:
    """读取配置文件，根据env_id自动返回对应环境的env

    Args:
        env_cfg (DictConfig): 环境配置文件

    Raises:
        ValueError: 当env_id无法识别时，抛出异常

    Returns:
        gym.Env: env_id对应的env
    """
    if env_cfg.env_id.startswith("FlyCraft"):
        return get_flycraft_env(
            seed=env_cfg.train_env.seed,
            config_file=str(PROJECT_ROOT_DIR / env_cfg.config_file),
            custom_config=OmegaConf.to_container(env_cfg.train_env.custom_config),
        )
    else:
        raise ValueError(f"Can not get vec_env for env: {env_cfg.env_id}!")


def get_flycraft_env(seed: int, config_file: str, custom_config: dict) -> gym.Env:
    return make_flycraft_env(
        rank=0,
        seed=seed,
        config_file=config_file,
        custom_config=custom_config,
    )()