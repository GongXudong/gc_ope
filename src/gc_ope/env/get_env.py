from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig

import gymnasium as gym
import flycraft
from flycraft.utils_common.dict_utils import update_nested_dict

from gc_ope.env.utils.flycraft.vec_env_helper import make_env as make_flycraft_env
from gc_ope.env.utils.my_reach.register_env import register_my_reach
from gc_ope.env.utils.my_point_maze.register_env import register_my_point_maze
from gc_ope.env.utils.my_point_maze.vec_env_helper import make_env as make_pointmaze_env


gym.register_envs(flycraft)
register_my_reach(goal_range=0.3, distance_threshold=0.02, control_type="joints", max_episode_steps=100)
register_my_point_maze()
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
    elif env_cfg.env_id.startswith("MyReach"):
        return get_gym_env(env_cfg.env_id)
    elif env_cfg.env_id.startswith("MyPointMaze"):
        return get_pointmaze_env(
            env_id=env_cfg.env_id,
            seed=env_cfg.train_env.seed,
            maze_map=env_cfg.maze_map,
            reward_type=env_cfg.reward_type,
            continuing_task=env_cfg.continuing_task,
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

def get_gym_env(env_id: str, **kwargs) -> gym.Env:
    return gym.make(
        id=env_id,
        **kwargs
    )

def get_pointmaze_env(env_id: str, seed: int=0, **kwargs) -> gym.Env:
    return make_pointmaze_env(
        env_id=env_id,
        rank=0,
        seed=seed,
        **kwargs
    )()
