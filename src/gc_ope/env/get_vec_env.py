from pathlib import Path
from copy import deepcopy
from omegaconf import OmegaConf, DictConfig

import gymnasium as gym
from stable_baselines3.common.vec_env import VecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import flycraft
from flycraft.utils_common.dict_utils import update_nested_dict
import gymnasium_robotics

from gc_ope.env.utils.flycraft.vec_env_helper import get_vec_env as get_flycraft_vec_env
from gc_ope.env.utils.my_reach.register_env import register_my_reach
from gc_ope.env.utils.my_push.register_env import register_my_push
from gc_ope.env.utils.my_slide.register_env import register_my_slide
from gc_ope.env.utils.my_maze.register_env import register_my_point_maze, register_my_ant_maze
# from gc_ope.env.utils.pointmaze.vec_env_helper import make_env as make_pointmaze_env
from gc_ope.algorithm.curriculum.mega_wrapper import MEGAWrapper
from gc_ope.algorithm.curriculum.omega_wrapper import OMEGAWrapper


gym.register_envs(flycraft)
register_my_reach(goal_range=0.3, distance_threshold=0.02, control_type="joints", max_episode_steps=100)
register_my_push(control_type="joints", goal_xy_range=0.5, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)
register_my_slide(control_type="joints", goal_xy_range=0.5, goal_x_offset=0.4, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)
register_my_point_maze()
register_my_ant_maze()
gym.register_envs(gymnasium_robotics)

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
    elif env_cfg.env_id.startswith("MyReach") or env_cfg.env_id.startswith("MyPush") or env_cfg.env_id.startswith("MySlide"):
        return get_my_reach_envs(env_cfg)
    elif env_cfg.env_id.startswith("MyPointMaze") or env_cfg.env_id.startswith("MyAntMaze"):
        return get_my_pointmaze_envs(env_cfg)
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

    # 训练使用的环境
    vec_env = get_flycraft_vec_env(
        use_curriculum=env_cfg.use_curriculum,
        curriculum_method=env_cfg.curriculum_method,
        curriculum_kwargs=env_cfg.curriculum_kwargs,
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

def get_my_reach_envs(env_cfg: DictConfig) -> tuple[VecEnv, VecEnv, VecEnv]:

    # 训练使用的环境
    if env_cfg.use_curriculum:
        
        curriculum_method = env_cfg.curriculum_method
        curriculum_kwargs = env_cfg.curriculum_kwargs

        if curriculum_method == "mega":
            curriculum_wrapper_class = MEGAWrapper
        else:
            raise ValueError(f"Can not process curriculum method: {curriculum_method}!")

        vec_env = make_vec_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.train_env.num_process,
            seed=env_cfg.train_env.seed,
            vec_env_cls=SubprocVecEnv,
            wrapper_class=curriculum_wrapper_class,
            wrapper_kwargs=curriculum_kwargs,
        )
    else:
        vec_env = make_vec_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.train_env.num_process,
            seed=env_cfg.train_env.seed,
            vec_env_cls=SubprocVecEnv,
        )

    # evaluate_policy使用的测试环境
    eval_env = make_vec_env(
        env_id=env_cfg.env_id,
        n_envs=env_cfg.evaluation_env.num_process,
        seed=env_cfg.evaluation_env.seed,
        vec_env_cls=SubprocVecEnv,
    )

    # 回调函数中使用的测试环境
    eval_env_in_callback = make_vec_env(
        env_id=env_cfg.env_id,
        n_envs=env_cfg.callback_env.num_process,
        seed=env_cfg.callback_env.seed,
        vec_env_cls=SubprocVecEnv,
    )

    return vec_env, eval_env, eval_env_in_callback

def get_my_pointmaze_envs(env_cfg: DictConfig) -> tuple[VecEnv, VecEnv, VecEnv]:

    # 训练使用的环境
    if env_cfg.use_curriculum:
        
        curriculum_method = env_cfg.curriculum_method
        curriculum_kwargs = env_cfg.curriculum_kwargs

        if curriculum_method == "mega":
            curriculum_wrapper_class = MEGAWrapper
        else:
            raise ValueError(f"Can not process curriculum method: {curriculum_method}!")

        vec_env = make_vec_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.train_env.num_process,
            seed=env_cfg.train_env.seed,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                "maze_map": env_cfg.maze_map,
                "reward_type": env_cfg.reward_type,
                "continuing_task": env_cfg.continuing_task,
            },
            wrapper_class=curriculum_wrapper_class,
            wrapper_kwargs=curriculum_kwargs,
        )
    else:
        vec_env = make_vec_env(
            env_id=env_cfg.env_id,
            n_envs=env_cfg.train_env.num_process,
            seed=env_cfg.train_env.seed,
            vec_env_cls=SubprocVecEnv,
            env_kwargs={
                "maze_map": env_cfg.maze_map,
                "reward_type": env_cfg.reward_type,
                "continuing_task": env_cfg.continuing_task,
            },
        )

    # evaluate_policy使用的测试环境
    eval_env = make_vec_env(
        env_id=env_cfg.env_id,
        n_envs=env_cfg.evaluation_env.num_process,
        seed=env_cfg.evaluation_env.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "maze_map": env_cfg.maze_map,
            "reward_type": env_cfg.reward_type,
            "continuing_task": env_cfg.continuing_task,
        },
    )

    # 回调函数中使用的测试环境
    eval_env_in_callback = make_vec_env(
        env_id=env_cfg.env_id,
        n_envs=env_cfg.callback_env.num_process,
        seed=env_cfg.callback_env.seed,
        vec_env_cls=SubprocVecEnv,
        env_kwargs={
            "maze_map": env_cfg.maze_map,
            "reward_type": env_cfg.reward_type,
            "continuing_task": env_cfg.continuing_task,
        },
    )

    return vec_env, eval_env, eval_env_in_callback
