from typing import Union, Any
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from flycraft.env import FlyCraftEnv
from gc_ope.env.utils.my_maze.my_point_maze import MyPointMazeEnv
from gc_ope.env.utils.my_maze.my_ant_maze import MyAntMazeEnv
from gc_ope.env.utils.my_reach.my_reach import MyPandaReachEnv
from gc_ope.env.utils.my_push.my_push_env import MyPandaPushEnv
from gc_ope.env.utils.my_slide.my_slide_env import MyPandaSlideEnv

from gc_ope.env.utils.flycraft import desired_goal_utils as flycraft_desired_goal_utils
from gc_ope.env.utils.my_maze import desired_goal_utils as maze_desired_goal_utils
from gc_ope.env.utils.my_reach import desired_goal_utils as reach_desired_goal_utils
from gc_ope.env.utils.my_push import desired_goal_utils as push_desired_goal_utils


def sample_a_desired_goal(env: Union[FlyCraftEnv, MyPointMazeEnv, MyAntMazeEnv, MyPandaReachEnv, MyPandaPushEnv, MyPandaSlideEnv, gym.Wrapper]) -> np.ndarray:
    """根据环境原有的采样目标方法，采样一个desired_goal

    Args:
        env (Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]): 环境

    Returns:
        np.ndarray: desired_goal
    """

    env_id = env.unwrapped.spec.id

    if env_id.startswith("FlyCraft"):
        return flycraft_desired_goal_utils.sample_a_desired_goal(env)
    elif env_id.startswith("MyReach"):
        return reach_desired_goal_utils.sample_a_desired_goal(env)
    elif env_id.startswith("MyPush") or env_id.startswith("MySlide"):
        return push_desired_goal_utils.sample_a_desired_goal(env)
    elif env_id.startswith("MyPointMaze") or env_id.startswith("MyAntMaze"):
        return maze_desired_goal_utils.sample_a_desired_goal(env)
    else:
        raise ValueError(f"Can not process env: {env_id}!")


def get_all_possible_dgs(env: Union[FlyCraftEnv, MyPointMazeEnv, MyAntMazeEnv, MyPandaReachEnv, MyPandaPushEnv, MyPandaSlideEnv, gym.Wrapper], **kwargs) -> list[tuple]:

    env_id = env.unwrapped.spec.id

    if env_id.startswith("FlyCraft"):
        return flycraft_desired_goal_utils.get_all_possible_dgs(env=env, step_v=kwargs["step_v"], step_mu=kwargs["step_mu"], step_chi=kwargs["step_chi"])
    elif env_id.startswith("MyReach"):
        return reach_desired_goal_utils.get_all_possible_dgs(env=env, step_x=kwargs["step_x"], step_y=kwargs["step_y"], step_z=kwargs["step_z"])
    elif env_id.startswith("MyPush") or env_id.startswith("MySlide"):
        return push_desired_goal_utils.get_all_possible_dgs(env=env, step_x=kwargs["step_x"], step_y=kwargs["step_y"], step_z=kwargs["step_z"])
    elif env_id.startswith("MyPointMaze") or env_id.startswith("MyAntMaze"):
        return maze_desired_goal_utils.generate_all_possible_dgs(env=env, n=kwargs["n"])
    else:
        raise ValueError(f"Can not process env: {env_id}!")

def reset_env_with_desired_goal(
    env: Union[FlyCraftEnv, MyPointMazeEnv, MyAntMazeEnv, MyPandaReachEnv, MyPandaPushEnv, MyPandaSlideEnv, gym.Wrapper],
    desired_goal: np.ndarray,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
) -> tuple[ObsType, dict[str, Any]]:
    """按指定的desired_goal初始化环境。

    Args:
        env (Union[FlyCraftEnv, MyPointMazeEnv, MyAntMazeEnv, MyReachEnv, MyPushEnv, MySlideEnv, gym.Wrapper]): 环境
        desired_goal (np.ndarray): 期望目标

    Returns:
        tuple[ObsType, dict[str, Any]]: 按期望目标重置环境后的观测、辅助信息
    """

    env_id = env.unwrapped.spec.id

    if env_id.startswith("FlyCraft"):
        return flycraft_desired_goal_utils.reset_env_with_desired_goal(env, desired_goal, seed=seed, options=options)
    elif env_id.startswith("MyPointMaze") or env_id.startswith("MyAntMaze"):
        return maze_desired_goal_utils.reset_env_with_desired_goal(env, desired_goal, seed=seed, options=options)
    elif env_id.startswith("MyReach"):
        return reach_desired_goal_utils.reset_env_with_desired_goal(env, desired_goal, seed=seed, options=options)
    elif env_id.startswith("MyPush") or env_id.startswith("MySlide"):
        return push_desired_goal_utils.reset_env_with_desired_goal(env, desired_goal, seed=seed, options=options)
    else:
        raise ValueError(f"Can not process env: {env_id}!")


def get_desired_goal_space_volumn(env: Union[FlyCraftEnv, MyPointMazeEnv, MyAntMazeEnv, MyPandaReachEnv, MyPandaPushEnv, MyPandaSlideEnv, gym.Wrapper]) -> float:
    """计算desired_goal空间的体积
    """
    env_id = env.unwrapped.spec.id

    if env_id.startswith("FlyCraft"):
        return flycraft_desired_goal_utils.get_desired_goal_space_volumn(env)
    elif env_id.startswith("MyPointMaze") or env_id.startswith("MyAntMaze"):
        return maze_desired_goal_utils.get_desired_goal_space_volumn(env)
    elif env_id.startswith("MyReach"):
        return reach_desired_goal_utils.get_desired_goal_space_volumn(env)
    elif env_id.startswith("MyPush") or env_id.startswith("MySlide"):
        return push_desired_goal_utils.get_desired_goal_space_volumn(env)
    else:
        raise ValueError(f"Can not process env: {env_id}!")
