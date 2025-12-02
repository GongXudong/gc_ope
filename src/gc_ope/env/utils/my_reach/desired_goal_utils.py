import itertools
from typing import Union, Any
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from panda_gym.envs import PandaReachEnv
from gc_ope.env.utils.my_reach.my_reach import MyPandaReachEnv


def sample_a_desired_goal(env: Union[MyPandaReachEnv, gym.Wrapper]) -> np.ndarray:
    """根据环境原有的采样目标方法，采样一个desired_goal

    Args:
        env (Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]): 环境

    Returns:
        np.ndarray: desired_goal
    """
    return env.unwrapped.task._sample_goal()


def get_all_possible_dgs(env: Union[PandaReachEnv], step_x: float=0.05, step_y: float=0.05, step_z: float=0.05) -> list[tuple]:
    """以step为间隔，在desired goal space中生成所有可能的goal。

    适用于Reach（x、y、z三个维度需要枚举）。
    """
    x_min = env.unwrapped.task.goal_range_low[0]
    x_max = env.unwrapped.task.goal_range_high[0]
    y_min = env.unwrapped.task.goal_range_low[1]
    y_max = env.unwrapped.task.goal_range_high[1]
    z_min = env.unwrapped.task.goal_range_low[2]
    z_max = env.unwrapped.task.goal_range_high[2]

    EPS = 1e-6

    xs = np.arange(x_min, x_max + EPS, step_x)
    ys = np.arange(y_min, y_max + EPS, step_y)
    zs = np.arange(z_min, z_max + EPS, step_z)

    all_dgs = list(itertools.product(xs, ys, zs))
    return all_dgs


def get_random_dgs(env: Union[PandaReachEnv], num_dg: int) -> list[tuple]:

    x_min = env.unwrapped.task.goal_range_low[0]
    x_max = env.unwrapped.task.goal_range_high[0]
    y_min = env.unwrapped.task.goal_range_low[1]
    y_max = env.unwrapped.task.goal_range_high[1]
    z_min = env.unwrapped.task.goal_range_low[2]
    z_max = env.unwrapped.task.goal_range_high[2]

    xs = [np.random.random() * (x_max - x_min) + x_min for i in range(num_dg)]
    ys = [np.random.random() * (y_max - y_min) + y_min for i in range(num_dg)]
    zs = [np.random.random() * (z_max - z_min) + z_min for i in range(num_dg)]

    random_dgs = list(zip(xs, ys, zs))
    return random_dgs


def get_random_dgs_2(env: Union[PandaReachEnv], num_dg: int) -> list[np.ndarray]:

    return [env.unwrapped.task._sample_goal() for _ in range(num_dg)]


def reset_env_with_desired_goal(env: Union[MyPandaReachEnv, gym.Wrapper], desired_goal: np.ndarray)  -> tuple[ObsType, dict[str, Any]]:
    """按指定的desired_goal初始化环境。

    Args:
        env (Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]): 环境
        desired_goal (np.ndarray): 期望目标

    Returns:
        tuple[ObsType, dict[str, Any]]: 按期望目标重置环境后的观测、辅助信息
    """
    obs, info = env.reset()

    tmp_episode_goal = desired_goal.copy()
    env.unwrapped.task.goal = tmp_episode_goal
    env.unwrapped.task.sim.set_base_pose("target", tmp_episode_goal, np.array([0.0, 0.0, 0.0, 1.0]))

    obs = {
        "achieved_goal": env.unwrapped.task.get_achieved_goal().astype(np.float32),
        "desired_goal": env.unwrapped.task.get_goal().astype(np.float32),
        "observation": env.unwrapped.robot.get_obs().astype(np.float32),
    }

    info = {"is_success": env.unwrapped.task.is_success(obs["achieved_goal"], env.unwrapped.task.get_goal())}

    return obs, info
