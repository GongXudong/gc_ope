from typing import Union, Any
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium_robotics.envs.maze.point_maze import MazeEnv
from gc_ope.env.utils.my_maze.my_point_maze import MyPointMazeEnv
from gc_ope.env.utils.my_maze.my_ant_maze import MyAntMazeEnv


def sample_a_desired_goal(env: Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]) -> np.ndarray:
    """根据环境原有的采样目标方法，采样一个desired_goal

    Args:
        env (Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]): 环境

    Returns:
        np.ndarray: desired_goal
    """
    tmp_dg = env.unwrapped.generate_target_goal()
    dg = env.unwrapped.add_xy_position_noise(tmp_dg)
    return dg


def generate_grid_points(center_x: float, center_y: float, side_length: float, n: int) -> list[tuple]:
    """
    在正方形内部等间隔生成n*n个点。左上角的点距离左侧的边与上侧的边的距离均为“间隔的一半”。

    参数:

    center_x: 正方形中心点的x坐标

    center_y: 正方形中心点的y坐标

    side_length: 正方形的边长. Maze类中，每个方格的边长是maze.maze_size_scaling，生成desired_goal时，是在以格子中心为中心，边长为2 * position_noise_range * maze.maze_size_scaling为边长的“小格子”内采样。

    n: 每边的点数 (n*n个点)

    返回:
    list: 包含所有点坐标的列表，每个点格式为(x, y)
    """
    # 计算正方形的边界
    half_size = side_length / 2
    left = center_x - half_size
    right = center_x + half_size
    bottom = center_y - half_size
    top = center_y + half_size

    # 计算点之间的间隔
    # 总共有n-1个间隔，边缘点到边的距离为间隔的一半
    if n == 1:
        # 如果只有一个点，直接返回中心点
        return [(center_x, center_y)]

    # 计算间隔：总可用空间是边长减去两倍的边缘距离
    # 边缘距离 = 间隔 / 2，所以总可用空间 = 边长 - 间隔
    # 间隔数量 = n - 1
    # 因此：边长 - 间隔 = (n - 1) * 间隔
    # 所以：边长 = n * 间隔
    spacing = side_length / n

    # 第一个点的位置：左边界 + 间隔/2
    start_x = left + spacing / 2
    start_y = bottom + spacing / 2

    # 生成所有点的坐标
    points = []
    for i in range(n):
        for j in range(n):
            # 计算当前点的坐标
            x = start_x + i * spacing
            y = start_y + j * spacing
            points.append((x, y))

    return points


def generate_all_possible_dgs(env: MazeEnv, n: int) -> list[tuple]:
    """对于MazeEnv中的所有标记为可生成goal的格子，在格子内等间隔生成n*n个goal。
    """
    dg_list = []

    all_goal_xys = env.maze.unique_goal_locations

    for tmp_goal in all_goal_xys:
        # tmp_dg_list = generate_grid_points(tmp_goal[0], tmp_goal[1], env.maze.maze_size_scaling, n)
        
        # TODO: check
        tmp_dg_list = generate_grid_points(tmp_goal[0], tmp_goal[1], 2 * env.position_noise_range * env.maze.maze_size_scaling, n)
        
        dg_list.extend(tmp_dg_list)

    return dg_list


def reset_env_with_desired_goal(
    env: Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper],
    desired_goal: np.ndarray,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
)  -> tuple[ObsType, dict[str, Any]]:
    """按指定的desired_goal初始化环境。

    Args:
        env (Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]): 环境
        desired_goal (np.ndarray): 期望目标

    Returns:
        tuple[ObsType, dict[str, Any]]: 按期望目标重置环境后的观测、辅助信息
    """
    obs, info = env.reset(seed=seed, options=options)
        
    env.unwrapped.goal = desired_goal.copy()

    if hasattr(env.unwrapped, "point_env"):
        tmp_obs, tmp_info = env.unwrapped.point_env.reset()
    elif hasattr(env.unwrapped, "ant_env"):
        tmp_obs, tmp_info = env.unwrapped.ant_env.reset()
    else:
        raise ValueError(f"This method is only applicable to AntMaze and PointMaze.")

    obs_dict = env.unwrapped._get_obs(tmp_obs)
    tmp_info["success"] = bool(
        np.linalg.norm(obs_dict["achieved_goal"] - env.unwrapped.goal) <= 0.45
    )

    return obs_dict, tmp_info


def get_desired_goal_space_volumn(env: Union[MyPointMazeEnv, MyAntMazeEnv, gym.Wrapper]) -> float:
    """计算desired_goal空间的体积
    """

    one_grid_volumn = env.unwrapped.position_noise_range * 2 * env.unwrapped.maze.maze_size_scaling

    return one_grid_volumn * len(env.unwrapped.maze.unique_goal_locations)
