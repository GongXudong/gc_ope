import numpy as np
import gymnasium as gym
import gymnasium_robotics
from gymnasium_robotics.envs.maze.point_maze import MazeEnv


gym.register_envs(gymnasium_robotics)


def generate_grid_points(center_x: float, center_y: float, maze_size_scaling: float, n: int) -> list[tuple]:
    """
    在正方形内部等间隔生成n*n个点。左上角的点距离左侧的边与上侧的边的距离均为“间隔的一半”。

    参数:
    center_x: 正方形中心点的x坐标
    center_y: 正方形中心点的y坐标
    maze_size_scaling: 正方形的边长
    n: 每边的点数 (n*n个点)

    返回:
    list: 包含所有点坐标的列表，每个点格式为(x, y)
    """
    # 计算正方形的边界
    half_size = maze_size_scaling / 2
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
    spacing = maze_size_scaling / n

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
        tmp_dg_list = generate_grid_points(tmp_goal[0], tmp_goal[1], env.maze.maze_size_scaling, n)
        dg_list.extend(tmp_dg_list)

    return dg_list
