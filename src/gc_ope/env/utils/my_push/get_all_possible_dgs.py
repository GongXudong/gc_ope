import itertools
from typing import Union
import numpy as np
from panda_gym.envs import PandaPushEnv, PandaSlideEnv

def get_all_possible_dgs(env: Union[PandaPushEnv, PandaSlideEnv], step_x: float=0.05, step_y: float=0.05, step_z: float=0.05) -> list[tuple]:
    """以step为间隔，在desired goal space中生成所有可能的goal。

    适用于Push和Slide（只有x、y两个维度需要枚举，z是固定值）。
    """
    x_min = env.unwrapped.task.goal_range_low[0]
    x_max = env.unwrapped.task.goal_range_high[0]
    y_min = env.unwrapped.task.goal_range_low[1]
    y_max = env.unwrapped.task.goal_range_high[1]
    z_min = env.unwrapped.task.object_size / 2  # z是固定值
    z_max = env.unwrapped.task.object_size / 2

    EPS = 1e-6

    xs = np.arange(x_min, x_max + EPS, step_x)
    ys = np.arange(y_min, y_max + EPS, step_y)
    zs = np.arange(z_min, z_max + EPS, step_z)

    all_dgs = list(itertools.product(xs, ys, zs))
    return all_dgs

def get_random_dgs(env: Union[PandaPushEnv, PandaSlideEnv], num_dg: int) -> list[tuple]:

    x_min = env.unwrapped.task.goal_range_low[0]
    x_max = env.unwrapped.task.goal_range_high[0]
    y_min = env.unwrapped.task.goal_range_low[1]
    y_max = env.unwrapped.task.goal_range_high[1]
    z_min = env.unwrapped.task.object_size / 2  # z是固定值
    z_max = env.unwrapped.task.object_size / 2

    xs = [np.random.random() * (x_max - x_min) + x_min for i in range(num_dg)]
    ys = [np.random.random() * (y_max - y_min) + y_min for i in range(num_dg)]
    zs = [np.random.random() * (z_max - z_min) + z_min for i in range(num_dg)]

    random_dgs = list(zip(xs, ys, zs))
    return random_dgs
