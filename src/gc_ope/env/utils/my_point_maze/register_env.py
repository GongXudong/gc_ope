from typing import Literal
from gymnasium.envs.registration import register
from gymnasium_robotics.envs.maze import maps

from gc_ope.env.utils.my_point_maze.my_point_maze import MyPointMazeEnv


def register_my_point_maze():

    register(
        id=f"MyPointMaze_Large_Diverse_G-v3",
        entry_point=MyPointMazeEnv,
        kwargs={
            "maze_map": maps.LARGE_MAZE_DIVERSE_G,
            "reward_type": "sparse",
        },
        max_episode_steps=800,
    )

    register(
        id=f"MyPointMaze_Large_Diverse_GDense-v3",
        entry_point=MyPointMazeEnv,
        kwargs={
            "maze_map": maps.LARGE_MAZE_DIVERSE_G,
            "reward_type": "dense",
        },
        max_episode_steps=800,
    )