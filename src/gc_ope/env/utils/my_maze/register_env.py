from typing import Literal
from gymnasium.envs.registration import register
from gymnasium_robotics.envs.maze import maps

from gc_ope.env.utils.my_maze.my_point_maze import MyPointMazeEnv
from gc_ope.env.utils.my_maze.my_ant_maze import MyAntMazeEnv

def register_my_point_maze():

    register(
        id=f"MyPointMaze_Large_Diverse_G-v3",
        entry_point=MyPointMazeEnv,
        kwargs={
            "maze_map": maps.LARGE_MAZE_DIVERSE_G,
            "reward_type": "sparse",
            "continuing_task": False,
        },
        max_episode_steps=800,
    )

    register(
        id=f"MyPointMaze_Large_Diverse_GDense-v3",
        entry_point=MyPointMazeEnv,
        kwargs={
            "maze_map": maps.LARGE_MAZE_DIVERSE_G,
            "reward_type": "dense",
            "continuing_task": False,
        },
        max_episode_steps=800,
    )

def register_my_ant_maze():

    register(
        id=f"MyAntMaze_Medium_Diverse_G-v3",
        entry_point=MyAntMazeEnv,
        kwargs={
            "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
            "reward_type": "sparse",
            "continuing_task": False,
        },
        max_episode_steps=1000,
    )

    register(
        id=f"MyAntMaze_Medium_Diverse_GDense-v3",
        entry_point=MyAntMazeEnv,
        kwargs={
            "maze_map": maps.MEDIUM_MAZE_DIVERSE_G,
            "reward_type": "dense",
            "continuing_task": False,
        },
        max_episode_steps=1000,
    )

    register(
        id=f"MyAntMaze_U_Diverse_G-v3",
        entry_point=MyAntMazeEnv,
        kwargs={
            "maze_map": maps.U_MAZE,
            "reward_type": "sparse",
            "continuing_task": False,
        },
        max_episode_steps=700,
    )

    register(
        id=f"MyAntMaze_U_Diverse_GDense-v3",
        entry_point=MyAntMazeEnv,
        kwargs={
            "maze_map": maps.U_MAZE,
            "reward_type": "dense",
            "continuing_task": False,
        },
        max_episode_steps=700,
    )