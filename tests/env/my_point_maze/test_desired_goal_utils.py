from pathlib import Path
from typing import Literal
import pytest
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import unwrap_wrapper
from gc_ope.env.get_env import get_env
from gc_ope.env.utils.my_maze import desired_goal_utils as my_pointmaze_desired_goal_utils
from gc_ope.env.utils import desired_goal_utils as common_desired_goal_utils
from gc_ope.utils.load_config_with_hydra import load_config


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "test_pkg",
    [("my_pointmaze"), ("common")],
)
def test_sample_a_desired_goal_1(test_pkg: Literal["my_pointmaze", "common"]):
    print("In test sample a desired goal:")

    env = gym.make("MyPointMaze_Large_Diverse_G-v3")

    print(env.unwrapped.maze.unique_goal_locations)

    for i in range(10):
        if test_pkg == "my_pointmaze":
            dg = my_pointmaze_desired_goal_utils.sample_a_desired_goal(env)
        else:
            dg = common_desired_goal_utils.sample_a_desired_goal(env)

        row_ij_of_dg = env.unwrapped.maze.cell_xy_to_rowcol(dg)

        row_xy_center_of_dg = env.unwrapped.maze.cell_rowcol_to_xy(row_ij_of_dg)

        print(type(dg), dg, row_ij_of_dg, row_xy_center_of_dg)

        assert any([(row_xy_center_of_dg[0] == task_dg[0] and row_xy_center_of_dg[1] == task_dg[1]) for task_dg in env.unwrapped.maze.unique_goal_locations])


@pytest.mark.parametrize(
    "test_pkg",
    [("my_pointmaze"), ("common")],
)
def test_reset_env_with_desired_goal(test_pkg: Literal["my_pointmaze", "common"]):
    print("In test reset env with desired goal:")

    env = gym.make("MyPointMaze_Large_Diverse_G-v3")

    EPS = 1e-10

    if test_pkg == "my_pointmaze":
        dg_list = [my_pointmaze_desired_goal_utils.sample_a_desired_goal(env) for i in range(100)]
    else:
        dg_list = [common_desired_goal_utils.sample_a_desired_goal(env) for i in range(100)]

    for dg in dg_list:
        if test_pkg == "my_pointmaze":
            obs, info = my_pointmaze_desired_goal_utils.reset_env_with_desired_goal(env, dg)
        else:
            obs, info = common_desired_goal_utils.reset_env_with_desired_goal(env, dg)

        assert np.allclose(obs["desired_goal"], dg, atol=EPS)

        for i in range(5):
            action = env.action_space.sample()
            next_obs, _, _, _, _ = env.step(action)

            assert np.allclose(obs["desired_goal"], dg, atol=EPS)


if __name__ == "__main__":
    test_sample_a_desired_goal_1("common")
    test_reset_env_with_desired_goal("common")
