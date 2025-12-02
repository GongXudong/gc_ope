from pathlib import Path
import numpy as np
import gymnasium as gym
from stable_baselines3.common.env_util import unwrap_wrapper
from gc_ope.env.get_env import get_env
from gc_ope.env.utils.my_maze.desired_goal_utils import sample_a_desired_goal, reset_env_with_desired_goal
from gc_ope.utils.load_config_with_hydra import load_config


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def test_sample_a_desired_goal_1():
    print("In test sample a desired goal:")

    env = gym.make("MyAntMaze_Medium_Diverse_G-v3")

    print(env.unwrapped.maze.unique_goal_locations)

    for i in range(10):
        dg = sample_a_desired_goal(env)

        row_ij_of_dg = env.unwrapped.maze.cell_xy_to_rowcol(dg)

        row_xy_center_of_dg = env.unwrapped.maze.cell_rowcol_to_xy(row_ij_of_dg)

        print(dg, row_ij_of_dg, row_xy_center_of_dg)

        assert any([(row_xy_center_of_dg[0] == task_dg[0] and row_xy_center_of_dg[1] == task_dg[1]) for task_dg in env.unwrapped.maze.unique_goal_locations])


def test_reset_env_with_desired_goal():
    print("In test reset env with desired goal:")

    env = gym.make("MyAntMaze_Medium_Diverse_G-v3")

    EPS = 1e-10

    dg_list = [sample_a_desired_goal(env) for i in range(100)]

    for dg in dg_list:
        obs, info = reset_env_with_desired_goal(env, dg)

        assert np.allclose(obs["desired_goal"], dg, atol=EPS)

        for i in range(5):
            action = env.action_space.sample()
            next_obs, _, _, _, _ = env.step(action)

            assert np.allclose(obs["desired_goal"], dg, atol=EPS)


if __name__ == "__main__":
    test_sample_a_desired_goal_1()
    test_reset_env_with_desired_goal()
