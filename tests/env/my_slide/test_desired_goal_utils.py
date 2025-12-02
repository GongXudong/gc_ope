import numpy as np
import gymnasium as gym
from gc_ope.env.utils.my_push.desired_goal_utils import sample_a_desired_goal, get_all_possible_dgs, get_random_dgs, reset_env_with_desired_goal
from gc_ope.env.utils.my_slide.register_env import register_my_slide


register_my_slide(control_type="joints", goal_xy_range=0.5, goal_x_offset=0.4, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)


def test_sample_a_desired_goal():
    print("In sample a desired goal:")

    env = gym.make("MySlideSparse-v0")

    dgs = [sample_a_desired_goal(env) for i in range(100)]

    for dg in dgs:
        assert env.unwrapped.task.goal_range_low[0] <= dg[0] <= env.unwrapped.task.goal_range_high[0]
        assert env.unwrapped.task.goal_range_low[1] <= dg[1] <= env.unwrapped.task.goal_range_high[1]
        assert dg[2] == env.unwrapped.task.object_size / 2


def test_get_all_possible_dgs():

    env = gym.make("MySlideSparse-v0")
    all_dgs = get_all_possible_dgs(env=env, step_x=0.02, step_y=0.02, step_z=0.02)
    print(len(all_dgs), np.array(all_dgs))

def test_get_random_dgs():

    env = gym.make("MySlideSparse-v0")
    random_dgs = get_random_dgs(env, 10)
    print(len(random_dgs), np.array(random_dgs))

    for dg in random_dgs:
        assert env.unwrapped.task.goal_range_low[0] <= dg[0] <= env.unwrapped.task.goal_range_high[0]
        assert env.unwrapped.task.goal_range_low[1] <= dg[1] <= env.unwrapped.task.goal_range_high[1]
        assert dg[2] == env.unwrapped.task.object_size / 2

def test_reset_env_with_desired_goal():
    print("In test reset env with desired goal:")

    env = gym.make("MySlideSparse-v0")

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
    test_sample_a_desired_goal()
    test_get_all_possible_dgs()
    test_get_random_dgs()
    test_reset_env_with_desired_goal()
