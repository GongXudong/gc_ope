import numpy as np
import gymnasium as gym
from gc_ope.env.utils.my_push.get_all_possible_dgs import get_all_possible_dgs, get_random_dgs
from gc_ope.env.utils.my_slide.register_env import register_my_slide


register_my_slide(control_type="joints", goal_xy_range=0.5, goal_x_offset=0.4, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

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

if __name__ == "__main__":
    test_get_all_possible_dgs()
    test_get_random_dgs()
