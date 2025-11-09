import gymnasium as gym

from gc_ope.env.utils.my_reach.register_env import register_my_env

def test_my_reach_init():
    register_my_env(goal_range=0.3, distance_threshold=0.01, reward_type="sparse", control_type="joints", max_episode_steps=100)

    env = gym.make("MyReach-v0")

    assert env.unwrapped.task.distance_threshold == 0.01

def test_my_reach_init_2():

    def func():
        register_my_env(goal_range=0.3, distance_threshold=0.01, reward_type="sparse", control_type="joints", max_episode_steps=100)

    func()

    env = gym.make("MyReach-v0")

    assert env.unwrapped.task.distance_threshold == 0.01
