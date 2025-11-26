import gymnasium as gym

from gc_ope.env.utils.my_push.register_env import register_my_push


register_my_push(control_type="joints", goal_xy_range=0.5, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)

def test_my_push_init():

    env = gym.make("MyPushSparse-v0")

    assert env.unwrapped.task.distance_threshold == 0.05

if __name__ == "__main__":
    test_my_push_init()
