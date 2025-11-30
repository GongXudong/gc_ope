from pathlib import Path
import numpy as np
import gymnasium as gym
import flycraft
from stable_baselines3.common.env_util import unwrap_wrapper
from gc_ope.env.get_env import get_env
from gc_ope.env.utils.flycraft.desired_goal_utils import sample_a_desired_goal, reset_env_with_desired_goal
from gc_ope.utils.load_config_with_hydra import load_config
from gc_ope.env.utils.flycraft.my_wrappers import ScaledObservationWrapper


gym.register_envs(flycraft)
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent

def test_sample_a_desired_goal_1():
    env = gym.make("FlyCraft-v0")

    dg_min = env.unwrapped.task.get_goal_lower_bounds()
    dg_max = env.unwrapped.task.get_goal_higher_bounds()

    for i in range(100):
        dg = sample_a_desired_goal(env)

        assert dg_min[0] <= dg[0] <= dg_max[0]
        assert dg_min[1] <= dg[1] <= dg_max[1]
        assert dg_min[2] <= dg[2] <= dg_max[2]
        
def test_sample_a_desired_goal_2():
    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    env = get_env(cfg.env)

    dg_min = env.unwrapped.task.get_goal_lower_bounds()
    dg_max = env.unwrapped.task.get_goal_higher_bounds()

    for i in range(100):
        dg = sample_a_desired_goal(env)

        assert dg_min[0] <= dg[0] <= dg_max[0]
        assert dg_min[1] <= dg[1] <= dg_max[1]
        assert dg_min[2] <= dg[2] <= dg_max[2]

def test_reset_env_with_desired_goal():
    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    EPS = 1e-10

    env = get_env(cfg.env)
    scaled_obs_wrapper_env = unwrap_wrapper(env, ScaledObservationWrapper)


    dg_list = [sample_a_desired_goal(env) for i in range(100)]

    for dg in dg_list:
        obs, info = reset_env_with_desired_goal(env, dg)
        
        if scaled_obs_wrapper_env is not None:
            # assert np.allclose(scaled_obs_wrapper_env.inverse_scale_state(obs)["desired_goal"], dg, atol=EPS)
            assert np.allclose(obs["desired_goal"], scaled_obs_wrapper_env.goal_scalar.transform(dg.reshape((1,-1))).reshape((-1)), atol=EPS)
        else:
            assert np.allclose(obs["desired_goal"], dg, atol=EPS)

        for i in range(5):
            action = env.action_space.sample()
            next_obs, _, _, _, _ = env.step(action)

            if scaled_obs_wrapper_env is not None:
                # assert np.allclose(scaled_obs_wrapper_env.inverse_scale_state(next_obs)["desired_goal"], dg, atol=EPS)
                assert np.allclose(next_obs["desired_goal"], scaled_obs_wrapper_env.goal_scalar.transform(dg.reshape((1,-1))).reshape((-1)), atol=EPS)
            else:
                assert np.allclose(next_obs["desired_goal"], dg, atol=EPS)

if __name__ == "__main__":
    test_sample_a_desired_goal_1()
    test_sample_a_desired_goal_2()
    test_reset_env_with_desired_goal()
