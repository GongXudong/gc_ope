from pathlib import Path
from typing import Literal
import numpy as np
import pytest
import gymnasium as gym
import flycraft
from stable_baselines3.common.env_util import unwrap_wrapper
from gc_ope.env.get_env import get_env
from gc_ope.env.utils.flycraft import desired_goal_utils as flycraft_desired_goal_utils
from gc_ope.env.utils import desired_goal_utils as common_desired_goal_utils
from gc_ope.utils.load_config_with_hydra import load_config
from gc_ope.env.utils.flycraft.my_wrappers import ScaledObservationWrapper


gym.register_envs(flycraft)
PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


@pytest.mark.parametrize(
    "test_pkg",
    [("flycraft"), ("common")],
)
def test_sample_a_desired_goal_1(test_pkg):
    print("In test sample a desired goal 1:")

    env = gym.make("FlyCraft-v0")
    print(env)

    dg_min = env.unwrapped.task.get_goal_lower_bounds()
    dg_max = env.unwrapped.task.get_goal_higher_bounds()

    for i in range(100):
        if test_pkg == "flycraft":
            dg = flycraft_desired_goal_utils.sample_a_desired_goal(env)
        else:
            dg = common_desired_goal_utils.sample_a_desired_goal(env)

        assert dg_min[0] <= dg[0] <= dg_max[0]
        assert dg_min[1] <= dg[1] <= dg_max[1]
        assert dg_min[2] <= dg[2] <= dg_max[2]


@pytest.mark.parametrize(
    "test_pkg",
    [("flycraft"), ("common")],
)
def test_sample_a_desired_goal_2(test_pkg: Literal["flycraft", "common"]):
    print("In test sample a desired goal 2:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    env = get_env(cfg.env)

    dg_min = env.unwrapped.task.get_goal_lower_bounds()
    dg_max = env.unwrapped.task.get_goal_higher_bounds()

    for i in range(100):
        if test_pkg == "flycraft":
            dg = flycraft_desired_goal_utils.sample_a_desired_goal(env)
        else:
            dg = common_desired_goal_utils.sample_a_desired_goal(env)

        assert dg_min[0] <= dg[0] <= dg_max[0]
        assert dg_min[1] <= dg[1] <= dg_max[1]
        assert dg_min[2] <= dg[2] <= dg_max[2]


def test_get_all_possible_dgs():
    print(f"In test get all possible dgs:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    env = get_env(cfg.env)

    all_dgs = flycraft_desired_goal_utils.get_all_possible_dgs(env)

    print(len(all_dgs))


def test_get_all_possible_dgs_and_dV():
    print("test get all possible dgs and dV:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    env = get_env(cfg.env)
    
    all_dgs, dV = flycraft_desired_goal_utils.get_all_possible_dgs_and_dV(env)
    print(len(all_dgs), dV, np.array(all_dgs))

    all_dgs, dV = flycraft_desired_goal_utils.get_all_possible_dgs_and_dV(env, step_list=[10.0, 2.0, 2.0])
    print(len(all_dgs), dV, np.array(all_dgs))


def test_get_all_possible_dgs_and_dV_common():
    print("test get all possible dgs:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    env = get_env(cfg.env)
    
    all_dgs, dV = common_desired_goal_utils.get_all_possible_dgs_and_dV(env)
    print(len(all_dgs), dV, np.array(all_dgs))

    all_dgs, dV = common_desired_goal_utils.get_all_possible_dgs_and_dV(env, step_list=[10.0, 2.0, 2.0])
    print(len(all_dgs), dV, np.array(all_dgs))


@pytest.mark.parametrize(
    "test_pkg",
    [("flycraft"), ("common")],
)
def test_reset_env_with_desired_goal(test_pkg: Literal["flycraft", "common"]):
    print("In test reset env with desired goal:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    EPS = 1e-10

    env = get_env(cfg.env)
    scaled_obs_wrapper_env = unwrap_wrapper(env, ScaledObservationWrapper)

    if test_pkg == "flycraft":
        dg_list = [flycraft_desired_goal_utils.sample_a_desired_goal(env) for i in range(100)]
    else:
        dg_list = [common_desired_goal_utils.sample_a_desired_goal(env) for i in range(100)]

    for dg in dg_list:
        if test_pkg == "flycraft":
            obs, info = flycraft_desired_goal_utils.reset_env_with_desired_goal(env, dg)
        else:
            obs, info = common_desired_goal_utils.reset_env_with_desired_goal(env, dg)

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


@pytest.mark.parametrize(
    "test_pkg",
    [("flycraft"), ("common")],
)
def test_get_desired_goal_space_volumn(test_pkg: Literal["flycraft", "common"]):
    print("In test get desired goal space volumn:")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "FlyCraft-v0"

    env = get_env(cfg.env)

    if test_pkg == "flycraft":
        volumn = flycraft_desired_goal_utils.get_desired_goal_space_volumn(env)
    else:
        volumn = common_desired_goal_utils.get_desired_goal_space_volumn(env)

    print(volumn)

if __name__ == "__main__":
    test_sample_a_desired_goal_1("common")
    test_sample_a_desired_goal_2("common")
    test_get_all_possible_dgs()
    test_reset_env_with_desired_goal("common")
    test_get_desired_goal_space_volumn("common")

    test_get_all_possible_dgs_and_dV()
    test_get_all_possible_dgs_and_dV_common()
