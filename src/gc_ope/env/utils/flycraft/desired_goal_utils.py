from typing import Union, Any
import itertools
import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from stable_baselines3.common.env_util import unwrap_wrapper, is_wrapped
from flycraft.env import FlyCraftEnv
from gc_ope.env.utils.flycraft.my_wrappers import ScaledObservationWrapper


def sample_a_desired_goal(env: Union[FlyCraftEnv, gym.Wrapper]) -> np.ndarray:
    """根据环境原有的采样目标方法，采样一个desired_goal

    Args:
        env (Union[FlyCraftEnv, gym.Wrapper]): 环境

    Returns:
        np.ndarray: desired_goal
    """
    goal_dict = env.unwrapped.task.goal_sampler.sample_goal()
    return np.array([goal_dict["v"], goal_dict["mu"], goal_dict["chi"]])


def get_all_possible_dgs(env: Union[FlyCraftEnv], step_v: float=10, step_mu: float=2, step_chi: float=2) -> list[tuple]:
    """以step为间隔，在desired goal space中生成所有可能的goal。
    """
    v_min = env.unwrapped.task.config["goal"]["v_min"]
    v_max = env.unwrapped.task.config["goal"]["v_max"]
    mu_min = env.unwrapped.task.config["goal"]["mu_min"]
    mu_max = env.unwrapped.task.config["goal"]["mu_max"]
    chi_min = env.unwrapped.task.config["goal"]["chi_min"]
    chi_max = env.unwrapped.task.config["goal"]["chi_max"]

    EPS = 1e-6

    vs = np.arange(v_min, v_max + EPS, step_v)
    mus = np.arange(mu_min, mu_max + EPS, step_mu)
    chis = np.arange(chi_min, chi_max + EPS, step_chi)

    all_dgs = list(itertools.product(vs, mus, chis))
    return all_dgs


def get_all_possible_dgs_and_dV(env: Union[FlyCraftEnv], step_list: list[float] = [10.0, 2.0, 2.0]) -> tuple[list[tuple], float]:
    """以step为间隔，在desired goal space中生成所有可能的goal。
    
    Returns:
        tuple[list[tuple], float]: 目标集合，间隔体积
    """
    assert len(step_list) == 3

    all_dgs = get_all_possible_dgs(env=env, step_v=step_list[0], step_mu=step_list[1], step_chi=step_list[2])
    dV = step_list[0] * step_list[1] * step_list[2]

    return all_dgs, dV


def reset_env_with_desired_goal(
    env: Union[FlyCraftEnv, gym.Wrapper],
    desired_goal: np.ndarray,
    seed: int | None = None,
    options: dict[str, Any] | None = None,
)  -> tuple[ObsType, dict[str, Any]]:
    """按指定的desired_goal初始化环境。

    Args:
        env (Union[FlyCraftEnv, gym.Wrapper]): 环境
        desired_goal (np.ndarray): 期望目标

    Returns:
        tuple[ObsType, dict[str, Any]]: 按期望目标重置环境后的观测、辅助信息
    """
    obs, info = env.reset(seed=seed, options=options)

    env.unwrapped.task.goal = desired_goal.copy()

    new_obs = env.unwrapped._get_obs()

    tmp_env = unwrap_wrapper(env, ScaledObservationWrapper)

    if tmp_env is not None:
        return tmp_env.scale_state(new_obs), info
    else:
        return new_obs, info


def get_desired_goal_space_volumn(env: Union[FlyCraftEnv, gym.Wrapper]) -> float:
    """计算desired_goal空间的体积
    """
    dg_mins = [
        env.unwrapped.env_config["goal"]["v_min"],
        env.unwrapped.env_config["goal"]["mu_min"],
        env.unwrapped.env_config["goal"]["chi_min"],
    ]
    dg_maxs = [
        env.unwrapped.env_config["goal"]["v_max"],
        env.unwrapped.env_config["goal"]["mu_max"],
        env.unwrapped.env_config["goal"]["chi_max"],
    ]

    return np.prod([(dim_max - dim_min) for dim_min, dim_max in zip(dg_mins, dg_maxs)])
