from typing import Union, Any
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
