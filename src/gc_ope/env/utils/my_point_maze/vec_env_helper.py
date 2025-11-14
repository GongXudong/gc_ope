import gymnasium as gym
import gymnasium_robotics
from stable_baselines3.common.utils import set_random_seed


gym.register_envs(gymnasium_robotics)


def make_env(
        env_id: str,
        rank: int,
        seed: int = 0,
        **kwargs
    ):
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = gym.make(env_id, **kwargs)
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed)
    return _init