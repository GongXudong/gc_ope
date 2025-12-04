import logging
import sys
from pathlib import Path
from gymnasium.envs.registration import EnvSpec
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from flycraft.env import FlyCraftEnv

from gc_ope.algorithm.curriculum.mega_wrapper import MEGAWrapper
from gc_ope.env.utils.flycraft.my_wrappers import ScaledActionWrapper, ScaledObservationWrapper


def make_env(rank: int, seed: int = 0, **kwargs):
    """
    Utility function for multiprocessed env.

    :param seed: the inital seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = FlyCraftEnv(
            config_file=kwargs["config_file"],
            custom_config=kwargs.get("custom_config", {})
        )

        # TODO: 该行代码是在完成FlyCraft上的训练、评估之后加上的，暂时不清楚该行代码对训练、评估可能产生的影响，还需系统性测试！！！
        env.unwrapped.spec = EnvSpec(
            id="FlyCraft-v0",
        )

        # 增加课程学习wrapper
        if kwargs.get("use_curriculum", False):
            curriculum_method = kwargs.get("curriculum_method", "")
            curriculum_kwargs = kwargs.get("curriculum_kwargs", {})

            if curriculum_method == "mega":
                curriculum_wrapper_class = MEGAWrapper
            else:
                raise ValueError(f"Can not process curriculum method: {curriculum_method}!")

            print(f"Check curriculum, method: {curriculum_method}, kwargs: {curriculum_kwargs}")
            env = curriculum_wrapper_class(env, **curriculum_kwargs)

        env = ScaledActionWrapper(ScaledObservationWrapper(env))
        env.reset(seed=seed + rank)
        print(seed+rank, env.unwrapped.task.np_random, env.unwrapped.task.goal_sampler.np_random)
        return env

    set_random_seed(seed)
    return _init

def get_vec_env(num_process: int=10, seed: int=0, **kwargs):
    return SubprocVecEnv([make_env(rank=i, seed=seed, **kwargs) for i in range(num_process)])
