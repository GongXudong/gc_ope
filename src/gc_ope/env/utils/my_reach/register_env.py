from typing import Literal
from gymnasium.envs.registration import register

from gc_ope.env.utils.my_reach.my_reach import MyPandaReachEnv


def register_my_env(goal_range: float=0.3, distance_threshold: float=0.01, reward_type: Literal["sparse", "dense"]="sparse", control_type: Literal["ee", "joints"]="ee", max_episode_steps: int=50):
    register(
        id="MyReach-v0",
        entry_point=MyPandaReachEnv,
        kwargs={"reward_type": reward_type, "control_type": control_type, "goal_range": goal_range, "distance_threshold": distance_threshold},
        max_episode_steps=max_episode_steps,
    )
