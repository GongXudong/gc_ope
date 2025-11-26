from gymnasium.envs.registration import register
from gc_ope.env.utils.my_push.my_push_env import MyPandaPushEnv


def register_my_push(
    control_type: str="joints",
    goal_xy_range: float=0.3,
    obj_xy_range: float=0.3,
    distance_threshold: float=0.01,
    max_episode_steps: int=50,
):
    register(
        id="MyPushSparse-v0",
        entry_point=MyPandaPushEnv,
        kwargs={
            "reward_type": "sparse",
            "control_type": control_type,
            "goal_xy_range": goal_xy_range,
            "obj_xy_range": obj_xy_range,
            "distance_threshold": distance_threshold,
        },
        max_episode_steps=max_episode_steps,
    )
