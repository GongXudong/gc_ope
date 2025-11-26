from gymnasium.envs.registration import register
from gc_ope.env.utils.my_slide.my_slide_env import MyPandaSlideEnv


def register_my_slide(
    control_type: str="joints",
    goal_xy_range: float=0.3,
    goal_x_offset: float=0.4,
    obj_xy_range: float=0.3,
    distance_threshold: float=0.01,
    max_episode_steps: int=50,
):
    register(
        id="MySlideSparse-v0",
        entry_point=MyPandaSlideEnv,
        kwargs={
            "reward_type": "sparse",
            "control_type": control_type,
            "goal_xy_range": goal_xy_range,
            "goal_x_offset": goal_x_offset,
            "obj_xy_range": obj_xy_range,
            "distance_threshold": distance_threshold,
        },
        max_episode_steps=max_episode_steps,
    )
