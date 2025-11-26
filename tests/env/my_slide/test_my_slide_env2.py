import gymnasium as gym

from gc_ope.env.get_env import get_env
from gc_ope.utils.load_config_with_hydra import load_config


def test_my_slide_init():
    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.env_id = "MySlideSparse-v0"

    env = get_env(cfg.env)

    assert env.unwrapped.task.distance_threshold == 0.05

if __name__ == "__main__":
    test_my_slide_init()
