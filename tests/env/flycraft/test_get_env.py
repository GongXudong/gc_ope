from pathlib import Path

from stable_baselines3.common.env_checker import check_env

from gc_ope.env.get_vec_env import get_flycraft_envs
from gc_ope.utils.load_config_with_hydra import load_config


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def test_get_flycraft_vec_env():
    cfg = load_config(
        config_path="../../../configs/train",
        config_name="config",
    )

    cfg.env.train_env.num_process = 2
    cfg.env.evaluation_env.num_process = 1
    cfg.env.callback_env.num_process = 1

    print(cfg.env)

    train_env, eval_env, callback_env = get_flycraft_envs(cfg.env)

    # check env
    obs = train_env.reset()

    for i in range(10):
        actions = [train_env.action_space.sample() for _ in range(train_env.num_envs)]
        # action = train_env.action_space.sample()
        # print(action.shape)
        obs, reward, done, info = train_env.step(actions)


if __name__ == "__main__":
    test_get_flycraft_vec_env()
