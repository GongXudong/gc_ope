from pathlib import Path
from stable_baselines3 import PPO
from gc_ope.env.get_vec_env import get_vec_env
from gc_ope.algorithm.utils.my_evaluate_policy import evaluate_policy_with_stat
from gc_ope.utils.load_config_with_hydra import load_config

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def test_my_evaluate_policy_stat():

    algo = PPO.load(PROJECT_ROOT_DIR / "checkpoints/my_push/ppo/seed_1/best_model.zip")

    cfg = load_config(
        config_path="../../../configs/train",
        config_name="conifg",
    )

    cfg.env.env_id = "MyPushSparse-v0"
    cfg.env.train_env.num_process = 2
    cfg.env.evaluation_env.num_process = 1
    cfg.env.callback_env.num_process = 1

    vec_env = get_vec_env(cfg)


if __name__ == "__main__":
    test_my_evaluate_policy_stat()
