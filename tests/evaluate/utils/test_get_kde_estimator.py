from pathlib import Path
from stable_baselines3.common.save_util import load_from_pkl
from gc_ope.evaluate.utils.get_kde_estimator import get_kde_estimator_for_replay_buffer


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent


def test_get_kde_estimator_for_replay_buffer():
    replay_buffer_path = "checkpoints/flycraft/easy/sac/seed_1/rl_model_replay_buffer_10000_steps.pkl"

    replay_buffer = load_from_pkl(PROJECT_ROOT_DIR / replay_buffer_path)
    achieved_goals = replay_buffer.observations["achieved_goal"].reshape((-1, replay_buffer.observations["achieved_goal"].shape[-1]))
    print(achieved_goals.shape)

    estimator = get_kde_estimator_for_replay_buffer(
        replay_buffer_path=replay_buffer_path,
        sample_num_for_replay_buffer=1000,
    )

if __name__ == "__main__":
    test_get_kde_estimator_for_replay_buffer()
