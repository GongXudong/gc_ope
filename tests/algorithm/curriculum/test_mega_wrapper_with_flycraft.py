from pathlib import Path
import gymnasium as gym
import flycraft
from gc_ope.algorithm.curriculum.mega_wrapper import MEGAWrapper
from gc_ope.env.get_env import get_flycraft_env
from gc_ope.env.utils.desired_goal_utils import sample_a_desired_goal


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent
gym.register_envs(flycraft)


def test_mega_wrapper_with_flycraft_1():
    print("In test mega wrapper with flycraft:")

    env = get_flycraft_env(
        seed=1,
        config_file=PROJECT_ROOT_DIR / "configs/env_configs/flycraft/env_config_for_sac_easy.json",
        custom_config={},
    )

    mega_env = MEGAWrapper(
        env=env,
        sample_N=10,
        kde_kernel="gaussian",
        kde_bandwidth=0.2,
        kde_data_discounted_factor=0.9,
    )

    dgs = [sample_a_desired_goal(env) for _ in range(9)]

    # 向测评数据中加入前3个点
    mega_env.sync_evaluation_stat([
        {
            "desired_goal": dgs[i],
            "success": True,
            "cumulative_reward": 10.0
        } for i in range(0, 3)
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=dgs))

    # 向测评数据中加入中间3个点
    mega_env.sync_evaluation_stat([
        {
            "desired_goal": dgs[i],
            "success": True,
            "cumulative_reward": 10.0
        } for i in range(3, 6)
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=dgs))

    # 向测评数据中加入后3个点
    mega_env.sync_evaluation_stat([
        {
            "desired_goal": dgs[i],
            "success": True,
            "cumulative_reward": 10.0
        } for i in range(6, 9)
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=dgs))


def test_mega_wrapper_sample_goal_with_flycraft_1():
    print("In test mega wrapper sample goal with flycraft:")

    env = get_flycraft_env(
        seed=1,
        config_file=PROJECT_ROOT_DIR / "configs/env_configs/flycraft/env_config_for_sac_easy.json",
        custom_config={},
    )

    mega_env = MEGAWrapper(
        env=env,
        sample_N=10,
        kde_kernel="gaussian",
        kde_bandwidth=0.2,
        kde_data_discounted_factor=0.9,
    )

    dgs = [sample_a_desired_goal(env) for _ in range(9)]

    # 向测评数据中加入前3个点
    mega_env.sync_evaluation_stat([
        {
            "desired_goal": dgs[i],
            "success": True,
            "cumulative_reward": 10.0
        } for i in range(0, 3)
    ])
    mega_env.estimate_kde()

    # print(mega_env.kde_estimator.evaluate(desired_goals=dgs))

    # 向测评数据中加入中间3个点
    mega_env.sync_evaluation_stat([
        {
            "desired_goal": dgs[i],
            "success": True,
            "cumulative_reward": 10.0
        } for i in range(3, 6)
    ])
    mega_env.estimate_kde()

    # print(mega_env.kde_estimator.evaluate(desired_goals=dgs))

    # 向测评数据中加入后3个点
    mega_env.sync_evaluation_stat([
        {
            "desired_goal": dgs[i],
            "success": True,
            "cumulative_reward": 10.0
        } for i in range(6, 9)
    ])
    mega_env.estimate_kde()

    # print(mega_env.kde_estimator.evaluate(desired_goals=dgs))

    print("Sample Goal......")
    
    for i in range(3):
        print(f"Trial {i}: \n", mega_env.sample_goal())


if __name__ =="__main__":
    test_mega_wrapper_with_flycraft_1()
    test_mega_wrapper_sample_goal_with_flycraft_1()
