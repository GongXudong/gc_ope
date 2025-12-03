import gymnasium as gym
from gc_ope.algorithm.curriculum.mega_wrapper import MEGAWrapper
from gc_ope.env.utils.my_push.register_env import register_my_push
from gc_ope.env.utils.desired_goal_utils import sample_a_desired_goal, reset_env_with_desired_goal


register_my_push(control_type="joints", goal_xy_range=0.5, obj_xy_range=0.0, distance_threshold=0.05, max_episode_steps=50)


def test_mega_wrapper_with_push_1():
    print("In test mega wrapper with push 1:")

    env = gym.make("MyPushSparse-v0")

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


def test_mega_wrapper_sample_goal_with_push_1():
    print("In test mega wrapper sample goal with push:")

    env = gym.make("MyPushSparse-v0")

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

    print(mega_env.kde_estimator.evaluate(desired_goals=dgs))

    print("Sample Goal......")
    
    for i in range(3):
        print(f"Trial {i}: \n", mega_env.sample_goal())


if __name__ == "__main__":
    test_mega_wrapper_with_push_1()
    test_mega_wrapper_sample_goal_with_push_1()
