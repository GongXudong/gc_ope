import gymnasium as gym
from gc_ope.algorithm.curriculum.mega_wrapper import MEGAWrapper
from gc_ope.env.utils.my_maze.register_env import register_my_ant_maze


register_my_ant_maze()


def test_mega_wrapper_with_antmaze_1():
    print("In test mega wrapper with antmaze 1:")

    env = gym.make(
        id="MyAntMaze_Medium_Diverse_G-v3",
        continuing_task=False,
        reward_type="sparse",
        maze_map=[
            [1, 1, 1, 1, 1],
            [1, "r", "g", "g", 1],
            [1, 1, 1, "g", 1],
            [1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1],
        ],
    )

    mega_env = MEGAWrapper(
        env=env,
        sample_N=10,
        kde_kernel="gaussian",
        kde_bandwidth=0.2,
        kde_data_discounted_factor=0.9,
    )

    # 总共6个点
    print(env.unwrapped.maze.unique_goal_locations)

    # 向测评数据中加入前2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[0], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[1], "success": True, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    # 向测评数据中加入中间2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[2], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[3], "success": True, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    # 向测评数据中加入后2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[4], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[5], "success": True, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))


def test_mega_wrapper_with_antmaze_2():
    # 与上面一个测试的区别在于：上面的测评数据中，每次包括的全是成功的目标；本测试中，包括部分失败的目标！
    print("In test mega wrapper with antmaze 2:")

    env = gym.make(
        id="MyAntMaze_Medium_Diverse_G-v3",
        continuing_task=False,
        reward_type="sparse",
        maze_map=[
            [1, 1, 1, 1, 1],
            [1, "r", "g", "g", 1],
            [1, 1, 1, "g", 1],
            [1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1],
        ],
    )

    mega_env = MEGAWrapper(
        env=env,
        sample_N=10,
        kde_kernel="gaussian",
        kde_bandwidth=0.2,
        kde_data_discounted_factor=0.9,
    )

    # 总共6个点
    print(env.unwrapped.maze.unique_goal_locations)

    # 向测评数据中加入前2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[0], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[1], "success": False, "cumulative_reward": 10.0},
    ])
    # 注意，调用estimate_kde前，需要保证至少有1条成功的评估数据，不然没有数据用来拟合KDE！！！
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    # 向测评数据中加入中间2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[2], "success": False, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[3], "success": False, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    # 向测评数据中加入后2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[4], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[5], "success": False, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))


def test_mega_wrapper_sample_goal_with_antmaze():
    print("In test mega wrapper sample goal with antmaze:")

    env = gym.make(
        id="MyAntMaze_Medium_Diverse_G-v3",
        continuing_task=False,
        reward_type="sparse",
        maze_map=[
            [1, 1, 1, 1, 1],
            [1, "r", "g", "g", 1],
            [1, 1, 1, "g", 1],
            [1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1],
        ],
    )

    mega_env = MEGAWrapper(
        env=env,
        sample_N=10,
        kde_kernel="gaussian",
        kde_bandwidth=0.2,
        kde_data_discounted_factor=0.9,
    )

    # 向测评数据中加入前2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[0], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[1], "success": True, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    # 向测评数据中加入中间2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[2], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[3], "success": True, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    # 向测评数据中加入后2个点
    mega_env.sync_evaluation_stat([
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[4], "success": True, "cumulative_reward": 10.0},
        {"desired_goal": env.unwrapped.maze.unique_goal_locations[5], "success": False, "cumulative_reward": 10.0},
    ])
    mega_env.estimate_kde()

    print(mega_env.kde_estimator.evaluate(desired_goals=env.unwrapped.maze.unique_goal_locations))

    print("Sample Goal......")
    print(mega_env.sample_goal())

if __name__ == "__main__":
    test_mega_wrapper_with_antmaze_1()
    test_mega_wrapper_with_antmaze_2()
    test_mega_wrapper_sample_goal_with_antmaze()
