import gymnasium as gym
import gymnasium_robotics

from gc_ope.env.utils.my_maze.register_env import register_my_ant_maze


gym.register_envs(gymnasium_robotics)
register_my_ant_maze()

def test_antmaze_init():
    env = gym.make(
        id="MyAntMaze_Medium_Diverse_G-v3",
        continuing_task=False,
        reward_type="sparse",
        maze_map=[
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", "g", 1, 1, "g", "g", 1],
            [1, "g", "g", 1, "g", "g", "g", 1],
            [1, 1, "g", "g", "g", 1, 1, 1],
            [1, "g", "g", 1, "g", "g", "g", 1],
            [1, "g", 1, "g", "g", 1, "g", 1],
            [1, "g", "g", "g", 1, "g", "g", 1],
            [1, 1, 1, 1, 1, 1, 1, 1],
        ],
    )

    obs, info = env.reset()
    print(obs)

    print(env.unwrapped.maze.maze_size_scaling)


if __name__ == "__main__":
    test_antmaze_init()
