import gymnasium as gym
import gymnasium_robotics

from gc_ope.env.utils.my_point_maze.register_env import register_my_point_maze


gym.register_envs(gymnasium_robotics)
register_my_point_maze()

def test_pointmaze_init():
    env = gym.make(
        id="MyPointMaze_Large_Diverse_G-v3",
        continuing_task=False,
        maze_map=[
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, "r", "g", "g", "g", 1, "g", "g", "g", "g", "g", 1],
            [1, "g", 1, 1, "g", 1, "g", 1, "g", 1, "g", 1],
            [1, "g", "g", "g", "g", "g", "g", 1, "g", "g", "g", 1],
            [1, "g", 1, 1, 1, 1, "g", 1, 1, 1, "g", 1],
            [1, "g", "g", 1, "g", 1, "g", "g", "g", "g", "g", 1],
            [1, 1, "g", 1, "g", 1, "g", 1, "g", 1, 1, 1],
            [1, "g", "g", 1, "g", "g", "g", 1, "g", "g", "g", 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        ],
    )

    obs, info = env.reset()
    print(obs)


if __name__ == "__main__":
    test_pointmaze_init()
