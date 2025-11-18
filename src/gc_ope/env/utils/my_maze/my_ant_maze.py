import numpy as np
import gymnasium as gym
from gymnasium_robotics.core import GoalEnv
from gymnasium_robotics.envs.maze.ant_maze_v5 import AntMazeEnv


class MyAntMazeEnv(AntMazeEnv):

    def __init__(self, render_mode = None, maze_map = ..., reward_type = "sparse", continuing_task = True, reset_target = False, xml_file = None, **kwargs):
        super().__init__(render_mode, maze_map, reward_type, continuing_task, reset_target, xml_file, **kwargs)

    def reset(self, *, seed = None, **kwargs):
        
        # from reset() of MazeEnv
        GoalEnv.reset(self, seed=seed)

        if ("options" not in kwargs) or (kwargs["options"] is None):
            goal = self.generate_target_goal()
            # Add noise to goal position
            self.goal = self.add_xy_position_noise(goal)
            reset_pos = self.generate_reset_pos()
        else:
            if "goal_xy" in kwargs["options"] and kwargs["options"]["goal_xy"] is not None:
                self.goal = np.array(kwargs["options"]["goal_xy"])
            else:
                if "goal_cell" in kwargs["options"] and kwargs["options"]["goal_cell"] is not None:
                    # assert that goal cell is valid
                    assert self.maze.map_length > kwargs["options"]["goal_cell"][0]
                    assert self.maze.map_width > kwargs["options"]["goal_cell"][1]
                    assert (
                        self.maze.maze_map[kwargs["options"]["goal_cell"][0]][kwargs["options"]["goal_cell"][1]]
                        != 1
                    ), f"Goal can't be placed in a wall cell, {kwargs["options"]['goal_cell']}"

                    goal = self.maze.cell_rowcol_to_xy(kwargs["options"]["goal_cell"])

                else:
                    goal = self.generate_target_goal()

                # Add noise to goal position
                self.goal = self.add_xy_position_noise(goal)

            if "reset_cell" in kwargs["options"] and kwargs["options"]["reset_cell"] is not None:
                # assert that goal cell is valid
                assert self.maze.map_length > kwargs["options"]["reset_cell"][0]
                assert self.maze.map_width > kwargs["options"]["reset_cell"][1]
                assert (
                    self.maze.maze_map[kwargs["options"]["reset_cell"][0]][
                        kwargs["options"]["reset_cell"][1]
                    ]
                    != 1
                ), f"Reset can't be placed in a wall cell, {kwargs["options"]['reset_cell']}"

                reset_pos = self.maze.cell_rowcol_to_xy(kwargs["options"]["reset_cell"])

            else:
                reset_pos = self.generate_reset_pos()

        # Update the position of the target site for visualization
        self.update_target_site_pos()

        self.reset_pos = reset_pos
        # Add noise to reset position
        # self.reset_pos = self.add_xy_position_noise(reset_pos)

        # Update the position of the target site for visualization
        self.update_target_site_pos()

        # from reset() of AntMazeEnv
        self.ant_env.init_qpos[:2] = self.reset_pos

        obs, info = self.ant_env.reset(seed=seed)
        obs_dict = self._get_obs(obs)
        info["success"] = bool(
            np.linalg.norm(obs_dict["achieved_goal"] - self.goal) <= 0.45
        )

        return obs_dict, info