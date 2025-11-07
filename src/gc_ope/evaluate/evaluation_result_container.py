from abc import ABC
import numpy as np


class EvaluationResultContainer(ABC):

    desired_goal_list: list
    success_list: list
    cumulative_reward_list: list
    discounted_cumulative_reward_list: list

    def __init__(self):
        self.reset()

    def reset(self):
        self.desired_goal_list = []
        self.success_list = []
        self.cumulative_reward_list = []
        self.discounted_cumulative_reward_list = []

    def add(self, desired_goal: list, success: bool, cumulative_reward: float, discounted_cumulative_reward: float, *args):
        self.before_add_hook()

        self.desired_goal_list.append(desired_goal)
        self.success_list.append(success)
        self.cumulative_reward_list.append(cumulative_reward)
        self.discounted_cumulative_reward_list.append(discounted_cumulative_reward)

        self.after_add_hook()

    def add_batch(self, desired_goal_batch: list, success_batch: list, cumulative_reward_batch: list, discounted_cumulative_reward_batch: list, *args):
        self.before_add_hook()

        self.desired_goal_list.extend(desired_goal_batch)
        self.success_list.extend(success_batch)
        self.cumulative_reward_list.extend(cumulative_reward_batch)
        self.discounted_cumulative_reward_list.extend(discounted_cumulative_reward_batch)

        self.after_add_hook()

    def before_add_hook(self, *args):
        pass

    def after_add_hook(self, *args):
        pass


class WeightedEvaluationResultContainer(EvaluationResultContainer):

    desired_goal_weights: np.ndarray
    discounted_factor: float

    def __init__(self, discounted_factor):
        self.reset()
        self.discounted_factor = discounted_factor

    def reset(self):
        super().reset()
        self.desired_goal_weights = np.array([])

    def add(self, desired_goal: list, success: bool, cumulative_reward: float, discounted_cumulative_reward: float, desired_goal_weight: float):
        super().add(desired_goal, success, cumulative_reward, discounted_cumulative_reward)
        self.desired_goal_weights = np.append(self.desired_goal_weights, desired_goal_weight)

    def add_batch(self, desired_goal_batch: list, success_batch: list, cumulative_reward_batch: list, discounted_cumulative_reward_batch: list, desired_goal_weight_batch: list):
        super().add_batch(desired_goal_batch, success_batch, cumulative_reward_batch, discounted_cumulative_reward_batch)
        self.desired_goal_weights = np.append(self.desired_goal_weights, desired_goal_weight_batch)

    def before_add_hook(self, *args):
        self.desired_goal_weights = self.desired_goal_weights * self.discounted_factor
