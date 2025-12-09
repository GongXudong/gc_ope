from typing import Union
from pathlib import Path
import numpy as np
import pandas as pd
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_kde import KDEEvaluator
from stable_baselines3.common.save_util import load_from_pkl


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent.parent.parent
PROJECT_ROOT_DIR


def get_kde_estimator_for_replay_buffer(
    replay_buffer_path: Union[str, Path],
    sample_num_for_replay_buffer: int = 5000,
    kde_data_discounted_factor: float = 0.9,
    kde_bandwidth: float = 1.0,
) -> KDEEvaluator:

    replay_buffer = load_from_pkl(PROJECT_ROOT_DIR / replay_buffer_path)

    # replay_buffer.observations原本的维度是(step_num, env_num, goal_dim)，将其维度转换为(step_num * env_num, goal_dim)
    achieved_goals = replay_buffer.observations["achieved_goal"].reshape((-1, replay_buffer.observations["achieved_goal"].shape[-1]))

    kde_evaluator_for_replay_buffer = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs=dict(
            discounted_factor=kde_data_discounted_factor,
        ),
        kde_bandwidth=kde_bandwidth,
        kde_kernel="gaussian",
    )

    random_index = np.random.randint(
        low=0,
        high=replay_buffer.size(),
        size=sample_num_for_replay_buffer,
    )

    kde_evaluator_for_replay_buffer.eval_res_container.add_batch(
        desired_goal_batch=achieved_goals[random_index],
        success_batch=[True] * sample_num_for_replay_buffer,
        cumulative_reward_batch=[0.0] * sample_num_for_replay_buffer,
        discounted_cumulative_reward_batch=[0.0] * sample_num_for_replay_buffer,
        desired_goal_weight_batch=[1.0] * sample_num_for_replay_buffer,
    )

    return kde_evaluator_for_replay_buffer

def get_kde_estimator_for_eval_res(
    res_dir: Union[str, Path],
    goal_keys_in_csv: list[str] = ["x", "y", "z"],
    kde_data_discounted_factor: float = 0.9,
    kde_bandwidth: float = 1.0,
) -> KDEEvaluator:
    eval_res = pd.read_csv(PROJECT_ROOT_DIR / res_dir)

    kde_evaluator_for_eval_res = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs=dict(
            discounted_factor=kde_data_discounted_factor,
        ),
        kde_bandwidth=kde_bandwidth,
        kde_kernel="gaussian",
    )

    kde_evaluator_for_eval_res.eval_res_container.add_batch(
        desired_goal_batch=eval_res.loc[:][goal_keys_in_csv].to_numpy(),
        success_batch=(eval_res.loc[:]["termination"].to_numpy() == "reach target"),
        cumulative_reward_batch=eval_res.loc[:]["cumulative_rewards"],
        discounted_cumulative_reward_batch=eval_res.loc[:]["discounted_cumulative_rewards"],
        desired_goal_weight_batch=[1.0] * eval_res.shape[0],
    )

    return kde_evaluator_for_eval_res

def get_kde_estimator_for_historical_eval(
    kde_data_discounted_factor: float = 0.9,
    kde_bandwidth: float = 1.0,
) -> KDEEvaluator:
    
    kde_evaluator_for_historical_eval = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs=dict(
            discounted_factor=kde_data_discounted_factor,
        ),
        kde_bandwidth=kde_bandwidth,
        kde_kernel="gaussian",
    )

    return kde_evaluator_for_historical_eval
