from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_kde import KDEEvaluator


PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


def get_kde_evaluator():
    discounted_factor = 0.9

    kde_evaluator = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={
            "discounted_factor": discounted_factor
        },
        kde_bandwidth=1.0,
        kde_kernel="gaussian",
    )

    return discounted_factor, kde_evaluator

def test_kde_evaluator_init():
    
    discounted_factor, kde_evaluator = get_kde_evaluator()

    assert kde_evaluator.eval_res_container.desired_goal_list == []
    assert kde_evaluator.eval_res_container.success_list == []
    assert kde_evaluator.eval_res_container.cumulative_reward_list == []
    assert kde_evaluator.eval_res_container.discounted_cumulative_reward_list == []
    assert kde_evaluator.eval_res_container.desired_goal_weights.size == 0
    assert kde_evaluator.eval_res_container.discounted_factor == discounted_factor


@pytest.mark.parametrize(
    "desired_goals, success_list, cumulative_reward_list, discounted_cumulative_reward_list, weights",
    [
        (
            [
                [[1, 1, 1],
                [2, 2, 2]],
                [[3, 3, 3],
                [4, 4, 4]],
            ],
            [[True, True], [True, True]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ),
        (
            [
                [[1, 1, 1],
                [1, -1, 1]],
                [[-1, 1, 1],
                [-1, -1, 1]],
            ],
            [[True, True], [True, True]],
            [[1.0, 2.0], [3.0, 4.0]],
            [[10.0, 20.0], [30.0, 40.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        ),
    ],
)
def test_kde_evaluator(desired_goals, success_list, cumulative_reward_list, discounted_cumulative_reward_list, weights):

    discounted_factor, kde_evaluator = get_kde_evaluator()

    for dgs, scs, crs, dcrs, ws in zip(desired_goals, success_list, cumulative_reward_list, discounted_cumulative_reward_list, weights):

        kde_evaluator.eval_res_container.add_batch(dgs, scs, crs, dcrs, ws)

    normalized_dgs, dg_weights, dg_densities = kde_evaluator.fit_evaluator()
    print(list(zip(normalized_dgs, dg_weights, dg_densities)))

    # evaluate
    tmp_dg_arr = np.array(desired_goals)
    squeezed_dgs = tmp_dg_arr.reshape(([-1, tmp_dg_arr.shape[-1]]))
    normalized_dgs_1, dg_densities_1 = kde_evaluator.evaluate(desired_goals=squeezed_dgs, scale=True)
    print(list(zip(normalized_dgs_1, dg_densities_1)))

    assert np.allclose(normalized_dgs, normalized_dgs_1)
    assert np.allclose(dg_densities, dg_densities_1)

    normalized_dgs_2, dg_densities_2 = kde_evaluator.evaluate(desired_goals=normalized_dgs, scale=False)

    assert np.allclose(normalized_dgs, normalized_dgs_2)
    assert np.allclose(dg_densities, dg_densities_2)


if __name__ == "__main__":
    test_kde_evaluator_init()
    test_kde_evaluator(
        [
            [[1, 1, 1],
            [1, -1, 1]],
            [[-1, 1, 1],
            [-1, -1, 1]],
        ],
        [[True, True], [True, True]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[10.0, 20.0], [30.0, 40.0]],
        [[1.0, 1.0], [1.0, 1.0]],
    )
