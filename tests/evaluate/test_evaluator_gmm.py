import numpy as np
import pytest

from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_gmm import GMMEvaluator, weighted_resample


def test_weighted_resample_is_reproducible_and_weighted():
    samples = np.arange(12, dtype=float).reshape(6, 2)
    weights = np.array([0, 0, 0, 0, 0, 1.0])
    a = weighted_resample(samples, weights, 20, random_state=7)
    b = weighted_resample(samples, weights, 20, random_state=7)
    assert np.array_equal(a, b)
    assert np.all(a == samples[-1])


@pytest.mark.parametrize("weights", [[1, -1], [np.nan, 1], [0, 0]])
def test_weighted_resample_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_resample(np.ones((2, 2)), np.asarray(weights), 2, random_state=0)


def test_gmm_evaluator_fit_and_evaluate():
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_components=4,
        resample_size=100,
        random_state=3,
    )
    goals = np.array([[0, 0], [0.1, 0], [3, 3], [3.1, 3], [2, 2]], dtype=float)
    evaluator.eval_res_container.add_batch(
        goals, [True, True, True, True, False], [0] * 5, [0] * 5, [1] * 5
    )
    positive, scaled, weights, densities = evaluator.fit_evaluator()
    scaled_again, densities_again = evaluator.evaluate(positive)
    assert positive.shape == (4, 2)
    assert scaled.shape == positive.shape
    assert np.all(np.isfinite(densities)) and np.all(densities > 0)
    assert np.allclose(scaled, scaled_again)
    assert np.allclose(densities, densities_again)
    assert evaluator.gmm.n_components == 4

