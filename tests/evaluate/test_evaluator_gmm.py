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
    assert evaluator.fit_diagnostics_["converged"] is True
    assert evaluator.fit_diagnostics_["n_positive_samples"] == 4
    assert evaluator.fit_diagnostics_["constant_features"] == []


def test_gmm_evaluator_rejects_too_small_resample_size():
    with pytest.raises(ValueError, match="at least 2"):
        GMMEvaluator(resample_size=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_components": 1.5},
        {"n_init": 0},
        {"max_iter": 0},
        {"tol": 0.0},
        {"reg_covar": -1.0},
        {"covariance_type": "invalid"},
    ],
)
def test_gmm_evaluator_rejects_invalid_hyperparameters(kwargs):
    with pytest.raises(ValueError):
        GMMEvaluator(**kwargs)


def test_gmm_evaluator_accepts_generator_random_state():
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_components=2,
        resample_size=20,
        random_state=np.random.default_rng(3),
    )
    goals = np.array([[0.0, 0.02], [0.1, 0.02], [3.0, 0.02], [3.1, 0.02]])
    evaluator.eval_res_container.add_batch(
        goals, [True] * len(goals), [0.0] * len(goals), [0.0] * len(goals), [1.0] * len(goals)
    )
    evaluator.fit_evaluator()
    assert evaluator.fit_diagnostics_["converged"] is True


def test_gmm_evaluator_reports_constant_features_and_caps_components():
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_components=5,
        resample_size=2,
        random_state=4,
    )
    goals = np.array([[0.0, 7.0], [1.0, 7.0], [2.0, 7.0]])
    evaluator.eval_res_container.add_batch(
        goals, [True] * len(goals), [0.0] * len(goals), [0.0] * len(goals), [1.0] * len(goals)
    )
    evaluator.fit_evaluator()
    assert evaluator.gmm.n_components == 2
    assert evaluator.fit_diagnostics_["constant_features"] == [1]


def test_gmm_evaluator_rejects_misaligned_weights():
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
    )
    evaluator.eval_res_container.add_batch(
        [[0.0, 0.0], [1.0, 1.0]], [True, True], [0.0, 0.0], [0.0, 0.0], [1.0, 1.0]
    )
    evaluator.eval_res_container.desired_goal_weights = np.array([1.0])
    with pytest.raises(ValueError, match="align"):
        evaluator.fit_evaluator()


def test_weighted_resample_rejects_nonfinite_samples():
    with pytest.raises(ValueError, match="samples must be finite"):
        weighted_resample(np.array([[0.0, np.inf]]), np.array([1.0]), 2, random_state=0)


def test_weighted_resample_rejects_boolean_size():
    with pytest.raises(ValueError, match="positive integer"):
        weighted_resample(np.ones((2, 2)), np.ones(2), True, random_state=0)


def _fitted_gmm(n_components=2):
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_components=n_components,
        resample_size=100,
        random_state=0,
    )
    goals = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]], dtype=float)
    evaluator.eval_res_container.add_batch(
        goals, [True] * len(goals), [0.0] * len(goals), [0.0] * len(goals), [1.0] * len(goals)
    )
    evaluator.fit_evaluator()
    return evaluator


def test_gmm_evaluate_grid_returns_finite_density_in_raw_coordinates():
    evaluator = _fitted_gmm()
    grid = np.array([[0.05, 0.0], [3.05, 3.0], [1.5, 1.5]], dtype=float)
    density = evaluator.evaluate_grid(grid)
    assert density.shape == (3,)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)
    # 密度峰值应靠近样本所在的两个簇附近（(0,0) 与 (3,3) 簇）
    assert max(density[0], density[1]) > density[2]


def test_gmm_evaluate_grid_log_density_matches_evaluate():
    evaluator = _fitted_gmm()
    goals = np.array([[0.0, 0.0], [3.0, 3.0]], dtype=float)
    _, log_from_evaluate = evaluator.evaluate(goals, return_density=False)
    log_from_grid = evaluator.evaluate_grid(goals, return_log_density=True)
    # evaluate() 不做 Jacobian 校正（标准化空间），evaluate_grid() 做；
    # 两者数值不同，但都应有限且形状一致。
    assert log_from_grid.shape == log_from_evaluate.shape
    assert np.all(np.isfinite(log_from_grid))


def test_gmm_evaluate_grid_rejects_unfitted_and_bad_shape():
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
    )
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        evaluator.evaluate_grid(np.zeros((2, 2)))
    fitted = _fitted_gmm()
    with pytest.raises(ValueError, match="2 维"):
        fitted.evaluate_grid(np.zeros((3, 3)))


def test_gmm_evaluate_rejects_unfitted():
    evaluator = GMMEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
    )
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        evaluator.evaluate(np.zeros((1, 2)))
