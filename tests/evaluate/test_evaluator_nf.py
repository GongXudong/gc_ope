"""Normalizing Flow evaluator targeted tests."""

import numpy as np

from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_nf import NormalizingFlowDensityEvaluator


def _fit(seed: int = 0, epochs: int = 3) -> NormalizingFlowDensityEvaluator:
    goals = np.array(
        [[-0.2, -0.1], [-0.1, 0.0], [0.0, 0.1], [0.1, 0.0], [0.2, 0.1]],
        dtype=float,
    )
    weights = np.array([0.7, 0.8, 1.0, 0.9, 0.6], dtype=float)
    ev = NormalizingFlowDensityEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=epochs,
        hidden_features=8,
        transforms=2,
        bins=4,
        random_state=seed,
    )
    ev.eval_res_container.add_batch(
        goals.tolist(), [True] * len(goals), [0.0] * len(goals), [0.0] * len(goals), weights.tolist()
    )
    ev.eval_res_container.desired_goal_weights = weights
    ev.fit_evaluator()
    return ev


def test_nf_density_is_finite_and_positive():
    ev = _fit()
    query = np.array([[-0.15, -0.05], [0.0, 0.0], [0.15, 0.05]])
    _, density = ev.evaluate(query)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)
    raw_density = ev.evaluate_grid(query)
    assert np.all(np.isfinite(raw_density))
    assert np.all(raw_density > 0)


def test_nf_reproducible_with_same_seed():
    first = _fit(seed=7).evaluate_grid(np.array([[-0.1, 0.0], [0.1, 0.0]]))
    second = _fit(seed=7).evaluate_grid(np.array([[-0.1, 0.0], [0.1, 0.0]]))
    assert np.allclose(first, second, rtol=1e-6, atol=1e-8)


def test_nf_rejects_unfitted_evaluate():
    ev = NormalizingFlowDensityEvaluator()
    try:
        ev.evaluate(np.zeros((1, 2)))
    except RuntimeError as exc:
        assert "fit_evaluator" in str(exc)
    else:
        raise AssertionError("unfitted NF should raise RuntimeError")
