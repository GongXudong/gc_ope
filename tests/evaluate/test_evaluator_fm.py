"""Flow Matching 估计器的最小数值验收。"""

import numpy as np
import pytest

from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_fm import FlowMatchingDensityEvaluator


def _fit(seed: int = 0, epochs: int = 2) -> FlowMatchingDensityEvaluator:
    rng = np.random.default_rng(123)
    goals = rng.normal(loc=0.0, scale=0.12, size=(24, 2))
    weights = np.linspace(0.5, 1.5, len(goals))
    evaluator = FlowMatchingDensityEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=epochs,
        hidden_features=8,
        samples_per_epoch=64,
        ode_steps=4,
        likelihood_batch_size=7,
        random_state=seed,
    )
    evaluator.eval_res_container.add_batch(
        goals.tolist(), [True] * len(goals), [0.0] * len(goals), [0.0] * len(goals), weights.tolist()
    )
    evaluator.eval_res_container.desired_goal_weights = weights
    evaluator.fit_evaluator()
    return evaluator


def test_fm_density_is_finite_and_positive():
    evaluator = _fit()
    query = np.array([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.05]])
    _, density = evaluator.evaluate(query)
    raw_density = evaluator.evaluate_grid(query)
    assert np.all(np.isfinite(density)) and np.all(density > 0)
    assert np.all(np.isfinite(raw_density)) and np.all(raw_density > 0)
    assert evaluator.fit_diagnostics_["early_stopping"] is False
    assert evaluator.fit_diagnostics_["validation_fraction"] == 0.0


def test_fm_reproducible_with_same_seed():
    query = np.array([[-0.1, 0.0], [0.0, 0.0], [0.1, 0.05]])
    first = _fit(seed=7).evaluate_grid(query)
    second = _fit(seed=7).evaluate_grid(query)
    assert np.allclose(first, second, rtol=1e-6, atol=1e-8)


def test_fm_likelihood_query_is_chunked():
    evaluator = _fit()
    query = np.tile(np.array([[0.0, 0.0]]), (19, 1))
    values = evaluator.evaluate_grid(query)
    assert values.shape == (19,)
    assert np.all(np.isfinite(values))


def test_fm_rejects_unfitted_evaluate():
    evaluator = FlowMatchingDensityEvaluator()
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        evaluator.evaluate(np.zeros((1, 2)))


def test_fm_uses_fixed_training_budget_without_validation():
    evaluator = _fit()
    assert evaluator.fit_diagnostics_["training_scheme"].endswith("no validation")


@pytest.mark.parametrize("a", [0.0, 0.4, -0.3])
def test_analytic_linear_flow_likelihood_sign_and_jacobian(a):
    """已知 v(x)=a*x 的解验证反向积分符号、raw Jacobian 和 no_grad。"""
    import torch

    class LinearVelocity(torch.nn.Module):
        def forward(self, xt):
            return a * xt[:, :2]

    evaluator = FlowMatchingDensityEvaluator(ode_steps=32, likelihood_batch_size=2)
    evaluator.model = LinearVelocity()
    evaluator._fitted = True
    evaluator.scaler.fit(np.array([[-2., -4.], [2., 4.]]))
    raw = np.array([[1., 2.], [-0.3, 0.7], [0., 0.]])
    scaled = evaluator.scaler.transform(raw)
    source = scaled * np.exp(-a)
    expected = -0.5 * (source ** 2).sum(axis=1) - np.log(2 * np.pi) - 2 * a
    with torch.no_grad():
        _, got = evaluator.evaluate(raw, return_density=False)
    np.testing.assert_allclose(got, expected, atol=2e-5)
    _, already_scaled = evaluator.evaluate(scaled, scale=False, return_density=False)
    np.testing.assert_allclose(got, already_scaled, atol=1e-6)
    np.testing.assert_allclose(
        evaluator.evaluate_grid(raw, return_log_density=True),
        expected - np.log(evaluator.scaler.scale_).sum(), atol=2e-5,
    )


def test_weighted_sampling_probabilities_and_failure_exclusion(monkeypatch):
    """直接截取 multinomial 输入，确认权重仅注入一次且失败目标不入模。"""
    import torch
    original = torch.multinomial
    seen = []

    def capture(weights, *args, **kwargs):
        seen.append(weights.detach().numpy().copy())
        return original(weights, *args, **kwargs)

    monkeypatch.setattr(torch, "multinomial", capture)
    ev = FlowMatchingDensityEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=2, hidden_features=4, samples_per_epoch=16, ode_steps=2,
    )
    ev.eval_res_container.add_batch(
        [[-1., -2.], [1., 2.], [999., 999.]], [True, True, False],
        [0.] * 3, [0.] * 3, [1., 3., 100.],
    )
    positive, _, weights, _ = ev.fit_evaluator()
    assert len(positive) == 2
    np.testing.assert_allclose(weights, [1., 3.])
    np.testing.assert_allclose(ev.scaler.mean_, [0., 0.])
    assert len(seen) == 2
    for probabilities in seen:
        np.testing.assert_allclose(probabilities, [0.25, 0.75])


@pytest.mark.parametrize("kwargs", [
    {"ode_steps": 0}, {"n_epochs": -1}, {"samples_per_epoch": 0},
    {"likelihood_batch_size": 0}, {"lr": float("nan")},
    {"weight_decay": -1}, {"random_state": -1}, {"device": "cuda"},
])
def test_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        FlowMatchingDensityEvaluator(**kwargs)


def test_fm_fit_preserves_global_rng_and_query_does_not_accumulate_gradients():
    import torch
    torch.manual_seed(32)
    before = torch.random.get_rng_state().clone()
    ev = _fit()
    assert torch.equal(before, torch.random.get_rng_state())
    for parameter in ev.model.parameters():
        parameter.grad = None
    ev.evaluate_grid(np.array([[0., 0.]]))
    assert all(parameter.grad is None for parameter in ev.model.parameters())
