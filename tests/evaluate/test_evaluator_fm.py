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


def test_job_propagates_fm_parameters_and_strict_history(tmp_path, monkeypatch):
    """合成 CSV 验证非默认参数、时间权重、采样和无历史跳过路径。"""
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import replacement_run_job as job

    def write(step, points, flags):
        pd.DataFrame({"x": [p[0] for p in points], "y": [p[1] for p in points],
                      "termination": ["reach target" if f else "timeout" for f in flags]}).to_csv(
            tmp_path / f"rl_model_{step}_steps_eval_res_on_fixed.csv", index=False)

    write(10000, [[-0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]], [1, 1, 0])
    write(20000, [[-0.1, -0.1], [0.1, 0.1], [0.1, -0.1]], [1, 1, 0])
    write(30000, [[999., 999.]], [1])
    monkeypatch.setattr(job, "_checkpoint_dir", lambda task, seed: tmp_path)
    made = []
    factory = job._make_estimator

    def capture(*args, **kwargs):
        ev = factory(*args, **kwargs)
        made.append(ev)
        return ev

    monkeypatch.setattr(job, "_make_estimator", capture)
    kwargs = dict(kappa=0.9, n_components=5, resample_size=1000, bandwidth=0.2,
                  random_state=3, gmm_reg_covar=1e-6, n_hist_bins=10,
                  gaussian_reg_covar=1e-6, nn_epochs=2, nn_lr=1e-3, nn_hidden=4,
                  fm_epochs=2, fm_hidden=4, fm_samples=24, fm_lr=0.002,
                  fm_ode_steps=3, fm_weight_decay=0.001, fm_likelihood_batch_size=5,
                  mc_samples=16, mc_repeats=1, n_sampled_goals=2, candidate_goals=4)
    row = job._run_one_job("push", 1, 20000, "flow_matching", **kwargs)
    assert row["status"] == "ok", row
    assert row["historical_successes"] == 2
    assert np.isfinite(row["kl"]) and np.isfinite(row["fixed_grid_kl"])
    d = row["_fit_diagnostics"]
    for key, value in {"n_epochs": 2, "hidden_features": 4, "samples_per_epoch": 24,
                       "lr": .002, "ode_steps": 3, "weight_decay": .001,
                       "likelihood_batch_size": 5}.items():
        assert d[key] == value
    np.testing.assert_allclose(made[0].eval_res_container.desired_goal_weights, [.9] * 3)
    assert np.asarray(row["_sampled_goals"]).shape == (2, 2)
    early = job._run_one_job("push", 1, 10000, "flow_matching", **kwargs)
    assert early["status"] == "skipped:no_historical_successes"
    assert np.isnan(early["_sampled_goals"]).all()



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


def test_plot_loader_recognizes_fm_without_relabeling_metric(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    import pandas as pd
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import plot_fig6_kde_gmm_nn as plot
    monkeypatch.setattr(plot, "METHOD_DIR", tmp_path)
    pd.DataFrame([dict(task="push", seed=1, checkpoint=100000, method="flow_matching",
                       kl=.5, fixed_grid_kl=1.5, status="ok", error="")]).to_csv(
        tmp_path / "flow_matching_push_seed1.csv", index=False)
    frame = plot._load_current_method("flow_matching", "push", [1])
    assert frame.method.tolist() == ["FM"]
    assert frame.kl.tolist() == [.5]
    frame = plot._load_current_method("flow_matching", "push", [1], metric="fixed_grid_kl")
    assert frame.kl.tolist() == [1.5]
