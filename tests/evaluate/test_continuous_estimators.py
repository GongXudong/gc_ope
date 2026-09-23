"""连续密度、MC 方向与采样的一致性检查；可单文件直接运行。"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gc_ope.evaluate.evaluator_factory import make_evaluator, fit_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch
from gc_ope.evaluate.utils.distribution_kl import monte_carlo_kl


PARAMETERS = {
    "kde": {}, "gmm": {"n_components": 2},
    "nn": {"n_epochs": 5, "early_stopping": False},
    "nf": {"n_epochs": 2, "hidden_layer_sizes": [8, 8], "transforms": 2},
    "fm": {"n_members": 2, "member_parameters": {"n_epochs": 2, "hidden_layer_sizes": [8, 8, 8], "samples_per_epoch": 32, "ode_steps": 4}},
}


def fitted(method):
    rng = np.random.default_rng(7)
    goals = rng.normal([2, -3], [0.3, 0.6], (100, 2))
    labels = goals[:, 0] > 2
    support = np.array(np.meshgrid(np.linspace(1, 3, 8), np.linspace(-5, -1, 8))).reshape(2, -1).T
    model = make_evaluator(method, support_goals=support, parameters=PARAMETERS[method])
    EvaluationBatch(goals, labels, np.linspace(.1, 1, 100)).fill(model)
    fit_evaluator(method, model)
    return model


@pytest.mark.parametrize("method", PARAMETERS)
def test_shared_density_contract_and_seeded_sampling(method):
    import torch
    before_numpy = np.random.get_state()
    before_torch = torch.random.get_rng_state().clone()
    model = fitted(method)
    samples = model.sample(100, 17)
    np.testing.assert_array_equal(samples, model.sample(100, 17))
    assert not np.array_equal(samples, model.sample(100, 18))
    assert samples.shape == (100, 2) and np.isfinite(samples).all()
    # 独立随机流不能扰动未来 SAC 的随机行为。
    np.testing.assert_array_equal(before_numpy[1], np.random.get_state()[1])
    assert before_numpy[2:] == np.random.get_state()[2:]
    assert torch.equal(before_torch, torch.random.get_rng_state())
    transformed, scaled_log = model.evaluate(samples, return_density=False)
    _, direct_log = model.evaluate(transformed, scale=False, return_density=False)
    np.testing.assert_allclose(scaled_log, direct_log, atol=1e-6)
    np.testing.assert_allclose(model.log_density(samples), scaled_log - np.log(model.scaler.scale_).sum())
    # 同一分布与自身 KL 必须为零，不能误用反向分布或另一个坐标系。
    metric = monte_carlo_kl(model, model, n_samples=100, repeats=2)
    assert metric["kl"] == 0
    assert metric["mc_kept"] == [100, 100]


def test_nn_density_integrates_to_one_and_sample_moments_match():
    model = fitted("nn")
    # 包含边界外高斯尾部；不能只在 [0,1]^2 或支持网格上检验连续归一化。
    axis_x = np.linspace(-1, 5, 301)
    axis_y = np.linspace(-8, 2, 401)
    grid = np.array(np.meshgrid(axis_x, axis_y)).reshape(2, -1).T
    density = np.exp(model.log_density(grid)).reshape(len(axis_y), len(axis_x))
    integral = np.trapezoid(np.trapezoid(density, axis_x, axis=1), axis_y)
    assert integral == pytest.approx(1, abs=1e-4)
    samples = model.sample(50000, 41)
    centers = model.support_goals_
    weights = model.support_weights_
    mean = (weights[:, None] * centers).sum(0)
    variance = (weights[:, None] * (centers - mean)**2).sum(0) + (model.bandwidth * model.scaler.scale_)**2
    np.testing.assert_allclose(samples.mean(0), mean, atol=.02)
    np.testing.assert_allclose(samples.var(0), variance, rtol=.04)


def test_mc_direction_against_analytic_gaussians():
    class Gaussian:
        def __init__(self, mean, variance):
            self.distribution = multivariate_normal(mean, np.eye(2) * variance)
        def sample(self, n, random_state):
            return self.distribution.rvs(n, random_state=random_state)
        def log_density(self, goals):
            return self.distribution.logpdf(goals)
    p, q = Gaussian([0, 0], 1), Gaussian([2, 0], 4)
    # KL(N(0,I) || N((2,0),4I)) = .5*(.5 + 1 - 2 + log(16))。
    expected = .5 * (.5 + 1 - 2 + np.log(16))
    got = monte_carlo_kl(p, q, 10000, 5)["kl"]
    assert got == pytest.approx(expected, abs=.025)
    assert abs(got - monte_carlo_kl(q, p, 10000, 5)["kl"]) > 1


def test_fm_forward_sampling_matches_known_linear_flow():
    import torch
    from gc_ope.evaluate.evaluator_fm import _FMMember
    class Velocity(torch.nn.Module):
        def forward(self, xt):
            return .4 * xt[:, :2]
    model = _FMMember(ode_steps=32)
    model.model, model._fitted = Velocity(), True
    model.scaler.fit([[-2, -4], [2, 4]])
    source = torch.randn((100, 2), generator=torch.Generator().manual_seed(9)).numpy()
    expected = model.scaler.inverse_transform(source * np.exp(.4))
    np.testing.assert_allclose(model.sample(100, 9), expected, atol=1e-5)


def test_legacy_mode_keeps_original_low_density_filter():
    from sklearn.preprocessing import StandardScaler
    class ConstantLogs:
        scaler = StandardScaler().fit([[-1, -1], [1, 1]])
        def sample(self, n, random_state):
            return np.zeros((n, 2))
        def evaluate(self, samples, scale=False, return_density=False):
            return samples, np.resize([0., -30.], len(samples))
    metric = monte_carlo_kl(ConstantLogs(), ConstantLogs(), 100, 2, mode="legacy")
    assert metric["mc_kept"] == [50, 50]
    assert metric["kl"] == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
