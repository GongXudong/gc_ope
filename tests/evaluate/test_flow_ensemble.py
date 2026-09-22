"""集成密度的归一化、采样与 raw/standardized 接口一致性。"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal
from gc_ope.evaluate.evaluator_flow_ensemble import FlowEnsembleEvaluator
from gc_ope.evaluate.evaluator_factory import make_evaluator, fit_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch


def test_density_is_arithmetic_mixture_and_sampling_matches_it():
    class Gaussian:
        def __init__(self, mean):
            self.distribution = multivariate_normal(mean, np.eye(2) * .04)
        def log_density(self, goals):
            return self.distribution.logpdf(goals)
        def sample(self, n, random_state):
            return np.asarray(self.distribution.rvs(n, random_state=random_state)).reshape(-1, 2)
    model = FlowEnsembleEvaluator(n_members=2)
    model.scaler.fit([[-3., -1.], [5., 3.]])
    model.members_ = [Gaussian([-2., 0.]), Gaussian([2., 0.])]
    points = np.array([[-2., 0.], [0., 0.], [2., 0.]])
    expected = np.mean([np.exp(member.log_density(points)) for member in model.members_], axis=0)
    np.testing.assert_allclose(np.exp(model.log_density(points)), expected)
    scaled, log = model.evaluate(points, return_density=False)
    np.testing.assert_allclose(model.evaluate(scaled, scale=False, return_density=False)[1], log)
    samples = model.sample(30000, 13)
    np.testing.assert_array_equal(samples, model.sample(30000, 13))
    np.testing.assert_allclose(samples.mean(0), [0, 0], atol=.035)
    np.testing.assert_allclose(samples.var(0), [4.04, .04], rtol=.04)


def test_one_member_matches_old_fm_and_multiple_members_are_independent():
    parameters = dict(n_epochs=2, hidden_features=8, samples_per_epoch=32, ode_steps=4)
    x = np.random.default_rng(4).normal(size=(30, 2))
    data = EvaluationBatch(x, np.ones(30, bool), np.linspace(.1, 1, 30))
    old = make_evaluator("fm", parameters=parameters)
    single = make_evaluator("fm_ensemble", parameters=dict(n_members=1, member_parameters=parameters))
    multi = make_evaluator("fm_ensemble", parameters=dict(n_members=2, member_parameters=parameters))
    for method, model in [("fm", old), ("fm_ensemble", single), ("fm_ensemble", multi)]:
        data.fill(model)
        fit_evaluator(method, model)
    np.testing.assert_allclose(old.log_density(x), single.log_density(x), atol=1e-6)
    np.testing.assert_array_equal(old.sample(20, 7), single.sample(20, 7))
    assert multi.members_[0] is not multi.members_[1]
    assert multi.fit_diagnostics_["member_seeds"] == [0, 1]
    assert not np.allclose(multi.members_[0].log_density(x), multi.members_[1].log_density(x))


@pytest.mark.parametrize("parameters", [{"n_members": 0}, {"n_members": True},
    {"member_parameters": {"random_state": 0}}])
def test_invalid_configuration_rejected(parameters):
    with pytest.raises(ValueError):
        FlowEnsembleEvaluator(**parameters)
