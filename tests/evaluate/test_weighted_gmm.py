"""直接加权 EM 的数学性质、概率接口和异常边界；单文件运行。"""

import numpy as np
import pytest
from scipy.special import logsumexp
from sklearn.mixture import GaussianMixture
from sklearn.exceptions import NotFittedError, ConvergenceWarning

from gc_ope.evaluate.utils.weighted_gmm import WeightedGaussianMixture
from gc_ope.evaluate.evaluator_factory import make_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch


@pytest.fixture
def mixture():
    rng = np.random.default_rng(9)
    X = np.r_[rng.normal([-3, 0], [.4, .8], (80, 2)),
              rng.normal([3, 1], [.6, .4], (80, 2))]
    weights = np.r_[np.ones(80), np.full(80, 4.)]
    return X, weights


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_one_component_matches_analytic_weighted_moments(covariance_type):
    X = np.array([[0., 1.], [2., 0.], [4., 3.], [-1., 2.]])
    w = np.array([1., 2., 5., 3.])
    model = WeightedGaussianMixture(1, covariance_type, reg_covar=1e-5).fit(X, sample_weight=w)
    mean = np.average(X, axis=0, weights=w)
    covariance = ((X-mean).T*w)@(X-mean)/w.sum()
    expected = covariance + np.eye(2)*1e-5 if covariance_type == "full" else np.diag(covariance)+1e-5
    np.testing.assert_allclose(model.means_[0], mean, atol=1e-12)
    np.testing.assert_allclose(model.covariances_[0], expected, atol=1e-12)
    assert model.lower_bound_ == pytest.approx(np.average(model.score_samples(X), weights=w))


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_integer_weights_equal_repeating_samples(mixture, covariance_type):
    X, w = mixture
    weighted = WeightedGaussianMixture(2, covariance_type, tol=1e-10).fit(X, sample_weight=w)
    repeated = GaussianMixture(2, covariance_type=covariance_type, tol=1e-10, random_state=0).fit(
        np.repeat(X, w.astype(int), axis=0))
    np.testing.assert_allclose(weighted.score_samples(X), repeated.score_samples(X), atol=1e-7)
    # 两群记录数相等，但统计质量为 1:4；权重必须真正进入模型参数。
    order = weighted.means_[:, 0].argsort()
    np.testing.assert_allclose(weighted.weights_[order], [.2, .8], atol=1e-5)


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_equal_weights_match_sklearn(mixture, covariance_type):
    X, _ = mixture
    model = WeightedGaussianMixture(2, covariance_type, tol=1e-10).fit(X)
    standard = GaussianMixture(2, covariance_type=covariance_type, tol=1e-10, random_state=0).fit(X)
    np.testing.assert_allclose(model.score_samples(X), standard.score_samples(X), atol=1e-7)


def test_weight_scaling_zero_weight_and_rng_invariance(mixture):
    X, w = mixture
    before = np.random.get_state()
    a = WeightedGaussianMixture(2).fit(X, sample_weight=w)
    b = WeightedGaussianMixture(2).fit(np.r_[X, [[100., -100.]]], sample_weight=np.r_[w*1e200, 0])
    np.testing.assert_allclose(a.score_samples(X), b.score_samples(X), atol=1e-12)
    np.testing.assert_array_equal(a.sample(100, 13)[0], a.sample(100, 13)[0])
    np.testing.assert_array_equal(before[1], np.random.get_state()[1])
    assert before[2:] == np.random.get_state()[2:]


def test_weighted_likelihood_improves_and_responsibilities_normalize():
    rng = np.random.default_rng(31)
    X = np.r_[rng.normal(-.5, 1, (100, 2)), rng.normal(.8, 1.2, (100, 2))]
    w = np.linspace(.01, 1, len(X))
    model = WeightedGaussianMixture(2, reg_covar=0, tol=1e-6, max_iter=500).fit(X, sample_weight=w)
    assert np.min(np.diff(model.lower_bounds_)) >= -1e-10
    np.testing.assert_allclose(model.predict_proba(X).sum(axis=1), 1, atol=1e-12)
    assert np.isfinite(model.score_samples([[100., -100.]])).all()


@pytest.mark.parametrize("covariance_type", ["full", "diag"])
def test_density_and_samples_are_same_distribution(mixture, covariance_type):
    X, w = mixture
    model = WeightedGaussianMixture(2, covariance_type).fit(X, sample_weight=w)
    samples, labels = model.sample(80000, 11)
    mean = model.weights_ @ model.means_
    variances = (np.diagonal(model.covariances_, axis1=1, axis2=2)
                 if covariance_type == "full" else model.covariances_)
    variance = model.weights_ @ (variances+model.means_**2) - mean**2
    np.testing.assert_allclose(samples.mean(0), mean, atol=.04)
    np.testing.assert_allclose(samples.var(0), variance, rtol=.04)
    from scipy.stats import multivariate_normal
    terms = [np.log(model.weights_[k])+multivariate_normal.logpdf(X, model.means_[k],
             model.covariances_[k] if covariance_type == "full" else np.diag(model.covariances_[k]))
             for k in range(2)]
    np.testing.assert_allclose(model.score_samples(X), logsumexp(terms, axis=0), atol=1e-10)


@pytest.mark.parametrize("weights", [[0, 0], [1, -1], [1, np.nan], [1], [[1, 2]]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        WeightedGaussianMixture(1).fit([[0, 0], [1, 1]], sample_weight=weights)


def test_singular_small_sample_and_refit_failure_are_explicit():
    model = WeightedGaussianMixture(1).fit([[0., 0.]])
    assert np.isfinite(model.score_samples([[0., 0.]])).all()
    with pytest.raises(ValueError):
        model.fit([[0., 0.]], sample_weight=[0])
    with pytest.raises(NotFittedError):
        model.score_samples([[0., 0.]])
    with pytest.raises(ValueError, match="正定"):
        WeightedGaussianMixture(1, reg_covar=0).fit([[0., 0.]])


def test_multiple_initializations_select_best_weighted_objective(mixture):
    X, w = mixture
    a = WeightedGaussianMixture(2, n_init=1).fit(X, sample_weight=w)
    b = WeightedGaussianMixture(2, n_init=3).fit(X, sample_weight=w)
    assert b.lower_bound_ >= a.lower_bound_ - 1e-12


def test_iteration_limit_is_reported():
    X = np.random.default_rng(1).normal(size=(100, 2))
    with pytest.warns(ConvergenceWarning):
        model = WeightedGaussianMixture(3, max_iter=1, tol=1e-15).fit(X)
    assert not model.converged_ and model.n_iter_ == 1


def test_evaluator_caps_components_and_keeps_raw_density_contract():
    X = np.array([[1., 2.], [1., 2.]])
    model = make_evaluator("gmm")
    EvaluationBatch(X, np.ones(2, dtype=bool), np.array([.2, 1.])).fill(model)
    model.fit_evaluator()
    assert model.gmm.n_components == 1
    assert model.fit_diagnostics_["fitting_scheme"] == "direct_weighted_em"
    assert model.fit_diagnostics_["weighted_nll"] == -model.gmm.lower_bound_
    samples = model.sample(30, 1)
    scaled, logs = model.evaluate(samples, return_density=False)
    np.testing.assert_allclose(model.evaluate(scaled, scale=False, return_density=False)[1], logs)
    np.testing.assert_allclose(model.log_density(samples), logs-np.log(model.scaler.scale_).sum())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
