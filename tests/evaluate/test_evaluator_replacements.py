"""Replacement estimators: weighted histogram and weighted multivariate Gaussian.

覆盖 AC-1.1（直方图概率归一）、AC-1.2（加权均值/协方差手算对照）、
AC-1.3（resampling 权重生效）、AC-5.2（协方差 PSD/对称）。
"""

import numpy as np
import pytest
from scipy.stats import multivariate_normal

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_gmm import weighted_resample
from gc_ope.evaluate.evaluator_kde import KDEEvaluator
from gc_ope.evaluate.evaluator_replacements import (
    WeightedGaussianEvaluator,
    WeightedHistogramEvaluator,
    gaussian_mean_cov,
)


def _make_weighted_container(discounted_factor: float = 0.9) -> WeightedEvaluationResultContainer:
    return WeightedEvaluationResultContainer(discounted_factor=discounted_factor)


# ---------------------------------------------------------------------------
# AC-1.1：直方图概率归一
# ---------------------------------------------------------------------------

def test_histogram_probability_sums_to_one():
    ev = WeightedHistogramEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_bins=4,
    )
    goals = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.5, 1.0]])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    ev.eval_res_container.add_batch(
        goals.tolist(), [True] * 4, [0.0] * 4, [0.0] * 4, weights.tolist()
    )
    ev.eval_res_container.desired_goal_weights = weights
    ev.fit_evaluator()
    assert abs(ev.fit_diagnostics_["probability_sum"] - 1.0) < 1e-9


def test_histogram_density_non_negative_and_finite_on_grid():
    ev = WeightedHistogramEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_bins=4,
    )
    goals = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.5, 1.0]])
    weights = np.array([0.4, 0.3, 0.2, 0.1])
    ev.eval_res_container.add_batch(
        goals.tolist(), [True] * 4, [0.0] * 4, [0.0] * 4, weights.tolist()
    )
    ev.eval_res_container.desired_goal_weights = weights
    ev.fit_evaluator()
    grid = np.array([[0.0, 0.0], [1.0, 2.0], [0.0, 1.0], [2.0, 3.0]])
    density = ev.evaluate_grid(grid)
    assert np.all(np.isfinite(density))
    assert np.all(density >= 0.0)


# ---------------------------------------------------------------------------
# AC-1.2：加权均值 / 协方差手算对照
# ---------------------------------------------------------------------------

def test_gaussian_weighted_mean_and_cov_match_hand_calculation():
    goals = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    weights = np.array([0.5, 0.3, 0.2])
    mu, cov = gaussian_mean_cov(goals, weights)
    assert np.allclose(mu, [0.3, 0.4])
    ref_cov = ((goals - mu) * weights[:, None]).T @ (goals - mu) / weights.sum()
    assert np.allclose(cov, ref_cov)


def test_gaussian_density_matches_scipy_raw_space():
    """evaluate_grid（Jacobian 校正后）应与 scipy 直接算 raw-space Gaussian 一致。"""
    goals = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 2.0]])
    weights = np.array([0.5, 0.3, 0.2])
    reg_covar = 1e-6

    ev = WeightedGaussianEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        reg_covar=reg_covar,
    )
    ev.eval_res_container.add_batch(
        goals.tolist(), [True] * 3, [0.0] * 3, [0.0] * 3, weights.tolist()
    )
    ev.eval_res_container.desired_goal_weights = weights
    ev.fit_evaluator()

    mu, cov = gaussian_mean_cov(goals, weights, reg_covar)
    query = np.array([mu])
    density = ev.evaluate_grid(query)
    ref = multivariate_normal(mean=mu, cov=cov).pdf(query[0])
    assert np.allclose(density, [ref], rtol=1e-3), (density[0], ref)


# ---------------------------------------------------------------------------
# AC-1.3：resampling-based GMM 权重确实生效
# ---------------------------------------------------------------------------

def test_resampling_moves_mass_toward_high_weight_samples():
    # 两个样本，权重悬殊：高权重样本应被采得显著更多
    samples = np.array([[0.0], [10.0]])
    weights = np.array([0.05, 0.95])
    resampled = weighted_resample(samples, weights, 200, random_state=42)
    frac_low = float(np.mean(resampled[:, 0] == 0.0))
    # 期望 freq ≈ 0.05，不应接近 0.5
    assert frac_low < 0.2, f"low-weight sample drawn {frac_low:.3f} of the time"


def test_resampling_zero_weight_never_drawn():
    samples = np.array([[0.0], [10.0]])
    resampled = weighted_resample(samples, np.array([1.0, 0.0]), 100, random_state=0)
    assert np.all(resampled == 0.0)


# ---------------------------------------------------------------------------
# AC-1.4：KDE baseline 不受影响
# ---------------------------------------------------------------------------

def test_kde_baseline_still_runs_independently():
    ev = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        kde_bandwidth=0.2,
    )
    goals = np.array([[0.0, 0.0], [0.1, 0.0], [3.0, 3.0], [3.1, 3.0]])
    ev.eval_res_container.add_batch(
        goals.tolist(), [True] * 4, [0.0] * 4, [0.0] * 4, [1.0] * 4
    )
    _, _, _, densities = ev.fit_evaluator()
    assert np.all(np.isfinite(densities)) and np.all(densities > 0)


# ---------------------------------------------------------------------------
# AC-5.2 / AC-5.3：协方差合法性与数值 sanity
# ---------------------------------------------------------------------------

def test_gaussian_cov_is_symmetric_psd():
    goals = np.array([[0.0, 0.0], [1.0, 0.2], [0.5, 0.8], [2.0, 1.0]])
    weights = np.array([0.2, 0.3, 0.25, 0.25])
    _, cov = gaussian_mean_cov(goals, weights, reg_covar=1e-6)
    assert np.allclose(cov, cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    assert np.all(eigvals > 0), eigvals


def test_gaussian_rejects_bad_hyperparameter():
    with pytest.raises(ValueError):
        WeightedGaussianEvaluator(reg_covar=-1.0)
    with pytest.raises(ValueError):
        WeightedHistogramEvaluator(n_bins=1)


def test_gaussian_evaluate_rejects_unfitted():
    ev = WeightedGaussianEvaluator()
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        ev.evaluate(np.zeros((2, 2)))


def test_histogram_evaluate_rejects_unfitted():
    ev = WeightedHistogramEvaluator()
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        ev.evaluate(np.zeros((2, 2)))
