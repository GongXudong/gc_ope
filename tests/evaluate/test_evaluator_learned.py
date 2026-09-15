"""P0.5 学习型估计器（加权 MLP / Flow Matching）正确性与确定性测试。"""

import numpy as np
import pytest

from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_learned import (
    FlowMatchingDensityEvaluator,
    WeightedMLPDensityEvaluator,
)
from gc_ope.evaluate.evaluator_kde import KDEEvaluator


def _make_data(n: int = 80, seed: int = 42, dim: int = 2):
    rng = np.random.RandomState(seed)
    pos = rng.randn(n, dim) * 2.0
    w = np.exp(-0.5 * np.linalg.norm(pos, axis=1) ** 2)  # 伪时间权重
    return pos, w


# ---------------------------------------------------------------------------
# 确定性：同 seed 重训结果一致
# ---------------------------------------------------------------------------

def test_nn_deterministic_same_seed():
    pos, w = _make_data()
    args = dict(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=5, random_state=7,
    )
    densities = []
    for _ in range(2):
        ev = WeightedMLPDensityEvaluator(**args)
        ev.eval_res_container.add_batch(
            pos.tolist(), [True] * len(pos), [0.0] * len(pos), [0.0] * len(pos), w.tolist()
        )
        ev.eval_res_container.desired_goal_weights = w
        _, _, _, d = ev.fit_evaluator()
        densities.append(d)
    assert np.allclose(densities[0], densities[1]), "NN 重训不确定"


def test_fm_deterministic_same_seed():
    pos, w = _make_data()
    args = dict(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=3, n_flow_samples=50, random_state=7,
    )
    densities = []
    for _ in range(2):
        ev = FlowMatchingDensityEvaluator(**args)
        ev.eval_res_container.add_batch(
            pos.tolist(), [True] * len(pos), [0.0] * len(pos), [0.0] * len(pos), w.tolist()
        )
        ev.eval_res_container.desired_goal_weights = w
        _, _, _, d = ev.fit_evaluator()
        densities.append(d)
    # FM 采样固定 seed，确定性应成立
    assert np.allclose(densities[0], densities[1]), "FM 重训不确定"


# ---------------------------------------------------------------------------
# 数值 sanity：密度非负有限、evaluate_grid 一致
# ---------------------------------------------------------------------------

def test_nn_density_non_negative_finite():
    pos, w = _make_data()
    ev = WeightedMLPDensityEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=10, random_state=0,
    )
    ev.eval_res_container.add_batch(
        pos.tolist(), [True] * len(pos), [0.0] * len(pos), [0.0] * len(pos), w.tolist()
    )
    ev.eval_res_container.desired_goal_weights = w
    _, _, _, d = ev.fit_evaluator()
    assert np.all(np.isfinite(d)) and np.all(d > 0)


def test_fm_density_non_negative_finite():
    pos, w = _make_data()
    ev = FlowMatchingDensityEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        n_epochs=5, n_flow_samples=50, random_state=0,
    )
    ev.eval_res_container.add_batch(
        pos.tolist(), [True] * len(pos), [0.0] * len(pos), [0.0] * len(pos), w.tolist()
    )
    ev.eval_res_container.desired_goal_weights = w
    _, _, _, d = ev.fit_evaluator()
    assert np.all(np.isfinite(d)) and np.all(d > 0)


def test_nn_evaluate_rejects_unfitted():
    ev = WeightedMLPDensityEvaluator()
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        ev.evaluate(np.zeros((2, 2)))


def test_fm_evaluate_rejects_unfitted():
    ev = FlowMatchingDensityEvaluator()
    with pytest.raises(RuntimeError, match="fit_evaluator"):
        ev.evaluate(np.zeros((2, 2)))


# ---------------------------------------------------------------------------
# KDE baseline 不受影响（AC-1.4）
# ---------------------------------------------------------------------------

def test_kde_baseline_still_runs():
    pos, w = _make_data()
    ev = KDEEvaluator(
        evaluation_result_container_class=WeightedEvaluationResultContainer,
        evaluation_result_container_kwargs={"discounted_factor": 0.9},
        kde_bandwidth=0.2,
    )
    ev.eval_res_container.add_batch(
        pos.tolist(), [True] * len(pos), [0.0] * len(pos), [0.0] * len(pos), w.tolist()
    )
    ev.eval_res_container.desired_goal_weights = w
    _, _, _, d = ev.fit_evaluator()
    assert np.all(np.isfinite(d)) and np.all(d > 0)
