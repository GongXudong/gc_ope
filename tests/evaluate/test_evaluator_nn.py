"""sklearn MLPClassifier 成功率分布估计器测试。"""

import numpy as np
import pytest

from gc_ope.evaluate.evaluator_nn import GoalSuccessMLPClassifierEvaluator


def _data():
    rng = np.random.RandomState(42)
    x = rng.uniform(-1.0, 1.0, size=(40, 2))
    y = ((x[:, 0] + 0.5 * x[:, 1]) > 0.0).astype(int)
    # 确保两个类别都足够多，满足 sklearn 内置 10% 分层验证集的最小样本数。
    y[:4] = 0
    y[-4:] = 1
    w = np.linspace(0.5, 1.5, len(x))
    gx = np.linspace(-1.0, 1.0, 8)
    gy = np.linspace(-1.0, 1.0, 5)
    support = np.array([(a, b) for a in gx for b in gy])
    return x, y, w, support


def test_classifier_uses_sample_weight_and_normalizes_grid_density():
    x, y, w, support = _data()
    ev = GoalSuccessMLPClassifierEvaluator(n_epochs=20, hidden_width=4, random_state=0)
    ev.fit_classifier(x, y, w, support)
    density = ev.evaluate_grid(support)
    assert np.all(np.isfinite(density))
    assert np.all(density > 0)
    assert np.isclose(density.sum() * ev.support_cell_area_, 1.0)
    assert ev.fit_diagnostics_["target_scheme"] == "goal_to_P(success)_with_sample_weight"
    assert ev.fit_diagnostics_["early_stopping"] is True
    assert ev.fit_diagnostics_["validation_fraction"] == 0.1
    assert ev.fit_diagnostics_["n_iter"] <= 20


def test_classifier_is_deterministic():
    x, y, w, support = _data()
    densities = []
    for _ in range(2):
        ev = GoalSuccessMLPClassifierEvaluator(n_epochs=20, hidden_width=4, random_state=7)
        ev.fit_classifier(x, y, w, support)
        densities.append(ev.evaluate_grid(support))
    assert np.allclose(densities[0], densities[1])


def test_classifier_rejects_single_class():
    x, _, w, support = _data()
    ev = GoalSuccessMLPClassifierEvaluator(n_epochs=2)
    with pytest.raises(ValueError, match="both success and failure"):
        ev.fit_classifier(x, np.ones(len(x)), w, support)


def test_classifier_rejects_unfitted():
    ev = GoalSuccessMLPClassifierEvaluator()
    with pytest.raises(RuntimeError, match="fit_classifier"):
        ev.evaluate_grid(np.zeros((4, 2)))


def test_classifier_can_disable_validation():
    x, y, w, support = _data()
    ev = GoalSuccessMLPClassifierEvaluator(
        n_epochs=5, hidden_width=4, random_state=0, early_stopping=False
    )
    ev.fit_classifier(x, y, w, support)
    assert ev.fit_diagnostics_["early_stopping"] is False
    assert ev.fit_diagnostics_["validation_fraction"] == 0.0
    assert ev.fit_diagnostics_["n_validation_samples"] == 0
    assert ev.fit_diagnostics_["best_validation_score"] is None
