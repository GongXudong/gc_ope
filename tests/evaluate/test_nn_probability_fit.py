"""概率拟合回归：accuracy 不变时仍应继续改善 log-loss，而非恢复第一轮。"""

import numpy as np
import pytest
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split

from gc_ope.evaluate.evaluator_factory import make_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch


def fit(parameters=None):
    rng = np.random.default_rng(4)
    goals = rng.uniform(-1, 1, (676, 2))
    labels = (goals[:, 0] > .35) & (goals[:, 1] > -.2)
    weights = np.linspace(.3, 1, len(goals))
    model = make_evaluator("nn", support_goals=goals, parameters=parameters)
    EvaluationBatch(goals, labels, weights).fill(model)
    model.fit_evaluator()
    return model, goals, labels, weights


def test_logloss_improves_and_restores_best_epoch():
    model, goals, labels, weights = fit()
    d = model.fit_diagnostics_
    assert d["early_stopping_metric"] == "weighted_log_loss"
    assert d["best_epoch"] > 12
    assert d["validation_log_loss"] < d["validation_baseline_log_loss"] * .65
    assert d["epochs_run"] >= 50
    assert d["best_epoch"] == np.argmin(model.validation_loss_curve_) + 1
    _, val = train_test_split(np.arange(len(labels)), test_size=.1, stratify=labels, random_state=0)
    actual = log_loss(labels[val], model.predict_success_probability(goals[val]), sample_weight=weights[val])
    assert actual == pytest.approx(min(model.validation_loss_curve_))


def test_patience_and_minimum_budget_restore_even_if_later_epochs_run():
    model, *_ = fit(dict(n_epochs=100, min_epochs=20, n_iter_no_change=3, tol=1.))
    d = model.fit_diagnostics_
    assert d["epochs_run"] == 20
    assert d["stop_reason"] == "validation_log_loss"
    assert d["validation_log_loss"] == min(model.validation_loss_curve_)


def test_explicit_layers_and_old_alias_are_equivalent():
    a, goals, *_ = fit(dict(hidden_layer_sizes=[8, 4], n_epochs=2, early_stopping=False))
    assert [m.shape for m in a.classifier.coefs_] == [(2, 8), (8, 4), (4, 1)]
    a, *_ = fit(dict(hidden_layer_sizes=[8, 8], n_epochs=2, early_stopping=False))
    b, *_ = fit(dict(hidden_width=8, n_epochs=2, early_stopping=False))
    np.testing.assert_array_equal(a.predict_success_probability(goals), b.predict_success_probability(goals))


@pytest.mark.parametrize("layers", [16, [], [0], [-1], [True], [1.5]])
def test_invalid_layers_rejected(layers):
    with pytest.raises(ValueError, match="hidden_layer_sizes"):
        make_evaluator("nn", parameters={"hidden_layer_sizes": layers})


def test_conflicting_alias_rejected():
    with pytest.raises(ValueError, match="不能同时"):
        make_evaluator("nn", parameters={"hidden_width": 8, "hidden_layer_sizes": [8, 8]})


def test_budget_exhaustion_visible_and_repeat_fit_deterministic():
    model, goals, *_ = fit(dict(n_epochs=2))
    assert model.fit_diagnostics_["stop_reason"] == "max_epochs"
    assert model.fit_diagnostics_["quality_warnings"]
    first = model.predict_success_probability(goals)
    model.fit_evaluator()
    np.testing.assert_array_equal(first, model.predict_success_probability(goals))
