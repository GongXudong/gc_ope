"""正则化流的旧版退化等价、验证分组、最优轮数和全量重拟合。"""

import numpy as np
import pytest
from gc_ope.evaluate.evaluator_factory import make_evaluator, fit_evaluator
from gc_ope.evaluate.offline_data import EvaluationBatch
from gc_ope.evaluate.flow_training import spatial_validation_split


def batch():
    x = np.random.default_rng(9).normal(size=(50, 2))
    return EvaluationBatch(x, np.ones(50, bool), np.linspace(.1, 1, 50))


def test_spatial_split_keeps_repeated_coordinates_together():
    x = np.repeat(np.random.default_rng(2).normal(size=(20, 2)), 5, axis=0)
    ti, vi = spatial_validation_split(x, .2, 701)
    assert not set(map(tuple, x[ti])) & set(map(tuple, x[vi]))
    np.testing.assert_array_equal(np.sort(np.r_[ti, vi]), np.arange(100))
    assert spatial_validation_split(x[:5], .2, 701) is None


@pytest.mark.parametrize("method", ["nf"])
def test_best_epoch_is_selected_then_all_data_refitted(method, monkeypatch):
    parameters = dict(n_epochs=40, hidden_layer_sizes=[8, 8], min_epochs=10, validation_interval=10,
                      patience=10, noise_std=.2, random_state=3)
    parameters.update(dict(transforms=2) if method == "nf" else dict(samples_per_epoch=32, ode_steps=4))
    model = make_evaluator(method, parameters=parameters)
    data = batch()
    data.fill(model)
    scores = iter([3., 1., 2.])
    # 强制验证先改善再恶化，检验选择的是最优轮数而非最后一次更新。
    monkeypatch.setattr(model, "_validation_log_density", lambda network, x: np.full(len(x), -next(scores)))
    fit_evaluator(method, model)
    diagnostic = model.fit_diagnostics_
    assert diagnostic["n_epochs"] == 20 and len(diagnostic["validation_curve"]) == 3
    assert diagnostic["final_refit_records"] == 50
    assert diagnostic["selection_training_records"] + diagnostic["selection_validation_records"] == 50
    np.testing.assert_allclose(model.scaler.mean_, data.goals.mean(0))
    refit = make_evaluator(method, parameters={**parameters, "n_epochs": 20, "early_stopping": False})
    data.fill(refit)
    fit_evaluator(method, refit)
    np.testing.assert_allclose(model.log_density(data.goals), refit.log_density(data.goals), atol=1e-6)


@pytest.mark.parametrize("method", ["nf"])
def test_sparse_fallback_and_nonfinite_validation_are_explicit(method, monkeypatch):
    parameters = dict(n_epochs=4, hidden_layer_sizes=[8, 8], min_epochs=1, validation_interval=1, fallback_epochs=2)
    parameters.update(dict(transforms=2) if method == "nf" else dict(samples_per_epoch=16, ode_steps=2))
    model = make_evaluator(method, parameters=parameters)
    data = batch()
    EvaluationBatch(data.goals[:3], data.successes[:3], data.weights[:3]).fill(model)
    fit_evaluator(method, model)
    assert model.fit_diagnostics_["n_epochs"] == 2
    assert model.fit_diagnostics_["selection_reason"] == "insufficient_unique_goals"
    data.fill(model)
    monkeypatch.setattr(model, "_validation_log_density", lambda network, x: np.full(len(x), np.nan))
    with pytest.raises(FloatingPointError, match="验证"):
        fit_evaluator(method, model)
    assert not model._fitted


@pytest.mark.parametrize("parameters", [{"noise_std": -1}, {"noise_std": np.nan},
    {"patience": 0}, {"validation_fraction": .8}, {"validation_interval": 1.5}])
def test_invalid_regularization_rejected(parameters):
    with pytest.raises(ValueError):
        make_evaluator("nf", parameters=parameters)
