"""历史抽样的科学口径、跨方法一致性与串并行边界回归。"""
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import replacement_run_job as job
import run_push_history100 as batch
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer


@pytest.fixture
def history(tmp_path, monkeypatch):
    for step in [10000, 20000, 30000]:
        pd.DataFrame({"x": np.arange(200) + step, "y": np.arange(200) % 3,
                      "termination": ["reach target" if i % 3 else "timeout" for i in range(200)]}).to_csv(
            tmp_path / f"rl_model_{step}_steps_eval_res_on_fixed.csv", index=False)
    monkeypatch.setattr(job, "_checkpoint_dir", lambda task, seed: tmp_path)
    return tmp_path


def test_inclusive_sampling_is_reproducible_and_keeps_both_labels(history):
    first = job._load_sampled_history("push", 1, 20000, 100, 0, True)
    later = job._load_sampled_history("push", 1, 30000, 100, 0, True)
    assert [step for step, _ in first] == [10000, 20000]
    for (_, left), (_, right) in zip(first, later):
        pd.testing.assert_frame_equal(left, right)
        assert len(left) == 100
        assert left.index.duplicated().any()
        assert set(left.termination) == {"reach target", "timeout"}
    other = job._load_sampled_history("push", 2, 20000, 100, 0, True)
    assert not first[0][1].equals(other[0][1])


def test_default_stays_all_strict_history_and_handles_small_file(history):
    old = job._load_sampled_history("push", 1, 20000)
    assert [step for step, _ in old] == [10000]
    assert len(old[0][1]) == 200
    file = history / "rl_model_10000_steps_eval_res_on_fixed.csv"
    pd.read_csv(file).head(2).to_csv(file, index=False)
    sampled = job._load_sampled_history("push", 1, 10000, 100, 0, True)
    assert len(sampled[0][1]) == 100
    assert sampled[0][1].index.nunique() == 2


def test_all_four_methods_get_identical_sampled_records_and_weights(history, monkeypatch):
    captured = {}

    class Fake:
        def __init__(self, method):
            self.method = method
            self.eval_res_container = WeightedEvaluationResultContainer(.9)

        def fit_classifier(self, x, labels, weights, support):
            captured[self.method] = (x, np.asarray(labels, dtype=bool), weights)
            raise RuntimeError("测试已捕获训练输入")

        def fit_evaluator(self):
            c = self.eval_res_container
            captured[self.method] = (np.asarray(c.desired_goal_list), np.asarray(c.success_list), c.desired_goal_weights)
            raise RuntimeError("测试已捕获训练输入")

    monkeypatch.setattr(job, "_make_estimator", lambda method, *a, **kw: Fake(method))
    kwargs = dict(kappa=.9, n_components=5, resample_size=1000, bandwidth=.2,
                  random_state=0, gmm_reg_covar=1e-6, n_hist_bins=10, gaussian_reg_covar=1e-6,
                  nn_epochs=100, nn_lr=.001, nn_hidden=16, mc_samples=10000, mc_repeats=5,
                  history_samples_per_checkpoint=100, history_include_current=True)
    for method in batch.METHODS:
        result = job._run_one_job("push", 1, 20000, method, **kwargs)
        assert "测试已捕获训练输入" in result["error"]
        assert result["_history_diagnostics"]["rows"] == 200
    for values in captured.values():
        for a, b in zip(values, captured["nn"]):
            np.testing.assert_array_equal(a, b)
        np.testing.assert_allclose(values[2], [.9] * 100 + [1.] * 100)


def test_scheduler_runs_five_seeds_together_but_never_overlaps_methods():
    barriers = {method: threading.Barrier(5, timeout=5) for method in batch.METHODS}
    lock = threading.Lock()
    active, finished = set(), []

    def run(method, seed):
        with lock:
            assert not active or {m for m, _ in active} == {method}
            assert len(finished) >= batch.METHODS.index(method) * 5
            active.add((method, seed))
        barriers[method].wait()
        with lock:
            active.remove((method, seed))
            finished.append((method, seed))
        return {"passed": True}

    assert len(batch.schedule(run)) == 4
    assert len(finished) == 20


def test_scheduler_stops_on_failed_seed_and_commands_exclude_kde(tmp_path):
    calls = []

    def run(method, seed):
        calls.append((method, seed))
        return {"passed": seed != 3, "exit_code": 7 if seed == 3 else 0}

    stages = batch.schedule(run)
    assert len(stages) == 1 and len(calls) == 5
    for method in batch.METHODS:
        command = batch.command_for(method, 1, tmp_path)
        assert command[command.index("--workers") + 1] == "4"
        assert command[command.index("--mc-samples") + 1] == "10000"
        assert command[command.index("--history-samples-per-checkpoint") + 1] == "100"
        assert "--history-include-current" in command
        assert "kde" not in command


def test_audit_rejects_missing_and_failed_checkpoints(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame([dict(method="nn", task="push", seed=1, checkpoint=10000, status="error")]).to_csv(path, index=False)
    result = batch.audit_csv(path, "nn", 1, [10000, 20000])
    assert result["missing"] == [20000] and len(result["errors"]) == 1
    assert not result["passed"]


def test_audit_rejects_nonfinite_success_metrics(tmp_path):
    path = tmp_path / "rows.csv"
    pd.DataFrame([dict(method="nn", task="push", seed=1, checkpoint=10000,
                       status="ok", kl=np.inf, fixed_grid_kl=.1)]).to_csv(path, index=False)
    result = batch.audit_csv(path, "nn", 1, [10000])
    assert result["invalid_metrics"] == [10000] and not result["passed"]
