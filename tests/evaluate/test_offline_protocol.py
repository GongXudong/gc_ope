"""抽样边界、同类参考、续跑和真实 CSV 对照。"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from gc_ope.evaluate.offline_data import load_pair, sampled_indices
from gc_ope.evaluate.offline_experiment import ExperimentConfig, fit_pair, run_checkpoint
from gc_ope.evaluate.offline_results import result_lock, verify_manifest, save_row, read_rows, audit_result


def write_checkpoint(root, step, offset=0):
    path = root / "my_push/sac/seed_1"
    path.mkdir(parents=True, exist_ok=True)
    goals = np.random.default_rng(step).normal(size=(676, 2)) + offset
    pd.DataFrame(dict(x=goals[:, 0], y=goals[:, 1], z=.02,
                      termination=np.where(np.arange(676) % 2, "reach target", "timeout"))).to_csv(
        path / f"rl_model_{step}_steps_eval_res_on_fixed.csv", index=False)


def test_history_is_shared_inclusive_and_stable(tmp_path):
    for step in [10000, 20000, 30000]:
        write_checkpoint(tmp_path, step, step / 10000)
    history, reference = load_pair(tmp_path, 1, 20000)
    later, _ = load_pair(tmp_path, 1, 30000)
    assert len(history.goals) == 200 and len(reference.goals) == 676
    assert not history.successes.all()  # 抽取全部记录，不只抽成功点。
    np.testing.assert_array_equal(history.goals, later.goals[:200])
    np.testing.assert_allclose(history.weights, [0.9]*100 + [1.]*100)
    np.testing.assert_allclose(later.weights[:200], history.weights * .9)
    indices = sampled_indices(676, 1, 20000)
    np.testing.assert_array_equal(history.goals[-100:], reference.goals[indices])
    # 与保存版本使用的 SeedSequence 协议逐索引对照。
    rng = np.random.default_rng(np.random.SeedSequence([0, 1, 1, 20000]))
    np.testing.assert_array_equal(indices, rng.integers(0, 676, size=100))


@pytest.mark.parametrize("method,parameters", [
    ("gmm", {}), ("nn", {"n_epochs": 2, "early_stopping": False}),
    ("nf", {"n_epochs": 1}), ("fm", {"n_epochs": 1, "samples_per_epoch": 16, "ode_steps": 2}),
])
def test_pair_is_same_family_independent_and_uses_correct_records(tmp_path, method, parameters):
    write_checkpoint(tmp_path, 10000)
    config = ExperimentConfig(str(tmp_path), parameters={method: parameters}, mc_samples=16, mc_repeats=1)
    history, reference = load_pair(tmp_path, 1, 10000)
    estimate, truth = fit_pair(method, history, reference, config)
    assert type(estimate) is type(truth) and estimate is not truth
    assert estimate.scaler is not truth.scaler
    assert len(estimate.eval_res_container.success_list) == 100
    assert len(truth.eval_res_container.success_list) == 676
    row, detail = run_checkpoint(config, method, 1, 10000)
    assert row["status"] == "ok", row["error"]
    assert detail["mc_kept"] == [16]


def test_results_upsert_lock_manifest_and_audit(tmp_path):
    path = tmp_path / "gmm_push_seed1.csv"
    row = dict(checkpoint=10000, status="error", kl=None, kl_seed_std=None)
    with result_lock(tmp_path):
        with pytest.raises(RuntimeError):
            with result_lock(tmp_path):
                pass
        verify_manifest(tmp_path, {"protocol": "new"})
        with pytest.raises(ValueError):
            verify_manifest(tmp_path, {"protocol": "old"})
        save_row(path, row)
        save_row(path, {**row, "status": "ok", "kl": -.01, "kl_seed_std": .02})
    assert len(read_rows(path)) == 1  # MC 小负值保留，重试不产生重复行。
    assert audit_result(path, [10000])["invalid"] == []
    assert audit_result(path, [10000, 20000])["missing"] == [20000]
    pd.concat([pd.read_csv(path)]*2).to_csv(path, index=False)
    with pytest.raises(ValueError, match="重复"):
        read_rows(path)


def test_reference_shortage_is_skipped_not_replaced_with_kde(tmp_path):
    write_checkpoint(tmp_path, 10000)
    path = next(tmp_path.rglob("*.csv"))
    frame = pd.read_csv(path)
    frame["termination"] = "timeout"
    frame.to_csv(path, index=False)
    row, _ = run_checkpoint(ExperimentConfig(str(tmp_path)), "gmm", 1, 10000)
    assert row["status"] == "skipped:insufficient_samples"


def test_nn_quality_warning_preserves_metric_and_is_audited(tmp_path):
    write_checkpoint(tmp_path, 10000)
    config = ExperimentConfig(str(tmp_path), parameters={"nn": {"n_epochs": 1}}, mc_samples=16, mc_repeats=1)
    row, detail = run_checkpoint(config, "nn", 1, 10000)
    assert row["status"] == "ok", row["error"]
    assert row["fit_quality"] == "warning"
    assert "训练上限" in row["fit_warnings"]
    assert detail["reference_fit"]["stop_reason"] == "max_epochs"
    path = tmp_path / "nn.csv"
    save_row(path, row)
    assert audit_result(path, [10000])["fit_warnings"] == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
