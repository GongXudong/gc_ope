"""历史选参的隔离、重复记录分组、时间权重和候选覆盖率测试。"""

import numpy as np
import pandas as pd
import pytest
from gc_ope.evaluate.offline_data import EvaluationBatch, load_pair
from gc_ope.evaluate.history_validation import sampled_history, split_successes, weighted_nll, select_candidates
from gc_ope.evaluate.evaluator_common import InsufficientSamples


def test_sampled_history_matches_production_and_ignores_future(tmp_path):
    directory = tmp_path / "my_push/sac/seed_1"
    directory.mkdir(parents=True)
    for step in [10000, 20000]:
        x = np.random.default_rng(step).normal(size=(676, 2))
        pd.DataFrame(dict(x=x[:, 0], y=x[:, 1], termination=np.where(
            np.arange(676) % 3, "reach target", "timeout"))).to_csv(
                directory / f"rl_model_{step}_steps_eval_res_on_fixed.csv", index=False)
    # 未来文件即便损坏，也不得被当前拟合读取。
    (directory / "rl_model_30000_steps_eval_res_on_fixed.csv").write_text("invalid")
    actual, identities = sampled_history(tmp_path, 1, 20000)
    expected, _ = load_pair(tmp_path, 1, 20000)
    for name in ["goals", "successes", "weights"]:
        np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))
    assert identities.shape == (200, 2)
    assert set(identities[:, 0]) == {10000, 20000}


def test_duplicate_observations_stay_together_and_weights_are_preserved():
    identities = np.repeat(np.column_stack([np.full(20, 10000), np.arange(20)]), 3, axis=0)
    goals = np.random.default_rng(3).normal(size=(60, 2))
    batch = EvaluationBatch(goals, np.ones(60, dtype=bool), np.linspace(.01, 1, 60))
    train, valid, ti, vi = split_successes(batch, identities, 1701)
    assert not set(map(tuple, identities[ti])) & set(map(tuple, identities[vi]))
    np.testing.assert_array_equal(np.sort(np.r_[ti, vi]), np.arange(60))
    np.testing.assert_array_equal(train.weights, batch.weights[ti])
    np.testing.assert_array_equal(valid.weights, batch.weights[vi])
    np.testing.assert_array_equal(split_successes(batch, identities, 1701)[3], vi)


def test_failures_never_enter_density_validation_and_shortage_is_explicit():
    batch = EvaluationBatch(np.zeros((12, 2)), np.arange(12) < 9, np.ones(12))
    identities = np.column_stack([np.ones(12), np.arange(12)])
    with pytest.raises(InsufficientSamples):
        split_successes(batch, identities, 1)
    batch.successes[9] = True
    train, valid, ti, vi = split_successes(batch, identities, 1)
    assert train.successes.all() and valid.successes.all()
    assert set(np.r_[ti, vi]) == set(range(10))


def test_weighted_nll_uses_raw_density_and_does_not_clip_negative_values():
    class Density:
        def log_density(self, goals):
            return goals[:, 0]
    batch = EvaluationBatch(np.array([[2., 0.], [4., 0.]]), np.ones(2, bool), np.array([1., 3.]))
    assert weighted_nll(Density(), batch) == -3.5
    batch.weights *= 50
    assert weighted_nll(Density(), batch) == -3.5


def test_selection_uses_only_complete_heldout_scores_not_kl():
    candidates = {"gmm": {"baseline": {}, "good": {}, "missing": {}, "error": {}}}
    cases = [(1, 10000, 1), (1, 20000, 1)]
    rows = []
    for name, value in [("baseline", 2.), ("good", 1.), ("missing", -100.), ("error", -200.)]:
        for seed, step, split in cases[:1] if name == "missing" else cases:
            rows.append(dict(method="gmm", candidate=name, seed=seed, checkpoint=step, split_seed=split,
                             status="error" if name == "error" else "ok", valid_nll=value,
                             kl=0. if name == "baseline" else 100.))
    selected, scores = select_candidates(rows, candidates, cases)
    assert selected == {"gmm": "good"}
    assert set(scores["gmm"]) == {"baseline", "good"}
    # 重复结果不能伪装成满足所有任务的完整候选。
    rows.append(rows[2].copy())
    assert select_candidates(rows, candidates, cases)[0] == {"gmm": "baseline"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
