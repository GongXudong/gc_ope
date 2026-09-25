"""离线超参数搜索入口的协议、输入边界和断点回归。"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import hyperparameter_search as search


ROOT = Path(__file__).resolve().parents[2]


def _small_config(task: str, method: str) -> dict:
    return {
        "method": method,
        "task": task,
        "protocol": {
            "seeds": [1], "samples_per_checkpoint": 4, "sampling_seed": 11,
            "mc_seed": 13, "global_seed": 17, "kappa": 0.9,
            "mc_samples": 16, "mc_repeats": 1,
        },
        "fixed": {"resample_size": 30} if method == "gmm" else {},
        "stages": {"screening": {"seeds": [1], "checkpoints": [20000], "max_candidates": 1}},
        "search_space": (
            {"n_components": [1], "covariance_type": ["full"], "reg_covar": [1e-4],
             "max_iter": [30], "tol": [1e-3], "n_init": [1]}
            if method == "gmm" else {}
        ),
    }


def _write_tiny_root(root: Path, task: str = "push") -> None:
    directory = root / "checkpoints" / f"my_{task}" / "sac" / "seed_1"
    directory.mkdir(parents=True)
    for step in (10000, 20000):
        x = np.linspace(-1, 1, 20) + step / 1e6
        y = np.linspace(1, -1, 20)
        pd.DataFrame({
            "x": x, "y": y, "z": 0.02,
            "termination": np.where(np.arange(20) % 2, "reach target", "timeout"),
        }).to_csv(directory / f"rl_model_{step}_steps_eval_res_on_fixed.csv", index=False)


def test_parameter_spaces_and_method_specific_validation():
    for task in ("push", "slide"):
        for method in search.METHODS:
            config = search._load_config(
                ROOT / "configs/evaluate/offline" / task / "hyperparameter_search" / f"{method}.yaml"
            )
            candidates = search.generate_candidates(method, config["search_space"], config.get("fixed"))
            assert candidates
            assert all(candidate["hidden_layer_sizes"] == [16, 16] for candidate in candidates if method in {"nn", "nf", "fm"})
    with pytest.raises(search.SearchProtocolError):
        search.validate_method_params("gmm", {
            "n_components": 0, "covariance_type": "full", "reg_covar": 1e-4,
            "max_iter": 10, "tol": 1e-3, "n_init": 1,
        })
    with pytest.raises(search.SearchProtocolError):
        search.validate_method_params("nf", {
            "transforms": 1, "bins": 1, "lr": 1e-3, "weight_decay": 0,
            "noise_std": 0, "n_epochs": 2, "patience": 1, "hidden_layer_sizes": [32, 32],
        })


@pytest.mark.parametrize("task", ["push", "slide"])
def test_real_push_and_slide_fixed_data_are_read(task):
    files = search.checkpoint_files(ROOT, task, 1)
    assert len(files) == 100
    assert files[0][0] == 10000 and files[-1][0] == 1000000
    data = search.load_trajectory(ROOT, task, 1, 20000, samples_per_checkpoint=4, mc_samples=16, mc_repeats=1)
    assert [source.step for source in data.history] == [10000, 20000]
    assert all(len(source.frame) == 4 for source in data.history)
    assert len(data.reference) == 676


def test_sampling_is_shared_inclusive_and_uses_absolute_time_weights():
    first = search.load_trajectory(ROOT, "push", 1, 20000, samples_per_checkpoint=4, mc_samples=16, mc_repeats=1)
    second = search.load_trajectory(ROOT, "push", 1, 20000, samples_per_checkpoint=4, mc_samples=16, mc_repeats=1)
    for left, right in zip(first.history, second.history):
        np.testing.assert_array_equal(left.sampled_indices, right.sampled_indices)
    goals, labels, weights = search._history_arrays(first.history, 20000, 0.9)
    assert goals.shape == (8, 2)
    np.testing.assert_allclose(weights, [0.9] * 4 + [1.0] * 4)
    np.testing.assert_array_equal(first.mc_indices, second.mc_indices)


def test_same_parameters_reach_four_independent_evaluator_instances():
    params = {
        "n_components": 2, "covariance_type": "diag", "reg_covar": 1e-4,
        "max_iter": 30, "tol": 1e-3, "n_init": 1, "resample_size": 20,
    }
    first = search.make_estimator("gmm", params, kappa=0.9, random_state=3)
    second = search.make_estimator("gmm", params, kappa=0.9, random_state=3)
    assert first is not second
    assert first.gmm.get_params()["n_components"] == second.gmm.get_params()["n_components"] == 2

    nn = search.make_estimator("nn", {
        "bandwidth": .1, "lr": 1e-3, "alpha": 1e-4, "n_epochs": 2,
        "n_iter_no_change": 1, "hidden_layer_sizes": [16, 16],
    }, kappa=.9, random_state=4)
    assert nn.hidden_layer_sizes == (16, 16) and nn.bandwidth == .1
    nf = search.make_estimator("nf", {
        "transforms": 1, "bins": 2, "lr": 1e-3, "weight_decay": 0,
        "noise_std": 0, "n_epochs": 2, "patience": 1, "hidden_layer_sizes": [16, 16],
    }, kappa=.9, random_state=5)
    assert nf.hidden_layer_sizes == (16, 16) and nf.patience == 1
    fm = search.make_estimator("fm", {
        "n_members": 2, "n_epochs": 1, "samples_per_epoch": 4, "lr": 1e-3,
        "weight_decay": 0, "noise_std": 0, "patience": 1, "ode_steps": 1,
        "likelihood_batch_size": 2, "hidden_layer_sizes": [16, 16],
    }, kappa=.9, random_state=6)
    assert fm.n_members == 2


def test_mc_kl_records_nonfinite_and_insufficient_results():
    class Fake:
        def sample(self, n_samples, random_state):
            return np.zeros((n_samples, 2))

        def evaluate_grid(self, goals, return_log_density=False):
            values = np.zeros(len(goals))
            return values

    reference = pd.DataFrame({"x": [0.0], "y": [0.0], "termination": ["reach target"]})
    too_short = search.calculate_mc_kl(Fake(), Fake(), reference, [np.arange(4)])
    assert too_short["status"] == "skipped:insufficient_samples"

    class Nonfinite(Fake):
        def evaluate_grid(self, goals, return_log_density=False):
            return np.full(len(goals), np.nan)

    invalid = search.calculate_mc_kl(Nonfinite(), Fake(), reference, [np.arange(10)])
    assert invalid["status"] == "invalid:nonfinite_kl"


def test_tiny_search_finishes_and_resume_does_not_duplicate(tmp_path):
    _write_tiny_root(tmp_path)
    config_path = tmp_path / "gmm.yaml"
    config = _small_config("push", "gmm")
    config_path.write_text(__import__("yaml").safe_dump(config, sort_keys=False), encoding="utf-8")
    output = tmp_path / "logs"
    kwargs = dict(
        method="gmm", task="push", stage="screening", config_path=config_path,
        output_dir=output, data_root=tmp_path,
    )
    first = search.run_search(**kwargs)
    second = search.run_search(**kwargs)
    assert first["dataset_count"] == second["dataset_count"] == 1
    results = pd.read_csv(output / "candidate_results.csv")
    assert len(results) == 1 and results.iloc[0]["status"] == "ok"
    details = [json.loads(line) for line in (output / "candidate_details.jsonl").read_text().splitlines()]
    assert len(details) == 1
    assert (output / "search_config.yaml").is_file()
    assert (output / "validation_summary.csv").is_file()
    assert (output / "best_params.yaml").is_file()
