"""只用历史成功目标选择 FM 固定预算，再运行独立 seed 的小规模检查。

调参留出按目标坐标分组，避免同一固定网格点跨时间重复进入两侧。
当前检查点的 oracle 不参与选择；选定配置先落盘，再读取检查用 oracle。
这是探索性调参，不代替全量五 seed 评估，也不改变正式训练的无 early stopping 协议。
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from replacement_run_job import ROOT, _historical_files, _run_one_job
from gc_ope.evaluate.evaluation_result_container import WeightedEvaluationResultContainer
from gc_ope.evaluate.evaluator_fm import FlowMatchingDensityEvaluator


class Tee:
    """同时向终端和日志输出。"""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)
            stream.flush()

    def flush(self):
        for stream in self.streams:
            stream.flush()


def save_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def history_split(seed, checkpoint):
    """只读严格早于 checkpoint 的文件，保留原协议的时间权重。"""
    frames, sources = [], []
    for step, path in _historical_files("push", seed, checkpoint):
        assert step < checkpoint
        frame = pd.read_csv(path)
        frame = frame.loc[frame.termination == "reach target", ["x", "y"]].copy()
        frame["weight"] = 0.9 ** ((checkpoint - step) / 10000)
        frames.append(frame)
        sources.append({"step": step, "path": str(path.relative_to(ROOT)),
                        "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    data = pd.concat(frames, ignore_index=True)
    goals = data[["x", "y"]].to_numpy()
    unique, group = np.unique(goals, axis=0, return_inverse=True)
    rng = np.random.default_rng(20260922)
    held_out = rng.permutation(len(unique))[:max(1, round(0.2 * len(unique)))]
    valid = np.isin(group, held_out)
    train = ~valid
    assert not set(map(tuple, goals[train])) & set(map(tuple, goals[valid]))
    return data.loc[train], data.loc[valid], sources


def run(out):
    candidates = [
        {"name": "baseline_e100_h32", "n_epochs": 100, "hidden_features": 32},
        {"name": "e500_h32", "n_epochs": 500, "hidden_features": 32},
        {"name": "e1500_h32", "n_epochs": 1500, "hidden_features": 32},
        {"name": "e500_h64", "n_epochs": 500, "hidden_features": 64},
    ]
    common = dict(lr=1e-3, samples_per_epoch=2000, ode_steps=32,
                  likelihood_batch_size=1024, weight_decay=0.0, random_state=0)
    manifest = {
        "task": "push", "tuning_seed": 1, "history_cutoffs": [500000, 800000],
        "confirmation_seed": 2, "confirmation_checkpoints": [500000, 800000],
        "candidates": candidates, "common": common,
        "selection": "历史内部按唯一目标坐标固定留出20%；跨两个历史截点平均加权 raw-space NLL 最低",
        "confirmation_mc": {"samples": 5000, "repeats": 3},
        "limits": "固定一次划分、一次模型随机种子的小规模探索；不代表全量结果或独立最终测试",
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [Path(__file__), ROOT / "src/gc_ope/evaluate/evaluator_fm.py",
                                    ROOT / "scripts/replacement_run_job.py"]},
    }
    save_json(out / "manifest.json", manifest)
    results, split_info = [], []
    for checkpoint in manifest["history_cutoffs"]:
        train, valid, sources = history_split(1, checkpoint)
        split_info.append({"checkpoint": checkpoint, "train_rows": len(train),
                           "validation_rows": len(valid), "sources": sources,
                           "validation_goals": valid[["x", "y"]].drop_duplicates().values.tolist()})
        print(f"历史截点 {checkpoint}: 训练 {len(train)} 条，留出 {len(valid)} 条；无当前检查点数据", flush=True)
        for candidate in candidates:
            start = time.perf_counter()
            model = FlowMatchingDensityEvaluator(
                evaluation_result_container_class=WeightedEvaluationResultContainer,
                evaluation_result_container_kwargs={"discounted_factor": 0.9},
                **common, **{k: v for k, v in candidate.items() if k != "name"},
            )
            n = len(train)
            model.eval_res_container.add_batch(
                train[["x", "y"]].to_numpy(), [True] * n, [0.] * n, [0.] * n,
                train.weight.to_numpy(),
            )
            model.fit_evaluator()
            query = valid[["x", "y"]].to_numpy()
            log_density = model.evaluate_grid(query, return_log_density=True)
            score = float(-np.average(log_density, weights=valid.weight))
            model.ode_steps = 64
            refined = model.evaluate_grid(query, return_log_density=True)
            row = {"checkpoint": checkpoint, "candidate": candidate["name"],
                   "validation_nll": score,
                   "ode32_vs64_max_log_density_diff": float(np.max(np.abs(refined - log_density))),
                   "elapsed_s": time.perf_counter() - start,
                   "fit_diagnostics": model.fit_diagnostics_, "loss_curve": model._loss_curve}
            results.append(row)
            save_json(out / "tuning_details.json", results)
            print(f"  {candidate['name']}: 历史留出 NLL={score:.6f}，耗时 {row['elapsed_s']:.1f}s", flush=True)
    save_json(out / "history_splits.json", split_info)
    scores = pd.DataFrame([{k: v for k, v in row.items() if k not in {"fit_diagnostics", "loss_curve"}}
                           for row in results])
    scores.to_csv(out / "tuning_scores.csv", index=False)
    ranking = scores.groupby("candidate").validation_nll.mean().sort_values()
    best = next(c for c in candidates if c["name"] == ranking.index[0])
    # 在访问当前 oracle 前固定结果，检查结果不反馈到本轮参数选择。
    save_json(out / "selected_config.json", {"selected": best, "common": common,
                                            "mean_validation_nll": ranking.to_dict()})
    print(f"已固定候选配置: {best['name']}；开始 seed 2 检查", flush=True)
    checks = []
    unique_candidates = {c["name"]: c for c in [candidates[0], best]}
    for checkpoint in manifest["confirmation_checkpoints"]:
        for candidate in unique_candidates.values():
            row = _run_one_job(
                "push", 2, checkpoint, "flow_matching", kappa=0.9,
                n_components=8, resample_size=10000, bandwidth=0.2, random_state=0,
                gmm_reg_covar=1e-6, n_hist_bins=20, gaussian_reg_covar=1e-6,
                nn_epochs=100, nn_lr=1e-3, nn_hidden=16,
                fm_epochs=candidate["n_epochs"], fm_hidden=candidate["hidden_features"],
                mc_samples=5000, mc_repeats=3,
            )
            row.pop("_sampled_goals", None)
            diagnostic = row.pop("_fit_diagnostics", {})
            row["candidate"] = candidate["name"]
            checks.append(row)
            pd.DataFrame(checks).to_csv(out / "confirmation.csv", index=False)
            if row["status"] != "ok":
                raise RuntimeError(f"检查失败: {row}")
            save_json(out / f"confirmation_fit_{checkpoint}_{candidate['name']}.json", diagnostic)
            print(f"  seed2/{checkpoint} {candidate['name']}: KL={row['kl']:.6f}，"
                  f"网格 KL={row['fixed_grid_kl']:.6f}，耗时 {row['job_time_s']}s", flush=True)
    print("本轮调参与小规模检查完成；全量原始结果未覆盖。", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    with (args.out_dir / "run.log").open("w") as log:
        with contextlib.redirect_stdout(Tee(sys.stdout, log)), contextlib.redirect_stderr(Tee(sys.stderr, log)):
            run(args.out_dir)


if __name__ == "__main__":
    main()
