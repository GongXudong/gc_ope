"""Job-level runner for the P0 replacement experiment (plan.md 第 4/8 节)。

一个 job = (task, estimator, seed, checkpoint)。4 个估计器走同一份历史
数据 + 同一时间权重（AC-2.1/2.2），结果追加写同一个 CSV，单 job 失败
不影响其它 job（AC-7.4）。

用法：
  # 单 job
  conda run -n gc_ope python scripts/replacement_run_job.py \
      --task push --seed 1 --checkpoint 500000

  # 一组 job（可并行多起几个进程，各自写自己的 log）
  conda run -n gc_ope python scripts/replacement_run_job.py \
      --task push --seeds 1 2 3 --checkpoints 100000 200000 300000 500000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_gmm_prediction import (
    GOAL_COLUMNS,
    add_records,
    checkpoint_file,
    discrete_kl,
    historical_files,
)
from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_gmm import GMMEvaluator
from gc_ope.evaluate.evaluator_kde import KDEEvaluator
from gc_ope.evaluate.evaluator_replacements import (
    WeightedGaussianEvaluator,
    WeightedHistogramEvaluator,
)

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "logs" / "replacement_experiment"
CSV_PATH = OUT_DIR / "kde_gmm_hist_gaussian_kl.csv"

# 各任务的目标列。Push/Slide 固定 z=0.02，只取前 2 维（AC-0.3 / AC-3.3）；
# Reach 3D；VVC=flycraft (v, mu, chi)。
TASK_GOAL_COLUMNS = {
    "push": ["x", "y"],
    "slide": ["x", "y"],
    "reach": ["x", "y", "z"],
    "vvc": ["v", "mu", "chi"],
}

METRIC_COLUMNS = [
    "task", "seed", "checkpoint", "method",
    "kl", "kl_seed_std", "fixed_grid_kl",
    "historical_successes", "reference_successes",
    "status", "error", "job_time_s",
]
ORACLE_MAX_REFERENCES = 200000


def _checkpoint_dir(task: str, seed: int) -> Path:
    if task in ("push", "slide"):
        return ROOT / "checkpoints" / f"my_{task}" / "sac" / f"seed_{seed}"
    if task == "reach":
        return ROOT / "checkpoints" / "myreach" / "easy" / "sac" / f"seed_{seed}"
    if task == "vvc":
        return ROOT / "checkpoints" / "flycraft" / "easy" / "sac" / f"seed_{seed}"
    raise ValueError(f"unknown task: {task}")


def _checkpoint_csv(task: str, seed: int, checkpoint: int) -> Path:
    return _checkpoint_dir(task, seed) / f"rl_model_{checkpoint}_steps_eval_res_on_fixed.csv"


def _historical_files(task: str, seed: int, checkpoint: int):
    """严格早于当前 checkpoint 的所有 fixed-eval CSV，按时间排序。"""
    directory = _checkpoint_dir(task, seed)
    found = []
    for path in directory.glob("rl_model_*_steps_eval_res_on_fixed.csv"):
        match = re_match_step(path.name)
        if match and int(match.group(1)) < checkpoint:
            found.append((int(match.group(1)), path))
    return sorted(found)


def re_match_step(filename: str):
    import re
    m = re.search(r"rl_model_(\d+)_steps_eval_res_on_fixed\.csv$", filename)
    return m


def _make_estimator(method: str, task: str, *, kappa: float, n_components: int,
                    resample_size: int, bandwidth: float, random_state: int,
                    gmm_reg_covar: float, n_hist_bins: int, gaussian_reg_covar: float,
                    nn_epochs: int = 100, nn_lr: float = 1e-3, nn_hidden: int = 16,
                    fm_epochs: int = 80, fm_hidden: int = 16, fm_samples: int = 2000):
    container_kwargs = {"discounted_factor": kappa}
    if method == "kde":
        return KDEEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            kde_bandwidth=bandwidth,
        )
    if method == "gmm":
        return GMMEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            n_components=n_components,
            resample_size=resample_size,
            random_state=random_state,
            reg_covar=gmm_reg_covar,
        )
    if method == "histogram":
        return WeightedHistogramEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            n_bins=n_hist_bins,
        )
    if method == "gaussian":
        return WeightedGaussianEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            reg_covar=gaussian_reg_covar,
        )
    if method == "nn":
        from gc_ope.evaluate.evaluator_learned import WeightedMLPDensityEvaluator
        return WeightedMLPDensityEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            n_epochs=nn_epochs, lr=nn_lr, hidden_width=nn_hidden,
            kde_bandwidth=bandwidth, random_state=random_state,
        )
    if method == "flow_matching":
        from gc_ope.evaluate.evaluator_learned import FlowMatchingDensityEvaluator
        return FlowMatchingDensityEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            n_epochs=fm_epochs, lr=1e-3, hidden_width=fm_hidden,
            n_flow_samples=fm_samples, random_state=random_state,
        )
    raise ValueError(f"unknown method: {method}")


def _run_one_job(
    task: str,
    seed: int,
    checkpoint: int,
    method: str,
    *,
    kappa: float,
    n_components: int,
    resample_size: int,
    bandwidth: float,
    random_state: int,
    gmm_reg_covar: float,
    n_hist_bins: int,
    gaussian_reg_covar: float,
    nn_epochs: int,
    nn_lr: float,
    nn_hidden: int,
    fm_epochs: int,
    fm_hidden: int,
    fm_samples: int,
    mc_samples: int,
    mc_repeats: int,
) -> dict:
    t0 = time.time()
    goal_columns = TASK_GOAL_COLUMNS[task]
    reference = pd.read_csv(_checkpoint_csv(task, seed, checkpoint))
    history = [(t, pd.read_csv(p)) for t, p in _historical_files(task, seed, checkpoint)]
    historical_successes = int(sum((f["termination"] == "reach target").sum() for _, f in history))
    reference_successes = int((reference["termination"] == "reach target").sum())

    base_row = {
        "task": task,
        "seed": seed,
        "checkpoint": checkpoint,
        "method": method,
        "kl": np.nan,
        "kl_seed_std": np.nan,
        "fixed_grid_kl": np.nan,
        "historical_successes": historical_successes,
        "reference_successes": reference_successes,
        "job_time_s": round(time.time() - t0, 2),
    }

    if not history or historical_successes == 0:
        base_row.update(status="skipped:no_historical_successes", error="")
        return base_row

    try:
        # 4 个估计器共享同一份历史数据 + 同一时间权重（AC-2.1/2.2）。
        estimator = _make_estimator(
            method, task,
            kappa=kappa, n_components=n_components, resample_size=resample_size,
            bandwidth=bandwidth, random_state=random_state,
            gmm_reg_covar=gmm_reg_covar, n_hist_bins=n_hist_bins,
            gaussian_reg_covar=gaussian_reg_covar,
            nn_epochs=nn_epochs, nn_lr=nn_lr, nn_hidden=nn_hidden,
            fm_epochs=fm_epochs, fm_hidden=fm_hidden, fm_samples=fm_samples,
        )
        # add_records 把历史批次写进它接收的估计器容器（add_batch + 显式绝对权重）。
        hist_goals, hist_success, weights = add_records(
            estimator, history, checkpoint, kappa, goal_columns=goal_columns
        )
        estimator.fit_evaluator()

        # 固定网格 KL（诊断）
        grid = reference[goal_columns].to_numpy(dtype=float)
        success = reference.loc[reference["termination"] == "reach target", goal_columns].to_numpy(dtype=float)
        if method in ("gmm", "nn", "flow_matching"):
            _, gd = estimator.evaluate(grid)
            fixed_grid_kl = discrete_kl(success, grid, gd) if len(success) else float("nan")
        elif method == "histogram":
            # evaluate 返回盒函数密度，但 bin_probs 概率与 bin_volume 是标准化
            # 空间的，需 Jacobian 校正（evaluate_grid 已内置）
            gd = estimator.evaluate_grid(grid)
            fixed_grid_kl = discrete_kl(success, grid, gd) if len(success) else float("nan")
        elif method == "gaussian":
            # Gaussian 的 evaluate 返回标准化空间密度；用 evaluate_grid 取 raw-space
            gd_raw = estimator.evaluate_grid(grid)
            fixed_grid_kl = discrete_kl(success, grid, gd_raw) if len(success) else float("nan")
        else:  # kde
            _, gd = estimator.evaluate(grid)
            fixed_grid_kl = discrete_kl(success, grid, gd) if len(success) else float("nan")

        # MC-KL：oracle KDE vs 估计器
        if len(success) == 0:
            base_row.update(status="skipped:no_reference_success", error="", fixed_grid_kl=fixed_grid_kl,
                            kl=float("nan"), kl_seed_std=float("nan"))
            base_row["job_time_s"] = round(time.time() - t0, 2)
            return base_row

        # oracle KDE 的参考成功目标超过 200k 时，MC-KL 计算代价过高（小时级），
        # 按计划降级策略降采样到 200k，保持实验可完成性。
        ref_used = success
        if len(ref_used) > ORACLE_MAX_REFERENCES:
            ref_used = ref_used[:ORACLE_MAX_REFERENCES]
        oracle = KDEEvaluator(
            evaluation_result_container_class=EvaluationResultContainer,
            kde_bandwidth=bandwidth,
        )
        oracle.eval_res_container.add_batch(
            ref_used, [True] * len(ref_used), [0.0] * len(ref_used), [0.0] * len(ref_used),
        )
        oracle.fit_evaluator()

        from plot_kde_gmm_comparison import monte_carlo_kl, log_density_in_raw_space
        mc_random_states = [mc_repeats + 7 * i for i in range(mc_repeats)]
        kls = [monte_carlo_kl(oracle, estimator, mc_samples, rs) for rs in mc_random_states]
        kl_mean = float(np.nanmean(kls))
        kl_std = float(np.nanstd(kls)) if len(kls) > 1 else float("nan")

        base_row.update(
            kl=kl_mean, kl_seed_std=kl_std, fixed_grid_kl=fixed_grid_kl,
            status="ok", error="",
        )
    except Exception as exc:
        base_row.update(status="error", error=f"{type(exc).__name__}: {exc}")
    base_row["job_time_s"] = round(time.time() - t0, 2)
    return base_row


def _append_row(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=METRIC_COLUMNS)
    # 多进程并发写同一 CSV：用文件锁（fcntl）防止行交错
    import fcntl
    with open(CSV_PATH, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        need_header = not CSV_PATH.exists() or CSV_PATH.stat().st_size == 0
        df.to_csv(f, mode="a", header=need_header, index=False)
        fcntl.flock(f, fcntl.LOCK_UN)
    print(json.dumps(row, ensure_ascii=False, default=str), flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["push", "slide", "reach", "vvc"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="multiple seeds; --seed takes precedence if both given")
    ap.add_argument("--checkpoint", type=int, default=None)
    ap.add_argument("--checkpoints", type=int, nargs="*", default=None)
    ap.add_argument("--methods", nargs="+", default=["kde", "gmm", "histogram", "gaussian"])
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--resample-size", type=int, default=1000)
    ap.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    ap.add_argument("--n-hist-bins", type=int, default=10)
    ap.add_argument("--gaussian-reg-covar", type=float, default=1e-6)
    ap.add_argument("--nn-epochs", type=int, default=100)
    ap.add_argument("--nn-lr", type=float, default=1e-3)
    ap.add_argument("--nn-hidden", type=int, default=16)
    ap.add_argument("--fm-epochs", type=int, default=80)
    ap.add_argument("--fm-hidden", type=int, default=16)
    ap.add_argument("--fm-samples", type=int, default=2000)
    ap.add_argument("--mc-samples", type=int, default=100000)
    ap.add_argument("--mc-repeats", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    seeds = [args.seed] if args.seed is not None else (args.seeds or [1])
    checkpoints = (
        [args.checkpoint]
        if args.checkpoint is not None
        else (args.checkpoints or [100000, 500000, 1000000])
    )

    for seed in seeds:
        for ckpt in checkpoints:
            for method in args.methods:
                row = _run_one_job(
                    args.task, seed, ckpt, method,
                    kappa=args.kappa,
                    n_components=args.n_components,
                    resample_size=args.resample_size,
                    bandwidth=args.bandwidth,
                    random_state=args.random_state,
                    gmm_reg_covar=args.gmm_reg_covar,
                    n_hist_bins=args.n_hist_bins,
                    gaussian_reg_covar=args.gaussian_reg_covar,
                    nn_epochs=args.nn_epochs,
                    nn_lr=args.nn_lr,
                    nn_hidden=args.nn_hidden,
                    fm_epochs=args.fm_epochs,
                    fm_hidden=args.fm_hidden,
                    fm_samples=args.fm_samples,
                    mc_samples=args.mc_samples,
                    mc_repeats=args.mc_repeats,
                )
                _append_row(row)

    print(f"done. results appended to {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
