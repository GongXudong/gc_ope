"""全量 100-checkpoint 补跑（4 个传统估计器：kde/gmm/histogram/gaussian）。

背景：原 P0 实验只用了 [100k, 300k, 500k, 700k, 1000k] 共 5 个 checkpoint，
但 my_push/my_slide 目录下每个 seed 实际有 100 个 checkpoint
（10k..990k 步进 10k 共 99 个 + 1000k）。本脚本把传统 4 估计器补齐到全部
100 个。

NN / Flow Matching 不在本次范围（学习型，全量重训成本过高；其原 5-checkpoint
数据保留不动，是否全量另议）。

输出：结果追加写入独立 CSV
  logs/replacement_experiment/all100/kde_gmm_hist_gaussian_kl.csv
（不污染原始 5-checkpoint CSV；列与 replacement_run_job.py 一致）。

用法：
  # 每 task 4 worker（默认），2 task × 5 seed × 100 ckpt × 4 method = 4000 jobs
  python scripts/replacement_run_all100.py

  # 断点续跑（跳过已成功 job）
  python scripts/replacement_run_all100.py --skip-done

  # 调参
  python scripts/replacement_run_all100.py --tasks slide --workers 8
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

# 10k..990k（步进 10k，共 99 个）+ 1000k（最终 checkpoint），共 100 个
ALL101 = [10000 + 10000 * i for i in range(99)] + [1000000]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TASKS = ["push", "slide"]
DEFAULT_METHODS = ["kde", "gmm", "histogram", "gaussian"]

# 模块级加载一次（每个 worker 进程也会重新 import，开销一次性）
import replacement_run_job as mod

OUT_DIR = ROOT / "logs" / "replacement_experiment" / "all100"
OUT_CSV = OUT_DIR / "kde_gmm_hist_gaussian_kl.csv"

METRIC_COLUMNS = [
    "task", "seed", "checkpoint", "method",
    "kl", "kl_seed_std", "fixed_grid_kl",
    "historical_successes", "reference_successes",
    "status", "error", "job_time_s",
]


def _completed_keys() -> set[tuple[str, int, int, str]]:
    """从输出 CSV 读已确定结果的 job（status==ok 或 skipped:*），用于断点续跑。

    skipped 是确定性结果（历史文件缺失/无成功点），重跑仍会得到同一行，
    视为已完成以避免在 resume 时重复 append；error 行重跑时有机会修复。
    """
    if not OUT_CSV.exists():
        return set()
    df = pd.read_csv(OUT_CSV)
    done_status = df["status"].eq("ok") | df["status"].str.startswith("skipped:", na=False)
    return set(
        (r.task, int(r.seed), int(r.checkpoint), r.method)
        for r in df[done_status].itertuples()
    )


def _run_job(task: str, seed: int, ckpt: int, method: str, args: argparse.Namespace) -> dict:
    """跑单个 job：直接调用 replacement_run_job._run_one_job，结果写独立 CSV。

    比 subprocess 快（省去每 job 的 import/conda 启动开销），且可加进程内
    文件锁保证并发追加安全。
    """
    row = mod._run_one_job(
        task, seed, ckpt, method,
        kappa=args.kappa,
        n_components=args.n_components,
        resample_size=args.resample_size,
        bandwidth=args.bandwidth,
        random_state=args.random_state,
        gmm_reg_covar=args.gmm_reg_covar,
        n_hist_bins=args.n_hist_bins,
        gaussian_reg_covar=args.gaussian_reg_covar,
        nn_epochs=100, nn_lr=1e-3, nn_hidden=16,
        fm_epochs=80, fm_hidden=16, fm_samples=2000,
        mc_samples=args.mc_samples,
        mc_repeats=args.mc_repeats,
    )
    return row


def _append_row(row: dict) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row], columns=METRIC_COLUMNS)
    # 多进程并发写同一 CSV：fcntl 文件锁（跨进程有效）防止行交错
    with open(OUT_CSV, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        need_header = not OUT_CSV.exists() or OUT_CSV.stat().st_size == 0
        df.to_csv(f, mode="a", header=need_header, index=False)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def _worker(task: str, seed: int, ckpt: int, method: str, args: argparse.Namespace) -> None:
    """单个 job：跑估算器 + 追加结果。运行在 ProcessPool worker 进程中。

    不往 stdout 打印进度（worker 进程的 stdout 经管道回收易 BrokenPipe），
    结果统一写 CSV；主进程结束后汇总打印。
    """
    row = _run_job(task, seed, ckpt, method, args)
    _append_row(row)


def _run_group(task: str, args: argparse.Namespace, skip_done: bool) -> None:
    done = _completed_keys() if skip_done else set()
    jobs = [(s, c, m) for s in args.seeds for c in ALL101 for m in args.methods]
    todo = [j for j in jobs if (task, j[0], j[1], j[2]) not in done]
    print(f"\n===== task {task}: {len(todo)}/{len(jobs)} jobs "
          f"({len(jobs)-len(todo)} already done) =====", flush=True)
    # ProcessPool：每个 job 独立进程，无共享锁；--workers 控制并发数
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [
            pool.submit(_worker, task, s, c, m, args)
            for s, c, m in todo
        ]
        for f in as_completed(futs):
            f.result()  # 抛出任何异常
    print(f"task {task} done", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-done", action="store_true",
                    help="跳过输出 CSV 中已 ok 的 (task,seed,ckpt,method)")
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--resample-size", type=int, default=1000)
    ap.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    ap.add_argument("--n-hist-bins", type=int, default=10)
    ap.add_argument("--gaussian-reg-covar", type=float, default=1e-6)
    ap.add_argument("--mc-samples", type=int, default=100000)
    ap.add_argument("--mc-repeats", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    total = len(args.tasks) * len(args.seeds) * len(ALL101) * len(args.methods)
    print(f"All100 run: {total} jobs total: {args.tasks} × seeds={args.seeds} × "
          f"{len(ALL101)} ckpt × {args.methods}; out={OUT_CSV}", flush=True)
    for task in args.tasks:
        _run_group(task, args, args.skip_done)
    print("\nAll tasks complete.", flush=True)


if __name__ == "__main__":
    main()
