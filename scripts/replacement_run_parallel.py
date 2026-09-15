"""高吞吐版：task × method 粒度并行，每个 job 只跑一种 method。

避免旧调度器里"同一 worker 串行 4 种 method"导致 KDE（~40s）阻塞
GMM/histogram/gaussian（各 ~5s）的问题。

用法：
  # 默认 16 worker 并行跑 4 task × 4 method × 5 seed × 5 checkpoint
  conda run -n gc_ope python scripts/replacement_run_parallel.py

  # 只补 slide/reach/vvc（push 已有 25/25）
  conda run -n gc_ope python scripts/replacement_run_parallel.py --tasks slide reach vvc
"""

from __future__ import annotations

import argparse
import concurrent.futures
import sys
import time
from multiprocessing import Process, Queue
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from replacement_run_job import _run_one_job, _append_row, CSV_PATH, TASK_GOAL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
ALL_CHECKPOINTS = [100000, 300000, 500000, 700000, 1000000]
ALL_SEEDS = [1, 2, 3, 4, 5]
ALL_METHODS = ["kde", "gmm", "histogram", "gaussian", "nn", "flow_matching"]
ALL_TASKS = ["push", "slide", "reach", "vvc"]


def _already_done() -> set[tuple[str, int, int, str]]:
    """读已有 CSV，返回已完成的 (task, seed, ckpt, method) 集合，避免重复跑。"""
    if not CSV_PATH.exists():
        return set()
    df = pd.read_csv(CSV_PATH)
    ok = df[(df.status == "ok") | (df.status.str.startswith("skipped", na=False))]
    return set(zip(ok.task, ok.seed, ok.checkpoint, ok.method))


def _job(args, task: str, seed: int, ckpt: int, method: str, done: set) -> None:
    if (task, seed, ckpt, method) in done:
        return
    row = _run_one_job(
        task, seed, ckpt, method,
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
    print(f"[{task} s{seed} c{ckpt} {method}] {row['status']} kl={row['kl']:.4f}"
          if pd.notna(row.get("kl")) else
          f"[{task} s{seed} c{ckpt} {method}] {row['status']} {row.get('error','')[:60]}",
          flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=ALL_TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=ALL_SEEDS)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=ALL_CHECKPOINTS)
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS)
    ap.add_argument("--workers", type=int, default=16)
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

    done = _already_done()
    print(f"已完成 {len(done)} 个 (task,seed,ckpt,method)，跳过；workers={args.workers}", flush=True)

    jobs = [
        (task, seed, ckpt, method)
        for task in args.tasks
        for seed in args.seeds
        for ckpt in args.checkpoints
        for method in args.methods
    ]
    jobs = [j for j in jobs if j not in done]
    total = len(jobs)
    print(f"待跑 {total} 个 job（共 {total*args.mc_repeats*args.mc_samples/1e6:.0f}M MC samples）", flush=True)

    t0 = time.time()

    # 用 ProcessPoolExecutor 绕过 GIL：每个 (task,seed,ckpt,method) job 跑在独立进程
    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_job, args, task, seed, ckpt, method, done): (task, seed, ckpt, method)
            for task, seed, ckpt, method in jobs
        }
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            try:
                fut.result()
            except Exception as exc:
                print(f"[{i}/{total}] {key} UNCAUGHT: {exc}", flush=True)
            if i % 10 == 0:
                el = time.time() - t0
                rate = i / el
                eta = (total - i) / rate if rate > 0 else 0
                print(f"  progress {i}/{total}  elapsed={el/60:.1f}m  ETA={eta/60:.1f}m", flush=True)

    print(f"\n全部完成，耗时 {(time.time()-t0)/60:.1f} 分钟，结果 -> {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
