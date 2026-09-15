"""全量 replacement 实验调度（plan.md 第 8/10/11 节）。

按 task 分组，每组起 N 个 worker，worker 内顺序跑所有 (seed × checkpoint × method)
job。同一 task 的 4 种估计器共享 CSV 文件（追加模式），AC-7.4 保证单 job
失败不影响其它 job。

用法：
  # 默认：push/slide 各 5 seed × 5 checkpoints × 4 methods，每 task 8 worker
  conda run -n gc_ope python scripts/replacement_run_all.py

  # 指定 task 范围和并行度
  conda run -n gc_ope python scripts/replacement_run_all.py --tasks push slide reach vvc --workers 4

checkpoint 默认取 [100k, 300k, 500k, 700k, 1000k]（与交接脚本对齐，避免 10 个
checkpoint × 4 method 的组合爆炸；需要全量可改 --checkpoints）。
"""

from __future__ import annotations

import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CHECKPOINTS = [100000, 300000, 500000, 700000, 1000000]
DEFAULT_SEEDS = [1, 2, 3, 4, 5]
DEFAULT_TASKS = ["push", "slide", "reach", "vvc"]
DEFAULT_METHODS = ["kde", "gmm", "histogram", "gaussian"]


def _run_group(task: str, seeds: list[int], checkpoints: list[int],
               methods: list[str], args: argparse.Namespace) -> None:
    """一个 task 的全部 job，按 checkpoint 分片，多 worker 并行。"""
    # 把 (seed, checkpoint) 对均分给 N 个 worker
    pairs = [(s, c) for s in seeds for c in checkpoints]
    n = min(args.workers, len(pairs))
    chunks = [pairs[i::n] for i in range(n)]

    def run_chunk(chunk: list[tuple[int, int]]) -> None:
        for seed, ckpt in chunk:
            cmd = [
                "conda", "run", "-n", "gc_ope",
                "python", "scripts/replacement_run_job.py",
                "--task", task,
                "--seed", str(seed),
                "--checkpoint", str(ckpt),
                "--methods", *methods,
                "--kappa", str(args.kappa),
                "--bandwidth", str(args.bandwidth),
                "--n-components", str(args.n_components),
                "--n-hist-bins", str(args.n_hist_bins),
                "--mc-samples", str(args.mc_samples),
                "--mc-repeats", str(args.mc_repeats),
                "--random-state", str(args.random_state),
            ]
            print(f"[{task}] seed={seed} ckpt={ckpt}", flush=True)
            result = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  ERROR: {result.stderr[-200:]}", flush=True)
            else:
                # 只打印最后几行
                last = [l for l in result.stdout.splitlines() if l.startswith("{")]
                if last:
                    print(f"  {last[-1][:120]}", flush=True)

    with ThreadPoolExecutor(max_workers=n) as pool:
        futures = [pool.submit(run_chunk, chunk) for chunk in chunks if chunk]
        for f in as_completed(futures):
            f.result()  # 抛出任何异常


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--checkpoints", type=int, nargs="+", default=DEFAULT_CHECKPOINTS)
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--workers", type=int, default=8,
                    help="parallel workers per task group")
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--resample-size", type=int, default=1000)
    ap.add_argument("--n-hist-bins", type=int, default=10)
    ap.add_argument("--mc-samples", type=int, default=100000)
    ap.add_argument("--mc-repeats", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    total_jobs = len(args.tasks) * len(args.seeds) * len(args.checkpoints) * len(args.methods)
    print(f"Starting {total_jobs} jobs: {args.tasks} × seeds={args.seeds} × "
          f"checkpoints={args.checkpoints} × methods={args.methods}", flush=True)
    print(f"Workers per task: {args.workers}", flush=True)

    for task in args.tasks:
        print(f"\n===== task: {task} =====", flush=True)
        _run_group(task, args.seeds, args.checkpoints, args.methods, args)
        print(f"task {task} done", flush=True)

    print("\nAll tasks complete.", flush=True)


if __name__ == "__main__":
    main()
