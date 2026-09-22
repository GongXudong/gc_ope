"""按 method × task × seed 运行一条轨迹的全量 100-checkpoint 评估。

设计约束：
- 一个方法、一条训练轨迹、一个结果 CSV，禁止多方法/多轨迹混写；
- checkpoint 可以并行计算，因为同一 seed 内历史输入只依赖 checkpoint < t；
- 输出列与 replacement_run_job.py 完全一致；
- `--skip-done` 把 ok 和 skipped:* 视为已完成，error 行会重跑；
- 可选地复现课程学习中的 MEGA/RIG/DISCERN 候选目标采样，并将每个
  checkpoint 的 sampled behavioral goals 写入独立 CSV；
- 当前 kde、gmm、nn、flow_matching 和 normalizing_flow 已接入。

示例（只计算 KL 结果）：
  conda run -n gc_ope env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python scripts/run_method_single_seed.py \\
    --method gmm --task slide --seed 1 --workers 4 --skip-done

示例（同时保存每个 checkpoint 的 100 个 MEGA behavioral goals）：
  conda run -n gc_ope env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \\
    python scripts/run_method_single_seed.py \\
    --method gmm --task slide --seed 1 --workers 4 --skip-done \\
    --save-sampled-goals --n-sampled-goals 100 --candidate-goals 100 \\
    --sampling-method mega --sampling-seed 0
"""

from __future__ import annotations

# 必须在导入 numpy/sklearn 前设置，避免每个 worker 的 BLAS 线程超额。
import os

_SINGLE_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
for _name in _SINGLE_THREAD_VARS:
    os.environ[_name] = "1"

import argparse
import csv
import fcntl
import json
import signal
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT_DIR = ROOT / "logs" / "replacement_experiment" / "all100" / "method_per_seed"
ALL100_CHECKPOINTS = [10000 + 10000 * i for i in range(99)] + [1000000]
METRIC_COLUMNS = [
    "task",
    "seed",
    "checkpoint",
    "method",
    "kl",
    "kl_seed_std",
    "fixed_grid_kl",
    "historical_successes",
    "reference_successes",
    "status",
    "error",
    "job_time_s",
]

ALL_METHODS = ["kde", "gmm", "nn", "flow_matching", "normalizing_flow"]
AVAILABLE_METHODS = {"kde", "gmm", "nn", "flow_matching", "normalizing_flow"}
UNAVAILABLE_REASONS = {}
SAMPLING_METHODS = ["mega", "rig", "discern"]
_ACTIVE_POOL: ProcessPoolExecutor | None = None
_INTERRUPT_REQUESTED = False


def _worker_init() -> None:
    """让 Ctrl-C 由主进程统一处理，再由主进程终止整个 worker 池。"""
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _handle_sigint(signum: int, frame) -> None:
    """不等待当前 future，立即终止 worker 池。"""
    global _INTERRUPT_REQUESTED
    if _INTERRUPT_REQUESTED:
        # 第二次 Ctrl-C 不再尝试清理，避免用户被卡在退出流程。
        os._exit(130)
    _INTERRUPT_REQUESTED = True
    print("\n收到 Ctrl-C，立即取消任务并终止所有 worker...", file=sys.stderr, flush=True)
    if _ACTIVE_POOL is not None:
        _terminate_pool(_ACTIVE_POOL)
    raise SystemExit(130)


def _terminate_pool(pool: ProcessPoolExecutor) -> None:
    """尽快终止仍在运行的 ProcessPool worker。

    ``ProcessPoolExecutor.__exit__`` 默认会等待正在运行的 future；这在
    用户按 Ctrl-C 时会造成主进程等待而 worker 继续计算。这里先取消未开始
    的 future，再终止已经启动的子进程，必要时升级为 SIGKILL。
    """
    process_map = getattr(pool, "_processes", None)
    processes = list(process_map.values()) if process_map else []
    if process_map is not None:
        pool.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=2.0)
    for process in processes:
        if process.is_alive():
            process.kill()
    for process in processes:
        if process.is_alive():
            process.join(timeout=1.0)


def _output_path(output_dir: Path, method: str, task: str, seed: int) -> Path:
    return output_dir / f"{method}_{task}_seed{seed}.csv"


def _sampled_goals_output_path(output_dir: Path, method: str, task: str, seed: int) -> Path:
    return output_dir / f"{method}_{task}_seed{seed}_sampled_goals.csv"


def _sampled_columns(n_sampled_goals: int) -> list[str]:
    if n_sampled_goals <= 0:
        raise ValueError("n_sampled_goals must be positive")
    return ["checkpoint"] + [f"goal_{i}" for i in range(1, n_sampled_goals + 1)]


def _read_rows(path: Path, columns: list[str]) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != columns:
            raise ValueError(f"existing output has unexpected columns: {path}: {reader.fieldnames!r}")
        return list(reader)


def _validate_output_scope(path: Path, task: str, seed: int, method: str) -> None:
    rows = _read_rows(path, METRIC_COLUMNS)
    for row in rows:
        if row["task"] != task or row["seed"] != str(seed) or row["method"] != method:
            raise ValueError(
                "existing output contains rows outside the requested single "
                f"method/task/seed scope: {path}: {row!r}"
            )


def _validate_sampled_scope(path: Path, n_sampled_goals: int) -> None:
    columns = _sampled_columns(n_sampled_goals)
    rows = _read_rows(path, columns)
    expected_prefix = ["checkpoint"]
    if columns[:1] != expected_prefix:
        raise AssertionError("unexpected sampled-goals schema")
    for row in rows:
        int(row["checkpoint"])


def _completed_checkpoints(path: Path) -> set[int]:
    completed = set()
    for row in _read_rows(path, METRIC_COLUMNS):
        status = row.get("status", "")
        if status == "ok" or status.startswith("skipped:"):
            completed.add(int(row["checkpoint"]))
    return completed


def _completed_sampled_checkpoints(path: Path, n_sampled_goals: int) -> set[int]:
    return {int(row["checkpoint"]) for row in _read_rows(path, _sampled_columns(n_sampled_goals))}


def _append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        need_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=METRIC_COLUMNS, lineterminator="\n")
        if need_header:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def _format_goal(goal: tuple[float, float]) -> str:
    x, y = goal
    if not (float(x) == float(x) and float(y) == float(y)):
        return "nan"
    return f"{float(x):.10g};{float(y):.10g}"


def _append_sampled_row(path: Path, goals: list[tuple[float, float]], n_sampled_goals: int, checkpoint: int) -> None:
    if len(goals) != n_sampled_goals:
        raise ValueError(
            f"checkpoint {checkpoint}: expected {n_sampled_goals} sampled goals, got {len(goals)}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = _sampled_columns(n_sampled_goals)
    row = {"checkpoint": checkpoint}
    row.update({f"goal_{i}": _format_goal(goal) for i, goal in enumerate(goals, start=1)})
    with path.open("a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        need_header = f.tell() == 0
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        if need_header:
            writer.writeheader()
        writer.writerow(row)
        f.flush()
        fcntl.flock(f, fcntl.LOCK_UN)


def _sort_csv_by_checkpoint(path: Path, columns: list[str]) -> None:
    rows = _read_rows(path, columns)
    if not rows:
        return
    rows.sort(key=lambda row: int(row["checkpoint"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _run_job(checkpoint: int, args: argparse.Namespace) -> dict:
    # 延迟导入，确保子进程中也先继承/设置单线程 BLAS 环境变量。
    import replacement_run_job as mod

    return mod._run_one_job(
        args.task,
        args.seed,
        checkpoint,
        args.method,
        kappa=args.kappa,
        n_components=args.n_components,
        resample_size=args.resample_size,
        bandwidth=args.bandwidth,
        random_state=args.random_state,
        gmm_reg_covar=args.gmm_reg_covar,
        # 兼容单 job 接口的旧参数；GMM/FM 均不使用这两项。
        n_hist_bins=10,
        gaussian_reg_covar=1e-6,
        nn_epochs=args.nn_epochs,
        nn_lr=args.nn_lr,
        nn_hidden=args.nn_hidden,
        nn_early_stopping=args.nn_early_stopping,
        nn_validation_fraction=args.nn_validation_fraction,
        nn_n_iter_no_change=args.nn_n_iter_no_change,
        fm_epochs=args.fm_epochs,
        fm_hidden=args.fm_hidden,
        fm_samples=args.fm_samples,
        fm_lr=args.fm_lr,
        fm_ode_steps=args.fm_ode_steps,
        fm_weight_decay=args.fm_weight_decay,
        fm_likelihood_batch_size=args.fm_likelihood_batch_size,
        nf_epochs=args.nf_epochs,
        nf_lr=args.nf_lr,
        nf_hidden=args.nf_hidden,
        nf_transforms=args.nf_transforms,
        nf_bins=args.nf_bins,
        nf_weight_decay=args.nf_weight_decay,
        mc_samples=args.mc_samples,
        mc_repeats=args.mc_repeats,
        history_samples_per_checkpoint=args.history_samples_per_checkpoint,
        history_sampling_seed=args.history_sampling_seed,
        history_include_current=args.history_include_current,
        n_sampled_goals=args.n_sampled_goals if args.save_sampled_goals else 0,
        candidate_goals=args.candidate_goals,
        sampling_method=args.sampling_method,
        sampling_seed=args.sampling_seed,
    )


def _summary(path: Path) -> str:
    rows = _read_rows(path, METRIC_COLUMNS)
    statuses = Counter(row["status"] for row in rows)
    ok = statuses.get("ok", 0)
    skipped = sum(count for status, count in statuses.items() if status.startswith("skipped:"))
    error = sum(count for status, count in statuses.items() if status not in {"ok"} and not status.startswith("skipped:"))
    return f"rows={len(rows)}, ok={ok}, skipped={skipped}, error={error}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", required=True, choices=ALL_METHODS)
    parser.add_argument("--task", required=True, choices=["push", "slide"])
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--checkpoints", type=int, nargs="+", help="只运行指定 checkpoint；默认全部 100 个")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--skip-done", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只打印待运行任务，不执行评估")
    parser.add_argument("--kappa", type=float, default=0.9)
    parser.add_argument("--bandwidth", type=float, default=0.2)
    parser.add_argument("--n-components", type=int, default=5)
    parser.add_argument("--resample-size", type=int, default=1000)
    parser.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--mc-samples", type=int, default=100000)
    parser.add_argument("--mc-repeats", type=int, default=5)
    parser.add_argument("--history-samples-per-checkpoint", type=int, default=0, help="每个历史 checkpoint 从全部记录有放回抽样；0 使用全量")
    parser.add_argument("--history-sampling-seed", type=int, default=0, help="所有方法共享的历史抽样种子")
    parser.add_argument("--history-include-current", action="store_true", help="训练也使用当前 checkpoint 的抽样记录；默认只用严格历史")
    # 采样功能默认关闭，不改变原有只输出 KL 的命令行为。
    parser.add_argument("--save-sampled-goals", action="store_true", help="保存每个 checkpoint 的 behavioral goals")
    parser.add_argument("--n-sampled-goals", type=int, default=100, help="每个 checkpoint 采样的 behavioral goal 数")
    parser.add_argument("--candidate-goals", type=int, default=100, help="每个 behavioral goal 的候选目标数，对应训练 wrapper 的 sample_n")
    parser.add_argument("--sampling-method", choices=SAMPLING_METHODS, default="mega", help="复现课程 wrapper 的目标选择规则")
    parser.add_argument("--sampling-seed", type=int, default=0, help="离线 behavioral-goal 采样随机种子")
    # 各学习型方法独立配置；不会改变其它方法的训练参数。
    parser.add_argument("--nn-epochs", type=int, default=100)
    parser.add_argument("--nn-lr", type=float, default=1e-3)
    parser.add_argument("--nn-hidden", type=int, default=16)
    parser.add_argument("--nn-early-stopping", action=argparse.BooleanOptionalAction, default=True, help="是否启用随机 validation 和 early stopping；默认开启")
    parser.add_argument("--nn-validation-fraction", type=float, default=0.1, help="NN 随机验证集比例")
    parser.add_argument("--nn-n-iter-no-change", type=int, default=10, help="验证集指标连续多少轮不改善后停止")
    parser.add_argument("--fm-epochs", type=int, default=100, help="FM 优化更新轮数；每轮加权抽样一次，不使用 validation")
    parser.add_argument("--fm-hidden", type=int, default=32)
    parser.add_argument("--fm-samples", type=int, default=2000, help="每个 FM epoch 的加权目标抽样数")
    parser.add_argument("--fm-lr", type=float, default=1e-3)
    parser.add_argument("--fm-ode-steps", type=int, default=32, help="FM likelihood 的 RK4 步数")
    parser.add_argument("--fm-weight-decay", type=float, default=0.0)
    parser.add_argument("--fm-likelihood-batch-size", type=int, default=1024, help="FM 密度查询分块大小")
    parser.add_argument("--nf-epochs", type=int, default=100)
    parser.add_argument("--nf-lr", type=float, default=1e-3)
    parser.add_argument("--nf-hidden", type=int, default=32)
    parser.add_argument("--nf-transforms", type=int, default=4)
    parser.add_argument("--nf-bins", type=int, default=8)
    parser.add_argument("--nf-weight-decay", type=float, default=0.0)
    args = parser.parse_args()

    if args.history_samples_per_checkpoint < 0 or args.history_sampling_seed < 0:
        parser.error("历史抽样数量和种子不能为负")
    if args.mc_samples < 1 or args.mc_repeats < 1:
        parser.error("MC 样本数和重复次数必须为正")

    if args.method == "flow_matching":
        if min(args.fm_epochs, args.fm_hidden, args.fm_samples, args.fm_ode_steps, args.fm_likelihood_batch_size) <= 0:
            parser.error("FM 的轮数、宽度、抽样数、积分步数和查询块大小必须为正")
        if not (0 < args.fm_lr < float("inf")) or not (0 <= args.fm_weight_decay < float("inf")):
            parser.error("FM 学习率必须为正有限数，weight decay 必须非负有限")
    if args.method not in AVAILABLE_METHODS:
        parser.error(
            f"method {args.method!r} is registered but not available yet: "
            f"{UNAVAILABLE_REASONS[args.method]}"
        )
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.n_sampled_goals <= 0:
        parser.error("--n-sampled-goals must be >= 1")
    if args.candidate_goals <= 0:
        parser.error("--candidate-goals must be >= 1")

    output = _output_path(args.output_dir, args.method, args.task, args.seed)
    # 抽样实验写独立配置，拒绝把不同协议的结果当成已完成任务混用。
    config_path = output.with_suffix(".config.json")
    if args.history_samples_per_checkpoint or args.history_include_current or config_path.exists():
        excluded = {"output_dir", "workers", "checkpoints", "skip_done", "dry_run"}
        config = {k: v for k, v in vars(args).items() if k not in excluded}
        if config_path.exists():
            if json.loads(config_path.read_text()) != config:
                parser.error(f"输出目录已存在不同参数的结果，请另选目录: {config_path}")
        elif output.exists() and output.stat().st_size:
            parser.error(f"已有 CSV 缺少抽样协议配置，不能安全续跑: {output}")
        elif not args.dry_run:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    _validate_output_scope(output, args.task, args.seed, args.method)
    sampled_output = _sampled_goals_output_path(args.output_dir, args.method, args.task, args.seed)
    if args.save_sampled_goals:
        _validate_sampled_scope(sampled_output, args.n_sampled_goals)

    metric_done = _completed_checkpoints(output) if args.skip_done else set()
    sampled_done = (
        _completed_sampled_checkpoints(sampled_output, args.n_sampled_goals)
        if args.skip_done and args.save_sampled_goals
        else set()
    )
    done = metric_done & sampled_done if args.save_sampled_goals and args.skip_done else metric_done
    checkpoints = args.checkpoints or ALL100_CHECKPOINTS
    if len(set(checkpoints)) != len(checkpoints) or not set(checkpoints) <= set(ALL100_CHECKPOINTS):
        parser.error("--checkpoints 必须是不重复的 10000..1000000（步长 10000）")
    todo = [checkpoint for checkpoint in sorted(checkpoints) if checkpoint not in done]

    print(f"method={args.method}, task={args.task}, seed={args.seed}")
    print(f"output: {output}")
    if args.save_sampled_goals:
        print(f"sampled goals output: {sampled_output}")
        print(
            f"sampling: method={args.sampling_method}, goals/checkpoint={args.n_sampled_goals}, "
            f"candidates/goal={args.candidate_goals}, seed={args.sampling_seed}"
        )
    print(f"jobs: {len(todo)}/{len(checkpoints)} to run ({len(set(checkpoints) & done)} already done)")
    if args.dry_run:
        preview = ", ".join(str(checkpoint) for checkpoint in todo[:10])
        suffix = " ..." if len(todo) > 10 else ""
        print(f"dry-run todo preview: {preview}{suffix}")
        print("dry-run complete; no evaluation was started and no file was changed")
        return

    metric_written = set(metric_done)
    sampled_written = set(sampled_done)
    started = time.time()
    global _ACTIVE_POOL
    signal.signal(signal.SIGINT, _handle_sigint)
    pool = ProcessPoolExecutor(max_workers=args.workers, initializer=_worker_init)
    _ACTIVE_POOL = pool
    interrupted = False
    futures = {}
    try:
        futures = {pool.submit(_run_job, checkpoint, args): checkpoint for checkpoint in todo}
        for future in as_completed(futures):
            checkpoint = futures[future]
            row = future.result()
            # 如果只是在补采样文件，已有最终指标行不重复追加；error 行则按
            # 原有语义追加一次新的重跑结果。
            if checkpoint not in metric_written or row.get("status") == "error":
                _append_row(output, row)
                metric_written.add(checkpoint)
            sampled_goals = row.get("_sampled_goals")
            if args.save_sampled_goals and sampled_goals is not None and checkpoint not in sampled_written:
                _append_sampled_row(sampled_output, sampled_goals, args.n_sampled_goals, checkpoint)
                sampled_written.add(checkpoint)
            print(
                f"checkpoint={checkpoint}: status={row['status']}, "
                f"time={row['job_time_s']}s",
                flush=True,
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n收到 Ctrl-C，正在取消任务并终止所有 worker...", file=sys.stderr, flush=True)
        for future in futures:
            future.cancel()
        _terminate_pool(pool)
        print("worker 已终止；已写入的 CSV 行保留，可用 --skip-done 续跑。", file=sys.stderr, flush=True)
        raise SystemExit(130)
    except SystemExit:
        interrupted = True
        raise
    except BaseException:
        interrupted = True
        _terminate_pool(pool)
        raise
    finally:
        if not interrupted:
            pool.shutdown(wait=True)
        _ACTIVE_POOL = None

    _sort_csv_by_checkpoint(output, METRIC_COLUMNS)
    if args.save_sampled_goals:
        _sort_csv_by_checkpoint(sampled_output, _sampled_columns(args.n_sampled_goals))
    print(f"done in {time.time() - started:.1f}s")
    print(f"summary: {_summary(output)}")
    if args.save_sampled_goals:
        print(f"sampled goals rows: {len(_read_rows(sampled_output, _sampled_columns(args.n_sampled_goals)))}")


if __name__ == "__main__":
    main()
