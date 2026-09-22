"""四方法串行、五 seed 并行、每 seed 四 worker 的离线全量调度。"""

from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
import json
import shlex
import shutil
import signal
import subprocess
import time
import traceback

from gc_ope.evaluate.offline_data import fixed_files
from gc_ope.evaluate.offline_experiment import ExperimentConfig
from gc_ope.evaluate.offline_results import (
    atomic_json, audit_result, read_rows, result_lock, sha256, verify_manifest,
)


METHODS = ("nn", "fm", "nf", "gmm")


def run_directory(output, method, seed):
    return Path(output) / "runs" / method / f"seed_{seed}"


def seed_command(args, method, seed, config_path):
    """每个子任务只负责一个 seed，内部并发严格限制为四个 checkpoint。"""
    return [sys.executable, "-u", str(ROOT / "scripts/evaluate_push_estimators.py"),
            "--checkpoint-root", str(args.checkpoint_root),
            "--output", str(run_directory(args.output, method, seed)),
            "--methods", method, "--seeds", str(seed),
            "--workers", str(args.workers_per_seed), "--config", str(config_path),
            "--checkpoints", *map(str, args.checkpoints)]


def stop_children(processes):
    """先让各 seed 入口正常清理进程池；超时再清理整个独立进程组。"""
    for process in processes.values():
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 15
    for process in processes.values():
        try:
            process.wait(timeout=max(.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    # 即使 seed 主进程意外退出，也检查其进程组，避免留下孤立 worker。
    for process in processes.values():
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def run_stage(args, method, config_path, report):
    """五个 seed 都通过覆盖率检查后，才允许进入下一个方法。"""
    processes, handles, finished, audits = {}, [], set(), {}
    try:
        for seed in args.seeds:
            directory = run_directory(args.output, method, seed)
            directory.mkdir(parents=True, exist_ok=True)
            handle = (directory / "console.log").open("a", buffering=1)
            handles.append(handle)
            command = seed_command(args, method, seed, config_path)
            print(shlex.join(command), file=handle, flush=True)
            # 独立进程组隔离 Ctrl-C，由总入口统一负责退出和回收。
            processes[seed] = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT,
                                                cwd=ROOT, start_new_session=True)
            report(f"启动 {method} seed={seed} pid={processes[seed].pid}，{args.workers_per_seed} workers")
        last_progress = time.monotonic()
        while len(finished) < len(processes):
            for seed, process in processes.items():
                if seed in finished or process.poll() is None:
                    continue
                directory = run_directory(args.output, method, seed)
                if process.returncode != 0:
                    raise RuntimeError(f"{method} seed={seed} 退出码 {process.returncode}，见 {directory / 'console.log'}")
                csv_path = directory / f"{method}_push_seed{seed}.csv"
                audit = audit_result(csv_path, args.checkpoints)
                if audit["missing"] or audit["extra"] or audit["invalid"]:
                    raise RuntimeError(f"{method} seed={seed} 覆盖率/有限值检查失败：{audit}")
                # 汇总 20 份 CSV 到一个目录，绘图时不必遍历调度内部的 runs 目录。
                target = args.output / "method_per_seed" / csv_path.name
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_suffix(".csv.tmp")
                shutil.copyfile(csv_path, temporary)
                os.replace(temporary, target)
                audits[f"{method}_seed{seed}"] = audit
                finished.add(seed)
                report(f"完成 {method} seed={seed}：正常 {audit['ok']}，数据不足跳过 {audit['skipped']}")
            if time.monotonic() - last_progress >= 30:
                counts = [f"seed {seed}: {len(read_rows(run_directory(args.output, method, seed) / f'{method}_push_seed{seed}.csv'))}/{len(args.checkpoints)}"
                          for seed in args.seeds]
                report(f"{method} 进度：" + "；".join(counts))
                last_progress = time.monotonic()
            time.sleep(.2)
        return audits
    finally:
        stop_children(processes)
        for handle in handles:
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT.parent / "gc_ope/checkpoints")
    parser.add_argument("--output", type=Path, default=ROOT / "logs/push_same_family_all100_5x4")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/evaluate/push_same_family_all100.json")
    parser.add_argument("--seeds", type=int, nargs="+", choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoints", type=int, nargs="+", default=list(range(10000, 1000001, 10000)))
    parser.add_argument("--workers-per-seed", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--dry-run", action="store_true", help="只检查输入并打印命令，不创建目录、不启动实验")
    args = parser.parse_args()
    args.checkpoint_root, args.output, args.config = args.checkpoint_root.resolve(), args.output.resolve(), args.config.resolve()
    args.seeds, args.checkpoints = sorted(set(args.seeds)), sorted(set(args.checkpoints))
    settings = json.loads(args.config.read_text())
    ExperimentConfig(checkpoint_root=str(args.checkpoint_root), **settings)

    # 启动前一次性检查五个 seed 的完整输入；避免运行到后面才发现缺文件。
    inputs = {}
    for seed in args.seeds:
        files = fixed_files(args.checkpoint_root, seed)
        missing = set(args.checkpoints) - files.keys()
        if missing:
            raise FileNotFoundError(f"seed {seed} 缺少 fixed checkpoint：{sorted(missing)}")
        inputs.update({str(path.resolve()): sha256(path) for step, path in files.items() if step <= max(args.checkpoints)})
    manifest = {"methods": list(METHODS), "seeds": args.seeds, "checkpoints": args.checkpoints,
                "workers_per_seed": args.workers_per_seed, "settings": settings, "inputs": inputs,
                "launcher_sha256": sha256(__file__),
                "runner_sha256": sha256(ROOT / "scripts/evaluate_push_estimators.py")}
    config_path = args.output / "experiment.json"
    if args.dry_run:
        print(f"检查通过：{len(inputs)} 份 fixed 输入；{len(METHODS)} 方法 × {len(args.seeds)} seed × {len(args.checkpoints)} checkpoint")
        print(f"方法串行：{' → '.join(METHODS)}；计算 worker 上限：{len(args.seeds) * args.workers_per_seed}")
        for method in METHODS:
            for seed in args.seeds:
                print(shlex.join(seed_command(args, method, seed, args.config)))
        return 0

    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    with result_lock(args.output), (args.output / "master.log").open("a", buffering=1) as log:
        def report(message):
            message = time.strftime("%Y-%m-%d %H:%M:%S ") + message
            print(message, flush=True)
            print(message, file=log, flush=True)
        verify_manifest(args.output, manifest)
        atomic_json(config_path, settings)
        # 清除上一轮的完成状态；逐方法更新，避免中断后残留旧的“全量已完成”。
        summary = {"status": "running", "audits": {}}
        atomic_json(args.output / "audit.json", summary)
        report(f"离线全量：{' → '.join(METHODS)}；{len(args.seeds)} seed 并行 × 每 seed {args.workers_per_seed} workers；CPU-only")
        try:
            for method in METHODS:
                summary["audits"].update(run_stage(args, method, config_path, report))
                atomic_json(args.output / "audit.json", summary)
            summary["status"] = "complete"
            report("全部方法通过覆盖率检查；汇总 CSV 位于 method_per_seed/")
            code = 0
        except KeyboardInterrupt:
            summary["status"] = "interrupted"
            report("已中断并清理全部 seed/worker；同一命令可续跑")
            code = 130
        except Exception:
            summary["status"] = "failed"
            report(traceback.format_exc())
            code = 1
        atomic_json(args.output / "audit.json", summary)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
