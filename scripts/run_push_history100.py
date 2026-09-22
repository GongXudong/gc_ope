"""Push/SAC 历史100实验：NN→FM→NF→GMM串行，每方法5个seed×4个worker。

历史含当前 checkpoint，每份完整评估记录有放回抽100，固定随机流跨方法共享。
MC为10000×5，FM为500次更新。KDE仅登记旧投稿CSV，不执行KDE任务。
用 --dry-run 检查20条命令；--checkpoints 可做真实小规模验证；支持 --skip-done 续跑。
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
METHODS = ["nn", "flow_matching", "normalizing_flow", "gmm"]
SEEDS = [1, 2, 3, 4, 5]
DEFAULT_OUT = ROOT / "logs/replacement_experiment/push_history100_inclusive_mc10000"


def command_for(method, seed, output, checkpoints=None):
    """显式记录关键参数，后续默认值调整不改变本轮预算。"""
    command = [sys.executable, "-u", str(ROOT / "scripts/run_method_single_seed.py"),
               "--method", method, "--task", "push", "--seed", str(seed), "--workers", "4",
               "--history-samples-per-checkpoint", "100", "--history-sampling-seed", "0",
               "--history-include-current", "--mc-samples", "10000", "--mc-repeats", "5",
               "--random-state", "0", "--kappa", "0.9", "--bandwidth", "0.2",
               "--fm-epochs", "500", "--fm-hidden", "32", "--fm-samples", "2000",
               "--fm-lr", "0.001", "--fm-ode-steps", "32", "--fm-likelihood-batch-size", "1024",
               "--fm-weight-decay", "0", "--nn-epochs", "100", "--nn-hidden", "16",
               "--nn-lr", "0.001", "--nn-early-stopping", "--nn-validation-fraction", "0.1",
               "--nn-n-iter-no-change", "10", "--nf-epochs", "100", "--nf-hidden", "32",
               "--nf-lr", "0.001", "--nf-transforms", "4", "--nf-bins", "8",
               "--nf-weight-decay", "0", "--n-components", "5", "--resample-size", "1000",
               "--gmm-reg-covar", "0.000001", "--skip-done", "--output-dir", str(output)]
    if checkpoints:
        command += ["--checkpoints", *map(str, checkpoints)]
    return command


def audit_csv(path, method, seed, checkpoints):
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    expected = set(checkpoints or range(10000, 1000001, 10000))
    # 续跑的 error 行可能保留；最后一条是该检查点的最终状态。
    latest = {}
    for row in rows:
        if row["method"] != method or row["task"] != "push" or int(row["seed"]) != seed:
            raise ValueError(f"CSV 的方法/任务/seed 不匹配: {path}")
        latest[int(row["checkpoint"])] = row
    missing = sorted(expected - latest.keys())
    errors = [latest[t] for t in sorted(expected & latest.keys())
              if latest[t]["status"] != "ok" and not latest[t]["status"].startswith("skipped:")]
    invalid_metrics = [t for t in sorted(expected & latest.keys()) if latest[t]["status"] == "ok"
                       and any(not math.isfinite(float(latest[t][column])) for column in ("kl", "fixed_grid_kl"))]
    counts = {}
    for t in sorted(expected & latest.keys()):
        status = latest[t]["status"]
        counts[status] = counts.get(status, 0) + 1
    return {"missing": missing, "errors": errors, "invalid_metrics": invalid_metrics,
            "counts": counts, "passed": not missing and not errors and not invalid_metrics}


def schedule(run_one):
    """stage 的五个 future 全部退出后，才进入下一方法；任何失败都会停止后续方法。"""
    stages = []
    for method in METHODS:
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
            results = list(pool.map(lambda seed: run_one(method, seed), SEEDS))
        stages.append({"method": method, "results": results})
        if any(not r["passed"] for r in results):
            break
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoints", type=int, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.checkpoints and (len(set(args.checkpoints)) != len(args.checkpoints)
                            or any(x not in range(10000, 1000001, 10000) for x in args.checkpoints)):
        parser.error("checkpoint 必须是不重复的10000..1000000，间隔10000")
    out = args.out_dir.resolve()
    output = out / "method_per_seed"
    commands = {(m, s): command_for(m, s, output, args.checkpoints) for m in METHODS for s in SEEDS}
    if args.dry_run:
        for method in METHODS:
            print(f"方法 {method}：以下五个命令同时执行，全部完成后再进入下一方法")
            for seed in SEEDS:
                print(shlex.join(commands[method, seed]))
        return 0

    out.mkdir(parents=True, exist_ok=True)
    lock = (out / "launch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    output.mkdir(exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run = out / "runs" / stamp
    run.mkdir(parents=True)
    (out / "latest_run.txt").write_text(str(run) + "\n")
    command_dir = out / "commands"
    command_dir.mkdir(exist_ok=True)
    for (method, seed), command in commands.items():
        shell_command = ["conda", "run", "--no-capture-output", "-n", "gc_ope", "env",
                         "OPENBLAS_NUM_THREADS=1", "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1", *command]
        (command_dir / f"{method}_push_seed{seed}.sh").write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\nexec " + shlex.join(shell_command) + "\n")
    legacy = [ROOT / f"plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/sac/eval_data/my_push_sac_seed_{s}_kde_0_9_eval_res_in_training.csv" for s in SEEDS]
    sources = [ROOT / "scripts/replacement_run_job.py", ROOT / "scripts/run_method_single_seed.py",
               Path(__file__), *sorted((ROOT / "src/gc_ope/evaluate").glob("evaluator_*.py"))]
    manifest = {
        "pid": os.getpid(), "started": datetime.datetime.now().isoformat(),
        "method_order": METHODS, "seeds": SEEDS, "workers_per_seed": 4,
        "history_protocol": "每个checkpoint含当前，全部记录有放回抽100；跨方法共享确定性抽样；kappa=0.9",
        "mc_samples": 10000, "mc_repeats": 5, "commands": {f"{m}/seed{s}": c for (m, s), c in commands.items()},
        "source_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "legacy_kde": [{"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                        "metric": "$D_{KL}$ [his]"} for p in legacy],
        "legacy_note": "KDE复用旧投稿；旧随机抽样索引未知，旧KL独立标准化/低密度截除、oracle带宽1.0与新raw-space KL/带宽0.2不同，旧MC单次而新重复5次；不能声称完全同口径",
    }
    (run / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    # 在正式训练前记录五条轨迹每份输入的哈希，便于复核抽样来源。
    inputs = sorted(p for seed in SEEDS for p in (ROOT / f"checkpoints/my_push/sac/seed_{seed}").glob("rl_model_*_steps_eval_res_on_fixed.csv"))
    (run / "input_hashes.json").write_text(json.dumps(
        {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}, indent=2))
    io_lock, process_lock = threading.Lock(), threading.Lock()
    processes = {}
    interrupted = threading.Event()
    combined = (run / "combined.log").open("w", buffering=1)

    def emit(message):
        with io_lock:
            print(message, flush=True)
            combined.write(message + "\n")

    def stop(signum, frame):
        interrupted.set()
        with process_lock:
            for proc in processes.values():
                if proc.poll() is None:
                    # 每个seed独立进程组；worker忽略SIGINT，由runner清理自己的池。
                    try:
                        os.killpg(proc.pid, signal.SIGINT)
                    except ProcessLookupError:
                        pass

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    emit(f"总启动器PID={os.getpid()}；日志={run}；最多20个计算worker，CPU-only")

    def run_one(method, seed):
        result = {"method": method, "seed": seed, "passed": False}
        if interrupted.is_set():
            return dict(result, exit_code=130)
        try:
            with (run / f"{method}_seed{seed}.log").open("w", buffering=1) as log:
                with process_lock:
                    if interrupted.is_set():
                        return dict(result, exit_code=130)
                    proc = subprocess.Popen(commands[method, seed], cwd=ROOT, env=env,
                                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                            text=True, bufsize=1, start_new_session=True)
                    processes[method, seed] = proc
                emit(f"[{method}/seed{seed}] 启动PID={proc.pid}")
                for line in proc.stdout:
                    log.write(line)
                    emit(f"[{method}/seed{seed}] {line.rstrip()}")
                result["exit_code"] = proc.wait()
            if result["exit_code"] == 0:
                result["audit"] = audit_csv(output / f"{method}_push_seed{seed}.csv", method, seed, args.checkpoints)
                result["passed"] = result["audit"]["passed"]
        except Exception as exc:
            result.update(exit_code=-1, error=repr(exc))
        finally:
            with process_lock:
                processes.pop((method, seed), None)
        (run / f"{method}_seed{seed}_exit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2))
        emit(f"[{method}/seed{seed}] 结束，退出码={result.get('exit_code')}，覆盖率检查={result['passed']}")
        return result

    stages = schedule(run_one)
    passed = len(stages) == len(METHODS) and all(r["passed"] for stage in stages for r in stage["results"])
    code = 130 if interrupted.is_set() else 0 if passed else 1
    (run / "completion.json").write_text(json.dumps({"exit_code": code, "finished": datetime.datetime.now().isoformat(),
                                                     "stages": stages}, ensure_ascii=False, indent=2))
    emit(f"调度结束，退出码={code}；完成记录={run / 'completion.json'}")
    combined.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
