"""Push/SAC 离线实验入口；算法逻辑位于 gc_ope.evaluate，不放在调度脚本里。"""

from pathlib import Path
import os
import sys

# 在导入 numpy/torch 前限制 BLAS，并固定到本 worktree，避免加载旧可编辑安装。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
for name in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[name] = "1"

import argparse
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
import json
import multiprocessing as mp
import signal
import platform
from importlib.metadata import version

from gc_ope.evaluate.offline_data import fixed_files
from gc_ope.evaluate.offline_experiment import ExperimentConfig, run_checkpoint
from gc_ope.evaluate.offline_results import (
    atomic_json, sha256, result_lock, verify_manifest, read_rows, save_row, completed, audit_result,
)


def run_job(job):
    config, method, seed, checkpoint, output = job
    log_path = Path(output) / "jobs" / f"{method}_seed{seed}_{checkpoint}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    # 每个任务保留独立日志；断点重试追加到原日志，不覆盖错误证据。
    with log_path.open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
        print(f"开始 {method} seed={seed} checkpoint={checkpoint}", flush=True)
        row, detail = run_checkpoint(config, method, seed, checkpoint)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    atomic_json(log_path.with_suffix(".json"), {"row": row, "detail": detail})
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=["nn", "fm", "nf", "gmm", "kde"], default=["nn", "fm", "nf", "gmm"])
    parser.add_argument("--seeds", nargs="+", type=int, choices=range(1, 6), default=[1, 2, 3, 4, 5])
    parser.add_argument("--checkpoints", nargs="+", type=int,
                        help="省略时要求完整的 100 个 checkpoint（10000 到 1000000）")
    parser.add_argument("--workers", type=int, choices=range(1, 21), default=4)
    parser.add_argument("--config", type=Path, help="可选 JSON 参数覆盖；保存完整展开配置供续跑校验")
    args = parser.parse_args()
    config_values = json.loads(args.config.read_text()) if args.config else {}
    config = ExperimentConfig(checkpoint_root=str(args.checkpoint_root.resolve()), **config_values)
    expected = sorted(set(args.checkpoints or range(10000, 1000001, 10000)))
    output = args.output.resolve()

    # 清单包含用到的所有历史输入及源代码摘要，禁止用 skip-done 混合不同协议。
    inputs = {}
    for seed in args.seeds:
        files = fixed_files(config.checkpoint_root, seed)
        missing = set(expected) - files.keys()
        if missing:
            raise FileNotFoundError(f"seed {seed} 缺少 checkpoint：{sorted(missing)}")
        inputs.update({str(path.resolve()): sha256(path) for step, path in files.items() if step <= max(expected)})
    source_files = sorted((ROOT / "src" / "gc_ope" / "evaluate").rglob("*.py")) + [Path(__file__)]
    manifest = {"config": asdict(config), "seeds": args.seeds, "methods": args.methods,
                "checkpoints": expected, "inputs": inputs,
                "source": {str(path.relative_to(ROOT)): sha256(path) for path in source_files},
                "runtime": {"python": platform.python_version(), **{name: version(name) for name in
                            ["numpy", "scipy", "scikit-learn", "torch", "zuko"]}}}

    # SIGTERM 与 Ctrl-C 都先清理工作进程；Pool 上下文退出时 terminate + join。
    def interrupt(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, interrupt)
    with result_lock(output):
        verify_manifest(output, manifest)
        with (output / "run.log").open("a", buffering=1) as log:
            def report(message):
                print(message, flush=True)
                print(message, file=log, flush=True)
            report(f"导入根目录：{ROOT / 'src'}；{args.workers} 个 CPU worker；BLAS 单线程")
            try:
                # 方法依次运行；只创建一个有界进程池，不再嵌套 seed/worker 两层调度。
                with mp.get_context("spawn").Pool(args.workers) as pool:
                    for method in args.methods:
                        known = {seed: read_rows(output / f"{method}_push_seed{seed}.csv") for seed in args.seeds}
                        jobs = [(config, method, seed, step, str(output)) for step in expected for seed in args.seeds
                                if step not in known[seed] or not completed(known[seed][step])]
                        report(f"{method}：待运行 {len(jobs)} 个 checkpoint")
                        for row in pool.imap_unordered(run_job, jobs, chunksize=1):
                            save_row(output / f"{method}_push_seed{row['seed']}.csv", row)
                            report(f"{method} seed={row['seed']} step={row['checkpoint']} {row['status']} KL={row['kl']}")
                            if row.get("fit_warnings"):
                                report(f"拟合质量提醒：{row['fit_warnings']}")
            except KeyboardInterrupt:
                report("收到中断：工作进程已清理，已落盘结果可续跑")
                return 130
        audit = {f"{method}_seed{seed}": audit_result(output / f"{method}_push_seed{seed}.csv", expected)
                 for method in args.methods for seed in args.seeds}
        atomic_json(output / "audit.json", audit)
        return int(any(value["missing"] or value["extra"] or value["invalid"] for value in audit.values()))


if __name__ == "__main__":
    raise SystemExit(main())
