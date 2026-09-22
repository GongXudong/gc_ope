"""历史留出选参 → 封存配置 → 跨 seed 验证 → 小规模同方法参考 KL。"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

# 独立串行诊断只占一个计算线程，避免与用户工作争抢算力。
for key in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ[key] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
from gc_ope.evaluate.evaluator_factory import make_evaluator, fit_evaluator
from gc_ope.evaluate.history_validation import sampled_history, split_successes, weighted_nll, select_candidates
from gc_ope.evaluate.offline_experiment import ExperimentConfig, run_checkpoint
from gc_ope.evaluate.offline_results import sha256
from gc_ope.evaluate.offline_data import fixed_files


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def validate_case(root, seed, step, split_seed, method, candidate, parameters):
    """拟合只看训练部分；标准化也只在训练部分拟合。"""
    started = time.perf_counter()
    row = dict(seed=seed, checkpoint=step, split_seed=split_seed, method=method,
               candidate=candidate, status="error", valid_nll=None)
    try:
        history, identities = sampled_history(root, seed, step)
        train, valid, ti, vi = split_successes(history, identities, split_seed)
        model = make_evaluator(method, parameters=parameters)
        train.fill(model)
        fit_evaluator(method, model)
        row.update(status="ok", train_nll=weighted_nll(model, train), valid_nll=weighted_nll(model, valid),
                   train_records=len(ti), valid_records=len(vi),
                   valid_ess=float(valid.weights.sum()**2 / (valid.weights**2).sum()),
                   fit=model.fit_diagnostics_)
        if method in {"gmm", "gmm_em"}:
            row["minimum_scaled_covariance_eigenvalue"] = float(np.linalg.eigvalsh(model.gmm.covariances_).min())
    except Exception:
        row["error"] = traceback.format_exc()
    row["seconds"] = time.perf_counter() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, default=ROOT.parent / "gc_ope/checkpoints")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-config", type=Path, default=ROOT / "configs/evaluate/push_same_family_all100.json")
    parser.add_argument("--candidates", type=Path, default=ROOT / "configs/evaluate/history_validation_candidates.json")
    parser.add_argument("--evaluate-selected", action="store_true", help="封存后计算小规模 MC KL；不用于选参")
    parser.add_argument("--smoke", action="store_true", help="仅检查加权 GMM 的小任务和完成记录")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    log = (args.output / "run.log").open("w", buffering=1)

    def report(message):
        print(message, flush=True)
        print(message, file=log, flush=True)

    started = time.perf_counter()
    write_json(args.output / "process.json", dict(pid=os.getpid(), command=sys.argv))
    base = json.loads(args.base_config.read_text())
    base["parameters"].update(json.loads((ROOT / "configs/evaluate/push_gmm_em_all100.json").read_text())["parameters"])
    candidates = json.loads(args.candidates.read_text())
    # 先声明候选和任务，再运行；跨 seed 检查及全量参考 KL 均不能回头修改胜者。
    cases = [(1, step, split) for step in [100000, 400000, 1000000] for split in [1701, 1702]]
    transfer_seeds, transfer_steps, pilot_steps = [2, 3, 4, 5], [100000, 1000000], [10000, 100000, 1000000]
    if args.smoke:
        candidates = {"gmm_em": {key: candidates["gmm_em"][key] for key in ["baseline", "reg_0p01"]}}
        cases = [(1, 100000, 1701)]
        transfer_seeds, transfer_steps, pilot_steps = [2], [100000], [100000]
    resolved = {method: {name: {**base["parameters"][method], **override}
                        for name, override in variants.items()} for method, variants in candidates.items()}
    write_json(args.output / "plan.json", dict(candidates=resolved, selection_cases=cases,
               validation_fraction=.2, group="source_checkpoint,csv_row", criterion="mean raw weighted heldout NLL",
               transfer_seeds=transfer_seeds, transfer_steps=transfer_steps,
               pilot_steps=pilot_steps, mc_samples=10000, mc_repeats=5))
    source_paths = [Path(__file__), args.candidates, args.base_config,
                    ROOT / "configs/evaluate/push_gmm_em_all100.json"]
    source_paths += list((ROOT / "src/gc_ope/evaluate").rglob("*.py"))
    input_paths = [path for seed in [1, *transfer_seeds] for step, path in fixed_files(args.checkpoint_root, seed).items()
                   if step <= max(case[1] for case in cases)]
    hashes = {str(path): sha256(path) for path in source_paths + input_paths}
    write_json(args.output / "source_sha256.json", hashes)
    rows = []
    for method, variants in resolved.items():
        for name, parameters in variants.items():
            for seed, step, split in cases:
                row = validate_case(args.checkpoint_root, seed, step, split, method, name, parameters)
                rows.append(row)
                write_json(args.output / "selection_rows.json", rows)
                report(f"历史留出 {method}/{name} seed={seed} step={step} split={split}: {row['status']} NLL={row['valid_nll']}")
    selected, scores = select_candidates(rows, resolved, cases)
    write_json(args.output / "selection.json", dict(selected=selected, scores=scores))
    chosen = {method: variants[selected[method]] for method, variants in resolved.items()}
    experiment = {**base, "parameters": {**base["parameters"], **chosen}}
    write_json(args.output / "selected_experiment.json", experiment)
    sealed_hash = sha256(args.output / "selected_experiment.json")
    report(f"配置已封存：{selected}；SHA256={sealed_hash}")

    # 跨 seed 的历史留出检查是迁移验证，不再从这些结果选候选。
    transfer = []
    for method in resolved:
        for name in dict.fromkeys(["baseline", selected[method]]):
            for seed in transfer_seeds:
                for step in transfer_steps:
                    row = validate_case(args.checkpoint_root, seed, step, 1703, method, name, resolved[method][name])
                    transfer.append(row)
                    write_json(args.output / "transfer_rows.json", transfer)
                    report(f"跨 seed 留出 {method}/{name} seed={seed} step={step}: {row['status']} NLL={row['valid_nll']}")

    # 只有在选择已写入磁盘后，才允许计算同方法的全量参考 KL。
    pilot = []
    if args.evaluate_selected:
        config = ExperimentConfig(str(args.checkpoint_root), **experiment)
        for method in resolved:
            for seed in transfer_seeds:
                for step in pilot_steps:
                    row, detail = run_checkpoint(config, method, seed, step)
                    pilot.append(row)
                    write_json(args.output / f"pilot_{method}_{seed}_{step}.json", dict(row=row, detail=detail))
                    pd.DataFrame(pilot).to_csv(args.output / "pilot.csv", index=False)
                    report(f"封存后 KL {method} seed={seed} step={step}: {row['status']} KL={row['kl']}")
    assert sha256(args.output / "selected_experiment.json") == sealed_hash
    if any(sha256(path) != value for path, value in hashes.items()):
        raise RuntimeError("诊断期间代码、配置或输入数据发生变化")
    failed = sum(row["status"] == "error" for row in rows + transfer + pilot)
    write_json(args.output / "completion.json", dict(status="complete" if failed == 0 else "errors",
               errors=failed, seconds=time.perf_counter()-started, config_sha256=sealed_hash,
               selection_jobs=len(rows), transfer_jobs=len(transfer), pilot_jobs=len(pilot)))
    report(f"诊断结束，错误数 {failed}；原配置及 Fig.6 未修改")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
