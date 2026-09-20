"""Job-level runner for the P0 replacement experiment (plan.md 第 4/8 节)。

一个 job = (task, estimator, seed, checkpoint)。各估计器走同一份历史
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


def _sample_behavioral_goals(
    estimator,
    reference: pd.DataFrame,
    positive_samples: np.ndarray,
    *,
    goal_columns: list[str],
    n_sampled_goals: int,
    candidate_goals: int,
    sampling_method: str,
    sampling_seed: int,
    task: str,
    seed: int,
    checkpoint: int,
) -> list[tuple[float, float]]:
    """复现 MEGA/DISCERN/RIG 的候选目标采样规则。

    训练 wrapper 从环境目标空间均匀产生 ``sample_N`` 个候选目标，再按
    p_ag 密度选择 behavioral goal。离线评估没有保活的 Gym 环境，因此
    使用 fixed-eval CSV 暴露出的目标空间边界生成同样的二维均匀候选点。
    目标空间边界不是从成功/失败频率估计的，只用于恢复环境采样范围。
    """
    if n_sampled_goals <= 0:
        raise ValueError("n_sampled_goals must be positive")
    if candidate_goals <= 0:
        raise ValueError("candidate_goals must be positive")
    if sampling_method not in {"mega", "rig", "discern"}:
        raise ValueError(f"unknown sampling_method: {sampling_method}")
    if len(goal_columns) != 2:
        raise ValueError("behavioral-goal sampling currently supports 2D push/slide goals only")
    if positive_samples is None or len(positive_samples) == 0:
        return [(float("nan"), float("nan")) for _ in range(n_sampled_goals)]

    # fixed-eval 的网格覆盖完整环境目标空间；这里只取边界，不用当前成功
    # 目标的分布来决定候选点，避免把 oracle 分布混入估计分布采样。
    bounds = reference[goal_columns].to_numpy(dtype=float)
    low = np.min(bounds, axis=0)
    high = np.max(bounds, axis=0)
    if not np.all(np.isfinite(low)) or not np.all(np.isfinite(high)) or np.any(high <= low):
        raise ValueError(f"invalid goal-space bounds for {task}/seed{seed}/checkpoint{checkpoint}")

    task_code = {"push": 11, "slide": 17}.get(task, 23)
    rng = np.random.default_rng(np.random.SeedSequence([int(sampling_seed), task_code, int(seed), int(checkpoint)]))
    _, fitted_densities = estimator.evaluate(positive_samples, return_density=True)
    threshold = float(np.min(fitted_densities))
    selected: list[tuple[float, float]] = []

    for _ in range(n_sampled_goals):
        candidates = rng.uniform(low=low, high=high, size=(candidate_goals, 2))
        _, scores = estimator.evaluate(candidates, return_density=True)
        scores = np.asarray(scores, dtype=float)
        valid = np.isfinite(scores) & (scores >= threshold)

        if sampling_method == "mega":
            if np.any(valid):
                masked = np.where(valid, scores, np.inf)
                index = int(np.argmin(masked))
            else:
                # 与 MEGAWrapper._sample_goal_mega 一致：候选全在阈值外时
                # 退化为选择密度最大的候选。
                index = int(np.argmax(np.where(np.isfinite(scores), scores, -np.inf)))
        elif sampling_method == "discern":
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices) == 0:
                index = int(rng.integers(candidate_goals))
            else:
                index = int(rng.choice(valid_indices))
        else:  # rig
            valid_indices = np.flatnonzero(valid)
            if len(valid_indices) == 0:
                index = int(rng.integers(candidate_goals))
            else:
                valid_scores = np.maximum(scores[valid_indices], 0.0)
                total = float(valid_scores.sum())
                if not np.isfinite(total) or total <= 0:
                    index = int(rng.choice(valid_indices))
                else:
                    index = int(rng.choice(valid_indices, p=valid_scores / total))

        selected.append((float(candidates[index, 0]), float(candidates[index, 1])))

    return selected


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


def _historical_arrays(
    frames: list[tuple[int, pd.DataFrame]],
    checkpoint: int,
    kappa: float,
    goal_columns: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构造学习型估计器需要的历史目标、成功标签和绝对时间权重。"""
    goals, successes, weights = [], [], []
    for timestep, frame in frames:
        batch_goals = frame[goal_columns].to_numpy(dtype=float)
        goals.append(batch_goals)
        successes.extend((frame["termination"].to_numpy() == "reach target").tolist())
        weights.extend([float(kappa ** ((checkpoint - timestep) / 10000.0))] * len(frame))
    if not goals:
        raise ValueError("no historical evaluation files precede the selected checkpoint")
    return (
        np.concatenate(goals, axis=0),
        np.asarray(successes, dtype=bool),
        np.asarray(weights, dtype=float),
    )


def _make_estimator(method: str, task: str, *, kappa: float, n_components: int,
                    resample_size: int, bandwidth: float, random_state: int,
                    gmm_reg_covar: float, n_hist_bins: int, gaussian_reg_covar: float,
                    nn_epochs: int = 100, nn_lr: float = 1e-3, nn_hidden: int = 16,
                    nn_early_stopping: bool = True, nn_validation_fraction: float = 0.1,
                    nn_n_iter_no_change: int = 10,
                    fm_epochs: int = 80, fm_hidden: int = 16, fm_samples: int = 2000,
                    nf_epochs: int = 100, nf_lr: float = 1e-3, nf_hidden: int = 32,
                    nf_transforms: int = 4, nf_bins: int = 8, nf_weight_decay: float = 0.0):
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
    if method == "nn":
        from gc_ope.evaluate.evaluator_nn import GoalSuccessMLPClassifierEvaluator
        return GoalSuccessMLPClassifierEvaluator(
            hidden_width=nn_hidden,
            n_epochs=nn_epochs,
            lr=nn_lr,
            random_state=random_state,
            early_stopping=nn_early_stopping,
            validation_fraction=nn_validation_fraction,
            n_iter_no_change=nn_n_iter_no_change,
        )
    if method == "flow_matching":
        raise ValueError("flow_matching 已移除旧实现，等待按新协议重新实现")
    if method == "normalizing_flow":
        from gc_ope.evaluate.evaluator_nf import NormalizingFlowDensityEvaluator
        return NormalizingFlowDensityEvaluator(
            evaluation_result_container_class=WeightedEvaluationResultContainer,
            evaluation_result_container_kwargs=container_kwargs,
            n_epochs=nf_epochs,
            lr=nf_lr,
            weight_decay=nf_weight_decay,
            hidden_features=nf_hidden,
            transforms=nf_transforms,
            bins=nf_bins,
            random_state=random_state,
            device="cpu",
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
    nn_early_stopping: bool = True,
    nn_validation_fraction: float = 0.1,
    nn_n_iter_no_change: int = 10,
    fm_epochs: int = 80,
    fm_hidden: int = 16,
    fm_samples: int = 2000,
    nf_epochs: int = 100,
    nf_lr: float = 1e-3,
    nf_hidden: int = 32,
    nf_transforms: int = 4,
    nf_bins: int = 8,
    nf_weight_decay: float = 0.0,
    mc_samples: int = 100000,
    mc_repeats: int,
    n_sampled_goals: int = 0,
    candidate_goals: int = 100,
    sampling_method: str = "mega",
    sampling_seed: int = 0,
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
        "status": "",
        "error": "",
        "job_time_s": round(time.time() - t0, 2),
        # 供通用单轨迹调度器取出；不会写进指标 CSV。
        "_sampled_goals": None,
    }

    if not history or historical_successes == 0:
        base_row.update(status="skipped:no_historical_successes", error="")
        if n_sampled_goals > 0:
            base_row["_sampled_goals"] = [(float("nan"), float("nan")) for _ in range(n_sampled_goals)]
        return base_row

    try:
        # 各估计器共享同一份历史数据 + 同一时间权重（AC-2.1/2.2）。
        estimator = _make_estimator(
            method, task,
            kappa=kappa, n_components=n_components, resample_size=resample_size,
            bandwidth=bandwidth, random_state=random_state,
            gmm_reg_covar=gmm_reg_covar, n_hist_bins=n_hist_bins,
            gaussian_reg_covar=gaussian_reg_covar,
            nn_epochs=nn_epochs, nn_lr=nn_lr, nn_hidden=nn_hidden,
            nn_early_stopping=nn_early_stopping,
            nn_validation_fraction=nn_validation_fraction,
            nn_n_iter_no_change=nn_n_iter_no_change,
            fm_epochs=fm_epochs, fm_hidden=fm_hidden, fm_samples=fm_samples,
            nf_epochs=nf_epochs, nf_lr=nf_lr, nf_hidden=nf_hidden,
            nf_transforms=nf_transforms, nf_bins=nf_bins,
            nf_weight_decay=nf_weight_decay,
        )
        if method == "nn":
            # NN 是独立的 sklearn 分类器，不使用旧 evaluator container。
            hist_goals, hist_success, weights = _historical_arrays(
                history, checkpoint, kappa, goal_columns
            )
            # 新 NN 协议：历史所有目标做成功率分类，时间折扣通过
            # MLPClassifier.fit(sample_weight=...) 注入；支持点来自当前 checkpoint。
            support_goals = reference[goal_columns].drop_duplicates().to_numpy(dtype=float)
            estimator.fit_classifier(
                hist_goals,
                hist_success.astype(int),
                weights,
                support_goals,
            )
            positive_samples = hist_goals[hist_success]
        else:
            # KDE/GMM/FM 等旧接口继续通过统一的 container 记录历史数据。
            hist_goals, hist_success, weights = add_records(
                estimator, history, checkpoint, kappa, goal_columns=goal_columns
            )
            positive_samples, _, _, _ = estimator.fit_evaluator()

        if n_sampled_goals > 0:
            base_row["_sampled_goals"] = _sample_behavioral_goals(
                estimator,
                reference,
                positive_samples,
                goal_columns=goal_columns,
                n_sampled_goals=n_sampled_goals,
                candidate_goals=candidate_goals,
                sampling_method=sampling_method,
                sampling_seed=sampling_seed,
                task=task,
                seed=seed,
                checkpoint=checkpoint,
            )

        # 固定网格 KL（诊断）
        grid = reference[goal_columns].to_numpy(dtype=float)
        success = reference.loc[reference["termination"] == "reach target", goal_columns].to_numpy(dtype=float)
        if method in ("gmm", "nn", "normalizing_flow"):
            _, gd = estimator.evaluate(grid)
            fixed_grid_kl = discrete_kl(success, grid, gd) if len(success) else float("nan")
        elif method == "kde":
            gd = estimator.evaluate_grid(grid)
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
    ap.add_argument("--methods", nargs="+", default=["kde", "gmm", "nn", "normalizing_flow"])
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
    ap.add_argument("--nn-early-stopping", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--nn-validation-fraction", type=float, default=0.1)
    ap.add_argument("--nn-n-iter-no-change", type=int, default=10)
    ap.add_argument("--fm-epochs", type=int, default=80)
    ap.add_argument("--fm-hidden", type=int, default=16)
    ap.add_argument("--fm-samples", type=int, default=2000)
    ap.add_argument("--nf-epochs", type=int, default=100)
    ap.add_argument("--nf-lr", type=float, default=1e-3)
    ap.add_argument("--nf-hidden", type=int, default=32)
    ap.add_argument("--nf-transforms", type=int, default=4)
    ap.add_argument("--nf-bins", type=int, default=8)
    ap.add_argument("--nf-weight-decay", type=float, default=0.0)
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
                    nn_early_stopping=args.nn_early_stopping,
                    nn_validation_fraction=args.nn_validation_fraction,
                    nn_n_iter_no_change=args.nn_n_iter_no_change,
                    fm_epochs=args.fm_epochs,
                    fm_hidden=args.fm_hidden,
                    fm_samples=args.fm_samples,
                    nf_epochs=args.nf_epochs,
                    nf_lr=args.nf_lr,
                    nf_hidden=args.nf_hidden,
                    nf_transforms=args.nf_transforms,
                    nf_bins=args.nf_bins,
                    nf_weight_decay=args.nf_weight_decay,
                    mc_samples=args.mc_samples,
                    mc_repeats=args.mc_repeats,
                )
                _append_row(row)

    print(f"done. results appended to {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
