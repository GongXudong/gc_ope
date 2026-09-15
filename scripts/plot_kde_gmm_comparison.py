"""Compare KDE and weighted-resampled GMM on vanilla SAC checkpoints.

使用 vanilla SAC 训练轨迹的固定评估 CSV 对比 KDE 与加权重采样 GMM 两个
p̂_ag 估计器。历史成功目标（κ 折扣）拟合估计器；选定 checkpoint 的成功目标
作为参考经验分布（oracle KDE）。

输出到 ``plots/kde_gmm_comparison/``：

* ``kde_gmm_kl.csv``：每个 (env, seed, checkpoint, method) 的 MC-KL（多次平均）
  与固定网格 KL；
* ``density_heatmap.csv``：density checkpoint 处参考/估计密度在 2D 目标网格上的值；
* ``fig5_density_comparison.png/pdf``：2D 密度热图三联（Reference / KDE / GMM）；
* ``fig6_kl_comparison.png/pdf``：MC-KL 随训练步数曲线，每个 seed 一条线 +
  95% 置信带（按 method 聚合）。

图 5/6 的命名沿用论文中对应概念的编号以便对照，但协议细节（如参考分布
用成功目标的 KDE 而非 RB 真实分布）与本脚本的离线对比目的有关，并不
逐字节复现论文图。Push/Slide 比较使用二维 ``x,y`` 目标空间（``z`` 由环境
固定，CSV 中仍保留）。两个估计器吃同一份历史数据、同一套 success 筛选，
只有密度拟合器不同。

已知统计限制：参考成功目标（尤其训练早期）数量少（Push 常 <200），oracle
KDE 的 Monte-Carlo 方差偏大；因此跨 seed 聚合用 95% 置信区间而非
均值±std，MC 采样默认 100k 并取 5 个独立随机种子平均以压低 MC 噪声。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from evaluate_gmm_prediction import add_records, checkpoint_file, historical_files, discrete_kl
from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_gmm import GMMEvaluator
from gc_ope.evaluate.evaluator_kde import KDEEvaluator

ROOT = Path(__file__).resolve().parents[1]
GOAL_COLUMNS = ["x", "y"]
METRIC_COLUMNS = [
    "env", "seed", "checkpoint", "method", "kl", "kl_seed_std", "fixed_grid_kl",
    "historical_successes", "reference_successes", "status", "error",
]
# 2D 密度网格的列（用于热力图）：网格坐标 + 方法 + 密度
GRID_COLS = ["env", "seed", "ix", "iy", "gx", "gy", "method", "density"]
GRID_RESOLUTION = 80


def mesh_grid(columns: list[str], data: np.ndarray, resolution: int = GRID_RESOLUTION) -> tuple[np.ndarray, np.ndarray]:
    """Build a regular 2-D goal grid spanning the min/max of *data*."""
    lo = data.min(axis=0)
    hi = data.max(axis=0)
    xs = np.linspace(lo[0], hi[0], resolution)
    ys = np.linspace(lo[1], hi[1], resolution)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
    return grid, (gx, gy, xs, ys)


def metric_row(
    env: str,
    seed: int,
    checkpoint: int,
    method: str,
    *,
    kl: float,
    kl_seed_std: float,
    fixed_grid_kl: float,
    historical_successes: int,
    reference_successes: int,
    status: str,
    error: str = "",
) -> dict:
    return {
        "env": env,
        "seed": seed,
        "checkpoint": checkpoint,
        "method": method,
        "kl": kl,
        "kl_seed_std": kl_seed_std,
        "fixed_grid_kl": fixed_grid_kl,
        "historical_successes": historical_successes,
        "reference_successes": reference_successes,
        "status": status,
        "error": error,
    }


def append_metrics(path: Path, rows: list[dict]) -> None:
    """Persist each checkpoint immediately for interruption-safe sweeps."""
    if rows:
        pd.DataFrame(rows, columns=METRIC_COLUMNS).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )


def append_grid(path: Path, rows: list[dict]) -> None:
    """Persist 2-D density grid rows as soon as the selected checkpoint is processed."""
    if rows:
        pd.DataFrame(rows, columns=GRID_COLS).to_csv(
            path, mode="a", header=not path.exists(), index=False
        )


def write_run_config(path: Path, args: argparse.Namespace) -> None:
    config = {
        **vars(args),
        "goal_columns": GOAL_COLUMNS,
        "data_scope": "vanilla SAC fixed-evaluation CSVs only",
        "history_rule": "files strictly before selected checkpoint",
        "kl_definition": "MC mean of KL(reference KDE || estimate), 5 random states x mc_samples each",
        "grid_resolution": GRID_RESOLUTION,
    }
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")


def fit_oracle(reference_success: np.ndarray, bandwidth: float) -> KDEEvaluator:
    """Fit the current checkpoint's successful goals as the reference KDE."""
    if reference_success.ndim != 2 or len(reference_success) == 0:
        raise ValueError("the reference checkpoint must contain successful goals")
    oracle = KDEEvaluator(
        evaluation_result_container_class=EvaluationResultContainer,
        kde_bandwidth=bandwidth,
    )
    oracle.eval_res_container.add_batch(
        reference_success,
        [True] * len(reference_success),
        [0.0] * len(reference_success),
        [0.0] * len(reference_success),
    )
    oracle.fit_evaluator()
    return oracle


def reference_density_on_grid(
    reference_success: np.ndarray,
    grid: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    """Empirical reference density: Gaussian-KDE smoothing of the reference
    successes, evaluated on the shared 2-D grid (original goal coordinates)."""
    ref_sc = StandardScaler().fit(reference_success)
    ref_kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
    ref_kde.fit(ref_sc.transform(reference_success))
    log_d = ref_kde.score_samples(ref_sc.transform(grid))
    # StandardScaler 的 Jacobian 是 diag(1/scale)，密度按 1/prod(scale) 换算
    log_d -= np.log(ref_sc.scale_).sum()
    return np.exp(np.clip(log_d, -745.0, 709.0))


def estimate_density_on_grid(evaluator, grid: np.ndarray) -> np.ndarray:
    """Raw-goal-space density of a fitted estimator on the 2-D grid.

    KDEEvaluator.evaluate() 返回标准化空间密度（单位是 1/标准化坐标），
    GMMEvaluator.evaluate_grid() 已内置 Jacobian 校正；这里为两者统一取
    log 密度再用 Jacobian 校正，保证同量纲可直接画在同一色标上。
    """
    if isinstance(evaluator, GMMEvaluator):
        return evaluator.evaluate_grid(grid)
    raw, log_d = evaluator.evaluate(grid, return_density=False)
    scale = np.asarray(evaluator.scaler.scale_, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("拟合的 scaler 存在非正或非有限的 scale")
    log_d -= np.log(scale).sum()
    return np.exp(np.clip(log_d, -745.0, 709.0))


def log_density_in_raw_space(evaluator, raw_goals: np.ndarray) -> np.ndarray:
    """Convert a standardized estimator's log density to raw goal coordinates."""
    _, log_density = evaluator.evaluate(raw_goals, return_density=False)
    scale = np.asarray(evaluator.scaler.scale_, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("the fitted scaler has a non-positive or non-finite scale")
    return np.asarray(log_density) - np.log(scale).sum()


def monte_carlo_kl(
    oracle: KDEEvaluator,
    estimate,
    n_samples: int,
    random_state: int,
) -> float:
    """Estimate KL(oracle || estimate) with samples from the oracle KDE.

    采样前做支撑过滤：oracle KDE 的支撑受孤立成功点支撑（两点距离大于
    带宽时存在支撑间隙），落在支撑外采样的对数密度为 NaN/-inf，
    直接把整条 KL 污染。sklearn KernelDensity 本身对支撑外点返回 NaN，
    因此这里统一用有限掩码过滤。
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    scaled_samples = oracle.kde.sample(n_samples, random_state=random_state)
    raw_samples = oracle.scaler.inverse_transform(scaled_samples)
    oracle_log_density = log_density_in_raw_space(oracle, raw_samples)
    estimate_log_density = log_density_in_raw_space(estimate, raw_samples)
    diff = oracle_log_density - estimate_log_density
    mask = np.isfinite(diff)
    if not mask.all():
        dropped = int((~mask).sum())
        print(f"    MC-KL dropped {dropped}/{n_samples} out-of-support samples", flush=True)
    if mask.sum() < 10:
        return float("nan")
    return float(np.mean(diff[mask]))


def fit_pair(env: str, seed: int, checkpoint: int, *, kappa: float, n_components: int,
              resample_size: int, bandwidth: float, random_state: int,
              mc_samples: int, mc_random_states: list[int], gmm_reg_covar: float,
              want_grid: bool = False):
    reference = pd.read_csv(checkpoint_file(env, seed, checkpoint))
    history = [(t, pd.read_csv(p)) for t, p in historical_files(env, seed, checkpoint)]
    kde = KDEEvaluator(evaluation_result_container_class=WeightedEvaluationResultContainer,
                       evaluation_result_container_kwargs={"discounted_factor": kappa},
                       kde_bandwidth=bandwidth)
    gmm = GMMEvaluator(evaluation_result_container_class=WeightedEvaluationResultContainer,
                       evaluation_result_container_kwargs={"discounted_factor": kappa},
                       n_components=n_components, resample_size=resample_size,
                       random_state=random_state, reg_covar=gmm_reg_covar)
    hist_goals, hist_success, weights = add_records(
        gmm, history, checkpoint, kappa, goal_columns=GOAL_COLUMNS
    )
    kde.eval_res_container.add_batch(hist_goals, hist_success.tolist(), [0.0] * len(hist_goals),
                                     [0.0] * len(hist_goals), [1.0] * len(hist_goals))
    kde.eval_res_container.desired_goal_weights = weights
    gmm.fit_evaluator(); kde.fit_evaluator()
    grid = reference[GOAL_COLUMNS].to_numpy(dtype=float)
    success = reference.loc[reference["termination"] == "reach target", GOAL_COLUMNS].to_numpy(dtype=float)
    _, gd = gmm.evaluate(grid); _, kd = kde.evaluate(grid)
    gkl = discrete_kl(success, grid, gd)
    kkl = discrete_kl(success, grid, kd)
    # 参考分布没有成功目标时，KL 无定义，仍保留密度估计用于诊断。
    if len(success) == 0:
        gmm_kls = kde_kls = []
        gmm_mc_kl = kde_mc_kl = float("nan")
    else:
        oracle = fit_oracle(success, bandwidth)
        gmm_kls = [monte_carlo_kl(oracle, gmm, mc_samples, rs) for rs in mc_random_states]
        kde_kls = [monte_carlo_kl(oracle, kde, mc_samples, rs) for rs in mc_random_states]
        gmm_mc_kl = float(np.nanmean(gmm_kls))
        kde_mc_kl = float(np.nanmean(kde_kls))
    grid_rows = []
    if want_grid:
        grid2d, (gx, gy, _, _) = mesh_grid(GOAL_COLUMNS, grid)
        ref_d = (reference_density_on_grid(success, grid2d, bandwidth)
                 if len(success) else np.zeros(len(grid2d)))
        # 各 seed 的目标空间不同，无法跨 seed 平均密度网格；
        # 图 5 因此只取每个 env 的第一个可用 seed 画热图。
        if ref_d.sum() > 0:
            kde_d = estimate_density_on_grid(kde, grid2d)
            gmm_d = estimate_density_on_grid(gmm, grid2d)
        else:
            kde_d = gmm_d = np.zeros(len(grid2d))
        for method, dens in (("Reference", ref_d), ("KDE", kde_d), ("GMM", gmm_d)):
            for i in range(len(grid2d)):
                grid_rows.append({
                    "env": env, "seed": seed,
                    "ix": int(i % GRID_RESOLUTION), "iy": int(i // GRID_RESOLUTION),
                    "gx": float(grid2d[i, 0]), "gy": float(grid2d[i, 1]),
                    "method": method, "density": float(dens[i]),
                })
    return (
        gkl,
        kkl,
        gmm_mc_kl,
        kde_mc_kl,
        gmm_kls,
        kde_kls,
        grid_rows,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", nargs="+", default=["push", "slide"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    ap.add_argument("--checkpoints", nargs="+", type=int, default=[100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000])
    ap.add_argument("--density-checkpoint", type=int, default=500000)
    ap.add_argument("--output-dir", default="plots/kde_gmm_comparison")
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--resample-size", type=int, default=1000)
    # The paper and all PE-GCRL environment configs use bandwidth 0.2.
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument(
        "--mc-samples", type=int, default=100000,
        help="Monte-Carlo samples per random state for KL estimation",
    )
    ap.add_argument(
        "--mc-repeats", type=int, default=5,
        help="independent random states to average per checkpoint, lowering MC noise",
    )
    ap.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    args = ap.parse_args()
    if args.density_checkpoint not in args.checkpoints:
        ap.error("--density-checkpoint must be one of the values in --checkpoints")
    mc_random_states = [args.mc_repeats + 7 * i for i in range(args.mc_repeats)]
    out = ROOT / args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "kde_gmm_kl.csv"
    grid_path = out / "density_heatmap.csv"
    # Results are ignored experiment artifacts.  Remove stale files so rows
    # from different parameter settings cannot be mixed silently.
    metrics_path.unlink(missing_ok=True)
    grid_path.unlink(missing_ok=True)
    write_run_config(out / "run_config.json", args)
    rows: list[dict] = []
    grid_rows: list[dict] = []
    for env in args.envs:
        for seed in args.seeds:
            for ckpt in args.checkpoints:
                want_grid = (ckpt == args.density_checkpoint)
                print(f"[{env} seed={seed} checkpoint={ckpt}] start", flush=True)
                try:
                    reference = pd.read_csv(checkpoint_file(env, seed, ckpt))
                    history = [(t, pd.read_csv(p)) for t, p in historical_files(env, seed, ckpt)]
                    historical_successes = sum(
                        int((frame["termination"] == "reach target").sum()) for _, frame in history
                    )
                    reference_successes = int((reference["termination"] == "reach target").sum())
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    checkpoint_rows = [
                        metric_row(
                            env, seed, ckpt, method,
                            kl=np.nan, kl_seed_std=np.nan, fixed_grid_kl=np.nan,
                            historical_successes=np.nan, reference_successes=np.nan,
                            status="error", error=message,
                        )
                        for method in ("KDE", "GMM")
                    ]
                    rows.extend(checkpoint_rows)
                    append_metrics(metrics_path, checkpoint_rows)
                    print(f"[{env} seed={seed} checkpoint={ckpt}] error: {message}", flush=True)
                    continue
                if not history or historical_successes == 0:
                    reason = "no_historical_files" if not history else "no_historical_successes"
                    checkpoint_rows = [
                        metric_row(
                            env, seed, ckpt, method,
                            kl=np.nan, kl_seed_std=np.nan, fixed_grid_kl=np.nan,
                            historical_successes=historical_successes,
                            reference_successes=reference_successes,
                            status=f"skipped:{reason}",
                        )
                        for method in ("KDE", "GMM")
                    ]
                    rows.extend(checkpoint_rows)
                    append_metrics(metrics_path, checkpoint_rows)
                    print(f"[{env} seed={seed} checkpoint={ckpt}] {reason}", flush=True)
                    continue
                try:
                    (gkl, kkl, gmc, kmc, g_kls, k_kls, cgrid) = fit_pair(
                        env, seed, ckpt,
                        kappa=args.kappa,
                        n_components=args.n_components,
                        resample_size=args.resample_size,
                        bandwidth=args.bandwidth,
                        random_state=args.random_state,
                        mc_samples=args.mc_samples,
                        mc_random_states=mc_random_states,
                        gmm_reg_covar=args.gmm_reg_covar,
                        want_grid=want_grid,
                    )
                    if want_grid:
                        append_grid(grid_path, cgrid)
                    status = "ok" if np.isfinite(gmc) and np.isfinite(kmc) else "no_reference_success"
                    checkpoint_rows = [
                        metric_row(
                            env, seed, ckpt, "KDE", kl=kmc,
                            kl_seed_std=float(np.nanstd(k_kls)) if k_kls else np.nan,
                            fixed_grid_kl=kkl,
                            historical_successes=historical_successes,
                            reference_successes=reference_successes, status=status,
                        ),
                        metric_row(
                            env, seed, ckpt, "GMM", kl=gmc,
                            kl_seed_std=float(np.nanstd(g_kls)) if g_kls else np.nan,
                            fixed_grid_kl=gkl,
                            historical_successes=historical_successes,
                            reference_successes=reference_successes, status=status,
                        ),
                    ]
                    rows.extend(checkpoint_rows)
                    append_metrics(metrics_path, checkpoint_rows)
                    print(f"[{env} seed={seed} checkpoint={ckpt}] {status}", flush=True)
                except Exception as exc:
                    message = f"{type(exc).__name__}: {exc}"
                    checkpoint_rows = [
                        metric_row(
                            env, seed, ckpt, method,
                            kl=np.nan, kl_seed_std=np.nan, fixed_grid_kl=np.nan,
                            historical_successes=historical_successes,
                            reference_successes=reference_successes,
                            status="error", error=message,
                        )
                        for method in ("KDE", "GMM")
                    ]
                    print(f"[{env} seed={seed} checkpoint={ckpt}] error: {message}", flush=True)
                rows.extend(checkpoint_rows)
                append_metrics(metrics_path, checkpoint_rows)
    # 循环内 append_metrics 已增量写盘；结束时读回以统一 schema 供绘图。
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[METRIC_COLUMNS]
    else:
        metrics = pd.DataFrame(rows, columns=METRIC_COLUMNS)
    if grid_path.exists():
        grid_df = pd.read_csv(grid_path)
    else:
        grid_df = pd.DataFrame(grid_rows, columns=GRID_COLS)
    _plot_figures(out, metrics, grid_df, args)
    print(f"saved {metrics_path}", flush=True)


def _plot_figures(out: Path, metrics: pd.DataFrame, grid_df: pd.DataFrame, args) -> None:
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    valid = metrics[metrics["status"] == "ok"]

    # ---- Fig. 5: 2-D density heatmaps -----------------------------------
    if not grid_df.empty:
        fig, axes = plt.subplots(1, len(args.envs), figsize=(7 * len(args.envs), 4), squeeze=False)
        for ax, env in zip(axes[0], args.envs):
            d = grid_df[grid_df.env == env]
            if d.empty:
                ax.text(0.5, 0.5, "No successful reference episodes", ha="center", va="center")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue
            # 各 seed 的目标空间可能不同，密度网格无法跨 seed 平均；
            # 取该 env 中第一个 Reference 有成功目标的 seed 画热图
            avail = d[d.method == "Reference"]
            avail = avail.groupby("seed").density.sum().sort_values(ascending=False).index
            if avail.empty:
                ax.text(0.5, 0.5, "No successful reference episodes", ha="center", va="center")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue
            pick = int(avail.iloc[0])
            d = d[d.seed == pick]
            ref_m = d[d.method == "Reference"].pivot(index="iy", columns="ix", values="density")
            kde_m = d[d.method == "KDE"].pivot(index="iy", columns="ix", values="density")
            gmm_m = d[d.method == "GMM"].pivot(index="iy", columns="ix", values="density")
            ys = d[d.ix == 0].sort_values("iy").gy.to_numpy()
            xs = d[d.iy == 0].sort_values("ix").gx.to_numpy()
            def _log10p(m: pd.DataFrame) -> np.ndarray:
                v = m.to_numpy(float)
                return np.where(v > 0, np.log10(v), np.nan)
            ref_l, kde_l, gmm_l = _log10p(ref_m), _log10p(kde_m), _log10p(gmm_m)
            vmax = float(np.nanmax([np.nanmax(ref_l), np.nanmax(kde_l), np.nanmax(gmm_l), 0.0]))
            for label, arr, lo in (("Reference (empirical KDE)", ref_l, np.nanmin(np.where(np.isfinite(ref_l), ref_l, np.inf))),
                                    (f"KDE (κ={args.kappa})", kde_l, np.nanmin(np.where(np.isfinite(kde_l), kde_l, np.inf))),
                                    (f"GMM (k={args.n_components})", gmm_l, np.nanmin(np.where(np.isfinite(gmm_l), gmm_l, np.inf)))):
                im = ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis",
                               vmin=lo, vmax=vmax, extent=(xs[0], xs[-1], ys[0], ys[-1]))
                ax.set_title(f"{label}\nseed {pick}", fontsize=10)
                cbar = fig.colorbar(im, ax=ax, shrink=0.85)
                cbar.set_label(r"log$_{10}$ density")
            # 该 seed 的成功目标散点
            ref_pts = d[d.method == "Reference"]
            ref_pts = ref_pts[ref_pts.density > 0]
            ax.scatter(ref_pts.gx, ref_pts.gy, s=2, color="w", marker="o", alpha=0.6)
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.suptitle(f"Figure 5: p$_{{ag}}$ estimate vs. empirical reference on {args.density_checkpoint:,}-step checkpoint", y=1.02, fontsize=12)
        fig.tight_layout()
        fig.savefig(out / "fig5_density_comparison.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / "fig5_density_comparison.pdf", bbox_inches="tight")
        plt.close(fig)

    # ---- Fig. 6: per-seed KL curves + 95% CI ----------------------------
    if valid.empty:
        print("fig6: no ok rows, skipping", flush=True)
    else:
        fig, axes = plt.subplots(1, len(args.envs), figsize=(7 * len(args.envs), 4), squeeze=False)
        method_colors = {"KDE": "#d62728", "GMM": "#2ca02c"}
        for ax, env in zip(axes[0], args.envs):
            v = valid[valid.env == env]
            if v.empty:
                ax.text(0.5, 0.5, "No valid KL measurements", ha="center", va="center")
                ax.set_xlim(0, 1); ax.set_ylim(0, 1)
                continue
            for method, color in method_colors.items():
                m = v[v.method == method]
                for seed, g in m.groupby("seed"):
                    g = g.drop_duplicates().sort_values("checkpoint")
                    ax.plot(g.checkpoint.to_numpy(dtype=float), g.kl.to_numpy(dtype=float),
                            color=color, alpha=0.28, linewidth=0.9,
                            label=f"{method} seed {seed}")
                # 跨 seed 的 95% CI：n_seeds>=2 用 2.5/97.5 分位，否则退化为均值线
                ci = m.groupby("checkpoint", as_index=False)["kl"].agg(
                    mean="mean", n="count"
                ).sort_values("checkpoint")
                n_seeds = int(ci["n"].max())
                if n_seeds >= 2:
                    q025 = m.groupby("checkpoint")["kl"].quantile(0.025).reindex(ci.checkpoint).to_numpy(dtype=float)
                    q975 = m.groupby("checkpoint")["kl"].quantile(0.975).reindex(ci.checkpoint).to_numpy(dtype=float)
                    ax.fill_between(ci.checkpoint.to_numpy(dtype=float), q025, q975,
                                    color=color, alpha=0.18, label=f"{method} 95% CI")
                ax.plot(ci.checkpoint.to_numpy(dtype=float), ci["mean"].to_numpy(dtype=float),
                        color=color, linewidth=2.2, label=f"{method} mean")
            ax.set_xlabel("Environment steps")
            ax.set_ylabel(r"$D_{KL}(p_{ag}\,||\,\tilde p_{ag})$")
            ax.set_title(f"{env.capitalize()}", fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7, loc="upper right", ncol=2)
        fig.suptitle(
            f"Figure 6: estimation error over training  (KDE bandwidth={args.bandwidth}, "
            f"GMM k={args.n_components}, MC n={args.mc_samples}×{args.mc_repeats})",
            fontsize=12,
        )
        fig.tight_layout()
        fig.savefig(out / "fig6_kl_comparison.png", dpi=180, bbox_inches="tight")
        fig.savefig(out / "fig6_kl_comparison.pdf", bbox_inches="tight")
        plt.close(fig)

    print(valid.groupby(["env", "checkpoint", "method"], as_index=False)
          .agg(mean=("kl", "mean"), std=("kl", "std")).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
