"""6 方法 fig5/fig6 风格对比图（P0/P0.5 全部估计器，vanilla SAC 轨迹）。

用途：reviewer 要求的替换/稳健性对比——把 6 个 p̂_ag 估计器在
**无课程学习**（vanilla SAC）固定评估轨迹上的效果，按论文 fig5/fig6
的形式并排展示，且所有面板同尺度、同数据密度：

* ``fig5_density_comparison_6methods_{push,slide}.png/pdf``：
  每 task（push/slide）一张图，7 个面板（Reference 经验 KDE +
  KDE / GMM / Histogram / Gaussian / NN / FlowMatching），全部面板
  共享同一 log10 密度色标（同尺度）；成功目标散点统一 resample 到
  **100 点**（同数据密度）。
* ``fig6_kl_curves_6methods.png/pdf``：
  MC-KL(oracle ‖ estimate) 随训练步数，6 条方法均值线 + 95% CI +
  每 seed 淡线；数据来自 replacement 实验 CSV（与 ``summary_table.csv``
  同源）。

seed 选择：各 task 取 500k checkpoint 参考成功数最多的 seed
（push=seed 3，slide=seed 5，可用 ``--pick-seed`` 覆盖）——各 seed 的
目标空间不同，密度网格无法跨 seed 平均（与 fig5 原始协议一致）。

协议与 ``replacement_run_job.py`` 完全一致（AC-2.1/2.2）：同一份历史数据、
同一 κ=0.9 时间权重、同一 hyperparam 默认值。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_gmm_prediction import (
    GOAL_COLUMNS,
    add_records,
    checkpoint_file,
    historical_files,
)

CSV = ROOT / "logs" / "replacement_experiment" / "kde_gmm_hist_gaussian_kl.csv"
OUT = ROOT / "logs" / "replacement_experiment" / "plots"

GRID_RESOLUTION = 80
N_SCATTER = 100  # 成功目标散点统一重采样到 100 点（同数据密度）

METHOD_ORDER = ["kde", "gmm", "histogram", "gaussian", "nn", "flow_matching"]
METHOD_LABELS = {
    "kde": "KDE (κ=0.9)", "gmm": "GMM (k=5)", "histogram": "Histogram (10²)",
    "gaussian": "Gaussian", "nn": "NN (2×16)", "flow_matching": "FlowMatching",
}


def pick_seed(task: str, checkpoint: int, override: int | None) -> int:
    """取参考成功数最多的 seed（密度网格无法跨 seed 平均，与 fig5 原始协议一致）。"""
    if override is not None:
        return override
    best, best_n = None, -1
    for seed in range(1, 6):
        ref = pd.read_csv(checkpoint_file(task, seed, checkpoint))
        n = int((ref["termination"] == "reach target").sum())
        if n > best_n:
            best, best_n = seed, n
    return best


def build_estimators(task: str, seed: int, checkpoint: int, args) -> dict[str, object]:
    """6 个估计器吃同一份历史数据 + 同一权重（AC-2.1/2.2），返回 fitted 字典。"""
    import replacement_run_job as rrj
    estimators = {}
    reference = pd.read_csv(checkpoint_file(task, seed, checkpoint))
    history = [(t, pd.read_csv(p)) for t, p in historical_files(task, seed, checkpoint)]
    for method in METHOD_ORDER:
        est = rrj._make_estimator(
            method, task,
            kappa=args.kappa, n_components=args.n_components,
            resample_size=args.resample_size, bandwidth=args.bandwidth,
            random_state=args.random_state, gmm_reg_covar=args.gmm_reg_covar,
            n_hist_bins=args.n_hist_bins, gaussian_reg_covar=args.gaussian_reg_covar,
            nn_epochs=args.nn_epochs, nn_lr=args.nn_lr, nn_hidden=args.nn_hidden,
            fm_epochs=args.fm_epochs, fm_hidden=args.fm_hidden, fm_samples=args.fm_samples,
        )
        add_records(est, history, checkpoint, args.kappa, goal_columns=GOAL_COLUMNS)
        est.fit_evaluator()
        estimators[method] = est
    return estimators, reference


def raw_grid_density(evaluator, method: str, grid: np.ndarray) -> np.ndarray:
    """统一取各估计器在 raw-space 网格上的 log 密度（与 job 的 fixed_grid_kl 同协议）。"""
    if method in ("gmm", "gaussian", "nn", "flow_matching", "histogram"):
        log_d = evaluator.evaluate_grid(grid, return_log_density=True)
    else:  # kde
        _, log_d = evaluator.evaluate(grid, return_density=False)
        scale = np.asarray(evaluator.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("fitted scaler has non-positive scale")
        log_d -= np.log(scale).sum()
    return log_d


def resample_scatter(points: np.ndarray, n: int, seed: int) -> np.ndarray:
    """把成功目标散点重采样/下采样到固定 n 点（上采样可重复），保证同数据密度。"""
    if len(points) == 0:
        return points
    rng = np.random.default_rng(seed)
    if len(points) >= n:
        idx = rng.choice(len(points), size=n, replace=False)
    else:
        idx = rng.integers(0, len(points), size=n)
    return points[idx]


def reference_success_xy(task: str, seed: int, checkpoint: int = 500000) -> np.ndarray:
    """参考 checkpoint 成功目标坐标（散点用）。"""
    ref = pd.read_csv(checkpoint_file(task, seed, checkpoint))
    return ref.loc[ref["termination"] == "reach target", GOAL_COLUMNS].to_numpy(float)


DENSITY_CSV = ROOT / "logs" / "replacement_experiment" / "plots" / "density_heatmap_6methods.csv"
DENSITY_METHODS = ["Reference", "KDE (κ=0.9)", "GMM (k=5)", "Histogram (10²)",
                   "Gaussian", "NN (2×16)", "FlowMatching"]


def build_figure5(args) -> None:
    """纯绘图：从 density_heatmap_6methods.csv 读密度网格 + KL CSV 画图，
    不重新训练任何估计器（数据由一次性数据生成脚本产出，见文件头注释）。"""
    df = pd.read_csv(DENSITY_CSV)
    for task in args.tasks:
        d = df[df.env == task].copy()
        seed = int(d.seed.iloc[0])
        success = reference_success_xy(task, seed, args.density_checkpoint)
        g0 = d[d.ix == 0].sort_values("iy")
        xs = g0.gx.to_numpy(); ys = g0.gy.to_numpy()
        ix = d.ix.astype(int).to_numpy(); iy = d.iy.astype(int).to_numpy()

        panels = []
        for label in DENSITY_METHODS:
            sub = d[d.method == label]
            arr = np.full((len(ys), len(xs)), np.nan)
            arr[sub.iy.astype(int).to_numpy(), sub.ix.astype(int).to_numpy()] = \
                sub.log_density.to_numpy(float)
            m = np.where(np.isfinite(arr), arr, -np.inf)
            panels.append((label, m))

        # 面板共享色标。Histogram 是盒函数，峰值比连续密度方法高 1~2 个
        # decade（bin 概率/体积），会把色标撑到 +26 使其它面板全部饱和。
        # 因此共享色标上限取"直方图外各面板峰值的 max"，Histogram 面板
        # 自身超限部分饱和（可接受，bin 结构仍清晰可见）。
        continuous_caps = [
            float(np.nanmax(p[1][np.isfinite(p[1])]))
            for p in panels if p[0] != "Histogram (10²)"
        ]
        cap = int(np.ceil(max(continuous_caps)))
        lo10 = cap - 20
        norm = plt.Normalize(lo10, cap)

        n_panels = len(panels)  # 7
        n_cols = 4
        n_rows = 2
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.1 * n_cols, 2.9 * n_rows))
        axes = axes.ravel()
        for ax, (label, arr) in zip(axes, panels):
            ax.imshow(arr, origin="lower", aspect="auto", cmap="viridis",
                      norm=norm,
                      extent=(xs[0], xs[-1], ys[0], ys[-1]))
            scatter = resample_scatter(success, N_SCATTER, seed=seed)
            ax.scatter(scatter[:, 0], scatter[:, 1], s=10, color="w",
                       edgecolors="k", linewidths=0.3, alpha=0.9, marker="o")
            ax.set_title(label, fontsize=9)
            ax.tick_params(labelsize=7)
        for ax in axes[n_panels:]:
            ax.axis("off")
        cax = fig.add_axes([0.90, 0.15, 0.015, 0.7])
        sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
        sm.set_array([])
        fig.colorbar(sm, cax=cax, label=r"log$_{10}$ density")

        fig.suptitle(
            f"Figure 5 ({task}): p$_{{ag}}$ estimates, {args.density_checkpoint:,}-step checkpoint, seed {seed}\n"
            f"vanilla SAC (no curriculum); scatter = {N_SCATTER} resampled successful goals; shared colorbar",
            fontsize=11, y=1.0,
        )
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(OUT / f"fig5_density_comparison_6methods_{task}.{ext}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        print(f"saved fig5 {task} (seed {seed}, {len(success)} successes -> {N_SCATTER} scatter pts)", flush=True)


def build_figure6(args) -> None:
    df = pd.read_csv(CSV)
    df = df.drop_duplicates(subset=["task", "seed", "checkpoint", "method"], keep="last")
    df = df[(df.task.isin(args.tasks)) & (df.status == "ok")]
    if df.empty:
        print("fig6: no ok rows", flush=True)
        return
    fig, axes = plt.subplots(1, len(args.tasks), figsize=(6.5 * len(args.tasks), 4.5), squeeze=False)
    colors = {"kde": "#d62728", "gmm": "#2ca02c", "histogram": "#ff7f0e",
              "gaussian": "#1f77b4", "nn": "#9467bd", "flow_matching": "#8c564b"}
    for ax, task in zip(axes[0], args.tasks):
        d = df[df.task == task]
        for method in METHOD_ORDER:
            m = d[d.method == method]
            if m.empty:
                continue
            for seed, g in m.groupby("seed"):
                g = g.sort_values("checkpoint")
                ax.plot(g.checkpoint, g.kl, color=colors[method], alpha=0.25, linewidth=0.8)
            q025 = m.groupby("checkpoint")["kl"].quantile(0.025)
            q975 = m.groupby("checkpoint")["kl"].quantile(0.975)
            mean = m.groupby("checkpoint")["kl"].mean()
            if int(m.groupby("checkpoint")["kl"].count().max()) >= 2:
                ax.fill_between(q025.index, q025, q975, color=colors[method], alpha=0.15)
            ax.plot(mean.index, mean, color=colors[method], linewidth=2.2,
                    label=METHOD_LABELS[method])
        ax.axhline(0.0, color="0.5", linewidth=0.7, linestyle=":")
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(r"$D_{KL}(p_{ag}\,\|\,\hat p_{ag})$  (MC estimate)")
        ax.set_title(task.capitalize(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle(
        f"Figure 6: estimation error of the 6 goal-distribution estimators on vanilla SAC\n"
        f"(reference = empirical KDE of current-checkpoint successes; "
        f"κ={args.kappa}, bandwidth={args.bandwidth})",
        fontsize=11, y=1.0,
    )
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fig6_kl_curves_6methods.{ext}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"saved fig6 ({len(df)} rows)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["push", "slide"])
    ap.add_argument("--density-checkpoint", type=int, default=500000)
    ap.add_argument("--pick-seed", type=int, default=None,
                    help="覆盖自动选择（默认取成功数最多的 seed）")
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
    ap.add_argument("--random-state", type=int, default=0)
    ap.add_argument("--fig", choices=["all", "fig5", "fig6"], default="all")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.05)
    config = {**vars(args), "goal_columns": GOAL_COLUMNS,
              "data_scope": "vanilla SAC fixed-evaluation CSVs (no curriculum)",
              "scatter_n_points": N_SCATTER,
              "grid_resolution": GRID_RESOLUTION}
    (OUT / "run_config_fig5_fig6.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n")
    if args.fig in ("all", "fig5"):
        build_figure5(args)
    if args.fig in ("all", "fig6"):
        build_figure6(args)
    print(f"all done -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
