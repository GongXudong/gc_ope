"""绘制 KDE + GMM + NN（可选 FM/NF）的论文 fig6 风格 KL 曲线。

默认绘图约定：slide NN 使用 fixed_grid_kl，push NN 和 GMM 使用 kl；
零刻度下保留约 5% 显示空间留白，不修改或平移数据值。
可用 ``--linear-ymax 0.5 --no-full-log`` 另出线性 0.5 上限版本，
上方额外保留少量空白，不生成对数图。

可用 ``--include-fm`` / ``--include-nf`` 从对应单方法 CSV 加入 FM/NF，使用 kl 列。
可用 ``--kde-source per-seed`` 从同一 method_per_seed 目录读取 KDE。
可用 ``--fm-dir <目录>`` 单独指定更新后的 FM 数据目录，其余方法来源不变。
可用 ``--method-dir <目录>`` 指定本轮所有方法的 CSV 目录；FM 单独指定时优先使用 --fm-dir。
可用 ``--smooth-window 3`` 对每个 seed 分别做居中移动平均后计算均值和 CI；
端点使用可用样本，原始数据另存，图中注明窗口，默认不平滑。
``--smooth-sigma 2`` 沿用 rebuttal 图的逐 seed 高斯平滑，不能与移动平均叠加。
``--style paper`` 使用论文 Fig.6 的 Seaborn deep 配色、自动横轴留白和无标题排版；
Fig.6 原 notebook 未平滑。若同时开启平滑，方法说明保存为独立图注。

数据口径：
- GMM/NN：``logs/replacement_experiment/all100/method_per_seed/`` 下的
  单方法、单轨迹 CSV，使用 ``status == ok`` 的 ``kl`` 列；
- KDE：默认使用 ``logs/replacement_experiment/all100/kde_gmm_hist_gaussian_kl.csv``
  中按同一 replacement runner 计算的 KDE（``--kde-source aligned``），这与当前
  GMM/NN 的 MC-KL 协议一致；
- ``--kde-source legacy`` 可读取上一轮投稿保存的 KDE 文件，取其中的
  ``$D_{KL}$ [his]``（历史评估/PE）列。该数据使用旧的历史数据抽样和 MC 协议，
  与当前 GMM/NN 不完全同口径，脚本会在终端和 run_config 中明确警告。

原论文绘图设置参考：
``plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/sac/sac.ipynb``
和 ``plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/plot_omega.ipynb``：
``sns.set_theme(context="notebook", style="darkgrid", font_scale=2.0)``、
``figsize=(10, 8)``、``sns.lineplot(errorbar=("ci", 95))``、右上角图例。

示例：
  conda run -n gc_ope python scripts/plot_fig6_kde_gmm_nn.py --task push

  # 读取上一轮投稿的 KDE（仅建议作为 legacy 对照图）
  conda run -n gc_ope python scripts/plot_fig6_kde_gmm_nn.py \\
    --task push --kde-source legacy --allow-protocol-mismatch
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
METHOD_DIR = ROOT / "logs" / "replacement_experiment" / "all100" / "method_per_seed"
ALIGNED_KDE_CSV = ROOT / "logs" / "replacement_experiment" / "all100" / "kde_gmm_hist_gaussian_kl.csv"
LEGACY_KDE_ROOT = ROOT / "plots" / "p_ag_dist_between_truth_and_estimated_in_training"
DEFAULT_OUT_DIR = ROOT / "logs" / "replacement_experiment" / "plots_fig6_kde_gmm_nn"
BASE_METHOD_ORDER = ["KDE", "GMM", "NN"]
METHOD_COLORS = {
    "KDE": "#1f77b4",
    "GMM": "#2ca02c",
    "NN": "#9467bd",
    "FM": "#ff7f0e",
    "NF": "#d62728",
}
PAPER_NOTEBOOKS = {
    "vanilla_sac_panel": "plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/sac/sac.ipynb",
    "paper_fig6": "plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/plot_omega.ipynb",
}
# 绘图默认约定：slide NN 使用 fixed_grid_kl，其余保持 kl；零线下保留留白。
ZERO_PADDING = 0.05

METRIC_COLUMNS = ["task", "seed", "checkpoint", "method", "kl", "source"]


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _legacy_kde_path(task: str, seed: int) -> Path:
    return (
        LEGACY_KDE_ROOT
        / f"my_{task}"
        / "sac"
        / "eval_data"
        / f"my_{task}_sac_seed_{seed}_kde_0_9_eval_res_in_training.csv"
    )


def _load_current_method(method: str, task: str, seeds: list[int], metric: str = "kl",
                         method_dir: Path | None = None) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    coverage = []
    for seed in seeds:
        path = (METHOD_DIR if method_dir is None else method_dir) / f"{method}_{task}_seed{seed}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"找不到 {method} 的 seed {seed} 文件: {path}")
        df = pd.read_csv(path)
        required = {"task", "seed", "checkpoint", "method", metric, "status"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} 缺少列: {sorted(missing)}")
        if not ((df["task"] == task) & (df["seed"] == seed) & (df["method"] == method)).all():
            raise ValueError(f"{path} 存在轨迹/方法不匹配的记录")
        if df.duplicated(["checkpoint"]).any():
            raise ValueError(f"{path} 存在重复 checkpoint，请先核实重跑结果")
        coverage.append({"source": _display_path(path), "seed": seed,
                         "status_counts": df.status.value_counts().to_dict(),
                         "excluded": df.loc[df.status != "ok", ["checkpoint", "status", "error"]].fillna("").to_dict("records")})
        df = df[df["status"] == "ok"].copy()
        # 仅将选定列映射到绘图用 kl，不改写源结果。
        df["kl"] = pd.to_numeric(df[metric], errors="raise")
        if not df["kl"].map(lambda x: float("-inf") < x < float("inf")).all():
            raise ValueError(f"{path} 的 {metric} 含非有限值")
        display_names = {"normalizing_flow": "NF", "flow_matching": "FM"}
        df["method"] = display_names.get(method, method.upper())
        df["source"] = _display_path(path)
        frames.append(df[["task", "seed", "checkpoint", "method", "kl", "source"]])
    result = pd.concat(frames, ignore_index=True)
    result = _deduplicate(result, f"{method}/{task}")
    result.attrs["coverage"] = coverage
    return result


def _load_aligned_kde(task: str, seeds: list[int]) -> pd.DataFrame:
    if not ALIGNED_KDE_CSV.is_file():
        raise FileNotFoundError(f"找不到同协议 KDE CSV: {ALIGNED_KDE_CSV}")
    df = pd.read_csv(ALIGNED_KDE_CSV)
    required = {"task", "seed", "checkpoint", "method", "kl", "status"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ALIGNED_KDE_CSV} 缺少列: {sorted(missing)}")
    df = df[
        (df["task"] == task)
        & (df["seed"].isin(seeds))
        & (df["method"] == "kde")
        & (df["status"] == "ok")
    ].copy()
    df["method"] = "KDE"
    df["source"] = _display_path(ALIGNED_KDE_CSV)
    return _deduplicate(df[["task", "seed", "checkpoint", "method", "kl", "source"]], "aligned KDE")


def _load_legacy_kde(task: str, seeds: list[int]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    missing: list[Path] = []
    for seed in seeds:
        path = _legacy_kde_path(task, seed)
        if not path.is_file():
            missing.append(path)
            continue
        df = pd.read_csv(path)
        required = {"evaluation_index", "$D_{KL}$ [his]"}
        missing_columns = required - set(df.columns)
        if missing_columns:
            raise ValueError(f"{path} 缺少列: {sorted(missing_columns)}")
        out = pd.DataFrame({
            "task": task,
            "seed": seed,
            "checkpoint": df["evaluation_index"].astype(int),
            "method": "KDE",
            "kl": pd.to_numeric(df["$D_{KL}$ [his]"], errors="coerce"),
            "source": _display_path(path),
        })
        frames.append(out)
    if missing:
        raise FileNotFoundError("缺少 legacy KDE 文件:\n" + "\n".join(map(str, missing)))
    result = pd.concat(frames, ignore_index=True)
    result = result.dropna(subset=["kl"])
    return _deduplicate(result, "legacy KDE")


def _deduplicate(df: pd.DataFrame, label: str) -> pd.DataFrame:
    keys = ["task", "seed", "checkpoint", "method"]
    duplicates = int(df.duplicated(keys).sum())
    if duplicates:
        print(f"警告：{label} 有 {duplicates} 个重复键，保留最后一行", file=sys.stderr)
        df = df.drop_duplicates(keys, keep="last")
    return df.sort_values(["method", "seed", "checkpoint"]).reset_index(drop=True)


def resolve_nn_metric(task: str, requested: str = "auto") -> str:
    """默认 slide NN 用离散网格 KL；显式参数仅供复现历史对照图。"""
    return ("fixed_grid_kl" if task == "slide" else "kl") if requested == "auto" else requested


def pad_below_zero(ax, values, upper: float | None = None) -> list[float]:
    """在坐标变换后的空间保留约 5% 下边距，不平移、截零或修改数据。

    论文 notebook 未强制 bottom=0，而是保留自动边距。这里显式复现留白，
    symlog 也按显示空间处理，避免用巨大 ymax 直接计算负下界。
    真实负值仍完整保留在坐标范围内，不假定它们只是数值误差。
    """
    upper = ax.get_ylim()[1] if upper is None else float(upper)
    lower = min(0.0, float(np.min(values)))
    if not np.isfinite(upper) or upper <= 0:
        raise ValueError("纵轴上界必须是正有限数")
    transform = ax.yaxis.get_transform()
    lo, hi = transform.transform(np.array([lower, upper]))
    padded = float(transform.inverted().transform(np.array([lo - ZERO_PADDING * (hi - lo)]))[0])
    ax.set_ylim(padded, upper)
    return [padded, upper]


def load_data(
    task: str,
    seeds: list[int],
    kde_source: str,
    nn_metric: str = "auto",
    include_nf: bool = False,
    include_fm: bool = False,
    fm_dir: Path | None = None,
    method_dir: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    method_dir = METHOD_DIR if method_dir is None else method_dir
    fm_dir = method_dir if fm_dir is None else fm_dir
    nn_metric = resolve_nn_metric(task, nn_metric)
    method_order = list(BASE_METHOD_ORDER)
    if include_fm:
        method_order.append("FM")
    if include_nf:
        method_order.append("NF")
    if kde_source == "per-seed":
        kde = _load_current_method("kde", task, seeds, method_dir=method_dir)
    elif kde_source == "aligned":
        kde = _load_aligned_kde(task, seeds)
    elif kde_source == "legacy":
        kde = _load_legacy_kde(task, seeds)
    else:
        raise ValueError(f"未知 KDE 数据来源: {kde_source}")
    gmm = _load_current_method("gmm", task, seeds, method_dir=method_dir)
    nn = _load_current_method("nn", task, seeds, metric=nn_metric, method_dir=method_dir)
    frames = [kde, gmm, nn]
    kde["source_metric"] = "$D_{KL}$ [his]" if kde_source == "legacy" else "kl"
    gmm["source_metric"] = "kl"
    nn["source_metric"] = nn_metric
    if include_fm:
        fm = _load_current_method("flow_matching", task, seeds, metric="kl", method_dir=fm_dir)
        fm["source_metric"] = "kl"
        frames.append(fm)
    if include_nf:
        nf = _load_current_method("normalizing_flow", task, seeds, metric="kl", method_dir=method_dir)
        nf["source_metric"] = "kl"
        frames.append(nf)
    coverage = {str(frame["method"].iloc[0]): frame.attrs.get("coverage", []) for frame in frames if len(frame)}
    df = pd.concat(frames, ignore_index=True)
    df["method"] = pd.Categorical(df["method"], categories=method_order, ordered=True)
    df = df.sort_values(["method", "seed", "checkpoint"]).reset_index(drop=True)

    counts = (
        df.groupby(["method", "seed"], observed=True)["checkpoint"]
        .nunique()
        .unstack(fill_value=0)
    )
    print(f"loaded task={task}, KDE source={kde_source}, rows={len(df)}")
    print("checkpoint counts by method/seed:")
    print(counts.to_string())

    config = {
        "task": task,
        "seeds": seeds,
        "methods": method_order,
        "metric": "kl" if nn_metric == "kl" else "mixed",
        "metric_by_method": {
            "KDE": "$D_{KL}$ [his]" if kde_source == "legacy" else "kl",
            "GMM": "kl",
            "NN": nn_metric,
            **({"FM": "kl"} if include_fm else {}),
            **({"NF": "kl"} if include_nf else {}),
        },
        "mixed_metric_warning": "NN 使用离散 fixed_grid_kl；KDE/GMM 保留原指标，不是同口径比较。" if nn_metric != "kl" else None,
        "kde_source": kde_source,
        "kde_series": "$D_{KL}$ [his]" if kde_source == "legacy" else "kl",
        "kde_protocol_comparable_to_current_gmm_nn": kde_source in {"aligned", "per-seed"} and nn_metric == "kl",
        "include_fm": include_fm,
        "include_nf": include_nf,
        "input_coverage": coverage,
        "aggregation": "每个 checkpoint 使用有效 seed 均值和 bootstrap 95% CI；error/skipped 不补值",
        "current_method_dir": _display_path(method_dir),
        "fm_method_dir": _display_path(fm_dir) if include_fm else None,
        "aligned_kde_csv": _display_path(ALIGNED_KDE_CSV),
        "legacy_kde_root": _display_path(LEGACY_KDE_ROOT),
        "paper_notebooks": PAPER_NOTEBOOKS,
        "rows_by_method": {method: int((df["method"] == method).sum()) for method in method_order},
        "checkpoints_by_method": {
            method: int(df.loc[df["method"] == method, "checkpoint"].nunique())
            for method in method_order
        },
    }
    return df, config


def smooth_per_seed(df: pd.DataFrame, window: int = 1, sigma: float | None = None) -> pd.DataFrame:
    """等间隔轨迹内局部平均；不跨 seed、不补缺失点、不截断负值。"""
    if window < 1 or window % 2 == 0:
        raise ValueError("平滑窗口必须为正奇数")
    if sigma is not None and (not np.isfinite(sigma) or sigma <= 0 or window != 1):
        raise ValueError("高斯 sigma 必须为正有限数，且不能与移动平均叠加")
    result = df.sort_values(["method", "seed", "checkpoint"]).copy()
    result["raw_kl"] = result["kl"]
    if window == 1 and sigma is None:
        return result
    steps = np.sort(result["checkpoint"].unique())
    if len(steps) < 2:
        raise ValueError("检查点不足，无法确定平滑间隔")
    interval = np.diff(steps).min()
    for _, group in result.groupby(["method", "seed"], observed=True):
        if len(group) < window or not np.all(np.diff(group["checkpoint"]) == interval):
            raise ValueError("平滑要求每条轨迹有足够且连续等间隔的检查点，不能跨越缺失点")
        if sigma is not None:
            # 与原 rebuttal notebook 完全相同：reflect 边界，truncate=4。
            result.loc[group.index, "kl"] = gaussian_filter1d(
                group["kl"].to_numpy(dtype=float), sigma=sigma, mode="reflect", truncate=4.0,
            )
        else:
            result.loc[group.index, "kl"] = group["kl"].rolling(
                window, center=True, min_periods=1,
            ).mean()
    return result


def _plot(
    df: pd.DataFrame,
    task: str,
    out_dir: Path,
    *,
    method_order: list[str],
    zoom_from: int,
    zoom_ymax: float | None,
    linear_ymax: float | None = None,
    top_padding_fraction: float = 0.05,
    make_full_log: bool = True,
    smooth_window: int = 1,
    smooth_sigma: float | None = None,
    style: str = "current",
    legacy_kde: bool = False,
) -> dict:
    smoothing_label = (
        f"Gaussian smoothing (sigma={smooth_sigma:g} checkpoints); 95% CI across seeds"
        if smooth_sigma is not None else
        f"{smooth_window}-checkpoint centered mean; 95% CI across seeds"
        if smooth_window > 1 else None
    )
    means = df.groupby(["method", "checkpoint"], observed=True)["kl"].mean()
    method_tag = "-".join(method.lower() for method in method_order)
    late = means[means.index.get_level_values("checkpoint") >= zoom_from]
    if late.empty:
        raise ValueError(f"--zoom-from={zoom_from} 之后没有数据")
    if linear_ymax is not None:
        if not np.isfinite(linear_ymax) or linear_ymax <= 0:
            raise ValueError("--linear-ymax 必须是正有限数")
        if not np.isfinite(top_padding_fraction) or top_padding_fraction < 0:
            raise ValueError("--top-padding-fraction 必须是非负有限数")
        ymax = float(linear_ymax)
        zoom_source = "explicit linear ymax"
    else:
        auto_ymax = float(max(1.0, int(late.max()) + (1 if late.max() > 0 else 0)))
        ymax = auto_ymax if zoom_ymax is None else float(zoom_ymax)
        zoom_source = "explicit zoom ymax" if zoom_ymax is not None else "auto"
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("绘图纵轴上限必须是正有限数")

    off_scale = [
        {"method": method, "checkpoint": int(checkpoint), "mean_kl": float(value)}
        for (method, checkpoint), value in means.items()
        if value > ymax
    ]

    plot_df = df.copy()
    plot_df["Estimation Method"] = plot_df["method"].astype(str)
    sns.set_theme(context="notebook", style="darkgrid", font_scale=2.0)
    # 原论文默认使用 Seaborn deep 调色板；保留既有方法的色相对应关系。
    deep = sns.color_palette("deep")
    palette = ({"KDE": deep[0], "GMM": deep[2], "NN": deep[4],
                "FM": deep[1], "NF": deep[3]} if style == "paper" else METHOD_COLORS)
    display_order = list(method_order)
    if legacy_kde:
        plot_df["Estimation Method"] = plot_df["Estimation Method"].replace({"KDE": "KDE (legacy)"})
        display_order = ["KDE (legacy)" if m == "KDE" else m for m in method_order]
    display_palette = {("KDE (legacy)" if legacy_kde and m == "KDE" else m): palette[m] for m in method_order}
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        data=plot_df,
        x="checkpoint",
        y="kl",
        hue="Estimation Method",
        hue_order=display_order,
        palette=display_palette,
        linewidth=1.5,
        err_kws={"alpha": 0.2},
        errorbar=("ci", 95),
        n_boot=1000,
        seed=0,
        ax=ax,
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(r"$D_{KL}(P_{ag} \| \tilde{p}_{ag})$")
    if style == "paper":
        ax.margins(x=0.05)
    else:
        ax.set_xlim(left=0, right=1_000_000)
    # 固定上限版本把 0.5 视为数据显示上限，再在其上方留一点空白。
    axis_top = ymax * (1.0 + top_padding_fraction) if linear_ymax is not None else ymax
    ax.set_ylim(top=axis_top)
    linear_limits = pad_below_zero(ax, df["kl"])
    sns.move_legend(ax, loc="upper right", fontsize=18)
    if smoothing_label and style != "paper":
        ax.set_title(smoothing_label, fontsize=12, pad=12)
    fig.tight_layout()

    outputs: list[str] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig6_kl_curves_{method_tag}_{task}.{ext}"
        fig.savefig(path, format=ext, bbox_inches="tight", dpi=180)
        outputs.append(_display_path(path))
    plt.close(fig)

    full_outputs: list[str] = []
    full_limits: list[float] | None = None
    if make_full_log:
        # 不裁切的伴随图：当前 GMM 早期可能有数量级异常值，保留全部数据。
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.lineplot(
            data=plot_df,
            x="checkpoint",
            y="kl",
            hue="Estimation Method",
            hue_order=display_order,
            palette=display_palette,
            linewidth=1.5,
            err_kws={"alpha": 0.2},
            errorbar=("ci", 95),
            n_boot=1000,
            seed=0,
            ax=ax,
        )
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(r"$D_{KL}(P_{ag} \| \tilde{p}_{ag})$")
        if style == "paper":
            ax.margins(x=0.05)
        else:
            ax.set_xlim(left=0, right=1_000_000)
        ax.set_yscale("symlog", linthresh=1e-3)
        full_limits = pad_below_zero(ax, df["kl"])
        sns.move_legend(ax, loc="upper right", fontsize=18)
        if smoothing_label and style != "paper":
            ax.set_title(smoothing_label, fontsize=12, pad=12)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            path = out_dir / f"fig6_kl_curves_{method_tag}_{task}_fulllog.{ext}"
            fig.savefig(path, format=ext, bbox_inches="tight", dpi=180)
            full_outputs.append(_display_path(path))
        plt.close(fig)

    return {
        "outputs": outputs,
        "full_range_outputs": full_outputs,
        "methods": method_order,
        "method_tag": method_tag,
        "zoom_from": zoom_from,
        "zoom_ymax": ymax,
        "zoom_ymax_source": zoom_source,
        "requested_linear_ymax": linear_ymax,
        "linear_top_padding_fraction": top_padding_fraction if linear_ymax is not None else 0.0,
        "linear_axis_top": linear_limits[1],
        "off_scale_means": sorted(off_scale, key=lambda row: -row["mean_kl"]),
        "method_summary": df.groupby("method", observed=True)["kl"].agg(["mean", "std", "min", "max"]).to_dict(orient="index"),
        "plot_settings": {
            "style": style,
            "legend_labels": display_order,
            "palette": {method: matplotlib.colors.to_hex(palette[method]) for method in method_order},
            "linewidth": 1.5,
            "ci_alpha": 0.2,
            "smoothing_label_in_title": bool(smoothing_label and style != "paper"),
            "sns_theme": "context=notebook, style=darkgrid, font_scale=2.0",
            "figsize": [10, 8],
            "errorbar": ["ci", 95],
            "n_boot": 1000,
            "ci_seed": 0,
            "legend": "upper right, fontsize=18",
            "zero_padding_fraction": ZERO_PADDING,
            "padding_space": "axis transformed coordinates",
            "linear_ylim": linear_limits,
            "symlog_ylim": full_limits,
            "full_log_generated": make_full_log,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nn-metric", choices=["auto", "kl", "fixed_grid_kl"], default="auto", help="默认 slide 使用 fixed_grid_kl，push 使用 kl；可显式覆盖以复现历史图")
    parser.add_argument("--task", choices=["push", "slide"], default="push")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--kde-source", choices=["aligned", "legacy", "per-seed"], default="aligned", help="per-seed 从 method_per_seed 目录读取 KDE；aligned 读取汇总 CSV；legacy 读取旧投稿数据")
    parser.add_argument("--allow-protocol-mismatch", action="store_true", help="允许使用旧 KDE 与当前 GMM/NN 混合绘图")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--zoom-from", type=int, default=100000)
    parser.add_argument("--zoom-ymax", type=float, default=None)
    parser.add_argument("--linear-ymax", type=float, default=None, help="线性图的数据上限；例如 0.5。上方按 --top-padding-fraction 留白")
    parser.add_argument("--top-padding-fraction", type=float, default=0.05, help="固定线性上限时的顶部留白比例")
    parser.add_argument("--no-full-log", action="store_true", help="不生成 symlog full-log 伴随图")
    parser.add_argument("--include-fm", action="store_true", help="加入 flow_matching 方法（显示名 FM）")
    parser.add_argument("--fm-dir", type=Path, default=None, help="单独指定 FM 的 method_per_seed 目录，需同时启用 --include-fm")
    parser.add_argument("--method-dir", type=Path, default=METHOD_DIR, help="本轮所有方法的 method_per_seed 目录；可用 --fm-dir 单独覆盖 FM")
    parser.add_argument("--include-nf", action="store_true", help="加入 normalizing_flow 方法（显示名 NF）")
    parser.add_argument("--smooth-window", type=int, default=1, help="每个 seed 的居中移动平均窗口（正奇数）；默认 1 不平滑")
    parser.add_argument("--smooth-sigma", type=float, default=None, help="逐 seed 高斯平滑的 sigma，单位是检查点；rebuttal 图使用 2")
    parser.add_argument("--style", choices=["current", "paper"], default="current", help="paper 对齐论文 Fig.6 的配色、横轴留白与无标题排版")
    args = parser.parse_args()

    if args.fm_dir is not None and not args.include_fm:
        parser.error("--fm-dir 需要同时启用 --include-fm")
    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds 不能重复")
    if args.smooth_window < 1 or args.smooth_window % 2 == 0:
        parser.error("--smooth-window 必须为正奇数")
    if args.smooth_sigma is not None and (
        not np.isfinite(args.smooth_sigma) or args.smooth_sigma <= 0 or args.smooth_window != 1
    ):
        parser.error("--smooth-sigma 必须为正有限数，且不能与 --smooth-window > 1 叠加")
    if args.kde_source == "legacy" and not args.allow_protocol_mismatch:
        parser.error(
            "legacy KDE 与当前 GMM/NN 的历史数据抽样、MC-KL 口径不完全一致；"
            "如需生成对照图，请显式加 --allow-protocol-mismatch"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, config = load_data(
        args.task, args.seeds, args.kde_source, args.nn_metric,
        include_nf=args.include_nf, include_fm=args.include_fm,
        fm_dir=args.fm_dir,
        method_dir=args.method_dir,
    )
    method_order = config["methods"]
    plot_df = smooth_per_seed(df, args.smooth_window, args.smooth_sigma)
    config["smoothing"] = {
        "window": args.smooth_window,
        "sigma": args.smooth_sigma,
        "method": ("逐 method/seed 的 gaussian_filter1d；mode=reflect，truncate=4"
                   if args.smooth_sigma is not None else
                   "每个 method/seed 分别做居中等权移动平均；端点使用可用样本"
                   if args.smooth_window > 1 else "不平滑"),
        "uncertainty": "对平滑后的 seed 数值进行 bootstrap，得到局部平均曲线的逐点 95% CI；不是原始逐点 KL 的 CI",
        "raw_data_preserved": True,
    }
    plot_config = _plot(
        plot_df,
        args.task,
        args.out_dir,
        method_order=method_order,
        zoom_from=args.zoom_from,
        zoom_ymax=args.zoom_ymax,
        linear_ymax=args.linear_ymax,
        top_padding_fraction=args.top_padding_fraction,
        make_full_log=not args.no_full_log,
        smooth_window=args.smooth_window,
        smooth_sigma=args.smooth_sigma,
        style=args.style,
        legacy_kde=args.kde_source == "legacy",
    )
    config["plot"] = plot_config
    if args.style == "paper":
        caption = (
            f"Push，seed {args.seeds}；曲线为均值，阴影为逐点 bootstrap 95% 置信区间。\n"
            + (f"每条 seed 曲线先做 gaussian_filter1d(sigma={args.smooth_sigma:g}, mode=reflect, truncate=4)；"
               "区间对应平滑后的曲线，不是原始逐点 KL。\n" if args.smooth_sigma is not None else
               f"每条 seed 曲线先做 {args.smooth_window} 点居中移动平均。\n" if args.smooth_window > 1 else
               "未做平滑，与论文 Fig.6 原 notebook 的聚合方式一致。\n")
            + f"纵轴显示范围 {plot_config['plot_settings']['linear_ylim']}；早期超出上界的均值点数 "
              f"{len(plot_config['off_scale_means'])}，完整数据另存 CSV，未裁剪数据本身。\n"
            + "对齐的是绘图样式；当前五种估计器与原论文 RB/PE 对照的内容不同，曲线形状不应被强行匹配。\n"
            + ("KDE (legacy) 使用旧论文的历史评估列，与当前方法在 KL 计算、oracle 带宽和抽样随机性等方面仍有差异，不能视为严格同口径比较。\n"
               if args.kde_source == "legacy" else "")
        )
        (args.out_dir / f"figure_caption_{args.task}.txt").write_text(caption)
    config["valid_seed_counts"] = [
        {"method": str(method), "checkpoint": int(step), "n_seeds": int(n)}
        for (method, step), n in df.groupby(["method", "checkpoint"], observed=True).seed.nunique().items()
    ]
    if args.kde_source == "legacy":
        config["warning"] = (
            "legacy KDE 来自上一轮投稿的 historical/PE 列；其抽样随机性、标准化空间、低密度截除、"
            "oracle 带宽及 MC 重复次数与新 runner 存在差异。新方法具体协议以所选输入目录的运行配置为准。"
        )
        print("警告：这是 legacy KDE 对照图，不应未经口径说明直接作为方法间公平比较的最终论文图。", file=sys.stderr)
    method_tag = "-".join(method.lower() for method in method_order)
    config_path = args.out_dir / f"run_config_fig6_{method_tag}_{args.task}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    long_path = args.out_dir / f"fig6_data_{method_tag}_{args.task}.csv"
    df.to_csv(long_path, index=False)
    if args.smooth_window > 1 or args.smooth_sigma is not None:
        smooth_tag = f"gaussian{args.smooth_sigma:g}" if args.smooth_sigma is not None else f"smooth{args.smooth_window}"
        smoothed_path = args.out_dir / f"fig6_data_{method_tag}_{args.task}_{smooth_tag}.csv"
        plot_df.to_csv(smoothed_path, index=False)
        print(f"平滑数据已保存: {smoothed_path}")
    print(f"saved: {config_path}")
    print(f"saved: {long_path}")


if __name__ == "__main__":
    main()
