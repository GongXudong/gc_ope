"""绘制 KDE + GMM + NN（可选 NF）的论文 fig6 风格 KL 曲线。

默认绘图约定：slide NN 使用 fixed_grid_kl，push NN 和 GMM 使用 kl；
零刻度下保留约 5% 显示空间留白，不修改或平移数据值。
可用 ``--linear-ymax 0.5 --no-full-log`` 另出线性 0.5 上限版本，
上方额外保留少量空白，不生成对数图。

可用 ``--include-nf`` 从 normalizing_flow_{task}_seed{seed}.csv 加入 NF，使用 kl 列。

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


def _load_current_method(method: str, task: str, seeds: list[int], metric: str = "kl") -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    coverage = []
    for seed in seeds:
        path = METHOD_DIR / f"{method}_{task}_seed{seed}.csv"
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
        df["method"] = "NF" if method == "normalizing_flow" else method.upper()
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
) -> tuple[pd.DataFrame, dict]:
    nn_metric = resolve_nn_metric(task, nn_metric)
    method_order = [*BASE_METHOD_ORDER, "NF"] if include_nf else list(BASE_METHOD_ORDER)
    kde = _load_aligned_kde(task, seeds) if kde_source == "aligned" else _load_legacy_kde(task, seeds)
    gmm = _load_current_method("gmm", task, seeds)
    nn = _load_current_method("nn", task, seeds, metric=nn_metric)
    frames = [kde, gmm, nn]
    kde["source_metric"] = "$D_{KL}$ [his]" if kde_source == "legacy" else "kl"
    gmm["source_metric"] = "kl"
    nn["source_metric"] = nn_metric
    if include_nf:
        nf = _load_current_method("normalizing_flow", task, seeds, metric="kl")
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
            **({"NF": "kl"} if include_nf else {}),
        },
        "mixed_metric_warning": "NN 使用离散 fixed_grid_kl；KDE/GMM 保留原指标，不是同口径比较。" if nn_metric != "kl" else None,
        "kde_source": kde_source,
        "kde_series": "$D_{KL}$ [his]" if kde_source == "legacy" else "kl",
        "kde_protocol_comparable_to_current_gmm_nn": kde_source == "aligned" and nn_metric == "kl",
        "include_nf": include_nf,
        "input_coverage": coverage,
        "aggregation": "每个 checkpoint 使用有效 seed 均值和 bootstrap 95% CI；error/skipped 不补值",
        "current_method_dir": _display_path(METHOD_DIR),
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
) -> dict:
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
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(
        data=plot_df,
        x="checkpoint",
        y="kl",
        hue="Estimation Method",
        hue_order=method_order,
        palette=METHOD_COLORS,
        errorbar=("ci", 95),
        n_boot=1000,
        seed=0,
        ax=ax,
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(r"$D_{KL}(P_{ag} \| \tilde{p}_{ag})$")
    ax.set_xlim(left=0, right=1_000_000)
    # 固定上限版本把 0.5 视为数据显示上限，再在其上方留一点空白。
    axis_top = ymax * (1.0 + top_padding_fraction) if linear_ymax is not None else ymax
    ax.set_ylim(top=axis_top)
    linear_limits = pad_below_zero(ax, df["kl"])
    sns.move_legend(ax, loc="upper right", fontsize=18)
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
            hue_order=method_order,
            palette=METHOD_COLORS,
            errorbar=("ci", 95),
            n_boot=1000,
            seed=0,
            ax=ax,
        )
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(r"$D_{KL}(P_{ag} \| \tilde{p}_{ag})$")
        ax.set_xlim(left=0, right=1_000_000)
        ax.set_yscale("symlog", linthresh=1e-3)
        full_limits = pad_below_zero(ax, df["kl"])
        sns.move_legend(ax, loc="upper right", fontsize=18)
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
    parser.add_argument("--kde-source", choices=["aligned", "legacy"], default="aligned")
    parser.add_argument("--allow-protocol-mismatch", action="store_true", help="允许使用旧 KDE 与当前 GMM/NN 混合绘图")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--zoom-from", type=int, default=100000)
    parser.add_argument("--zoom-ymax", type=float, default=None)
    parser.add_argument("--linear-ymax", type=float, default=None, help="线性图的数据上限；例如 0.5。上方按 --top-padding-fraction 留白")
    parser.add_argument("--top-padding-fraction", type=float, default=0.05, help="固定线性上限时的顶部留白比例")
    parser.add_argument("--no-full-log", action="store_true", help="不生成 symlog full-log 伴随图")
    parser.add_argument("--include-nf", action="store_true", help="加入 normalizing_flow 方法（显示名 NF）")
    args = parser.parse_args()

    if len(args.seeds) != len(set(args.seeds)):
        parser.error("--seeds 不能重复")
    if args.kde_source == "legacy" and not args.allow_protocol_mismatch:
        parser.error(
            "legacy KDE 与当前 GMM/NN 的历史数据抽样、MC-KL 口径不完全一致；"
            "如需生成对照图，请显式加 --allow-protocol-mismatch"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df, config = load_data(args.task, args.seeds, args.kde_source, args.nn_metric, args.include_nf)
    method_order = config["methods"]
    plot_config = _plot(
        df,
        args.task,
        args.out_dir,
        method_order=method_order,
        zoom_from=args.zoom_from,
        zoom_ymax=args.zoom_ymax,
        linear_ymax=args.linear_ymax,
        top_padding_fraction=args.top_padding_fraction,
        make_full_log=not args.no_full_log,
    )
    config["plot"] = plot_config
    config["valid_seed_counts"] = [
        {"method": str(method), "checkpoint": int(step), "n_seeds": int(n)}
        for (method, step), n in df.groupby(["method", "checkpoint"], observed=True).seed.nunique().items()
    ]
    if args.kde_source == "legacy":
        config["warning"] = (
            "legacy KDE 来自上一轮投稿的 historical/PE 列；其生成协议使用每个 checkpoint 随机抽取 100 条历史评估记录、"
            "累计折扣并以 10000 个 MC 样本估计 KL，与当前 GMM/NN 的全历史记录、100000×5 MC 协议不完全一致。"
        )
        print("警告：这是 legacy KDE 对照图，不应未经口径说明直接作为方法间公平比较的最终论文图。", file=sys.stderr)
    method_tag = "-".join(method.lower() for method in method_order)
    config_path = args.out_dir / f"run_config_fig6_{method_tag}_{args.task}.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n")
    long_path = args.out_dir / f"fig6_data_{method_tag}_{args.task}.csv"
    df.to_csv(long_path, index=False)
    print(f"saved: {config_path}")
    print(f"saved: {long_path}")


if __name__ == "__main__":
    main()
