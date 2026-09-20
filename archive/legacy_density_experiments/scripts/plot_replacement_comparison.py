"""P0 replacement 实验结果可视化 + 汇总表格（plan.md 第 7/22.8 节）。

数据源：logs/replacement_experiment/kde_gmm_hist_gaussian_kl.csv
（4 任务 × 4 估计器 × 5 seed × 5 checkpoint，job 粒度追加写，可部分完成时先看已有结果）

输出到 logs/replacement_experiment/plots/：
* fig_kl_curves.png/pdf   — 每 task 一子图：4 条估计器均值线 + 跨 seed 95% CI + 每 seed 淡线
* fig_density_panel.csv   — density checkpoint 处 4 估计器 × 任务 的 2D 网格密度（供热图）
* summary_table.csv       — task × method 的 KL 均值/标准差（所有 checkpoint 均值），供论文/reviewer 对比
* conclusion.json         — 自动填写的一句话结论（AC-8.3），留待人工审阅
"""

from __future__ import annotations

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
CSV = ROOT / "logs" / "replacement_experiment" / "kde_gmm_hist_gaussian_kl.csv"
OUT = ROOT / "logs" / "replacement_experiment" / "plots"

METHOD_ORDER = ["kde", "gmm", "histogram", "gaussian", "nn", "flow_matching"]
METHOD_COLORS = {
    "kde": "#d62728", "gmm": "#2ca02c", "histogram": "#ff7f0e",
    "gaussian": "#1f77b4", "nn": "#9467bd", "flow_matching": "#8c564b",
}


def load() -> pd.DataFrame:
    if not CSV.exists():
        sys.exit(f"找不到数据文件 {CSV}")
    df = pd.read_csv(CSV)
    # 去重：同一 (task,seed,checkpoint,method) 若被多 worker 重复写，保留最新一条
    df = df.drop_duplicates(subset=["task", "seed", "checkpoint", "method"], keep="last")
    return df


def kl_curies(df: pd.DataFrame) -> None:
    """fig_kl_curves：每 task 一个子图，x=checkpoint，y=KL，4 条估计器线。"""
    tasks = sorted(df.task.unique(), key=["push", "slide", "reach", "vvc"].index)
    fig, axes = plt.subplots(1, len(tasks), figsize=(6 * len(tasks), 4.5), squeeze=False)
    for ax, task in zip(axes[0], tasks):
        d = df[(df.task == task) & (df.status == "ok")]
        if d.empty:
            ax.text(0.5, 0.5, "no valid rows", ha="center", va="center")
            continue
        for method in METHOD_ORDER:
            m = d[d.method == method]
            if m.empty:
                continue
            for seed, g in m.groupby("seed"):
                g = g.sort_values("checkpoint")
                ax.plot(g.checkpoint, g.kl, color=METHOD_COLORS[method], alpha=0.25, linewidth=0.8)
            mean = m.groupby("checkpoint")["kl"].mean()
            n = m.groupby("checkpoint")["kl"].count()
            q025 = m.groupby("checkpoint")["kl"].quantile(0.025)
            q975 = m.groupby("checkpoint")["kl"].quantile(0.975)
            if int(n.max()) >= 2:
                ax.fill_between(q025.index, q025, q975, color=METHOD_COLORS[method], alpha=0.15)
            ax.plot(mean.index, mean, color=METHOD_COLORS[method], linewidth=2.2,
                    label={"gmm": "GMM", "nn": "NN", "flow_matching": "FlowMatching"}.get(method, method.title()))
        ax.set_xlabel("Environment steps")
        ax.set_ylabel(r"$D_{KL}(p_{ag} \,||\, \hat{p}_{ag})$")
        ax.set_title(task.capitalize(), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc="upper right")
    fig.suptitle("KL divergence of goal-distribution estimators over training "
                 "(KDE bandwidth=0.2, GMM k=5, Histogram bins=10^d)", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_kl_curves.png", dpi=180, bbox_inches="tight")
    fig.savefig(OUT / "fig_kl_curves.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"saved {OUT / 'fig_kl_curves.png'}")


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """task × method 的 KL 均值/标准差（跨 checkpoint、seed），reviewer-ready 表。"""
    ok = df[df.status == "ok"]
    table = (
        ok.groupby(["task", "method"])["kl"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    table["method"] = pd.Categorical(
        table["method"],
        categories=[m for m in METHOD_ORDER if m in set(table["method"])],
        ordered=True,
    )
    table = table.sort_values(["task", "method"]).reset_index(drop=True)
    table.to_csv(OUT / "summary_table.csv", index=False)
    print(table.to_string(index=False))
    return table


def auto_conclusion(df: pd.DataFrame) -> dict:
    """AC-8.3：根据数据自动填充一句话结论（数值部分，定性部分留给人）。"""
    ok = df[df.status == "ok"]
    # 各 task 下 4 估计器 KL 均值的相对排序（KL 越小 = 越接近参考 oracle KDE）
    ranking = {}
    for task in ok.task.unique():
        d = ok[ok.task == task]
        means = d.groupby("method")["kl"].mean().sort_values()
        ranking[task] = means.index.tolist()
    return {
        "note": "自动生成的估计器排序（按平均 KL 升序，越小越接近参考 oracle KDE），供撰写结论时参考，定性判断需人工确认。",
        "task_kl_ranking": ranking,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.1)
    df = load()
    print(f"loaded {len(df)} rows, tasks={sorted(df.task.unique())}, "
          f"status_ok={(df.status=='ok').sum()}, status_err={(df.status=='error').sum()}", flush=True)
    kl_curies(df)
    table = summary_table(df)
    conclusion = auto_conclusion(df)
    (OUT / "conclusion.json").write_text(json.dumps(conclusion, indent=2, ensure_ascii=False))
    print(f"saved {OUT / 'conclusion.json'}")
    print(f"all done -> {OUT}")


if __name__ == "__main__":
    main()
