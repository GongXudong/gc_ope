"""正式五方法 Fig.6：论文原文字、Seaborn 默认顺序配色、实线、不平滑。"""

from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from gc_ope.evaluate.offline_results import audit_result, sha256

METHODS = ["KDE", "GMM", "NN", "FM", "NF"]
EXPECTED = list(range(10000, 1000001, 10000))
X_LABEL = "Environment Steps"
Y_LABEL = r"$D_{KL}(P_{ag} \| \tilde{p}_{ag})$"
LEGEND_TITLE = "Estimation Method"


def load_data(sources):
    """来源清单显式映射旧实验身份到正式名称；不改写任何实验 CSV。"""
    if [source["name"] for source in sources] != METHODS:
        raise ValueError("必须按 KDE、GMM、NN、FM、NF 顺序指定且只指定五个来源")
    frames, audit, hashes = [], {}, {}
    for source in sources:
        name, method = source["name"], source["method"]
        root = Path(source["root"])
        if not root.is_absolute():
            root = ROOT / root
        for relative, expected_hash in source.get("sha256", {}).items():
            path = root / relative
            if sha256(path) != expected_hash:
                raise ValueError(f"来源摘要不符：{path}")
            hashes[str(path.resolve())] = expected_hash
        for seed in range(1, 6):
            if source["protocol"] == "legacy":
                if name != "KDE":
                    raise ValueError("legacy 数据只适用于 KDE")
                path = root / f"my_push_sac_seed_{seed}_kde_0_9_eval_res_in_training.csv"
                original = pd.read_csv(path)
                frame = original[["evaluation_index", "$D_{KL}$ [his]"]].rename(
                    columns={"evaluation_index": "checkpoint", "$D_{KL}$ [his]": "kl"})
                if frame.checkpoint.duplicated().any() or not set(frame.checkpoint).issubset(EXPECTED):
                    raise ValueError(f"旧 KDE 的 checkpoint 异常：{path}")
                valid = np.isfinite(frame.kl.to_numpy(dtype=float))
                audit[f"{name}_seed{seed}"] = dict(ok=int(valid.sum()), nonfinite=int((~valid).sum()),
                    missing=sorted(set(EXPECTED)-set(frame.loc[valid, "checkpoint"])))
                frame = frame.loc[valid].copy()
                frame["seed"] = seed
            else:
                path = root / "method_per_seed" / f"{method}_push_seed{seed}.csv"
                check = audit_result(path, EXPECTED)
                if check["missing"] or check["extra"] or check["invalid"]:
                    raise ValueError(f"覆盖率检查失败：{path} {check}")
                original = pd.read_csv(path)
                if not ((original.task == "push") & (original.seed == seed) &
                        (original.method == method) & (original.protocol == source["protocol"]) &
                        (original.kl_mode == "raw")).all():
                    raise ValueError(f"实验身份或协议不一致：{path}")
                check["quality_warnings"] = (original.loc[original.fit_quality == "warning",
                    ["checkpoint", "fit_warnings"]].to_dict("records") if "fit_quality" in original else [])
                check["excluded"] = original.loc[original.status != "ok",
                    ["checkpoint", "status", "error"]].to_dict("records")
                audit[f"{name}_seed{seed}"] = check
                frame = original.loc[original.status == "ok", ["seed", "checkpoint", "kl"]].copy()
            frame["method"], frame["source_method"], frame["source"] = name, method, str(path.resolve())
            frames.append(frame)
            hashes[str(path.resolve())] = sha256(path)
    return pd.concat(frames, ignore_index=True).sort_values(["method", "seed", "checkpoint"]), audit, hashes


def summarize(data):
    """对各 checkpoint 的有效 seed 做 bootstrap，而不是重采样 MC 重复值。"""
    rows = []
    for (method, checkpoint), group in data.groupby(["method", "checkpoint"]):
        values = group.kl.to_numpy()
        indices = np.random.default_rng(0).integers(0, len(values), (1000, len(values)))
        low, high = np.percentile(values[indices].mean(axis=1), [2.5, 97.5])
        rows.append(dict(method=method, checkpoint=int(checkpoint), n_seeds=len(values),
                         mean=float(values.mean()), ci_low=float(low), ci_high=float(high)))
    return pd.DataFrame(rows)


def draw(data, summary, output, ymax):
    """论文使用 Seaborn darkgrid；按图例顺序取默认 deep 色，不再手工跳色。"""
    sns.set_theme(context="notebook", style="darkgrid", font_scale=2.0)
    colors = dict(zip(METHODS, sns.color_palette("deep", n_colors=5)))
    limits = {}
    for full in [False, True]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for method in METHODS:
            series = summary.loc[summary.method == method].set_index("checkpoint").reindex(EXPECTED)
            ax.plot(series.index, series["mean"], label=method, color=colors[method],
                    linewidth=1.5, linestyle="-")
            ax.fill_between(series.index, series.ci_low, series.ci_high, color=colors[method], alpha=.2)
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(Y_LABEL)
        ax.legend(title=LEGEND_TITLE, loc="upper right")
        ax.margins(x=.05)
        if full:
            ax.set_yscale("symlog", linthresh=1e-3)
            upper = max(float(data.kl.max()), float(summary.ci_high.max())) * 1.15
        else:
            upper = ymax * 1.05
        transform = ax.yaxis.get_transform()
        low, high = transform.transform([min(0., float(data.kl.min())), upper])
        lower = float(transform.inverted().transform([low-.05*(high-low)])[0])
        ax.set_ylim(lower, upper)
        tag = "fig6_fullrange" if full else "fig6"
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{tag}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        limits[tag] = [lower, upper]
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, default=ROOT / "configs/evaluate/fig6_sources.json")
    parser.add_argument("--result-root", type=Path, help="可选：四方法使用正式命名新跑出的结果目录；KDE 仍读来源清单")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-ymax", type=float)
    args = parser.parse_args()
    sources = json.loads(args.sources.read_text())["sources"]
    if args.result_root:
        for source in sources:
            if source["name"] != "KDE":
                source.update(root=str(args.result_root.resolve()), method=source["name"].lower(),
                              protocol="push_final_five_v1", sha256={})
    data, audit, hashes = load_data(sources)
    summary = summarize(data)
    late_upper = summary.loc[summary.checkpoint >= 200000, "ci_high"].max()
    ymax = args.linear_ymax if args.linear_ymax is not None else max(.5, np.ceil(late_upper*2)/2)
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("线性显示上限必须有限且为正")
    args.output.mkdir(parents=True, exist_ok=False)
    data.to_csv(args.output / "data_raw.csv", index=False)
    summary.to_csv(args.output / "summary_raw.csv", index=False)
    limits = draw(data, summary, args.output, ymax)
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("绘图期间输入文件发生变化")
    metadata = dict(sources=sources, coverage=audit, input_sha256=hashes,
        script_sha256=sha256(__file__), smoothing=None, methods=METHODS,
        palette=dict(zip(METHODS, sns.color_palette("deep",5).as_hex())), linestyle="-",
        xlabel=X_LABEL, ylabel=Y_LABEL, legend_title=LEGEND_TITLE,
        aggregation="五 seed 的原始 KL 均值，1000 次 bootstrap 逐点 95% CI", limits=limits,
        points_above_linear_view=summary.loc[summary.ci_high > limits["fig6"][1]].to_dict("records"),
        protocol_note="KDE 复用旧投稿数据；其计算口径有差异。其他方法各自拟合参考分布。名称简化不意味着共同参考。")
    (args.output / "plot_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+"\n")
    (args.output / "README.md").write_text(
        "# 正式五方法 Fig.6\n\n"
        "KDE、GMM、NN、FM、NF；实线、无平滑、Seaborn 默认 deep 顺序配色。\n"
        "横轴 Environment Steps；纵轴与论文 Fig.6 一致；图例标题 Estimation Method。\n"
        "均值和 95% CI 来自逐 checkpoint 的有效 seed，跳过项不补零、不删质量提醒。\n"
        "fig6.png/pdf 为线性主图，fig6_fullrange.png/pdf 为保留早期尖峰的 symlog 完整范围。\n"
        "名称简化仅影响展示：KDE 的 legacy 来源，以及各方法原始身份、配置摘要与警告见 plot_config.json。\n")
    print(f"完成：{args.output}；{len(data)} 条有效记录；五条未平滑实线")


if __name__ == "__main__":
    main()
