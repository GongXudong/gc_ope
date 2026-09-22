"""绘制 Push 五方法 Fig.6：原始版、逐 seed 高斯平滑版及完整范围图。"""

from pathlib import Path
import sys
import argparse
import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from gc_ope.evaluate.offline_results import audit_result, sha256


METHODS = ["KDE (legacy)", "GMM", "NN", "FM", "NF"]
EXPECTED = list(range(10000, 1000001, 10000))


def load_data(result_root, legacy_root):
    """核对每份 CSV 后读取有效记录；不把跳过项补成零。"""
    frames, audit, hashes = [], {}, {}
    for method in ["gmm", "nn", "fm", "nf"]:
        for seed in range(1, 6):
            path = result_root / "method_per_seed" / f"{method}_push_seed{seed}.csv"
            check = audit_result(path, EXPECTED)
            if check["missing"] or check["extra"] or check["invalid"]:
                raise ValueError(f"结果未通过覆盖率检查：{path} {check}")
            frame = pd.read_csv(path)
            if not ((frame.task == "push") & (frame.seed == seed) & (frame.method == method)
                    & (frame.protocol == "push_same_family_inclusive_v1") & (frame.kl_mode == "raw")).all():
                raise ValueError(f"实验身份或协议不一致：{path}")
            check["excluded"] = frame.loc[frame.status != "ok", ["checkpoint", "status", "error"]].to_dict("records")
            audit[f"{method}_seed{seed}"] = check
            frame = frame.loc[frame.status == "ok", ["seed", "checkpoint", "kl"]].copy()
            frame["method"], frame["source"] = method.upper(), str(path)
            frames.append(frame)
            hashes[str(path)] = sha256(path)
    for seed in range(1, 6):
        path = legacy_root / f"my_push_sac_seed_{seed}_kde_0_9_eval_res_in_training.csv"
        original = pd.read_csv(path)
        frame = original[["evaluation_index", "$D_{KL}$ [his]"]].rename(
            columns={"evaluation_index": "checkpoint", "$D_{KL}$ [his]": "kl"})
        if frame.checkpoint.duplicated().any() or not set(frame.checkpoint).issubset(EXPECTED):
            raise ValueError(f"旧 KDE 的 checkpoint 异常：{path}")
        valid = np.isfinite(frame.kl.to_numpy(dtype=float))
        audit[f"kde_legacy_seed{seed}"] = {"ok": int(valid.sum()), "nonfinite": int((~valid).sum()),
                                           "missing": sorted(set(EXPECTED) - set(frame.loc[valid, "checkpoint"]))}
        frame = frame.loc[valid].copy()
        frame["seed"], frame["method"], frame["source"] = seed, "KDE (legacy)", str(path)
        frames.append(frame)
        hashes[str(path)] = sha256(path)
    data = pd.concat(frames, ignore_index=True).sort_values(["method", "seed", "checkpoint"])
    return data, audit, hashes


def smooth_seeds(data, sigma=2):
    """各方法各 seed 独立平滑；存在断点时分段处理，不跨缺失点插值。"""
    smoothed = data.copy()
    smoothed["raw_kl"] = smoothed.kl
    for _, group in data.groupby(["method", "seed"]):
        boundaries = np.flatnonzero(np.diff(group.checkpoint) != 10000) + 1
        for segment in np.split(group.index.to_numpy(), boundaries):
            smoothed.loc[segment, "kl"] = gaussian_filter1d(
                data.loc[segment, "kl"].to_numpy(), sigma=sigma, mode="reflect", truncate=4,
            )
    return smoothed


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


def draw(data, summary, output, tag, ymax):
    """沿用原 Fig.6 的大小、配色、线宽与图例；完整范围另用 symlog 展示。"""
    sns.set_theme(context="notebook", style="darkgrid", font_scale=2.0)
    palette = sns.color_palette("deep")
    colors = dict(zip(METHODS, [palette[i] for i in [0, 2, 4, 1, 3]]))
    limits = {}
    for full in [False, True]:
        fig, ax = plt.subplots(figsize=(10, 8))
        for method in METHODS:
            series = summary[summary.method == method].sort_values("checkpoint")
            # 防止线段跨过真正缺失的 checkpoint。
            series = series.set_index("checkpoint").reindex(EXPECTED)
            ax.plot(series.index, series["mean"], color=colors[method], linewidth=1.5, label=method)
            ax.fill_between(series.index, series.ci_low, series.ci_high, color=colors[method], alpha=.2)
        ax.set_xlabel("Environment Steps")
        ax.set_ylabel(r"$D_{\mathrm{KL}}(p_{\mathrm{reference}}\,\|\,p_{\mathrm{history}})$")
        ax.margins(x=.05)
        ax.legend(title="Estimation Method", loc="upper right", fontsize=18)
        if full:
            ax.set_yscale("symlog", linthresh=1e-3)
            upper = max(float(data.kl.max()), float(summary.ci_high.max())) * 1.15
        else:
            upper = ymax * 1.05
        # 在显示坐标中留白；保留真实负值，不把纵轴固定到零。
        transform = ax.yaxis.get_transform()
        low, high = transform.transform([min(0., float(data.kl.min())), upper])
        lower = float(transform.inverted().transform([low - .05 * (high - low)])[0])
        ax.set_ylim(lower, upper)
        name = f"fig6_{tag}" + ("_fullrange" if full else "")
        fig.tight_layout()
        for extension in ["png", "pdf"]:
            fig.savefig(output / f"{name}.{extension}", dpi=180, bbox_inches="tight")
        plt.close(fig)
        limits[name] = [lower, upper]
    return limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=ROOT / "logs/push_same_family_all100_5x4")
    parser.add_argument("--legacy-root", type=Path, default=ROOT.parent / "gc_ope/plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/sac/eval_data")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--linear-ymax", type=float, default=None)
    args = parser.parse_args()
    # 每次另存，保护原图；用户可以显式选新的输出目录重复绘图。
    args.output.mkdir(parents=True, exist_ok=False)
    data, audit, hashes = load_data(args.result_root, args.legacy_root)
    smoothed = smooth_seeds(data)
    summaries = {"raw": summarize(data), "gaussian2": summarize(smoothed)}
    # 自动线性范围覆盖 200000 步以后两版的全部 95% CI。
    # 早期百万级尖峰会被平滑扩散到邻点，不能据此把主图范围放大到上千。
    late_upper = max(frame.loc[frame.checkpoint >= 200000, "ci_high"].max() for frame in summaries.values())
    ymax = args.linear_ymax if args.linear_ymax is not None else max(.5, np.ceil(late_upper * 2) / 2)
    if not np.isfinite(ymax) or ymax <= 0:
        raise ValueError("线性显示上限必须有限且为正")
    data.to_csv(args.output / "data_raw.csv", index=False)
    smoothed.to_csv(args.output / "data_gaussian2.csv", index=False)
    limits, excluded_from_view = {}, {}
    for tag, frame in [("raw", data), ("gaussian2", smoothed)]:
        summary = summaries[tag]
        summary.to_csv(args.output / f"summary_{tag}.csv", index=False)
        limits.update(draw(frame, summary, args.output, tag, ymax))
        excluded_from_view[tag] = summary.loc[summary["mean"] > ymax, ["method", "checkpoint", "mean"]].to_dict("records")
    # 读取和画图过程中原始实验结果必须保持不变。
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("绘图过程中输入结果发生变化，请重新检查")
    metadata = {"coverage": audit, "input_sha256": hashes, "script_sha256": sha256(__file__),
                "linear_range_reference_from": 200000,
                "smoothing": "逐方法、逐 seed、连续片段：gaussian_filter1d(sigma=2, mode=reflect, truncate=4)",
                "aggregation": "有效 seed 均值及 1000 次 bootstrap 的逐点 95% CI；先平滑 seed，再汇总",
                "limits": limits, "means_above_linear_view": excluded_from_view,
                "protocol_note": "新方法各自拟合同类参考分布，参考不共享；KDE 为旧投稿 legacy，不能作严格共同参考下的方法排名"}
    (args.output / "plot_config.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    current_audits = [value for key, value in audit.items() if not key.startswith("kde_legacy")]
    n_ok = sum(value["ok"] for value in current_audits)
    n_skipped = sum(value["skipped"] for value in current_audits)
    (args.output / "README.md").write_text(
        "# Fig.6：Push/SAC，seed 1～5\n\n"
        "主展示版：`fig6_gaussian2.png/pdf`；未平滑版：`fig6_raw.png/pdf`。\n"
        "对应的 `_fullrange` 为完整范围 symlog 图，包含早期巨大 KL，不截断原始数值。\n\n"
        "每种方法先逐 seed 做高斯平滑（σ=2 checkpoint，reflect，truncate=4），"
        "再计算跨 seed 均值和逐点 bootstrap 95% CI。原论文 Fig.6 未做此平滑；"
        "阴影表示平滑曲线的不确定性，不是原始单点的 CI。缺失/跳过不补零、不插值。\n\n"
        f"线性图上界为 {ymax * 1.05:g}；早期超界部分见完整范围图。纵轴保留零线以下留白。\n\n"
        "新方法的 KL 为各自全量拟合参考分布到历史估计分布；不同方法的参考分布不同。"
        "KDE (legacy) 来自旧投稿的 `$D_{KL}$ [his]` 列，标准化、过滤、带宽和 MC 协议有差异。"
        "本图不能据 KL 大小直接推断共同真实分布下的方法优劣。\n\n"
        f"四种新方法共 {n_ok + n_skipped} 条记录：{n_ok} 有效、{n_skipped} 条样本不足跳过。逐文件检查、有效 seed 数、"
        "数据与汇总表、超界点和源文件 SHA256 均另存，原实验文件未修改。\n"
    )
    print(f"完成：{args.output}\n绘图数据 {len(data)} 条；线性显示上限 {ymax * 1.05:g}")


if __name__ == "__main__":
    main()
