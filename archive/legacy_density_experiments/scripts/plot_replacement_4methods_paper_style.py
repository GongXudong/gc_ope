"""非课程 Push/SAC 轨迹上 p̂_ag 估计器的论文 fig5 / fig6 风格对比图。

## 一、论文 fig5 / fig6 的绘图脚本定位（已核对，非按文件名猜测）

* **fig5**（Distribution of sampled behavioral goals on p_ag）：
  `plots/density_of_behavioral_goals/my_push/sac/plot_behavioral_goals_on_p_ag_omega.ipynb`；
  产出与 `paper/ICML2026/pics/behavioral_goals_on_p_ag/push/sac/behavioral_goals_on_p_ag_omega.pdf`
  **MD5 相同**（49ec73e554f6c66498937a988c823811）。
* **fig6**（Estimation error of p_ag under different estimation methods）：
  `plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/plot_omega.ipynb`
  （GC-SAC 单面板见同目录 `sac/sac.ipynb`）；产出与论文
  `paper/ICML2026/pics/KL_p_ag_truth_and_p_ag_estimated/push/*.pdf` MD5 相同。

沿用的绘图设置：全局 `sns.set_theme(context="notebook", style="darkgrid", font_scale=2.0)`；
fig5 每列一个进度、每行一个分布、`Progress = xx%` 列标题、最右列竖排方法名；
fig6 `figsize=(10, 8)`、`sns.lineplot(errorbar=("ci", 95))`、
`move_legend(loc="upper right", fontsize=18)`。

## 二、本实验协议

数据源 `logs/replacement_experiment/all100/kde_gmm_hist_gaussian_kl copy.csv`：
vanilla SAC（无课程）Push、5 seed × 100 checkpoint，4 个估计器均为
**用历史评估成功目标（κ=0.9 折扣）拟合、对当前 checkpoint 成功目标
（oracle KDE）算 MC-KL**。

## 三、为什么默认只画 fig6、且只保留 KDE/GMM/Gaussian

这是本脚本最重要的口径问题，直接决定图能支撑什么结论：

1. **fig5 在非课程轨迹上测不到论文想论证的东西。** 论文 fig5 的黄点是
   "课程采样器实际选出的 behavioral goals"，它要说明的是
   *用历史评估数据估计的 p̂_ag 去采样* 比 *用 replay buffer 估计的 p̂_ag 去采样*
   更好。而本实验是 vanilla SAC、**没有课程、没有采样器**：
   `checkpoints/my_push/sac/seed_*/` 下不存在 behavioral-goal 采样日志
   （`logs_in_process/my_push/sac/` 里只有 mega/omega/orig/odiscern 等带课程
   运行的日志，无裸 vanilla SAC 日志）。因此本目录里的 fig5 把黄点画成
   "该 checkpoint 的成功目标" 只是**为了有参考散点**的替代画法，
   它对应的是"历史评估拟合 vs 当前真实分布"的估计误差，**不是课程采样效果**。
   换句话说：本实验对 fig5 只支持"拟合误差"这一半，不支持论文 fig5 的结论。
   故 `--fig` 默认是 `fig6`，fig5 需显式指定才画，且图上/JSON 都标注了单 seed 口径。
2. **Histogram 被排除，是因为指标对它失效，不是因为"它更差"。**
   直方图密度是盒函数：在 oracle（当前 checkpoint 成功目标）支撑外大量取 0。
   MC-KL 的现有实现会把 `q=0` 的样本过滤掉（避免 `-inf`），等价于**只在 q>0 的
   子集上求一个条件 KL**，这个量没有下界，于是出现系统性负值
   （515 个 ok 行里 435 行为负，最负 −3.09）。同一 checkpoint 上：
   KDE/GMM/Gaussian 被过滤 0%，Histogram 被过滤 16.6%。
   同时 Histogram 估计器还有一处实现问题：`bin_volume` 用
   `np.prod(np.diff(edges))` 把"同一维内 10 个 bin 的宽度"连乘了（应为跨维连乘），
   2D 下把密度放大约 1e8 倍；其 `evaluate(return_density=False)` 分支与
   `return_density=True` 分支自相差 23.03 nats（其余三个方法自差 0）。
   所以 Histogram 的 KL 数值不能与另外三个方法并列比较。
   需要复现旧图时用 `--methods kde gmm histogram gaussian`。

## 四、产出（默认 `--methods kde gmm gaussian`）

输出目录 `logs/replacement_experiment/all100/plots_fig5_fig6_4methods/`：

* `fig6_kl_curves_kde-gmm-gaussian_push.{png,pdf}` —— 论文 fig6 单面板设置；
* `fig6_kl_curves_kde-gmm-gaussian_push_fulllog.{png,pdf}` —— symlog 全量程，不裁点；
* `fig5_density_kde-gmm-gaussian_push.{png,pdf}` —— 仅在 `--fig fig5`/`--fig all` 时生成；
* `run_config_fig5_fig6.json` —— 输入、seed 规则、超参、被裁掉的均值等。

## 五、用法

    conda run -n gc_ope --no-capture-output env OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python scripts/plot_replacement_4methods_paper_style.py

    # 只重画 fig6（不拟合估计器，秒级）
    ... --fig fig6
    # 连 fig5 一起画（需显式开启）
    ... --fig all
    # 复现旧的四方法版本（含 Histogram，数值不可与前三者并列比较）
    ... --methods kde gmm histogram gaussian

只读操作：只读固定评估 CSV 与结果 CSV；缓存与输出全部落在 `logs/` 下。

## 六、口径限制（写入结论时必须保留）

1. **fig6 才是可用的结论图**：5 seed 均值 + 95% CI，衡量"历史评估数据估计的
   p̂_ag 与当前真实 p_ag 的距离随训练如何变化"。
2. **fig5 在本实验里只是拟合误差图**，不含课程采样；不要用它论证
   "历史评估优于 replay buffer"。
3. **> 80k 步之后的 KL 才是稳定可读的**：更早的 checkpoint 历史成功样本极少，
   KDE oracle 支撑高度碎片化，出现量级异常值（GMM @ 40k 均值 2.35e6）。
   这些点全部落在 `<= 80k`，已在图注与 JSON 中显式列出。
4. **估计器是离线重拟合的**，入参与 `replacement_run_job._run_one_job` 完全一致；
   已用 `fixed_grid_kl` 逐行对拍，3 checkpoint × 4 方法最大绝对误差 1.8e-15。
5. **参考带宽**：默认 `--reference-renderer oracle` 用 `KDEEvaluator(bw=0.2)`，
   与 MC-KL 里被评分的 oracle 严格同一个分布；论文 fig5 原样的 `sns.kdeplot`
   用 Scott 带宽，两者不是同一个分布，对照时不要混用。
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from evaluate_gmm_prediction import (  # noqa: E402
    GOAL_COLUMNS,
    add_records,
    checkpoint_file,
    historical_files,
)

DEFAULT_CSV = ROOT / "logs" / "replacement_experiment" / "all100" / "kde_gmm_hist_gaussian_kl copy.csv"
DEFAULT_OUT = ROOT / "logs" / "replacement_experiment" / "all100" / "plots_fig5_fig6_4methods"

# 默认对比的三个估计器。Histogram 不在默认列表里，原因见文件头"口径限制"第 5/6 条：
# 它的密度是盒函数，在 oracle 的成功目标支撑外大量取 0，MC-KL 对它失效
# （过滤掉 q=0 的样本后会在 q>0 子集上求一个条件 KL，可得到负值）。
# 需要时可用 --methods kde gmm histogram gaussian 显式加回来。
METHOD_ORDER = ["kde", "gmm", "gaussian"]
ALL_METHODS = ["kde", "gmm", "histogram", "gaussian"]
METHOD_LABELS = {
    "kde": "KDE",
    "gmm": "GMM",
    "histogram": "Histogram",
    "gaussian": "Gaussian",
}
# 论文 fig5/fig6 的绘图脚本（用于 provenance 记录）
PAPER_NOTEBOOKS = {
    "fig5": "plots/density_of_behavioral_goals/my_push/sac/plot_behavioral_goals_on_p_ag_omega.ipynb",
    "fig6": "plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/plot_omega.ipynb",
    "fig6_gc_sac_panel": "plots/p_ag_dist_between_truth_and_estimated_in_training/my_push/sac/sac.ipynb",
}
PAPER_FIGURES = {
    "fig5_pdf": "paper/ICML2026/pics/behavioral_goals_on_p_ag/push/sac/behavioral_goals_on_p_ag_omega.pdf",
    "fig6_pdf": "paper/ICML2026/pics/KL_p_ag_truth_and_p_ag_estimated/push/my_push_dist_between_p_ag_SAC.pdf",
}


def make_tag(methods: list[str]) -> str:
    """把方法列表压成文件名后缀，例如 ['kde','gmm','gaussian'] -> 'kde-gmm-gaussian'。"""
    return "-".join(methods)


def rel_or_abs(path: Path) -> str:
    """尽力给出相对仓库根目录的路径；输出目录在仓库外时退回绝对路径。"""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(Path(path).resolve())


# --------------------------------------------------------------------------- #
# 数据与估计器
# --------------------------------------------------------------------------- #
def load_metrics(csv_path: Path, task: str) -> pd.DataFrame:
    """读取 replacement 结果 CSV：去重（保留最新一条）、只保留 status == ok。"""
    if not csv_path.is_file():
        sys.exit(f"找不到数据文件 {csv_path}")
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset=["task", "seed", "checkpoint", "method"], keep="last")
    df = df[(df["task"] == task) & (df["status"] == "ok")].copy()
    if df.empty:
        sys.exit(f"{csv_path} 中没有 {task} 的可用行（status == ok）")
    return df


def make_estimator_args(args: argparse.Namespace) -> argparse.Namespace:
    """把绘图参数折叠成 replacement_run_job._make_estimator 需要的名字空间。"""
    return argparse.Namespace(
        kappa=args.kappa,
        n_components=args.n_components,
        resample_size=args.resample_size,
        bandwidth=args.bandwidth,
        random_state=args.random_state,
        gmm_reg_covar=args.gmm_reg_covar,
        n_hist_bins=args.n_hist_bins,
        gaussian_reg_covar=args.gaussian_reg_covar,
        nn_epochs=100, nn_lr=1e-3, nn_hidden=16,
        fm_epochs=80, fm_hidden=16, fm_samples=2000,
    )


def fit_four_estimators(task: str, seed: int, checkpoint: int, est_args: argparse.Namespace):
    """用与 replacement_run_job 完全一致的历史数据 + 权重拟合所选估计器。

    方法列表取自 ``est_args.methods``（默认三方法）。"""
    import replacement_run_job as rrj

    estimator_args = make_estimator_args(est_args)
    methods = list(getattr(est_args, "methods", METHOD_ORDER))
    reference = pd.read_csv(checkpoint_file(task, seed, checkpoint))
    history = [(t, pd.read_csv(p)) for t, p in historical_files(task, seed, checkpoint)]
    estimators = {}
    for method in methods:
        est = rrj._make_estimator(
            method, task,
            kappa=estimator_args.kappa,
            n_components=estimator_args.n_components,
            resample_size=estimator_args.resample_size,
            bandwidth=estimator_args.bandwidth,
            random_state=estimator_args.random_state,
            gmm_reg_covar=estimator_args.gmm_reg_covar,
            n_hist_bins=estimator_args.n_hist_bins,
            gaussian_reg_covar=estimator_args.gaussian_reg_covar,
        )
        add_records(est, history, checkpoint, estimator_args.kappa, goal_columns=GOAL_COLUMNS)
        est.fit_evaluator()
        estimators[method] = est
    return estimators, reference


def oracle_reference_field(
    task: str, seed: int, checkpoint: int, grid: np.ndarray, est_args: argparse.Namespace,
    cache_dir: Path | None,
) -> np.ndarray:
    """参考密度场：与 MC-KL / fixed_grid_kl 里完全同一个 oracle。

    `replacement_run_job._run_one_job` 的 oracle 是
    ``KDEEvaluator(kde_bandwidth=bandwidth)`` 拟合当前 checkpoint 的成功目标。
    这里用同一个构造，保证图上的参考密度与 CSV 里被评分的参考分布一致
    （而不是 seaborn `kdeplot` 默认的 Scott 带宽 —— 那会是另一个分布）。
    """
    path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = density_cache_path(
            cache_dir, task, seed, checkpoint, "oracle_ref", est_args, grid
        )
        if path.is_file():
            with np.load(path) as data:
                return data["density"]

    from gc_ope.evaluate.evaluation_result_container import EvaluationResultContainer
    from gc_ope.evaluate.evaluator_kde import KDEEvaluator

    ref = pd.read_csv(checkpoint_file(task, seed, checkpoint))
    success = ref.loc[ref["termination"] == "reach target", GOAL_COLUMNS].to_numpy(float)
    oracle = KDEEvaluator(
        evaluation_result_container_class=EvaluationResultContainer,
        kde_bandwidth=est_args.bandwidth,
    )
    oracle.eval_res_container.add_batch(
        success, [True] * len(success), [0.0] * len(success), [0.0] * len(success)
    )
    oracle.fit_evaluator()
    _, log_d = oracle.evaluate(grid, return_density=False)
    scale = np.asarray(oracle.scaler.scale_, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("oracle scaler 存在非正或非有限的 scale")
    field = np.exp(np.clip(log_d - np.log(scale).sum(), -745.0, 709.0))
    if path is not None:
        np.savez_compressed(path, density=field)
    return field


def density_cache_path(
    cache_dir: Path, task: str, seed: int, checkpoint: int, method: str,
    est_args: argparse.Namespace, grid: np.ndarray,
) -> Path:
    """密度场缓存文件名；键包含任务/seed/checkpoint/方法/超参/网格。"""
    tag = (
        f"{task}_seed{seed}_ckpt{checkpoint}_{method}"
        f"_k{est_args.kappa:g}_bw{est_args.bandwidth:g}"
        f"_gmm{est_args.n_components}x{est_args.resample_size}"
        f"_hist{est_args.n_hist_bins}"
        f"_rs{est_args.random_state}_grid{grid.shape[0]}"
        f"_{grid[:, 0].min():.4f}_{grid[:, 0].max():.4f}"
    )
    return cache_dir / f"{tag}.npz"


def density_fields_for_checkpoint(
    task: str, seed: int, checkpoint: int, grid: np.ndarray,
    est_args: argparse.Namespace, cache_dir: Path | None,
) -> dict[str, np.ndarray]:
    """一次拿到该 checkpoint 上 4 个估计器的密度场（带磁盘缓存）。

    冷启动时只拟合一遍 4 个估计器并把 4 份密度场写入缓存；之后重跑全部命中
    缓存，不再重新拟合。缓存键含全部超参与网格，键一致即数值一致。
    独立校验脚本已确认重拟合结果与 CSV 记录的 ``fixed_grid_kl`` 一致到 1e-15。
    """
    methods = list(getattr(est_args, "methods", METHOD_ORDER))
    fields: dict[str, np.ndarray] = {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for method in methods:
            path = density_cache_path(cache_dir, task, seed, checkpoint, method, est_args, grid)
            if path.is_file() and method not in fields:
                with np.load(path) as data:
                    fields[method] = data["density"]
    missing = [m for m in methods if m not in fields]
    if missing:
        estimators, _ = fit_four_estimators(task, seed, checkpoint, est_args)
        for method in missing:
            field = raw_density_on_grid(estimators[method], method, grid)
            fields[method] = field
            if cache_dir is not None:
                np.savez_compressed(
                    density_cache_path(cache_dir, task, seed, checkpoint, method, est_args, grid),
                    density=field,
                )
    return fields


def raw_density_on_grid(estimator, method: str, grid: np.ndarray) -> np.ndarray:
    """在 raw-space 网格上取估计器的概率密度（与 job 的 fixed_grid_kl 同协议）。"""
    if method == "kde":
        _, log_d = estimator.evaluate(grid, return_density=False)
        scale = np.asarray(estimator.scaler.scale_, dtype=float)
        if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("KDE scaler 存在非正或非有限的 scale")
        log_d = log_d - np.log(scale).sum()
    else:
        log_d = estimator.evaluate_grid(grid, return_log_density=True)
    return np.exp(np.clip(np.asarray(log_d, dtype=float), -745.0, 709.0))


def pick_seed(df: pd.DataFrame, checkpoints: list[int]) -> int:
    """取参考成功目标总数最多的 seed（密度场无法跨 seed 平均，与论文单 seed 协议一致）。"""
    sub = df[(df["method"] == "kde") & (df["checkpoint"].isin(checkpoints))]
    totals = sub.groupby("seed")["reference_successes"].sum()
    if totals.empty:
        sys.exit("无法从 CSV 推断 seed（缺少 kde 行），请显式指定 --seed")
    return int(totals.idxmax())


# --------------------------------------------------------------------------- #
# fig5 风格的 2D 密度层级
# --------------------------------------------------------------------------- #
def quantile_to_level(density: np.ndarray, isoprop: np.ndarray) -> np.ndarray:
    """把 iso-proportion 层级换算为密度层级（与 seaborn 内部规则一致）。

    与 ``seaborn.distributions._DistributionPlotter._quantile_to_level``
    使用同一算法：密度值降序排列后按累积质量归一化，再按 ``1 - isoprop``
    取分位点。这样任意估计器的密度场都能用和 ``sns.kdeplot`` 相同的
    "等概率质量"层级来渲染，跨方法可比。

    与 seaborn 的差别：这里对层级做 ``np.unique`` 去重并保证严格递增。
    Histogram 的密度场是分段常数（至多 ``n_bins^2`` 个取值），重复层级会
    触发 matplotlib 的 "Contour levels must be increasing"；去重后层级数
    变少是该方法本身的分辨率限制，属于如实反映，不做插值美化。
    """
    values = np.ravel(np.asarray(density, dtype=float))
    values = values[np.isfinite(values)]
    sorted_values = np.sort(values)[::-1]
    if sorted_values.size == 0 or sorted_values.sum() <= 0:
        raise ValueError("密度场为空或总质量非正，无法换算层级")
    normalized = np.cumsum(sorted_values) / sorted_values.sum()
    idx = np.searchsorted(normalized, 1.0 - np.asarray(isoprop, dtype=float))
    levels = np.unique(np.take(sorted_values, idx, mode="clip"))
    if levels.size < 2:
        lo = float(np.min(values))
        hi = float(np.max(values))
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            raise ValueError("密度场取值退化（常数），无法构造层级")
        levels = np.array([lo, hi])
    return levels


def cmap_from_color(color: str):
    """复刻 seaborn 由颜色生成顺序色图的规则（失败时退回 white->color 渐变）。"""
    try:
        from seaborn.distributions import _DistributionPlotter
        return _DistributionPlotter._cmap_from_color(None, color)
    except Exception:  # pragma: no cover - 仅依赖 seaborn 私有 API 时的兜底
        return sns.blend_palette(["white", color], as_cmap=True)


# --------------------------------------------------------------------------- #
# fig5
# --------------------------------------------------------------------------- #
def build_figure5(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict:
    task = args.task
    progress = list(args.progress_checkpoints)
    labels = list(args.progress_labels)
    if len(progress) != len(labels):
        sys.exit("--progress-checkpoints 与 --progress-labels 数量必须一致")

    seed = int(df["seed"].iloc[0]) if args.seed is None else int(args.seed)
    if args.seed is None:
        seed = pick_seed(df, progress)
    print(f"[fig5] task={task} seed={seed} checkpoints={progress}", flush=True)

    # 固定目标空间（fixed eval 网格）为所有子图共享坐标范围
    lims = (args.axis_min, args.axis_max)
    res = args.grid_resolution
    xs = np.linspace(lims[0], lims[1], res)
    ys = np.linspace(lims[0], lims[1], res)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], axis=1)

    ref_frames: dict[int, pd.DataFrame] = {}
    density_fields: dict[tuple[str, int], np.ndarray] = {}
    ref_fields: dict[int, np.ndarray] = {}
    for ck in progress:
        ref = pd.read_csv(checkpoint_file(task, seed, ck))
        ref = ref[ref["termination"] == "reach target"].copy()
        if ref.empty:
            sys.exit(f"{task} seed={seed} checkpoint={ck} 没有成功目标，无法画 fig5")
        ref_frames[ck] = ref
        fitted = density_fields_for_checkpoint(
            task, seed, ck, grid, args, args.cache_dir
        )
        for method in args.methods:
            density_fields[(method, ck)] = fitted[method].reshape(res, res)
        ref_fields[ck] = oracle_reference_field(
            task, seed, ck, grid, args, args.cache_dir
        ).reshape(res, res)
        print(
            f"[fig5]   ckpt={ck} 参考成功目标 n={len(ref)}；"
            f"已完成 {len(args.methods)} 个估计器拟合",
            flush=True,
        )

    n_rows = len(args.methods) + 1
    n_cols = len(progress)
    tag = make_tag(args.methods)
    isoprop = np.linspace(0.05, 1.0, 10)  # sns.kdeplot 默认 levels=10, thresh=0.05
    fill_cmap = cmap_from_color("C0")
    scatter_color, truth_color = "C1", "C0"

    # 与论文 fig5 相同：单 panel 4in x 4in（论文是 2 行 x 5 列 -> figsize=(20, 8)）
    f, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(4.0 * n_cols, 4.0 * n_rows)
    )
    row_labels = [r"Reference ($p_{ag}$)"] + [METHOD_LABELS[m] for m in args.methods]

    for col, ck in enumerate(progress):
        ref = ref_frames[ck]
        ref_field = ref_fields[ck]
        # ---- 第一行：参考分布 p_ag -------------------------------------------
        ax = axes[0, col]
        if args.reference_renderer == "oracle":
            # 与 MC-KL / fixed_grid_kl 同一个 oracle：KDEEvaluator(bw=args.bandwidth)。
            # 用与其它行相同的 iso-proportion 层级规则填充，保证视觉语言一致。
            ax.contourf(
                gx, gy, ref_field,
                levels=quantile_to_level(ref_field, isoprop),
                cmap=fill_cmap,
            )
        else:  # seaborn：完全复刻论文 fig5 的 kdeplot(fill=True) 画法（Scott 带宽）
            sns.kdeplot(data=ref, x="x", y="y", fill=True, color=truth_color, ax=ax)
        sns.scatterplot(data=ref, x="x", y="y", color=scatter_color, ax=ax)
        # ---- 其余行：各估计器的自身密度（同一 iso-proportion 层级） ---------
        for row, method in enumerate(args.methods, start=1):
            ax = axes[row, col]
            field = density_fields[(method, ck)]
            levels = quantile_to_level(field, isoprop)
            ax.contourf(gx, gy, field, levels=levels, cmap=fill_cmap)
            # 参考轮廓线：与第一行/指标同源的 oracle 场，便于直接读出估计偏差
            if args.reference_contour:
                ax.contour(
                    gx, gy, ref_field,
                    levels=quantile_to_level(ref_field, np.linspace(0.4, 0.95, 4)),
                    colors="0.15", linewidths=0.9, alpha=0.8,
                )
            sns.scatterplot(data=ref, x="x", y="y", color=scatter_color, ax=ax)

    # 与论文 fig5 相同：只有底行显示 x 标签、只有最左列显示 y 标签
    for row in range(n_rows):
        for col in range(n_cols):
            ax = axes[row, col]
            ax.set_xlim(*lims)
            ax.set_ylim(*lims)
            ax.set_xlabel("x" if row == n_rows - 1 else "")
            ax.set_ylabel("y" if col == 0 else "")
            if col != 0:
                ax.tick_params(labelleft=False)
            if row != n_rows - 1:
                ax.tick_params(labelbottom=False)

    for col, lab in enumerate(labels):
        axes[0, col].set_title(f"Progress = {lab}", fontsize=16, pad=8)
    for row, lab in enumerate(row_labels):
        axes[row, -1].text(
            1.05, 0.5, lab, transform=axes[row, -1].transAxes,
            fontsize=16, ha="left", va="center", rotation=-90,
        )

    density_patch = Patch(facecolor="C0", alpha=0.4, label=r"$p_{ag}$ / $\hat{p}_{ag}$ density")
    scatter_handle = Line2D([0], [0], marker="o", color=scatter_color,
                            markerfacecolor=scatter_color, markersize=8,
                            linestyle="", label="Successful goals")
    handles = [density_patch, scatter_handle]
    if args.reference_contour:
        handles.append(Line2D([0], [0], color="0.15", linewidth=1.2,
                              label=r"oracle $p_{ag}$ contour"))
    axes[0, 0].legend(handles=handles, fontsize=12, loc="upper right")

    # 脚注：单 seed 协议与每列的参考成功目标数必须写在图上，避免被误读为 5-seed 平均
    counts = " / ".join(str(int(len(ref_frames[ck]))) for ck in progress)
    f.text(
        0.5, 0.005,
        f"vanilla SAC (no curriculum), {task.capitalize()}, single seed = {seed} "
        f"(chosen by total reference successes over the five columns). "
        f"Rows 2-5 fit $\\hat{{p}}_{{ag}}$ on strictly earlier evaluations with "
        f"$\\kappa$ = {args.kappa:g}; black contours in those rows are the same oracle "
        f"$p_{{ag}}$ scored in the KL metrics "
        f"({'oracle KDEEvaluator, bw = ' + format(args.bandwidth, 'g') if args.reference_renderer == 'oracle' else 'seaborn kdeplot default bandwidth'}). "
        f"Successful goals per column: {counts}.",
        ha="center", va="bottom", fontsize=13, color="0.25",
    )
    plt.tight_layout(rect=(0, 0.02, 1, 1))
    outs = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig5_density_{tag}_{task}.{ext}"
        f.savefig(path, format=ext, bbox_inches="tight", dpi=180)
        outs.append(path)
    plt.close(f)

    return {
        "seed": seed,
        "seed_selection": "argmax over seeds of total reference successes across the five columns",
        "progress_checkpoints": progress,
        "progress_labels": labels,
        "reference_successes": {str(ck): int(len(ref_frames[ck])) for ck in progress},
        "axis_limits": list(lims),
        "grid_resolution": int(res),
        "kappa": float(args.kappa),
        "kde_bandwidth": float(args.bandwidth),
        "gmm_n_components": int(args.n_components),
        "gmm_resample_size": int(args.resample_size),
        "histogram_n_bins": int(args.n_hist_bins),
        "reference_renderer": args.reference_renderer,
        "reference_contour_on_method_rows": bool(args.reference_contour),
        "outputs": [rel_or_abs(p) for p in outs],
    }


# --------------------------------------------------------------------------- #
# fig6
# --------------------------------------------------------------------------- #
def _kl_lineplot(d: pd.DataFrame, order: list[str], *, ylim=None, ax=None, title=None):
    """论文 fig6 的单面板画法：均值线 + 95% CI（seaborn 默认 errorbar）。"""
    ax = sns.lineplot(
        data=d,
        x="checkpoint",
        y="kl",
        hue="Estimation Method",
        hue_order=order,
        errorbar=("ci", 95),
        ax=ax,
    )
    ax.set_xlabel("Environment Steps")
    ax.set_ylabel(r"$D_{KL}(P_{ag} \| \hat{p}_{ag})$")
    if title is not None:
        ax.set_title(title, fontsize=18)
    if ylim is not None:
        ax.set_ylim(*ylim)
    sns.move_legend(ax, loc="upper right", fontsize=18)
    return ax


def build_figure6(df: pd.DataFrame, args: argparse.Namespace, out_dir: Path) -> dict:
    """论文 fig6 风格：KL 对训练步数的均值曲线 + 95% CI（5 seed）。

    主图 ``fig6_kl_curves_4methods_<task>.pdf`` 完全沿用论文 fig6 的单面板设置
    （``figsize=(10, 8)``、``sns.lineplot(errorbar=("ci", 95))``、
    ``move_legend(loc="upper right", fontsize=18)``）。

    与论文的唯一差异是 y 轴量程：本实验数据在训练最早期（<= 80k 步）存在
    若干量级异常值（GMM 在 40k 步的 (方法, checkpoint) 均值达 2.3e6），
    线性全量程会把其余曲线压成一条直线。因此：

    * 主图按 ``--zoom-from`` 之后的最大 (方法, checkpoint) 均值自动定量程，
      被裁掉的点全部落在训练最早期，并在图注中显式说明；
    * 另出 ``..._fulllog.{png,pdf}`` 用 symlog y 轴展示全量程、不丢任何点。

    这两个图互为补充，不是互相替代。
    """
    task = args.task
    tag = make_tag(args.methods)
    d = df.copy()
    d = d[d["method"].isin(args.methods)].copy()
    d["Estimation Method"] = d["method"].map(METHOD_LABELS)
    order = [METHOD_LABELS[m] for m in args.methods]

    means = d.groupby(["method", "checkpoint"])["kl"].mean()
    if args.zoom_ymax is None:
        # 训练进入稳定期后，所有 (方法, checkpoint) 均值都落在量程内；向上取整。
        late_means = means[means.index.get_level_values("checkpoint") >= args.zoom_from]
        if late_means.empty:
            sys.exit(f"--zoom-from={args.zoom_from} 之后没有任何 checkpoint")
        zoom_ymax = float(max(np.ceil(float(late_means.max())), 1.0))
        zoom_source = (
            f"auto: ceil of the largest per-(method, checkpoint) mean among "
            f"checkpoints >= {args.zoom_from // 1000}k"
        )
    else:
        zoom_ymax = float(args.zoom_ymax)
        zoom_source = "explicit --zoom-ymax"

    off_scale = [
        {"method": method, "checkpoint": int(ck), "mean_kl": float(value)}
        for (method, ck), value in means[means > zoom_ymax].items()
    ]
    late_means_all = means[means.index.get_level_values("checkpoint") >= args.zoom_from]

    # ---- 主图：论文 fig6 单面板设置 ------------------------------------------
    lo = float(np.nanmin(d["kl"]))
    f, ax = plt.subplots(figsize=(10, 8))
    _kl_lineplot(d, order, ylim=(lo - 0.05 * (zoom_ymax - lo), zoom_ymax), ax=ax)
    ax.axhline(0.0, color="0.5", linewidth=0.7, linestyle=":")
    if off_scale:
        worst = max(off_scale, key=lambda r: r["mean_kl"])
        max_off_ckpt = max(r["checkpoint"] for r in off_scale)
        f.text(
            0.5, -0.02,
            f"Linear y-axis clipped at y = {zoom_ymax:g} ({zoom_source}); "
            f"{len(off_scale)} of {len(means)} (estimator, checkpoint) means exceed it, "
            f"all at checkpoints <= {max_off_ckpt // 1000}k steps. Largest off-scale mean = "
            f"{worst['mean_kl']:.3g} ({METHOD_LABELS[worst['method']]} @ "
            f"{worst['checkpoint'] // 1000}k steps). See the companion *_fulllog figure for the "
            f"unclipped range; every off-scale value is listed in "
            f"run_config_fig5_fig6.json.",
            ha="center", va="top", fontsize=13, color="0.25",
        )
    plt.tight_layout()
    outs = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig6_kl_curves_{tag}_{task}.{ext}"
        f.savefig(path, format=ext, bbox_inches="tight", dpi=180)
        outs.append(path)
    plt.close(f)

    # ---- 伴随诊断图：symlog 全量程，不丢任何点 ------------------------------
    f, ax = plt.subplots(figsize=(10, 8))
    _kl_lineplot(d, order, ax=ax)
    ax.set_yscale("symlog")
    ax.set_ylim(bottom=lo - 0.1 * abs(lo), top=float(np.ceil(means.max())))
    ax.set_title("All 99 checkpoints, symlog y-axis (no values clipped)", fontsize=18)
    ax.axhline(0.0, color="0.5", linewidth=0.7, linestyle=":")
    plt.tight_layout()
    full_outs = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig6_kl_curves_{tag}_{task}_fulllog.{ext}"
        f.savefig(path, format=ext, bbox_inches="tight", dpi=180)
        full_outs.append(path)
    plt.close(f)

    summary = (
        d.groupby("method")["kl"]
        .agg(["mean", "std", "min", "max"])
        .reindex(args.methods)
        .round(4)
    )
    return {
        "n_rows": int(len(d)),
        "n_checkpoints": int(d["checkpoint"].nunique()),
        "seeds": sorted(int(s) for s in d["seed"].unique()),
        "zoom_ymax": zoom_ymax,
        "zoom_ymax_source": zoom_source,
        "zoom_from": int(args.zoom_from),
        "n_method_checkpoint_means_total": int(len(means)),
        "off_scale_means": sorted(off_scale, key=lambda r: -r["mean_kl"]),
        "off_scale_max_checkpoint": (
            int(max(r["checkpoint"] for r in off_scale)) if off_scale else None
        ),
        "off_scale_all_at_or_before_zoom_from": bool(
            all(r["checkpoint"] < args.zoom_from for r in off_scale)
        ) if off_scale else True,
        "kl_mean_std_min_max_by_method": summary.to_dict(orient="index"),
        "max_mean_kl_after_zoom_from": float(late_means_all.max()),
        "outputs": [rel_or_abs(p) for p in outs],
        "full_range_outputs": [rel_or_abs(p) for p in full_outs],
    }


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="push", help="任务（当前数据可用于 push；slide 尚未跑完）")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="replacement 结果 CSV")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--methods", nargs="+", choices=ALL_METHODS, default=list(METHOD_ORDER),
                    help="参与对比的估计器；默认 kde gmm gaussian。Histogram 因 MC-KL 对它失效"
                         "（盒函数在 oracle 支撑外取 0）而不在默认列表内")
    ap.add_argument("--fig", choices=["fig5", "fig6", "all"], default="fig6",
                    help="默认只画 fig6；fig5 见文件头口径限制，需显式指定才画")
    ap.add_argument("--seed", type=int, default=None, help="fig5 的 seed；默认取参考成功目标最多的 seed")
    ap.add_argument("--progress-checkpoints", type=int, nargs="+",
                    default=[100000, 150000, 210000, 250000, 310000])
    ap.add_argument("--progress-labels", nargs="+",
                    default=["10%", "15%", "20%", "25%", "30%"])
    ap.add_argument("--grid-resolution", type=int, default=200)
    ap.add_argument("--axis-min", type=float, default=-0.25)
    ap.add_argument("--axis-max", type=float, default=0.25)
    ap.add_argument("--reference-contour", action=argparse.BooleanOptionalAction, default=True,
                    help="方法子图上叠加 oracle KDE 轮廓线（默认开启）")
    ap.add_argument("--reference-renderer", choices=["oracle", "seaborn"], default="oracle",
                    help="参考分布画法：oracle = 与 MC-KL 指标同一个 KDEEvaluator(bw=bandwidth)；"
                         "seaborn = 论文 fig5 原样的 sns.kdeplot(fill=True)")
    ap.add_argument("--cache-dir", type=Path, default=None,
                    help="fig5 密度场缓存目录；默认 <out-dir>/density_cache，传空字符串可禁用")
    ap.add_argument("--no-cache", action="store_true", help="禁用密度场缓存（每次重新拟合）")
    ap.add_argument("--zoom-ymax", type=float, default=None,
                    help="fig6 右面板线性 y 轴上界；默认按 --zoom-from 之后的 (方法, checkpoint) 均值自动确定")
    ap.add_argument("--zoom-from", type=int, default=100000,
                    help="fig6 自动量程的起始 checkpoint；该步数之后的所有方法均值都应落在图内")
    # 估计器超参（与 replacement_run_all100.py 默认值一致）
    ap.add_argument("--kappa", type=float, default=0.9)
    ap.add_argument("--bandwidth", type=float, default=0.2)
    ap.add_argument("--n-components", type=int, default=5)
    ap.add_argument("--resample-size", type=int, default=1000)
    ap.add_argument("--gmm-reg-covar", type=float, default=1e-6)
    ap.add_argument("--n-hist-bins", type=int, default=10)
    ap.add_argument("--gaussian-reg-covar", type=float, default=1e-6)
    ap.add_argument("--random-state", type=int, default=0)
    args = ap.parse_args()

    args.methods = list(dict.fromkeys(args.methods))  # 去重并保序
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.no_cache:
        args.cache_dir = None
    elif args.cache_dir is None:
        args.cache_dir = args.out_dir / "density_cache"
    print(f"density cache: {args.cache_dir}", flush=True)
    sns.set_theme(context="notebook", style="darkgrid", font_scale=2.0)

    df = load_metrics(args.csv, args.task)
    available = set(df["method"].unique())
    missing_methods = [m for m in args.methods if m not in available]
    if missing_methods:
        sys.exit(f"CSV 中缺少这些方法的数据: {missing_methods}（可用: {sorted(available)}）")
    print(
        f"loaded {len(df)} ok rows: task={args.task}, "
        f"methods={sorted(df['method'].unique())}, "
        f"seeds={sorted(df['seed'].unique())}, ckpts={df['checkpoint'].nunique()}",
        flush=True,
    )

    config = {
        "input_csv": rel_or_abs(args.csv),
        "task": args.task,
        "protocol": "vanilla SAC fixed-evaluation CSVs, p̂_ag fitted on historical "
                    "successful goals with κ-discont, KL vs current-checkpoint oracle KDE",
        "paper_notebooks": PAPER_NOTEBOOKS,
        "paper_figures": PAPER_FIGURES,
        "fig5": None,
        "fig6": None,
    }
    if args.fig in ("all", "fig5"):
        print(
            "[fig5] 注意：非课程（vanilla SAC）轨迹没有 behavioral-goal 采样日志，"
            "本图只能画「历史评估拟合的 p̂_ag vs 当前 checkpoint 的 oracle p_ag」，"
            "不能反映课程采样效果。详见脚本 docstring 的口径限制。",
            flush=True,
        )
        config["fig5"] = build_figure5(df, args, args.out_dir)
    if args.fig in ("all", "fig6"):
        config["fig6"] = build_figure6(df, args, args.out_dir)

    (args.out_dir / "run_config_fig5_fig6.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n"
    )
    print(f"all done -> {rel_or_abs(args.out_dir)}", flush=True)


if __name__ == "__main__":
    main()
