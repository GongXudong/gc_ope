"""一次性：生成 fig5 所需的密度网格数据 density_heatmap_6methods.csv。

对 push/slide 各自"成功数最多的 seed"（可 --pick-seed 覆盖）在 500k
checkpoint：
- 用与 replacement_run_job.py 完全一致的协议（AC-2.1/2.2）拟合 6 个估计器；
- 在 80x80 raw-space 网格上取各估计器 raw-space log 密度；
- Reference = 成功目标的经验 KDE（与 MC-KL 的 oracle 同协议）。

本文件是数据生成脚本（只跑一次，输出 CSV 后 fig5/fig6 绘图脚本
plot_replacement_fig5_fig6.py 只读 CSV、不重训）。
"""
import sys; sys.path.insert(0, 'scripts')
import argparse
import numpy as np, pandas as pd
import plot_replacement_fig5_fig6 as p

ap = argparse.ArgumentParser()
ap.add_argument("--tasks", nargs="+", default=["push", "slide"])
ap.add_argument("--checkpoint", type=int, default=500000)
ap.add_argument("--pick-seed", type=int, default=None)
args_cli = ap.parse_args()

args = argparse.Namespace(
    tasks=args_cli.tasks, density_checkpoint=args_cli.checkpoint, pick_seed=args_cli.pick_seed,
    kappa=0.9, bandwidth=0.2, n_components=5, resample_size=1000, gmm_reg_covar=1e-6,
    n_hist_bins=10, gaussian_reg_covar=1e-6, nn_epochs=100, nn_lr=1e-3, nn_hidden=16,
    fm_epochs=80, fm_hidden=16, fm_samples=2000, random_state=0)
CSVOUT = p.ROOT / 'logs' / 'replacement_experiment' / 'plots' / 'density_heatmap_6methods.csv'
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity

rows = []
for task in args_cli.tasks:
    seed = p.pick_seed(task, args_cli.checkpoint, args_cli.pick_seed)
    ests, reference = p.build_estimators(task, seed, args_cli.checkpoint, args)
    success = reference.loc[reference["termination"] == "reach target", p.GOAL_COLUMNS].to_numpy(float)
    allg = reference[p.GOAL_COLUMNS].to_numpy(float)
    lo, hi = allg.min(0), allg.max(0)
    xs = np.linspace(lo[0], hi[0], p.GRID_RESOLUTION); ys = np.linspace(lo[1], hi[1], p.GRID_RESOLUTION)
    gx, gy = np.meshgrid(xs, ys)
    grid = np.stack([gx.ravel(), gy.ravel()], 1)
    sc = StandardScaler().fit(success)
    ref_log = KernelDensity(bandwidth=args.bandwidth, kernel="gaussian") \
        .fit(sc.transform(success)).score_samples(sc.transform(grid)) - np.log(sc.scale_).sum()
    panels = [("Reference", ref_log)]
    for m in p.METHOD_ORDER:
        panels.append((p.METHOD_LABELS[m], p.raw_grid_density(ests[m], m, grid)))
    for label, log_d in panels:
        for i in range(len(grid)):
            rows.append(dict(env=task, seed=seed, ix=int(i % p.GRID_RESOLUTION),
                             iy=int(i // p.GRID_RESOLUTION),
                             gx=float(grid[i, 0]), gy=float(grid[i, 1]),
                             method=label, log_density=float(log_d[i])))
    vmax = float(np.nanmax(np.concatenate([ld[np.isfinite(ld)] for _, ld in panels])))
    print(f"{task} seed={seed}: {len(success)} successes, log10(vmax)={np.log10(max(vmax, 1.0)):.2f}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(CSVOUT, index=False)
print("saved", CSVOUT, len(df), "rows")
