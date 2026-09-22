"""在全新输出目录执行旧 notebook 的原函数；不修改 notebook、旧 CSV 或 checkpoint。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    baseline, output = args.baseline.resolve(), args.output.resolve()
    # 旧函数会依次使用所有 checkpoint；先查完整输入，避免中途才发现缺 RB。
    for seed in range(1, 6):
        folder = baseline / f"checkpoints/my_push/sac/seed_{seed}"
        for checkpoint in range(10000, 1000001, 10000):
            for name in [f"rl_model_{checkpoint}_steps_eval_res_on_fixed.csv",
                         f"rl_model_replay_buffer_{checkpoint}_steps.pkl"]:
                if not (folder / name).is_file():
                    raise FileNotFoundError(folder / name)
    # 禁止覆盖；每次复现保留独立随机状态和完整来源。
    output.mkdir(parents=True, exist_ok=False)
    notebook = baseline / "plots/p_ag_dist_between_truth_and_estimated_in_training/generate_eval_data_for_sac.ipynb"
    cells = json.loads(notebook.read_text())["cells"]
    function = next("".join(c["source"]) for c in cells
                    if "".join(c["source"]).startswith("def generate_eval_data("))
    (output / "generate_eval_data_original.py").write_text(function)
    revision = subprocess.check_output(["git", "-C", str(baseline), "rev-parse", "dev"], text=True).strip()
    archive = subprocess.check_output(["git", "-C", str(baseline), "archive", revision, "src"])
    snapshot = output / "old_source"
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(snapshot, filter="data")
    legacy_dir = notebook.parent / "my_push/sac/eval_data"
    old_paths = sorted(legacy_dir.glob("my_push_sac_seed_*_kde_0_9_eval_res_in_training.csv"))
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = {str(p): digest(p) for p in [notebook, *old_paths]}
    manifest = dict(old_revision=revision, random_seed=args.random_seed,
                    rng="每个 seed 独立进程，从 random_seed + seed 初始化 numpy 全局随机流",
                    notebook_and_original_csv=hashes, function_sha256=digest(output / "generate_eval_data_original.py"))
    manifest["runtime"] = {name: version(name) for name in ["numpy", "scipy", "scikit-learn", "stable-baselines3"]}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    # 直接执行提取的原函数，保留 RB 抽样、历史抽样、RB MC、历史 MC 的顺序。
    driver = '''import os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from gc_ope.evaluate.utils.kde_dist import kl_divergence_kde_3d_monte_carlo
from gc_ope.evaluate.utils.get_kde_estimator import get_kde_estimator_for_replay_buffer, get_kde_estimator_for_eval_res, get_kde_estimator_for_historical_eval
PROJECT_ROOT_DIR = Path(sys.argv[1])
np.random.seed(int(sys.argv[3]) + int(sys.argv[2]))
exec(Path("../generate_eval_data_original.py").read_text())
generate_eval_data(exp_seed=int(sys.argv[2]), env_str="my_push", env_str_for_save_csv="my_push")
'''
    (output / "driver.py").write_text(driver)
    env = {**os.environ, "PYTHONPATH": str(snapshot / "src"), "OPENBLAS_NUM_THREADS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}

    def run(seed):
        folder = output / f"seed_{seed}"
        folder.mkdir()
        print(f"开始旧 KDE seed {seed}", flush=True)
        with (folder / "run.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(output / "driver.py"), str(baseline),
                            str(seed), str(args.random_seed)], cwd=folder, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f"完成旧 KDE seed {seed}", flush=True)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(run, range(1, 6)))
    import pandas as pd
    import numpy as np
    comparisons = []
    audits = []
    for seed in range(1, 6):
        name = f"my_push_sac_seed_{seed}_kde_0_9_eval_res_in_training.csv"
        old, new = pd.read_csv(legacy_dir / name), pd.read_csv(output / f"seed_{seed}" / name)
        if not new.evaluation_index.is_unique or not np.isfinite(new.to_numpy()).all():
            raise ValueError(f"seed {seed} 输出存在重复 checkpoint 或非有限值")
        audits.append(dict(seed=seed, processed=100, valid_rows=len(new),
                           skipped=sorted(set(range(10000, 1000001, 10000)) - set(new.evaluation_index))))
        joined = old.merge(new, on="evaluation_index", suffixes=("_old", "_reproduced"), validate="one_to_one")
        joined.to_csv(output / f"comparison_seed{seed}.csv", index=False)
        for column in ["$D_{KL}$ [his]", "$D_{KL}$ [rb]"]:
            a, b = joined[column + "_old"], joined[column + "_reproduced"]
            comparisons.append(dict(seed=seed, metric=column, old_rows=len(old), new_rows=len(new),
                                    overlap=len(joined), old_mean=a.mean(), reproduced_mean=b.mean(),
                                    mae=(a-b).abs().mean(), max_abs_difference=(a-b).abs().max()))
    pd.DataFrame(comparisons).to_csv(output / "comparison_summary.csv", index=False)
    assert all(digest(Path(path)) == value for path, value in hashes.items()), "旧文件发生变化"
    (output / "audit.json").write_text(json.dumps(dict(original_hashes_unchanged=True, seeds=audits),
                                                    ensure_ascii=False, indent=2))
    print("五个 seed 复现与对照完成；旧 notebook、旧 CSV 摘要未变。", flush=True)


if __name__ == "__main__":
    main()
