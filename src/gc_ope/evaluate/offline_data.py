"""Push 离线实验的数据边界：每个 fixed checkpoint 抽 100 条并累计。"""

from dataclasses import dataclass
from pathlib import Path
import re
import numpy as np
import pandas as pd


@dataclass
class EvaluationBatch:
    goals: np.ndarray
    successes: np.ndarray
    weights: np.ndarray

    def fill(self, estimator):
        """一次装入绝对时间权重，避免容器重复执行批次折扣。"""
        container = estimator.eval_res_container
        container.reset()
        container.add_batch(self.goals.tolist(), self.successes.tolist(),
                            [0.0] * len(self.goals), [0.0] * len(self.goals),
                            self.weights.tolist())


def fixed_files(checkpoint_root, seed):
    """只读取 vanilla Push/SAC 的 fixed CSV，不混入 random 评估。"""
    directory = Path(checkpoint_root) / "my_push" / "sac" / f"seed_{seed}"
    pattern = re.compile(r"rl_model_(\d+)_steps_eval_res_on_fixed\.csv$")
    files = {int(match[1]): path for path in directory.glob("*.csv")
             if (match := pattern.fullmatch(path.name))}
    if not files:
        raise FileNotFoundError(f"未找到 fixed 评估数据：{directory}")
    return dict(sorted(files.items()))


def sampled_indices(n_rows, seed, source_checkpoint, sampling_seed=0, n_samples=100):
    """随机流不依赖方法、待评估 checkpoint 或调度顺序。"""
    if n_rows <= 0 or n_samples <= 0:
        raise ValueError("不能从空文件或按非正数量抽样")
    rng = np.random.default_rng(np.random.SeedSequence([sampling_seed, 1, seed, source_checkpoint]))
    return rng.integers(0, n_rows, size=n_samples)


def read_evaluation(path):
    frame = pd.read_csv(path)
    goals = frame[["x", "y"]].to_numpy(dtype=float)
    if not len(goals) or not np.isfinite(goals).all() or frame["termination"].isna().any():
        raise ValueError(f"评估文件为空或包含缺失/非有限数据：{path}")
    # Push 的 z 固定；沿用旧 desired_goal_utils 的平面目标空间定义。
    if "z" in frame and (not np.isfinite(frame["z"]).all() or np.ptp(frame["z"]) > 1e-6):
        raise ValueError(f"此入口只支持固定高度的 Push 目标：{path}")
    return goals, frame["termination"].to_numpy() == "reach target"


def load_pair(checkpoint_root, seed, checkpoint, *, kappa=0.9, sampling_seed=0, n_samples=100):
    """估计侧包含当前 checkpoint 的抽样；参考侧仅含当前全量数据。"""
    files = fixed_files(checkpoint_root, seed)
    if checkpoint not in files:
        raise FileNotFoundError(f"缺少 seed {seed} 的 checkpoint {checkpoint}")
    all_goals, all_successes, all_weights = [], [], []
    reference = None
    for source_step, path in files.items():
        if source_step > checkpoint:
            break
        goals, successes = read_evaluation(path)
        if source_step == checkpoint:
            reference = EvaluationBatch(goals, successes, np.ones(len(goals)))
        indices = sampled_indices(len(goals), seed, source_step, sampling_seed, n_samples)
        all_goals.append(goals[indices])
        all_successes.append(successes[indices])
        all_weights.extend([kappa ** ((checkpoint - source_step) / 10000)] * len(indices))
    history = EvaluationBatch(np.concatenate(all_goals), np.concatenate(all_successes), np.array(all_weights))
    return history, reference
