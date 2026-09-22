"""只用已抽到的历史记录做留出验证，不访问全量参考分布的拟合结果。"""

import numpy as np
from gc_ope.evaluate.offline_data import EvaluationBatch, fixed_files, read_evaluation, sampled_indices
from gc_ope.evaluate.evaluator_common import InsufficientSamples


def sampled_history(checkpoint_root, seed, checkpoint, kappa=.9, sampling_seed=0):
    """返回原协议的 100 条/轮记录，以及 (来源 checkpoint, CSV 行号)。

    有放回抽样会重复抽到同一条评估。保留身份是为了让这些副本进入同一侧，
    避免训练集里已见过的那一次评估又被当作独立验证数据。
    """
    goals, flags, weights, identities = [], [], [], []
    files = fixed_files(checkpoint_root, seed)
    if checkpoint not in files:
        raise FileNotFoundError(f"缺少 checkpoint {checkpoint}")
    for step, path in files.items():
        if step > checkpoint:
            break
        x, y = read_evaluation(path)
        indices = sampled_indices(len(x), seed, step, sampling_seed, 100)
        goals.append(x[indices])
        flags.append(y[indices])
        weights.extend([kappa ** ((checkpoint - step) / 10000)] * len(indices))
        identities.extend((step, int(i)) for i in indices)
    return (EvaluationBatch(np.concatenate(goals), np.concatenate(flags), np.array(weights)),
            np.asarray(identities, dtype=np.int64))


def split_successes(history, identities, random_state, fraction=.2):
    """对历史成功评估的身份分组后留出 20%；原时间权重和重复次数均保留。

    同一个目标在不同 checkpoint 的评估是不同观察，允许分到两侧。
    本接口只用于成功目标密度模型；不改变 NN 使用成功/失败标签的协议。
    """
    identities = np.asarray(identities)
    if identities.shape != (len(history.goals), 2) or not 0 < fraction < 1:
        raise ValueError("样本身份形状或留出比例不正确")
    positive = np.flatnonzero(history.successes)
    groups, inverse = np.unique(identities[positive], axis=0, return_inverse=True)
    if len(groups) < 10:
        raise InsufficientSamples("历史成功评估身份不足 10 个，不能可靠留出")
    permutation = np.random.default_rng(random_state).permutation(len(groups))
    n_valid = max(2, int(np.ceil(len(groups) * fraction)))
    valid_mask = np.isin(inverse, permutation[:n_valid])
    train_indices, valid_indices = positive[~valid_mask], positive[valid_mask]

    def take(indices):
        return EvaluationBatch(history.goals[indices], history.successes[indices], history.weights[indices])

    return take(train_indices), take(valid_indices), train_indices, valid_indices


def weighted_nll(model, batch):
    """在同一个 raw 目标空间计分；密度可以大于 1，因此 NLL 可以为负。"""
    values = model.log_density(batch.goals)
    if not np.isfinite(values).all() or np.any(batch.weights <= 0):
        raise FloatingPointError("验证对数密度或权重无效")
    return -float(np.average(values, weights=batch.weights))


def select_candidates(rows, candidates, expected_cases):
    """只按预先指定的全部历史留出任务的平均 NLL 选配置，不读取 KL 列。

    每个 checkpoint/划分等权，避免晚期样本多而完全主导选择。
    任一任务出错或缺失的候选没有资格获选；不通过删掉坏点提高均值。
    """
    scores, selected = {}, {}
    for method, variants in candidates.items():
        scores[method] = {}
        for name in variants:
            records = [row for row in rows if row["method"] == method and row["candidate"] == name]
            cases = {(row["seed"], row["checkpoint"], row["split_seed"]) for row in records}
            if (cases == set(expected_cases) and len(records) == len(expected_cases)
                    and all(row["status"] == "ok" and np.isfinite(row["valid_nll"]) for row in records)):
                scores[method][name] = float(np.mean([row["valid_nll"] for row in records]))
        if not scores[method]:
            raise RuntimeError(f"{method} 没有覆盖全部历史留出任务的有效候选")
        selected[method] = min(scores[method], key=scores[method].get)
    return selected, scores
