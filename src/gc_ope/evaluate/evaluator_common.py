"""估计器共用的数据准备函数。"""

from __future__ import annotations

import numpy as np

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)


class InsufficientSamples(ValueError):
    """数据尚不足以拟合；离线记录跳过，在线退回环境原本的目标采样。"""


def validate_hidden_layers(values):
    """一个元素表示一个隐藏层；拒绝含糊的单个宽度，避免隐式重复层数。"""
    if not isinstance(values, (list, tuple)) or not values:
        raise ValueError("hidden_layer_sizes 必须为非空列表或元组，例如 [16, 16]")
    if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1
           for value in values):
        raise ValueError("每个隐藏层宽度必须为正整数")
    return tuple(int(value) for value in values)


def uniform_grid_kl(log_density, dV, u_density):
    """保留课程学习的网格积分协议，使用 logsumexp 避免密度下溢。"""
    from scipy.special import logsumexp
    if dV <= 0 or u_density <= 0 or not np.isfinite([dV, u_density]).all():
        raise ValueError("面积和均匀密度必须有限且为正")
    values = np.asarray(log_density, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise FloatingPointError("网格对数密度必须非空且有限")
    normalized_log_density = values - logsumexp(values) - np.log(dV)
    return float(u_density * np.sum(np.log(u_density) - normalized_log_density) * dV)


def positive_samples_and_weights(
    container: EvaluationResultContainer,
) -> tuple[np.ndarray, np.ndarray]:
    """从评估结果容器中取成功目标的原始坐标与时间权重。"""
    all_samples = np.asarray(container.desired_goal_list, dtype=float)
    flags = np.asarray(container.success_list, dtype=bool)
    if all_samples.ndim != 2 or all_samples.shape[0] == 0:
        raise ValueError("no evaluation samples are available")
    if flags.ndim != 1 or flags.shape[0] != all_samples.shape[0]:
        raise ValueError("success_list must align with desired_goal_list")
    if not np.all(np.isfinite(all_samples)):
        raise ValueError("evaluation samples must be finite")
    positive = all_samples[flags]
    if positive.shape[0] == 0:
        raise ValueError("no successful evaluation samples are available")
    if isinstance(container, WeightedEvaluationResultContainer):
        all_weights = np.asarray(container.desired_goal_weights, dtype=float)
        if all_weights.ndim != 1 or all_weights.shape[0] != all_samples.shape[0]:
            raise ValueError("desired_goal_weights must align with desired_goal_list")
        weights = all_weights[flags]
    elif isinstance(container, EvaluationResultContainer):
        weights = np.ones(positive.shape[0], dtype=float)
    else:
        raise ValueError(f"cannot process container type: {type(container)}")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample weights must be finite and non-negative")
    if not np.any(weights > 0):
        raise ValueError("at least one successful sample weight must be positive")
    return positive, weights
