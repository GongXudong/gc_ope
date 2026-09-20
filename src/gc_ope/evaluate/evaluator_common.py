"""估计器共用的数据准备函数。"""

from __future__ import annotations

import numpy as np

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)


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
