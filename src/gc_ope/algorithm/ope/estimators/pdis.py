"""Per-Decision Importance Sampling (PDIS) estimators.

This module provides PDIS and Self-Normalized PDIS estimators for OPE.
PDIS uses step-wise cumulative weights instead of trajectory-level weights,
resulting in lower variance than TIS.
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ..ope_input import OPEInputs
from .base import (
    BaseISEstimator,
    EstimateResult,
    traj_slices,
    self_normalize_weights,
)


class PDISEstimator(BaseISEstimator):
    """Per-Decision Importance Sampling (PDIS) estimator.

    PDIS estimates policy value using step-wise cumulative importance weights:

        V^PDIS = (1/M) * sum_τ sum_t γ^t * w_{0:t} * r_t

    where w_{0:t} = prod_{t'=0}^t [π_e(a_t'|s_t') / π_b(a_t'|s_t')]

    PDIS has lower variance than TIS because weights grow more slowly.
    When use_kernel=True, uses kernel similarity instead of density ratio.
    """

    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        """Compute trajectory-level PDIS values.

        Args:
            inputs: OPE inputs.

        Returns:
            Array of PDIS values (M,).
        """
        traj_map = traj_slices(inputs.traj_id)
        gamma = inputs.gamma if inputs.gamma is not None else self.gamma
        bandwidth = self._get_bandwidth(inputs) if self.use_kernel else None

        all_weights = []
        all_values = []

        for idxs in traj_map.values():
            r = inputs.rewards[idxs]
            discounts = np.power(gamma, np.arange(len(r)))

            if self.use_kernel:
                # Kernel-based cumulative weights
                similarity = self._compute_kernel_similarity(
                    inputs, idxs, bandwidth
                )
                cum_weights = np.cumprod(similarity)
            else:
                # Log-probability based cumulative weights
                logp_b = inputs.behavior_log_prob[idxs]
                logp_e = inputs.eval_log_prob[idxs]
                log_ratios = np.clip(logp_e - logp_b, -10.0, 10.0)
                log_cum_weights = np.cumsum(log_ratios)
                log_cum_weights = np.clip(log_cum_weights, -20.0, 10.0)
                cum_weights = np.exp(log_cum_weights)
                cum_weights = np.clip(cum_weights, 0.0, 1e4)

            # Store mean weight for self-normalization
            all_weights.append(cum_weights.mean())
            # Compute PDIS value for this trajectory
            all_values.append(np.sum(discounts * cum_weights * r))

        weights = np.asarray(all_weights, dtype=np.float32)
        values = np.asarray(all_values, dtype=np.float32)

        if self.self_normalize:
            # Normalize by mean weight across trajectories
            norm_factor = weights.mean() + 1e-10
            values = values / norm_factor

        return values


class SelfNormalizedPDIS(PDISEstimator):
    """Self-Normalized PDIS estimator.

    Automatically enables self-normalization to bound variance.
    Trades unbiasedness for numerical stability.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        use_kernel: bool = True,
        kernel: str = "gaussian",
        bandwidth: Union[float, str] = "auto",
    ):
        super().__init__(
            gamma=gamma,
            use_kernel=use_kernel,
            kernel=kernel,
            bandwidth=bandwidth,
            self_normalize=True,
        )
