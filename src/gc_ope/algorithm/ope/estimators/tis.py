"""Trajectory-wise Importance Sampling (TIS) estimators.

This module provides TIS and Self-Normalized TIS estimators for OPE.
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


class TISEstimator(BaseISEstimator):
    """Trajectory-wise Importance Sampling (TIS) estimator.

    TIS estimates policy value using trajectory-level importance weights:

        V^TIS = (1/M) * sum_τ w_τ * G_τ

    where w_τ = prod_t [π_e(a_t|s_t) / π_b(a_t|s_t)]

    When use_kernel=True, uses kernel similarity instead of density ratio.
    """

    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        """Compute trajectory-level weighted returns.

        Args:
            inputs: OPE inputs.

        Returns:
            Array of weighted returns (M,).
        """
        traj_map = traj_slices(inputs.traj_id)
        gamma = inputs.gamma if inputs.gamma is not None else self.gamma
        bandwidth = self._get_bandwidth(inputs) if self.use_kernel else None

        weights = []
        returns = []

        for idxs in traj_map.values():
            r = inputs.rewards[idxs]
            discounts = np.power(gamma, np.arange(len(r)))
            G = np.sum(discounts * r)
            returns.append(G)

            if self.use_kernel:
                # Kernel-based weight
                similarity = self._compute_kernel_similarity(
                    inputs, idxs, bandwidth
                )
                # Trajectory weight = product of step similarities
                weight = similarity.prod()
            else:
                # Log-probability based weight
                logp_b = inputs.behavior_log_prob[idxs]
                logp_e = inputs.eval_log_prob[idxs]
                log_weights = np.clip(logp_e - logp_b, -10.0, 10.0)
                weight = np.exp(log_weights.sum())

                if not np.isfinite(weight):
                    weight = 0.0
                weight = np.clip(weight, 0.0, 1e4)

            weights.append(weight)

        weights = np.asarray(weights, dtype=np.float32)
        returns = np.asarray(returns, dtype=np.float32)

        if self.self_normalize:
            weights = self_normalize_weights(weights)

        return weights * returns


class SelfNormalizedTIS(TISEstimator):
    """Self-Normalized TIS estimator.

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
