"""Doubly Robust (DR) estimators.

This module provides DR and Self-Normalized DR estimators for OPE.
DR combines importance sampling with a control variate (Q-function)
to reduce variance while maintaining unbiasedness.
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


class DREstimator(BaseISEstimator):
    """Doubly Robust (DR) estimator.

    DR combines importance sampling with a control variate (Q-function):

        V_τ^DR = sum_t γ^t [w_t * (r_t - Q(s_t, a_t)) + w_{t-1} * Q(s_t, π_e(s_t))]

    where w_t = prod_{k=0}^t [π_e(a_k|s_k) / π_b(a_k|s_k)] and w_{-1} = 1.

    DR is "doubly robust" because it is unbiased if either the importance
    weights are correct or the Q-function is correct.

    When use_kernel=True, uses kernel similarity instead of density ratio.
    """

    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        """Compute trajectory-level DR values.

        Args:
            inputs: OPE inputs.

        Returns:
            Array of DR values (M,).
        """
        traj_map = traj_slices(inputs.traj_id)
        gamma = inputs.gamma if inputs.gamma is not None else self.gamma
        bandwidth = self._get_bandwidth(inputs) if self.use_kernel else None

        all_weights = []
        all_values = []

        for idxs in traj_map.values():
            r = inputs.rewards[idxs]
            q_sa = inputs.q_sa_behavior[idxs]
            v_eval = inputs.q_sa_eval[idxs]
            discounts = np.power(gamma, np.arange(len(r)))

            if self.use_kernel:
                # Kernel-based cumulative weights
                similarity = self._compute_kernel_similarity(
                    inputs, idxs, bandwidth
                )
                w_step = np.cumprod(similarity)
            else:
                # Log-probability based cumulative weights
                logp_b = inputs.behavior_log_prob[idxs]
                logp_e = inputs.eval_log_prob[idxs]
                ratios = np.exp(np.clip(logp_e - logp_b, -10.0, 5.0))
                w_step = np.cumprod(ratios)
                w_step = np.clip(w_step, 0.0, 1e4)

            # Previous step weights (w_{-1} = 1)
            w_prev = np.concatenate([[1.0], w_step[:-1]])

            # DR term: w_t * (r_t - Q(s_t, a_t)) + w_{t-1} * V(s_t)
            term = w_step * (r - q_sa) + w_prev * v_eval

            # Store mean weight for self-normalization
            all_weights.append(w_step.mean())
            all_values.append(np.sum(discounts * term))

        weights = np.asarray(all_weights, dtype=np.float32)
        values = np.asarray(all_values, dtype=np.float32)

        if self.self_normalize:
            norm_factor = weights.mean() + 1e-10
            values = values / norm_factor

        return values


class SelfNormalizedDR(DREstimator):
    """Self-Normalized DR estimator.

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
