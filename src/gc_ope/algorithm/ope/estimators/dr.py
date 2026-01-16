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
                # Kernel-based cumulative weights (use log-space to avoid underflow)
                similarity = self._compute_kernel_similarity(
                    inputs, idxs, bandwidth
                )
                # Add epsilon to avoid log(0) for bounded kernels
                log_sim = np.log(similarity + 1e-10)
                log_cum_weights = np.cumsum(log_sim)
                # Clip to avoid overflow/underflow
                log_cum_weights = np.clip(log_cum_weights, -20.0, 10.0)
                w_step = np.exp(log_cum_weights)
            else:
                # Log-probability based cumulative weights
                logp_b = inputs.behavior_log_prob[idxs]
                logp_e = inputs.eval_log_prob[idxs]
                ratios = np.exp(np.clip(logp_e - logp_b, -10.0, 5.0))
                w_step = np.cumprod(ratios)
                w_step = np.clip(w_step, 0.0, 1e4)

            # Previous step weights (w_{-1} = 1)
            w_prev = np.concatenate([[1.0], w_step[:-1]])

            if self.self_normalize:
                # For SN-DR: use a modified normalization that doesn't explode
                # The issue is w_prev[0]=1.0 is a boundary condition, not a weight
                # We normalize w_step, and scale w_prev consistently
                mean_w = w_step.mean() + 1e-10
                w_step_norm = w_step / mean_w
                # For w_prev, normalize the actual weights but keep w_prev[0] = 1
                # This maintains the DR structure while avoiding explosion
                w_prev_norm = np.concatenate([[1.0], w_step_norm[:-1]])
                # DR term with normalized weights
                term = w_step_norm * (r - q_sa) + w_prev_norm * v_eval
                all_values.append(np.sum(discounts * term))
            else:
                # DR term: w_t * (r_t - Q(s_t, a_t)) + w_{t-1} * V(s_t)
                term = w_step * (r - q_sa) + w_prev * v_eval
                all_values.append(np.sum(discounts * term))

        values = np.asarray(all_values, dtype=np.float32)
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
