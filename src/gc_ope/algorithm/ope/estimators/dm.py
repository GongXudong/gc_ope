"""Direct Method (DM) estimator.

This module provides the DM estimator for OPE, which uses a learned
Q-function to estimate policy value directly.
"""

from __future__ import annotations

import numpy as np

from ..ope_input import OPEInputs
from .base import BaseOPEEstimator, EstimateResult


class DMEstimator(BaseOPEEstimator):
    """Direct Method (DM) estimator.

    DM estimates policy value using the FQE Q-function:

        V^DM = (1/N) * sum_i Q(s_i, π_eval(s_i))

    If initial_only=True, only uses initial states (step_index == 0).

    DM is a biased estimator but has low variance.
    """

    def __init__(self, gamma: float = 0.99, initial_only: bool = False):
        """Initialize DM estimator.

        Args:
            gamma: Discount factor.
            initial_only: If True, only use initial states.
        """
        super().__init__(gamma)
        self.initial_only = initial_only

    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        """Compute Q-values for evaluation policy actions.

        Args:
            inputs: OPE inputs.

        Returns:
            Array of Q-values.
        """
        if self.initial_only:
            mask = inputs.step_index == 0
            values = inputs.q_sa_eval[mask]
        else:
            values = inputs.q_sa_eval

        return np.asarray(values, dtype=np.float32)
