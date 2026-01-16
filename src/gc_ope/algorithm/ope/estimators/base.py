"""Base classes for OPE estimators.

This module provides abstract base classes for OPE estimators, including
support for kernel-based importance sampling and self-normalization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Literal, Union

import numpy as np
from scipy import stats

from ..ope_input import OPEInputs
from ..kernel_utils import get_similarity
from ..bandwidth_selection import select_bandwidth


# CI method types
CI_METHOD = Literal["bootstrap", "normal", "t_test"]


@dataclass
class EstimateResult:
    """Result of an OPE estimate with confidence interval.

    Attributes:
        mean: Mean estimate of policy value.
        ci_lower: Lower bound of confidence interval (default: 95%).
        ci_upper: Upper bound of confidence interval (default: 95%).
    """
    mean: float
    ci_lower: float
    ci_upper: float


def compute_estimate_with_ci(
    trajectory_values: np.ndarray,
    ci_method: CI_METHOD = "bootstrap",
    alpha: float = 0.05,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> EstimateResult:
    """Compute mean estimate and confidence interval from trajectory values.

    Args:
        trajectory_values: Array of trajectory-level values (M,).
        ci_method: Method for computing confidence interval.
        alpha: Significance level (default: 0.05 for 95% CI).
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed for bootstrap.

    Returns:
        EstimateResult with mean estimate and confidence interval.
    """
    samples = np.asarray(trajectory_values, dtype=np.float32)
    mean = float(samples.mean())

    if ci_method == "bootstrap":
        rng = np.random.default_rng(seed)
        boot = []
        for _ in range(n_bootstrap):
            idx = rng.integers(0, len(samples), size=len(samples))
            boot.append(samples[idx].mean())
        lower = float(np.percentile(boot, alpha / 2 * 100))
        upper = float(np.percentile(boot, (1 - alpha / 2) * 100))
    elif ci_method == "normal":
        std_err = float(samples.std() / np.sqrt(len(samples)))
        z_score = stats.norm.ppf(1 - alpha / 2)
        lower = float(mean - z_score * std_err)
        upper = float(mean + z_score * std_err)
    elif ci_method == "t_test":
        std_err = float(samples.std() / np.sqrt(len(samples)))
        df = len(samples) - 1
        t_score = stats.t.ppf(1 - alpha / 2, df)
        lower = float(mean - t_score * std_err)
        upper = float(mean + t_score * std_err)
    else:
        raise ValueError(f"Unsupported ci_method: {ci_method}")

    return EstimateResult(mean=mean, ci_lower=lower, ci_upper=upper)


def self_normalize_weights(
    weights: np.ndarray, epsilon: float = 1e-10
) -> np.ndarray:
    """Self-normalize importance weights.

    w_normalized = w / (mean(w) + epsilon)

    This bounds variance at the cost of introducing bias.

    Args:
        weights: Raw importance weights, shape (n_trajectories,).
        epsilon: Small constant for numerical stability.

    Returns:
        Self-normalized weights.
    """
    mean_weight = weights.mean()
    return weights / (mean_weight + epsilon)


def traj_slices(traj_id: np.ndarray) -> Dict[int, np.ndarray]:
    """Group transition indices by trajectory ID.

    Args:
        traj_id: Trajectory ID for each transition (N,).

    Returns:
        Dictionary mapping trajectory ID to array of transition indices.
    """
    slices: Dict[int, list] = {}
    for idx, tid in enumerate(traj_id):
        slices.setdefault(int(tid), []).append(idx)
    return {k: np.asarray(v, dtype=np.int64) for k, v in slices.items()}


class BaseOPEEstimator(ABC):
    """Abstract base class for OPE estimators."""

    def __init__(self, gamma: float = 0.99):
        """Initialize estimator.

        Args:
            gamma: Discount factor.
        """
        self.gamma = gamma

    @abstractmethod
    def compute_trajectory_values(self, inputs: OPEInputs) -> np.ndarray:
        """Compute trajectory-level values.

        Args:
            inputs: OPE inputs.

        Returns:
            Array of trajectory values (M,).
        """
        pass

    def estimate(
        self,
        inputs: OPEInputs,
        ci_method: CI_METHOD = "bootstrap",
        **ci_kwargs,
    ) -> EstimateResult:
        """Compute estimate with confidence interval.

        Args:
            inputs: OPE inputs.
            ci_method: Method for computing confidence interval.
            **ci_kwargs: Additional arguments for CI computation.

        Returns:
            EstimateResult with mean and confidence interval.
        """
        trajectory_values = self.compute_trajectory_values(inputs)
        return compute_estimate_with_ci(
            trajectory_values, ci_method=ci_method, **ci_kwargs
        )


class BaseISEstimator(BaseOPEEstimator):
    """Base class for importance sampling estimators.

    Supports kernel-based similarity weights and self-normalization.
    """

    def __init__(
        self,
        gamma: float = 0.99,
        use_kernel: bool = True,
        kernel: str = "gaussian",
        bandwidth: Union[float, str] = "auto",
        self_normalize: bool = False,
    ):
        """Initialize IS estimator.

        Args:
            gamma: Discount factor.
            use_kernel: Whether to use kernel similarity weights.
            kernel: Kernel name ("gaussian", "epanechnikov", etc.).
            bandwidth: Bandwidth value or "auto" for automatic selection.
            self_normalize: Whether to self-normalize weights.
        """
        super().__init__(gamma)
        self.use_kernel = use_kernel
        self.kernel = kernel
        self.bandwidth = bandwidth
        self.self_normalize = self_normalize

    def _get_bandwidth(self, inputs: OPEInputs) -> float:
        """Get bandwidth value, computing automatically if needed.

        Args:
            inputs: OPE inputs.

        Returns:
            Bandwidth value.
        """
        if isinstance(self.bandwidth, (int, float)):
            return float(self.bandwidth)
        elif self.bandwidth == "auto":
            # Use median heuristic for OPE (Silverman's rule is too small)
            # Median heuristic gives bandwidth ~ typical action distance,
            # which keeps similarity values reasonable for long trajectories
            return select_bandwidth(inputs.actions, method="median")
        else:
            raise ValueError(f"Invalid bandwidth: {self.bandwidth}")

    def _compute_kernel_similarity(
        self, inputs: OPEInputs, idxs: np.ndarray, bandwidth: float
    ) -> np.ndarray:
        """Compute kernel similarity for given indices.

        Args:
            inputs: OPE inputs.
            idxs: Transition indices.
            bandwidth: Bandwidth value.

        Returns:
            Similarity values for each transition.
        """
        similarity_fn = get_similarity(self.kernel)
        actions = inputs.actions[idxs]
        eval_actions = inputs.eval_action[idxs]
        return similarity_fn(eval_actions, actions, bandwidth=bandwidth)
