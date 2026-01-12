"""OPE estimators package.

This package provides class-based OPE estimators with support for:
- Kernel-based similarity weights for continuous action spaces
- Self-normalization for numerical stability
- Automatic bandwidth selection

Estimators:
    - DMEstimator: Direct Method
    - TISEstimator: Trajectory-wise Importance Sampling
    - PDISEstimator: Per-Decision Importance Sampling
    - DREstimator: Doubly Robust

Self-Normalized variants:
    - SelfNormalizedTIS
    - SelfNormalizedPDIS
    - SelfNormalizedDR
"""

from .base import (
    BaseOPEEstimator,
    BaseISEstimator,
    EstimateResult,
    compute_estimate_with_ci,
    self_normalize_weights,
    traj_slices,
)
from .dm import DMEstimator
from .tis import TISEstimator, SelfNormalizedTIS
from .pdis import PDISEstimator, SelfNormalizedPDIS
from .dr import DREstimator, SelfNormalizedDR

__all__ = [
    # Base classes
    "BaseOPEEstimator",
    "BaseISEstimator",
    "EstimateResult",
    "compute_estimate_with_ci",
    "self_normalize_weights",
    "traj_slices",
    # Estimators
    "DMEstimator",
    "TISEstimator",
    "PDISEstimator",
    "DREstimator",
    # Self-normalized variants
    "SelfNormalizedTIS",
    "SelfNormalizedPDIS",
    "SelfNormalizedDR",
]
