"""Automatic bandwidth selection for kernel-based OPE estimators.

This module provides methods for automatically selecting the bandwidth
hyperparameter for kernel functions used in OPE estimators.

References:
    - Silverman, B.W. (1986). Density Estimation for Statistics and Data Analysis.
    - Scott, D.W. (1992). Multivariate Density Estimation.
"""

from __future__ import annotations

from typing import Literal

import numpy as np


def silverman_bandwidth(data: np.ndarray) -> float:
    """Silverman's rule of thumb for bandwidth selection.

    h = 0.9 * min(std, IQR/1.34) * n^(-1/5)

    This rule is optimal for Gaussian distributions and provides a good
    starting point for most applications.

    Args:
        data: Input data, shape (n_samples, n_dim).

    Returns:
        Optimal bandwidth.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = data.shape[0]
    std = data.std(axis=0).mean()
    iqr = (np.percentile(data, 75, axis=0) - np.percentile(data, 25, axis=0)).mean()

    # Silverman's rule
    h = 0.9 * min(std, iqr / 1.34) * (n ** (-1 / 5))
    return max(h, 1e-6)  # Prevent too small bandwidth


def scott_bandwidth(data: np.ndarray) -> float:
    """Scott's rule for bandwidth selection.

    h = 1.06 * std * n^(-1/5)

    This rule is simpler than Silverman's and works well for unimodal
    distributions.

    Args:
        data: Input data, shape (n_samples, n_dim).

    Returns:
        Optimal bandwidth.
    """
    if data.ndim == 1:
        data = data.reshape(-1, 1)

    n = data.shape[0]
    std = data.std(axis=0).mean()
    h = 1.06 * std * (n ** (-1 / 5))
    return max(h, 1e-6)


def median_bandwidth(data: np.ndarray, subsample: int = 1000) -> float:
    """Median heuristic for bandwidth selection.

    h = median(||x_i - x_j||) for all pairs i, j

    This is a robust method that works well for various distributions.

    Args:
        data: Input data, shape (n_samples, n_dim).
        subsample: Maximum number of samples to use for computing pairwise
            distances (for efficiency).

    Returns:
        Optimal bandwidth.
    """
    from scipy.spatial.distance import pdist

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    # Subsample for efficiency if data is large
    if len(data) > subsample:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(data), subsample, replace=False)
        data = data[indices]

    distances = pdist(data)
    h = float(np.median(distances))
    return max(h, 1e-6)


def select_bandwidth(
    data: np.ndarray,
    method: Literal["silverman", "scott", "median"] = "silverman",
) -> float:
    """Select bandwidth using specified method.

    Args:
        data: Input data, shape (n_samples, n_dim).
        method: Selection method:
            - "silverman": Silverman's rule of thumb (default)
            - "scott": Scott's rule
            - "median": Median heuristic

    Returns:
        Selected bandwidth.

    Raises:
        ValueError: If method is not supported.
    """
    if method == "silverman":
        return silverman_bandwidth(data)
    elif method == "scott":
        return scott_bandwidth(data)
    elif method == "median":
        return median_bandwidth(data)
    else:
        raise ValueError(
            f"Unknown method: {method}. "
            f"Available: ['silverman', 'scott', 'median']"
        )
