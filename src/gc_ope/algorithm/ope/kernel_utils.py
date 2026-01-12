"""Kernel functions for continuous action OPE estimators.

This module provides kernel functions for computing similarity weights
in importance sampling estimators for continuous action spaces.

Reference: scope-rl/scope_rl/utils.py
"""

from __future__ import annotations

from typing import Callable, Dict

import numpy as np


def l2_distance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Calculate squared L2 distance between x and y.

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).

    Returns:
        Squared L2 distance, shape (n_samples,).
    """
    x_2 = (x**2).sum(axis=1)
    y_2 = (y**2).sum(axis=1)
    x_y = (x[:, np.newaxis, :] @ y[:, :, np.newaxis]).flatten()
    return x_2 + y_2 - 2 * x_y


def gaussian_kernel(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Gaussian kernel similarity.

    K(x, y) = exp(-||x-y||^2 / (2h^2)) / sqrt(2πh^2)

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Kernel density, shape (n_samples,).
    """
    distance = l2_distance(x, y)
    return np.exp(-distance / (2 * bandwidth**2)) / np.sqrt(
        2 * np.pi * bandwidth**2
    )


def epanechnikov_kernel(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Epanechnikov kernel similarity (MSE optimal).

    K(x, y) = 0.75 * (1 - (||x-y||/h)^2) / h  if ||x-y|| < h, else 0

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Kernel density, shape (n_samples,).
    """
    distance = np.sqrt(l2_distance(x, y))
    clipped_norm_dist = np.clip(distance / bandwidth, None, 1.0)
    return 0.75 * (1 - clipped_norm_dist**2) / bandwidth


# Kernel function registry
KERNEL_FUNCTIONS: Dict[str, Callable] = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
}


def get_kernel(name: str) -> Callable:
    """Get kernel function by name.

    Args:
        name: Kernel name ("gaussian" or "epanechnikov").

    Returns:
        Kernel function.

    Raises:
        ValueError: If kernel name is not supported.
    """
    if name not in KERNEL_FUNCTIONS:
        raise ValueError(
            f"Unsupported kernel: {name}. "
            f"Available: {list(KERNEL_FUNCTIONS.keys())}"
        )
    return KERNEL_FUNCTIONS[name]
