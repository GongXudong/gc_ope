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


# =============================================================================
# Pure similarity functions (without normalization factor)
# These return values in [0, 1] range, solving the small weight problem
# =============================================================================


def gaussian_similarity(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Gaussian similarity (without normalization factor).

    S(x, y) = exp(-||x-y||^2 / (2h^2))

    Returns values in (0, 1] range, where 1 means identical.

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Similarity values in (0, 1], shape (n_samples,).
    """
    distance = l2_distance(x, y)
    return np.exp(-distance / (2 * bandwidth**2))


def epanechnikov_similarity(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Epanechnikov similarity (without normalization factor).

    S(x, y) = 1 - (||x-y||/h)^2  if ||x-y|| < h, else 0

    Returns values in [0, 1] range.

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Similarity values in [0, 1], shape (n_samples,).
    """
    distance = np.sqrt(l2_distance(x, y))
    u = distance / bandwidth
    return np.where(u < 1, 1 - u**2, 0.0)


def triangular_similarity(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Triangular similarity. Returns values in [0, 1] range.

    S(x, y) = 1 - ||x-y||/h  if ||x-y|| < h, else 0

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Similarity values in [0, 1], shape (n_samples,).
    """
    distance = np.sqrt(l2_distance(x, y))
    u = distance / bandwidth
    return np.where(u < 1, 1 - u, 0.0)


def cosine_similarity_kernel(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Cosine similarity kernel. Returns values in [0, 1] range.

    S(x, y) = cos(π * ||x-y|| / (2h))  if ||x-y|| < h, else 0

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Similarity values in [0, 1], shape (n_samples,).
    """
    distance = np.sqrt(l2_distance(x, y))
    u = distance / bandwidth
    return np.where(u < 1, np.cos(np.pi * u / 2), 0.0)


def uniform_similarity(
    x: np.ndarray, y: np.ndarray, bandwidth: float = 1.0
) -> np.ndarray:
    """Uniform similarity. Returns 1 if within bandwidth, 0 otherwise.

    S(x, y) = 1  if ||x-y|| < h, else 0

    Args:
        x: Input array 1, shape (n_samples, n_dim).
        y: Input array 2, shape (n_samples, n_dim).
        bandwidth: Bandwidth hyperparameter (default: 1.0).

    Returns:
        Similarity values in {0, 1}, shape (n_samples,).
    """
    distance = np.sqrt(l2_distance(x, y))
    return np.where(distance < bandwidth, 1.0, 0.0)


# Kernel function registry (with normalization factor - for density estimation)
KERNEL_FUNCTIONS: Dict[str, Callable] = {
    "gaussian": gaussian_kernel,
    "epanechnikov": epanechnikov_kernel,
}

# Similarity function registry (without normalization factor - for OPE)
SIMILARITY_FUNCTIONS: Dict[str, Callable] = {
    "gaussian": gaussian_similarity,
    "epanechnikov": epanechnikov_similarity,
    "triangular": triangular_similarity,
    "cosine": cosine_similarity_kernel,
    "uniform": uniform_similarity,
}


def get_kernel(name: str) -> Callable:
    """Get kernel function by name (with normalization factor).

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


def get_similarity(name: str) -> Callable:
    """Get similarity function by name (without normalization factor).

    These functions return values in [0, 1] range, suitable for OPE.

    Args:
        name: Similarity name ("gaussian", "epanechnikov", "triangular",
              "cosine", or "uniform").

    Returns:
        Similarity function.

    Raises:
        ValueError: If similarity name is not supported.
    """
    if name not in SIMILARITY_FUNCTIONS:
        raise ValueError(
            f"Unsupported similarity: {name}. "
            f"Available: {list(SIMILARITY_FUNCTIONS.keys())}"
        )
    return SIMILARITY_FUNCTIONS[name]
