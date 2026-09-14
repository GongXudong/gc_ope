"""Gaussian-mixture evaluator for weighted achievable-goal estimates."""

from typing import Any, Callable, Literal, Union

import numpy as np
from sklearn.mixture import GaussianMixture

from gc_ope.evaluate.evaluation_result_container import (
    EvaluationResultContainer,
    WeightedEvaluationResultContainer,
)
from gc_ope.evaluate.evaluator_base import EvaluatorBase


def weighted_resample(
    samples: np.ndarray,
    weights: np.ndarray,
    size: int,
    random_state: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Sample rows with replacement according to non-negative finite weights.

    This keeps the input size bounded and is the approximation used because
    ``GaussianMixture.fit`` has no ``sample_weight`` argument.
    """
    data = np.asarray(samples)
    if data.ndim != 2 or data.shape[0] == 0:
        raise ValueError("samples must be a non-empty 2-D array")
    if not isinstance(size, (int, np.integer)) or size <= 0:
        raise ValueError("size must be a positive integer")
    probability = np.asarray(weights, dtype=float)
    if probability.ndim != 1 or probability.shape[0] != data.shape[0]:
        raise ValueError("weights must be a vector aligned with samples")
    if not np.all(np.isfinite(probability)) or np.any(probability < 0):
        raise ValueError("weights must be finite and non-negative")
    total = probability.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError("at least one weight must be positive")
    probability = probability / total
    rng = random_state if isinstance(random_state, np.random.Generator) else np.random.default_rng(random_state)
    indices = rng.choice(data.shape[0], size=int(size), replace=True, p=probability)
    return data[indices]


class GMMEvaluator(EvaluatorBase):
    """Fit a Gaussian mixture to successful desired goals.

    The public fit/evaluate methods intentionally mirror :class:`KDEEvaluator`.
    """

    gmm: GaussianMixture

    def __init__(
        self,
        evaluation_result_container_class: type[EvaluationResultContainer] = EvaluationResultContainer,
        evaluation_result_container_kwargs: dict[str, Any] | None = None,
        n_components: int = 2,
        resample_size: int = 1000,
        covariance_type: Literal["full", "tied", "diag", "spherical"] = "full",
        n_init: int = 1,
        max_iter: int = 200,
        tol: float = 1e-3,
        random_state: int | None = 0,
        reg_covar: float = 1e-6,
    ):
        super().__init__(evaluation_result_container_class, evaluation_result_container_kwargs or {})
        if n_components <= 0:
            raise ValueError("n_components must be positive")
        if resample_size <= 0:
            raise ValueError("resample_size must be positive")
        self.n_components = int(n_components)
        self.resample_size = int(resample_size)
        self.random_state = random_state
        self.gmm = GaussianMixture(
            n_components=self.n_components,
            covariance_type=covariance_type,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            reg_covar=reg_covar,
        )

    def fit_evaluator(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        all_samples = np.asarray(self.eval_res_container.desired_goal_list, dtype=float)
        flags = np.asarray(self.eval_res_container.success_list, dtype=bool)
        if all_samples.ndim != 2 or all_samples.shape[0] == 0:
            raise ValueError("no evaluation samples are available")
        positive_samples = all_samples[flags]
        if positive_samples.shape[0] == 0:
            raise ValueError("no successful evaluation samples are available")
        if isinstance(self.eval_res_container, WeightedEvaluationResultContainer):
            sample_weights = np.asarray(self.eval_res_container.desired_goal_weights, dtype=float)[flags]
        elif isinstance(self.eval_res_container, EvaluationResultContainer):
            sample_weights = np.ones(positive_samples.shape[0], dtype=float)
        else:
            raise ValueError(f"Can not process EvaluationResultContainer type: {type(self.eval_res_container)}!")
        scaled_positive_samples = self.scaler.fit_transform(positive_samples)
        effective_components = min(self.n_components, scaled_positive_samples.shape[0])
        if effective_components != self.gmm.n_components:
            self.gmm.set_params(n_components=effective_components)
        fitted_samples = weighted_resample(
            scaled_positive_samples, sample_weights, self.resample_size, self.random_state
        )
        self.gmm.fit(fitted_samples)
        log_densities = self.gmm.score_samples(scaled_positive_samples)
        densities = np.exp(np.clip(log_densities, -745.0, 709.0))
        return positive_samples, scaled_positive_samples, sample_weights, densities

    def evaluate(self, desired_goals: np.ndarray, scale: bool = True, return_density: bool = True):
        goals = np.asarray(desired_goals, dtype=float)
        if goals.ndim == 1:
            goals = goals.reshape(1, -1)
        scaled_goals = self.scaler.transform(goals) if scale else goals
        log_densities = self.gmm.score_samples(scaled_goals)
        if return_density:
            return scaled_goals, np.exp(np.clip(log_densities, -745.0, 709.0))
        return scaled_goals, log_densities

    def kl_divergence_uniform_to_kde_integrate(
        self, samples: Union[list, np.ndarray], dV: float, u_density: float
    ) -> float:
        """Compatibility implementation of the evaluator base integral KL."""
        if dV <= 0 or u_density <= 0:
            raise ValueError("dV and u_density must be positive")
        _, densities = self.evaluate(np.asarray(samples), return_density=True)
        densities = np.maximum(densities, np.finfo(float).tiny)
        normalized = (densities / np.sum(densities)) / dV
        return float(u_density * np.sum(np.log(u_density) - np.log(normalized)) * dV)

