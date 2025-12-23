"""OPE estimators (DM / TIS / DR) for continuous GC tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Tuple

import numpy as np
from scipy import stats

from .ope_input import OPEInputs

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

    Separates the computation of trajectory values from the computation of
    mean and confidence interval, allowing different CI methods to be used.

    Args:
        trajectory_values: Array of trajectory-level values (M,) where M is
            the number of trajectories.
        ci_method: Method for computing confidence interval.
            - "bootstrap": Bootstrap resampling (default).
            - "normal": Assumes normal distribution, uses standard error.
            - "t_test": Uses t-distribution (suitable for small samples).
        alpha: Significance level (default: 0.05 for 95% CI).
        n_bootstrap: Number of bootstrap resamples (only for "bootstrap" method).
        seed: Random seed for bootstrap (only for "bootstrap" method).

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
        # Assume normal distribution, use standard error
        std_err = float(samples.std() / np.sqrt(len(samples)))
        z_score = stats.norm.ppf(1 - alpha / 2)
        lower = float(mean - z_score * std_err)
        upper = float(mean + z_score * std_err)
    elif ci_method == "t_test":
        # Use t-distribution (suitable for small samples)
        std_err = float(samples.std() / np.sqrt(len(samples)))
        df = len(samples) - 1  # degrees of freedom
        t_score = stats.t.ppf(1 - alpha / 2, df)
        lower = float(mean - t_score * std_err)
        upper = float(mean + t_score * std_err)
    else:
        raise ValueError(f"Unsupported ci_method: {ci_method}")

    return EstimateResult(mean=mean, ci_lower=lower, ci_upper=upper)


def dm_compute_trajectory_values(
    inputs: OPEInputs, initial_only: bool = False
) -> np.ndarray:
    r"""Compute trajectory-level values for Direct Method (DM) estimator.

    DM estimates the policy value using the FQE Q-function. This function
    computes the Q-values for each transition (or initial state if initial_only=True).

    Args:
        inputs: OPE inputs containing Q-values and metadata.
        initial_only: If True, only use initial states (default: False).

    Returns:
        Array of Q-values (N,) where N is the number of transitions (or initial states).
    """
    mask = inputs.step_index == 0 if initial_only else slice(None)
    values = inputs.q_sa_eval[mask]
    return np.asarray(values, dtype=np.float32)


def dm_estimate(
    inputs: OPEInputs,
    initial_only: bool = False,
    ci_method: CI_METHOD = "bootstrap",
    **ci_kwargs,
) -> EstimateResult:
    r"""Direct Method (DM) estimator.

    DM estimates the policy value using the FQE Q-function:

    .. math::

        \hat{V}^{\text{DM}} = \frac{1}{N} \sum_{i=1}^N Q(s_i, \pi_{\text{eval}}(s_i))

    Plain text: V^DM = (1/N) * sum_i Q(s_i, π_eval(s_i))

    If ``initial_only=True``, only uses initial states (step_index == 0):

    .. math::

        \hat{V}^{\text{DM}} = \frac{1}{|\mathcal{I}_0|} \sum_{i \in \mathcal{I}_0} Q(s_i, \pi_{\text{eval}}(s_i))

    Plain text: V^DM = (1/|I_0|) * sum_{i in I_0} Q(s_i, π_eval(s_i))

    where :math:`\mathcal{I}_0` is the set of initial state indices.

    References:
        * `Jiang & Li, Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.
          <https://arxiv.org/abs/1511.03722>`_

    Args:
        inputs: OPE inputs containing Q-values and metadata.
        initial_only: If True, only use initial states (default: False).
        ci_method: Method for computing confidence interval (default: "bootstrap").
        **ci_kwargs: Additional arguments for CI computation (alpha, n_bootstrap, seed).

    Returns:
        EstimateResult with mean estimate and 95% confidence interval.
    """
    values = dm_compute_trajectory_values(inputs, initial_only=initial_only)
    return compute_estimate_with_ci(values, ci_method=ci_method, **ci_kwargs)


def _traj_slices(traj_id: np.ndarray) -> Dict[int, np.ndarray]:
    """Group transition indices by trajectory ID.

    Args:
        traj_id: Trajectory ID for each transition (N,).

    Returns:
        Dictionary mapping trajectory ID to array of transition indices.
    """
    slices: Dict[int, np.ndarray] = {}
    for idx, tid in enumerate(traj_id):
        slices.setdefault(int(tid), []).append(idx)
    return {k: np.asarray(v, dtype=np.int64) for k, v in slices.items()}


def tis_compute_trajectory_values(inputs: OPEInputs) -> np.ndarray:
    r"""Compute trajectory-level values for Trajectory-wise Importance Sampling (TIS) estimator.

    TIS estimates the policy value using trajectory-level importance weights.
    This function computes the weighted returns for each trajectory.

    Args:
        inputs: OPE inputs containing rewards, log-probs, and trajectory info.

    Returns:
        Array of weighted returns (M,) where M is the number of trajectories.
    """
    traj_map = _traj_slices(inputs.traj_id)
    gamma = inputs.gamma
    returns = []
    for idxs in traj_map.values():
        r = inputs.rewards[idxs]
        logp_b = inputs.behavior_log_prob[idxs]
        logp_e = inputs.eval_log_prob[idxs]
        
        # Clip log-importance weights to prevent overflow/underflow
        # max_log_weight = 10.0 (approx exp(10) ~ 22000)
        log_weights = np.clip(logp_e - logp_b, -10.0, 10.0)
        
        # Calculate trajectory weight
        # Still can overflow if trajectory is long, but less likely
        weight = np.exp(log_weights.sum())
        
        # Additional safety check for infinity
        if not np.isfinite(weight):
            weight = 0.0  # or some large constant, but 0 is safer for stability
        
        # Hard clip the cumulative weight for TIS to prevent overflow
        # Values > 1e4 usually imply numerical instability or zero support
        weight = np.clip(weight, 0.0, 1e4)
            
        discounts = np.power(gamma, np.arange(len(r)))
        returns.append(weight * np.sum(discounts * r))
    return np.asarray(returns, dtype=np.float32)


def tis_estimate(
    inputs: OPEInputs, ci_method: CI_METHOD = "bootstrap", **ci_kwargs
) -> EstimateResult:
    r"""Trajectory-wise Importance Sampling (TIS) estimator.

    TIS estimates the policy value using trajectory-level importance weights:

    .. math::

        \hat{V}^{\text{TIS}} = \frac{1}{M} \sum_{\tau=1}^M w_\tau G_\tau

    Plain text: V^TIS = (1/M) * sum_τ w_τ * G_τ

    where :math:`M` is the number of trajectories, :math:`G_\tau` is the
    discounted return of trajectory :math:`\tau`, and the importance weight is:

    .. math::

        w_\tau = \prod_{t=0}^{T_\tau-1} \frac{\pi_{\text{eval}}(a_t | s_t)}{\pi_{\text{behavior}}(a_t | s_t)}
            = \exp\left( \sum_{t=0}^{T_\tau-1} \left( \log \pi_{\text{eval}}(a_t | s_t)
                - \log \pi_{\text{behavior}}(a_t | s_t) \right) \right)

    Plain text: w_τ = prod_t [π_eval(a_t|s_t) / π_behavior(a_t|s_t)]
                = exp(sum_t [log π_eval(a_t|s_t) - log π_behavior(a_t|s_t)])

    References:
        * `Precup et al., Eligibility Traces for Off-Policy Policy Evaluation.
          <https://www.cs.mcgill.ca/~jpineau/files/Precup-ICML2000.pdf>`_

    Args:
        inputs: OPE inputs containing rewards, log-probs, and trajectory info.
        ci_method: Method for computing confidence interval (default: "bootstrap").
        **ci_kwargs: Additional arguments for CI computation (alpha, n_bootstrap, seed).

    Returns:
        EstimateResult with mean estimate and 95% confidence interval.
    """
    trajectory_values = tis_compute_trajectory_values(inputs)
    return compute_estimate_with_ci(trajectory_values, ci_method=ci_method, **ci_kwargs)


def dr_compute_trajectory_values(inputs: OPEInputs) -> np.ndarray:
    r"""Compute trajectory-level values for Doubly Robust (DR) estimator.

    DR combines importance sampling with a control variate (Q-function) to
    reduce variance. This function computes the DR estimate for each trajectory.

    Args:
        inputs: OPE inputs containing rewards, log-probs, Q-values, and trajectory info.

    Returns:
        Array of DR estimates (M,) where M is the number of trajectories.
    """
    traj_map = _traj_slices(inputs.traj_id)
    gamma = inputs.gamma
    est = []
    for idxs in traj_map.values():
        r = inputs.rewards[idxs]
        logp_b = inputs.behavior_log_prob[idxs]
        logp_e = inputs.eval_log_prob[idxs]
        q_sa = inputs.q_sa_behavior[idxs]
        v_eval = inputs.q_sa_eval[idxs]

        # Clip importance ratios per step
        # Common practice in PPO (e.g., clip to [0.8, 1.2]) or more generous here for IS
        # Let's clip to [0, 100] to prevent single-step explosion
        ratios = np.exp(np.clip(logp_e - logp_b, -10.0, 5.0))  # exp(5) ~ 148
        
        # Cumulative product with safety checks
        w_step = np.cumprod(ratios)
        
        # If cumulative weight becomes too large, clip it
        # This biases the estimator but reduces variance and prevents NaN/Inf
        w_step = np.clip(w_step, 0.0, 1e4) 
        
        w_prev = np.concatenate([[1.0], w_step[:-1]])
        discounts = np.power(gamma, np.arange(len(r)))

        term = w_step * (r - q_sa) + w_prev * v_eval
        est.append(np.sum(discounts * term))

    return np.asarray(est, dtype=np.float32)


def dr_estimate(
    inputs: OPEInputs, ci_method: CI_METHOD = "bootstrap", **ci_kwargs
) -> EstimateResult:
    r"""Doubly Robust (DR) estimator.

    DR combines importance sampling with a control variate (Q-function) to
    reduce variance. For each trajectory :math:`\tau`, the estimate is:

    .. math::

        \hat{V}_\tau^{\text{DR}} = \sum_{t=0}^{T_\tau-1} \gamma^t \left[
            w_t (r_t - Q(s_t, a_t)) + w_{t-1} Q(s_t, \pi_{\text{eval}}(s_t))
        \right]

    Plain text: V_τ^DR = sum_t γ^t [w_t * (r_t - Q(s_t, a_t)) + w_{t-1} * Q(s_t, π_eval(s_t))]

    where :math:`w_t = \prod_{k=0}^t \frac{\pi_{\text{eval}}(a_k | s_k)}{\pi_{\text{behavior}}(a_k | s_k)}`
    is the step-wise importance weight, :math:`w_{-1} = 1`, and
    :math:`Q(s_t, a_t)` is the Q-value for the behavior action while
    :math:`Q(s_t, \pi_{\text{eval}}(s_t))` is for the evaluation policy action.

    Plain text weight: w_t = prod_{k=0}^t [π_eval(a_k|s_k) / π_behavior(a_k|s_k)]

    The overall estimate is:

    .. math::

        \hat{V}^{\text{DR}} = \frac{1}{M} \sum_{\tau=1}^M \hat{V}_\tau^{\text{DR}}

    Plain text: V^DR = (1/M) * sum_τ V_τ^DR

    DR is "doubly robust" because it is unbiased if either the importance
    weights are correct or the Q-function is correct.

    References:
        * `Jiang & Li, Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.
          <https://arxiv.org/abs/1511.03722>`_

    Args:
        inputs: OPE inputs containing rewards, log-probs, Q-values, and trajectory info.
        ci_method: Method for computing confidence interval (default: "bootstrap").
        **ci_kwargs: Additional arguments for CI computation (alpha, n_bootstrap, seed).

    Returns:
        EstimateResult with mean estimate and 95% confidence interval.
    """
    trajectory_values = dr_compute_trajectory_values(inputs)
    return compute_estimate_with_ci(trajectory_values, ci_method=ci_method, **ci_kwargs)
