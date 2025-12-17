"""OPE estimators (DM / TIS / DR) for continuous GC tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Tuple

import numpy as np

from .ope_input import OPEInputs


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


def _bootstrap_ci(
    samples: np.ndarray, alpha: float = 0.05, n_bootstrap: int = 200, seed: int = 0
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval.

    Args:
        samples: Sample values (N,).
        alpha: Significance level (default: 0.05 for 95% CI).
        n_bootstrap: Number of bootstrap resamples (default: 200).
        seed: Random seed (default: 0).

    Returns:
        Tuple of (mean, ci_lower, ci_upper).
    """
    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(samples), size=len(samples))
        boot.append(samples[idx].mean())
    lower = np.percentile(boot, alpha / 2 * 100)
    upper = np.percentile(boot, (1 - alpha / 2) * 100)
    return float(samples.mean()), float(lower), float(upper)


def dm_estimate(inputs: OPEInputs, initial_only: bool = False) -> EstimateResult:
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

    Returns:
        EstimateResult with mean estimate and 95% confidence interval.
    """
    mask = inputs.step_index == 0 if initial_only else slice(None)
    values = inputs.q_sa_eval[mask]
    mean, low, up = _bootstrap_ci(values)
    return EstimateResult(mean=mean, ci_lower=low, ci_upper=up)


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


def tis_estimate(inputs: OPEInputs) -> EstimateResult:
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

    Returns:
        EstimateResult with mean estimate and 95% confidence interval.
    """
    traj_map = _traj_slices(inputs.traj_id)
    gamma = inputs.gamma
    returns = []
    for idxs in traj_map.values():
        r = inputs.rewards[idxs]
        logp_b = inputs.behavior_log_prob[idxs]
        logp_e = inputs.eval_log_prob[idxs]
        weight = np.exp((logp_e - logp_b).sum())
        discounts = np.power(gamma, np.arange(len(r)))
        returns.append(weight * np.sum(discounts * r))
    samples = np.asarray(returns, dtype=np.float32)
    mean, low, up = _bootstrap_ci(samples)
    return EstimateResult(mean=mean, ci_lower=low, ci_upper=up)


def dr_estimate(inputs: OPEInputs) -> EstimateResult:
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

    Returns:
        EstimateResult with mean estimate and 95% confidence interval.
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

        ratios = np.exp(logp_e - logp_b)
        w_step = np.cumprod(ratios)
        w_prev = np.concatenate([[1.0], w_step[:-1]])
        discounts = np.power(gamma, np.arange(len(r)))

        term = w_step * (r - q_sa) + w_prev * v_eval
        est.append(np.sum(discounts * term))

    samples = np.asarray(est, dtype=np.float32)
    mean, low, up = _bootstrap_ci(samples)
    return EstimateResult(mean=mean, ci_lower=low, ci_upper=up)

