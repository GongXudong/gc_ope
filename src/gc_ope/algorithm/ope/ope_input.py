"""Construct OPE-ready inputs akin to CreateOPEInput for continuous GC tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm

from .fqe import FQETrainer
from .logged_dataset import LoggedDataset, _compute_action_log_prob


@dataclass
class OPEInputs:
    """Inputs for off-policy evaluation estimators.

    Contains all data needed to compute DM, TIS, and DR estimates.

    Attributes:
        obs_flat: Flattened observations (N, obs_dim).
        actions: Actions from behavior policy (N, act_dim).
        rewards: Rewards (N,).
        next_obs_flat: Next observations (N, obs_dim).
        dones: Episode termination flags (N,).
        traj_id: Trajectory ID for each transition (N,).
        step_index: Step index within trajectory (N,).
        behavior_log_prob: Log-probability of actions under behavior policy (N,).
        eval_action: Evaluation policy actions at s_t (N, act_dim).
        eval_log_prob: Log-probability of eval actions at s_t (N,).
        q_sa_behavior: Q-values Q(s_t, a_t) where a_t is behavior action (N,).
        q_sa_eval: Q-values Q(s_t, π_eval(s_t)) (N,).
        gamma: Discount factor.
    """
    obs_flat: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs_flat: np.ndarray
    dones: np.ndarray
    traj_id: np.ndarray
    step_index: np.ndarray
    behavior_log_prob: np.ndarray
    eval_action: np.ndarray
    eval_log_prob: np.ndarray
    q_sa_behavior: np.ndarray
    q_sa_eval: np.ndarray
    gamma: float


def _ensure_eval_actions(
    dataset: LoggedDataset, eval_algo: BaseAlgorithm
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure evaluation policy actions and log-probs are available.

    Returns cached values if present, otherwise computes them on-the-fly.

    Args:
        dataset: Logged dataset (may or may not have eval_action_curr cached).
        eval_algo: Evaluation policy.

    Returns:
        Tuple of (eval_actions, eval_log_probs) at each state s_t.
    """
    if dataset.eval_action_curr is not None and dataset.eval_log_prob_curr is not None:
        return dataset.eval_action_curr, dataset.eval_log_prob_curr

    eval_actions: list[np.ndarray] = []
    eval_log_probs: list[float] = []
    for obs in dataset.obs_dict:
        action, _ = eval_algo.predict(obs, deterministic=True)
        _, logp = _compute_action_log_prob(eval_algo, obs, action)
        eval_actions.append(action.astype(np.float32))
        eval_log_probs.append(logp)
    return np.asarray(eval_actions, dtype=np.float32), np.asarray(eval_log_probs, dtype=np.float32)


def build_ope_inputs(
    dataset: LoggedDataset, eval_algo: BaseAlgorithm, fqe: FQETrainer, gamma: float
) -> OPEInputs:
    """Build OPE inputs from logged dataset and trained FQE model.

    Computes evaluation policy actions/log-probs and Q-values needed for
    DM, TIS, and DR estimators.

    Args:
        dataset: Logged dataset from behavior policy.
        eval_algo: Evaluation policy.
        fqe: Trained FQE model.
        gamma: Discount factor.

    Returns:
        OPEInputs containing all data needed for OPE estimators.
    """
    eval_actions, eval_log_prob = _ensure_eval_actions(dataset, eval_algo)

    device = fqe.device
    obs_t = th.as_tensor(dataset.obs_flat, device=device)
    act_t = th.as_tensor(dataset.actions, device=device)
    eval_act_t = th.as_tensor(eval_actions, device=device)

    with th.no_grad():
        q_sa_behavior = fqe.predict_value(obs_t, act_t).cpu().numpy()
        q_sa_eval = fqe.predict_value(obs_t, eval_act_t).cpu().numpy()

    return OPEInputs(
        obs_flat=dataset.obs_flat,
        actions=dataset.actions,
        rewards=dataset.rewards,
        next_obs_flat=dataset.next_obs_flat,
        dones=dataset.dones,
        traj_id=dataset.traj_id,
        step_index=dataset.step_index,
        behavior_log_prob=dataset.behavior_log_prob,
        eval_action=eval_actions,
        eval_log_prob=eval_log_prob,
        q_sa_behavior=q_sa_behavior,
        q_sa_eval=q_sa_eval,
        gamma=gamma,
    )

