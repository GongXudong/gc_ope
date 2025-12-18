"""Construct OPE-ready inputs akin to CreateOPEInput for continuous GC tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch as th
import pickle
from stable_baselines3.common.base_class import BaseAlgorithm

from .fqe import FQETrainer
from .logged_dataset import LoggedDataset, _compute_action_log_prob, compute_eval_policy_cache


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

    Returns cached values if present, otherwise computes them using compute_eval_policy_cache.

    Args:
        dataset: Logged dataset (may or may not have eval_action_curr cached).
        eval_algo: Evaluation policy.

    Returns:
        Tuple of (eval_actions, eval_log_probs) at each state s_t.
    """
    if dataset.eval_action_curr is not None and dataset.eval_log_prob_curr is not None:
        return dataset.eval_action_curr, dataset.eval_log_prob_curr

    # Use compute_eval_policy_cache to compute and cache eval policy actions/log-probs
    compute_eval_policy_cache(dataset, eval_algo)
    with open('dataset.pkl', 'wb') as f:
        pickle.dump(dataset, f)
    print("Eval policy actions/log-probs are computed & saved to dataset.pkl")
    return dataset.eval_action_curr, dataset.eval_log_prob_curr


def build_ope_inputs(
    dataset: LoggedDataset,
    eval_algo: BaseAlgorithm,
    gamma: float,
    fqe: Optional[FQETrainer] = None,
    q_function_method: str = "fqe",
    fqe_train_kwargs: Optional[Dict[str, Any]] = None,
    fqe_kwargs: Optional[Dict[str, Any]] = None,
) -> OPEInputs:
    """Build OPE inputs from logged dataset and trained FQE model.

    Computes evaluation policy actions/log-probs and Q-values needed for
    DM, TIS, and DR estimators. If `fqe` is not provided, automatically creates
    and trains an FQE model (if `q_function_method="fqe"`).

    Args:
        dataset: Logged dataset from behavior policy.
        eval_algo: Evaluation policy.
        gamma: Discount factor.
        fqe: Optional pre-trained FQE model. If None and `q_function_method="fqe"`,
            a new FQE model will be created and trained.
        q_function_method: Method for computing Q-values. Currently only "fqe" is supported.
            Default: "fqe".
        fqe_train_kwargs: Optional dictionary of arguments for FQE training (fit method).
            Keys: batch_size, n_epochs, shuffle, logger. Default: batch_size=256, n_epochs=500.
        fqe_kwargs: Optional dictionary of arguments for FQETrainer initialization.
            Keys: tau, lr, hidden_sizes, obs_state_dim, goal_dim, device.
            Default: tau=0.005, lr=3e-4, hidden_sizes=(256, 256).

    Returns:
        OPEInputs containing all data needed for OPE estimators.

    Note:
        The FQE training process is now integrated into this function and not exposed
        to the user. This simplifies the API and ensures consistent usage.
    """
    # Ensure eval policy actions/log-probs are available
    print("Ensure eval policy actions/log-probs are available")
    eval_actions, eval_log_prob = _ensure_eval_actions(dataset, eval_algo)
    print("Eval policy actions/log-probs are available")

    # Handle Q-function computation
    if q_function_method == "fqe":
        # Create and train FQE if not provided
        if fqe is None:
            obs_dim = dataset.obs_flat.shape[1]
            act_dim = dataset.actions.shape[1]

            # Default FQE initialization arguments
            default_fqe_kwargs: Dict[str, Any] = {
                "tau": 0.005,
                "lr": 3e-4,
                "hidden_sizes": (256, 256),
                "obs_state_dim": None, # 默认使用simple mode，将observation和action拼接作为输入
                "goal_dim": None,
                "device": None,
            }
            if fqe_kwargs is not None:
                default_fqe_kwargs.update(fqe_kwargs)

            fqe = FQETrainer(
                obs_dim=obs_dim,
                act_dim=act_dim,
                eval_algo=eval_algo,
                gamma=gamma,
                **default_fqe_kwargs,
            )

            # Default training arguments
            default_train_kwargs: Dict[str, Any] = {
                "batch_size": 1024,
                "n_epochs": 500,
                "shuffle": True,
                "logger": None,
            }
            if fqe_train_kwargs is not None:
                default_train_kwargs.update(fqe_train_kwargs)

            # Train FQE
            fqe.fit(dataset, **default_train_kwargs)
    else:
        raise ValueError(f"Unsupported q_function_method: {q_function_method}")

    # Compute Q-values using trained FQE
    device = fqe.device
    # Convert numpy arrays to tensors and move to device
    obs_t = th.as_tensor(dataset.obs_flat, device=device, dtype=th.float32)
    act_t = th.as_tensor(dataset.actions, device=device, dtype=th.float32)
    eval_act_t = th.as_tensor(eval_actions, device=device, dtype=th.float32)

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

