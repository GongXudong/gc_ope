"""Behavior data collection utilities for goal-conditioned continuous-control OPE.

This module gathers transitions with a behavior policy, optionally precomputing
evaluation-policy actions/log-probs for later FQE and IS-based estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm


def flatten_obs(obs: Dict[str, Any]) -> np.ndarray:
    """Flatten a goal-conditioned observation dict into a 1D vector.

    Concatenates observation, desired_goal, and achieved_goal in that order.

    Args:
        obs: Dictionary with keys "observation", "desired_goal", "achieved_goal".

    Returns:
        Flattened 1D numpy array of dtype float32.
    """
    return np.concatenate(
        [obs["observation"].ravel(), obs["desired_goal"].ravel(), obs["achieved_goal"].ravel()],
        axis=0,
    ).astype(np.float32)


def _compute_action_log_prob(
    algo: BaseAlgorithm, obs: Dict[str, Any], action: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute action log-probability under an SB3 continuous actor policy.

    For continuous actions (e.g., SAC), computes :math:`\log \pi(a|s)` where
    :math:`\pi` is the policy network and the action distribution is typically
    a squashed Gaussian.

    Args:
        algo: Stable-Baselines3 algorithm with a continuous actor.
        obs: Observation dictionary (goal-conditioned).
        action: Action array (1D or 2D).

    Returns:
        Tuple of (action_tensor, log_probability) where action_tensor is the
        batched action and log_probability is a scalar float.
    """
    policy = algo.policy #TODO: algo要适配除了SAC以外的算法（eg. PPO、HER）
    obs_tensor = policy.obs_to_tensor(obs)[0]  # type: ignore[arg-type]
    # Ensure action is 2D: (batch_size=1, action_dim)
    if action.ndim == 1:
        action = action[np.newaxis, :]
    act_tensor = th.as_tensor(action, device=policy.device, dtype=th.float32)
    with th.no_grad():
        mean, log_std, kwargs = policy.actor.get_action_dist_params(obs_tensor)
        dist = policy.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
        log_prob = dist.log_prob(act_tensor)
        # log_prob may be scalar or 1D, ensure we get a scalar
        if log_prob.numel() == 1:
            log_prob_val = log_prob.item()
        else:
            log_prob_val = log_prob[0].item()
    # Return original 1D action (not batched)
    return act_tensor[0].cpu().numpy().astype(np.float32), log_prob_val


@dataclass
class LoggedDataset:
    """Logged dataset from behavior policy rollouts.

    Contains transitions :math:`(s_t, a_t, r_{t+1}, s_{t+1}, \text{done}_t)`
    collected by rolling out a behavior policy, along with optional
    precomputed evaluation policy actions and log-probabilities.

    Attributes:
        obs_flat: Flattened observations (N, obs_dim).
        actions: Actions taken by behavior policy (N, act_dim).
        rewards: Rewards (N,).
        next_obs_flat: Next observations (N, obs_dim).
        dones: Episode termination flags (N,).
        traj_id: Trajectory ID for each transition (N,).
        step_index: Step index within trajectory (N,).
        obs_dict: Original dict observations (list of N dicts).
        next_obs_dict: Original dict next observations (list of N dicts).
        behavior_log_prob: Log-probability of actions under behavior policy (N,).
        eval_action_curr: Evaluation policy actions at s_t (N, act_dim) or None.
        eval_action_next: Evaluation policy actions at s_{t+1} (N, act_dim) or None.
        eval_log_prob_curr: Log-probability of eval actions at s_t (N,) or None.
        eval_log_prob_next: Log-probability of eval actions at s_{t+1} (N,) or None.
    """
    obs_flat: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_obs_flat: np.ndarray
    dones: np.ndarray
    traj_id: np.ndarray
    step_index: np.ndarray
    obs_dict: list[Dict[str, Any]]
    next_obs_dict: list[Dict[str, Any]]
    behavior_log_prob: np.ndarray
    eval_action_curr: Optional[np.ndarray] = None
    eval_action_next: Optional[np.ndarray] = None
    eval_log_prob_curr: Optional[np.ndarray] = None
    eval_log_prob_next: Optional[np.ndarray] = None


def collect_logged_dataset(
    env,
    behavior_algo: BaseAlgorithm,
    eval_algo: Optional[BaseAlgorithm],
    n_episodes: int,
    max_steps: int,
) -> LoggedDataset:
    """Collect logged dataset by rolling out behavior policy.

    Rolls out the behavior policy to collect transitions
    :math:`(s_t, a_t, r_{t+1}, s_{t+1}, \text{done}_t)` from the environment.
    Optionally precomputes evaluation policy actions and log-probabilities
    at each state for later use in FQE and importance sampling estimators.

    Args:
        env: Gymnasium environment (goal-conditioned).
        behavior_algo: Behavior policy (e.g., early checkpoint of SAC).
        eval_algo: Evaluation policy (e.g., later checkpoint of SAC). If None,
            eval actions/log-probs are not computed.
        n_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.

    Returns:
        LoggedDataset containing all transitions and optional eval policy data.
    """
    transitions: Dict[str, list] = {
        "obs_flat": [],
        "action": [],
        "reward": [],
        "next_obs_flat": [],
        "done": [],
        "traj_id": [],
        "step_index": [],
        "obs_dict": [],
        "next_obs_dict": [],
        "behavior_log_prob": [],
    }

    eval_action_curr: list[np.ndarray] = []
    eval_action_next: list[np.ndarray] = []
    eval_log_prob_curr: list[float] = []
    eval_log_prob_next: list[float] = []

    traj_id = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminate, truncated = False, False
        step_idx = 0
        while not (terminate or truncated) and step_idx < max_steps:
            action, _ = behavior_algo.predict(obs, deterministic=True)
            next_obs, reward, terminate, truncated, _ = env.step(action)
            done = terminate or truncated

            # log-prob under behavior
            _, beh_logp = _compute_action_log_prob(behavior_algo, obs, action)

            transitions["obs_flat"].append(flatten_obs(obs))
            transitions["action"].append(action.astype(np.float32))
            transitions["reward"].append(float(reward))
            transitions["next_obs_flat"].append(flatten_obs(next_obs))
            transitions["done"].append(done)
            transitions["traj_id"].append(traj_id)
            transitions["step_index"].append(step_idx)
            transitions["obs_dict"].append(obs)
            transitions["next_obs_dict"].append(next_obs)
            transitions["behavior_log_prob"].append(beh_logp)

            if eval_algo is not None:
                # a_t under eval policy on s_t and s_{t+1}
                a_curr, _ = eval_algo.predict(obs, deterministic=True)
                a_next, _ = eval_algo.predict(next_obs, deterministic=True)
                a_curr_t, eval_logp_curr = _compute_action_log_prob(eval_algo, obs, a_curr)
                a_next_t, eval_logp_next = _compute_action_log_prob(eval_algo, next_obs, a_next)
                eval_action_curr.append(a_curr_t)
                eval_action_next.append(a_next_t)
                eval_log_prob_curr.append(eval_logp_curr)
                eval_log_prob_next.append(eval_logp_next)

            obs = next_obs
            step_idx += 1
        traj_id += 1

    def _to_np(name: str, dtype) -> np.ndarray:
        return np.asarray(transitions[name], dtype=dtype)

    dataset = LoggedDataset(
        obs_flat=_to_np("obs_flat", np.float32),
        actions=_to_np("action", np.float32),
        rewards=_to_np("reward", np.float32),
        next_obs_flat=_to_np("next_obs_flat", np.float32),
        dones=_to_np("done", np.bool_),
        traj_id=_to_np("traj_id", np.int32),
        step_index=_to_np("step_index", np.int32),
        obs_dict=transitions["obs_dict"],
        next_obs_dict=transitions["next_obs_dict"],
        behavior_log_prob=_to_np("behavior_log_prob", np.float32),
    )

    if eval_algo is not None:
        dataset.eval_action_curr = np.asarray(eval_action_curr, dtype=np.float32)
        dataset.eval_action_next = np.asarray(eval_action_next, dtype=np.float32)
        dataset.eval_log_prob_curr = np.asarray(eval_log_prob_curr, dtype=np.float32)
        dataset.eval_log_prob_next = np.asarray(eval_log_prob_next, dtype=np.float32)

    return dataset

