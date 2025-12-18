"""Behavior data collection utilities for goal-conditioned continuous-control OPE.

This module gathers transitions with a behavior policy, optionally precomputing
evaluation-policy actions/log-probs for later FQE and IS-based estimators.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

import numpy as np
import torch as th
from tqdm import tqdm
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


# Import compute_action_log_prob from algorithm_adapter
# This function supports multiple algorithms (SAC, PPO, HER, etc.)
from .algorithm_adapter import (
    compute_action_log_prob as _compute_action_log_prob,
    compute_action_log_prob_batch,
    predict_eval_action,
    predict_eval_action_batch_dict,
    compute_action_log_prob_batch_dict,
)


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
    n_episodes: int = 1000,
    max_steps: int = 400,
) -> LoggedDataset:
    """Collect logged dataset by rolling out behavior policy.

    Rolls out the behavior policy to collect transitions
    :math:`(s_t, a_t, r_{t+1}, s_{t+1}, \text{done}_t)` from the environment.
    
    Note: Evaluation policy actions and log-probabilities are not computed here.
    Use `compute_eval_policy_cache` separately to compute them.

    Args:
        env: Gymnasium environment (goal-conditioned).
        behavior_algo: Behavior policy (e.g., early checkpoint of SAC).
        n_episodes: Number of episodes to collect.
        max_steps: Maximum steps per episode.

    Returns:
        LoggedDataset containing all transitions from behavior policy.
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

    traj_id = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        terminate, truncated = False, False
        step_idx = 0
        while not (terminate or truncated) and step_idx < max_steps:
            action, _ = behavior_algo.predict(obs, deterministic=True)
            next_obs, reward, terminate, truncated, _ = env.step(action)
            done = terminate or truncated

            # log-prob under behavior (supports SAC, PPO, HER, etc.)
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

    return dataset


def compute_eval_policy_cache(
    dataset: LoggedDataset, eval_algo: BaseAlgorithm, batch_size: int = 256
) -> LoggedDataset:
    """Compute and cache evaluation policy actions and log-probabilities.

    This function computes evaluation policy actions and log-probabilities for
    all states in the dataset and caches them in the dataset. This allows
    separation of data collection and evaluation policy computation, making
    the code more modular and reusable.

    Args:
        dataset: Logged dataset from behavior policy (must have obs_dict and next_obs_dict).
        eval_algo: Evaluation policy (e.g., later checkpoint of SAC).
        batch_size: Batch size for batch processing (default: 256).

    Returns:
        The same dataset with eval_action_curr, eval_action_next, eval_log_prob_curr,
        and eval_log_prob_next populated. If these fields were already populated,
        they will be recomputed.

    Note:
        This function modifies the dataset in-place and also returns it for convenience.
    """
    policy = eval_algo.policy
    device = policy.device
    
    # Convert dictionary observations to batch dictionary tensors for policy
    def obs_dict_list_to_batch_dict(obs_dict_list: list[Dict[str, Any]]) -> Dict[str, th.Tensor]:
        """Convert a list of observation dictionaries to a batch dictionary tensor."""
        # Stack each key's values into a batch tensor
        batch_dict = {}
        for key in obs_dict_list[0].keys():
            values = [obs_dict[key] for obs_dict in obs_dict_list]
            # Stack numpy arrays and convert to tensor
            stacked = np.stack(values, axis=0)
            batch_dict[key] = th.as_tensor(stacked, device=device, dtype=th.float32)
        return batch_dict
    
    print("Computing eval policy cache (batch processing)")
    n_samples = len(dataset.obs_dict)
    eval_action_curr_list: list[np.ndarray] = []
    eval_action_next_list: list[np.ndarray] = []
    eval_log_prob_curr_list: list[float] = []
    eval_log_prob_next_list: list[float] = []
    
    with tqdm(total=n_samples, desc="Computing eval policy cache") as pbar:
        for i in range(0, n_samples, batch_size):
            end_idx = min(i + batch_size, n_samples)
            batch_obs_dict = dataset.obs_dict[i:end_idx]
            batch_next_obs_dict = dataset.next_obs_dict[i:end_idx]
            
            # Convert to batch dictionary tensors
            obs_batch_dict = obs_dict_list_to_batch_dict(batch_obs_dict)
            next_obs_batch_dict = obs_dict_list_to_batch_dict(batch_next_obs_dict)
            
            # Use policy's preprocessing to convert batch dict to tensor
            # The policy's preprocess_obs should handle batch dict correctly
            from stable_baselines3.common.preprocessing import preprocess_obs
            obs_tensor = preprocess_obs(
                obs_batch_dict, 
                policy.observation_space, 
                normalize_images=policy.normalize_images
            )
            next_obs_tensor = preprocess_obs(
                next_obs_batch_dict,
                policy.observation_space,
                normalize_images=policy.normalize_images
            )
            
            # Move to device if not already
            if isinstance(obs_tensor, dict):
                obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
                next_obs_tensor = {k: v.to(device) for k, v in next_obs_tensor.items()}
            else:
                obs_tensor = obs_tensor.to(device)
                next_obs_tensor = next_obs_tensor.to(device)
            
            # Use batch dictionary processing functions
            a_curr_tensor = predict_eval_action_batch_dict(
                eval_algo, obs_batch_dict, deterministic=True
            )
            a_next_tensor = predict_eval_action_batch_dict(
                eval_algo, next_obs_batch_dict, deterministic=True
            )
            
            # Compute log-probabilities (batch)
            eval_logp_curr_tensor = compute_action_log_prob_batch_dict(
                eval_algo, obs_batch_dict, a_curr_tensor
            )
            eval_logp_next_tensor = compute_action_log_prob_batch_dict(
                eval_algo, next_obs_batch_dict, a_next_tensor
            )
            
            # Convert to numpy and store
            a_curr_np = a_curr_tensor.cpu().numpy().astype(np.float32)
            a_next_np = a_next_tensor.cpu().numpy().astype(np.float32)
            eval_logp_curr_np = eval_logp_curr_tensor.cpu().numpy().astype(np.float32)
            eval_logp_next_np = eval_logp_next_tensor.cpu().numpy().astype(np.float32)
            
            # Append to lists
            for j in range(end_idx - i):
                eval_action_curr_list.append(a_curr_np[j])
                eval_action_next_list.append(a_next_np[j])
                eval_log_prob_curr_list.append(eval_logp_curr_np[j])
                eval_log_prob_next_list.append(eval_logp_next_np[j])
            
            pbar.update(end_idx - i)

    # Cache the computed values
    dataset.eval_action_curr = np.asarray(eval_action_curr_list, dtype=np.float32)
    dataset.eval_action_next = np.asarray(eval_action_next_list, dtype=np.float32)
    dataset.eval_log_prob_curr = np.asarray(eval_log_prob_curr_list, dtype=np.float32)
    dataset.eval_log_prob_next = np.asarray(eval_log_prob_next_list, dtype=np.float32)

    return dataset

