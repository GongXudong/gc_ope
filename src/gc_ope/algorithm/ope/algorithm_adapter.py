"""Algorithm adapter for supporting multiple RL algorithms in OPE.

This module provides adapters to compute action log-probabilities and
predict actions for different RL algorithms (SAC, PPO, HER, etc.).
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3 import SAC, PPO
from stable_baselines3.her import HerReplayBuffer


def detect_algorithm_type(algo: BaseAlgorithm) -> str:
    """Detect the type of algorithm.

    Args:
        algo: Stable-Baselines3 algorithm instance.

    Returns:
        Algorithm type string: "sac", "ppo", "her", or "unknown".
    """
    # Check for HER (which wraps other algorithms)
    if hasattr(algo, 'replay_buffer') and isinstance(algo.replay_buffer, HerReplayBuffer):
        return "her"
    
    # Check for SAC
    if isinstance(algo, SAC):
        return "sac"
    
    # Check for PPO
    if isinstance(algo, PPO):
        return "ppo"
    
    # Fallback: check by class name
    class_name = type(algo).__name__.lower()
    if "sac" in class_name:
        return "sac"
    elif "ppo" in class_name:
        return "ppo"
    elif "her" in class_name:
        return "her"
    
    return "unknown"


def compute_action_log_prob_sac(
    algo: BaseAlgorithm, obs: Dict[str, Any], action: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute action log-probability for SAC algorithm.

    Args:
        algo: SAC algorithm instance.
        obs: Observation dictionary (goal-conditioned).
        action: Action array (1D or 2D).

    Returns:
        Tuple of (action_tensor, log_probability).
    """
    policy = algo.policy
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


def compute_action_log_prob_ppo(
    algo: BaseAlgorithm, obs: Dict[str, Any], action: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute action log-probability for PPO algorithm.

    Args:
        algo: PPO algorithm instance.
        obs: Observation dictionary (goal-conditioned).
        action: Action array (1D or 2D).

    Returns:
        Tuple of (action_tensor, log_probability).
    """
    policy = algo.policy
    obs_tensor = policy.obs_to_tensor(obs)[0]  # type: ignore[arg-type]
    # Ensure action is 2D: (batch_size=1, action_dim)
    if action.ndim == 1:
        action = action[np.newaxis, :]
    act_tensor = th.as_tensor(action, device=policy.device, dtype=th.float32)
    with th.no_grad():
        # PPO uses policy.action_dist directly
        # Get distribution parameters
        features = policy.extract_features(obs_tensor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        mean_actions = policy.action_net(latent_pi)
        
        # PPO uses a learnable log_std or fixed log_std
        if hasattr(policy, 'log_std'):
            log_std = policy.log_std
        else:
            # Fallback: try to get from action_dist
            log_std = getattr(policy.action_dist, 'log_std', None)
            if log_std is None:
                # Default: use a small value
                log_std = th.zeros_like(mean_actions)
        
        dist = policy.action_dist.proba_distribution(mean_actions, log_std)
        log_prob = dist.log_prob(act_tensor)
        # log_prob may be scalar or 1D, ensure we get a scalar
        if log_prob.numel() == 1:
            log_prob_val = log_prob.item()
        else:
            log_prob_val = log_prob[0].item()
    # Return original 1D action (not batched)
    return act_tensor[0].cpu().numpy().astype(np.float32), log_prob_val


def compute_action_log_prob(
    algo: BaseAlgorithm, obs: Dict[str, Any], action: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Compute action log-probability under an SB3 policy (supports multiple algorithms).

    For continuous actions, computes :math:`\\log \\pi(a|s)` where :math:`\\pi` is the
    policy network. Supports SAC, PPO, and HER (which wraps SAC or DDPG).

    Args:
        algo: Stable-Baselines3 algorithm with a continuous actor.
        obs: Observation dictionary (goal-conditioned).
        action: Action array (1D or 2D).

    Returns:
        Tuple of (action_tensor, log_probability) where action_tensor is the
        batched action and log_probability is a scalar float.

    Raises:
        ValueError: If algorithm type is not supported.
    """
    algo_type = detect_algorithm_type(algo)
    
    # HER typically wraps SAC or DDPG, so we use the underlying algorithm
    if algo_type == "her":
        # HER wraps another algorithm, use the underlying one
        # The underlying algorithm is typically SAC
        return compute_action_log_prob_sac(algo, obs, action)
    elif algo_type == "sac":
        return compute_action_log_prob_sac(algo, obs, action)
    elif algo_type == "ppo":
        return compute_action_log_prob_ppo(algo, obs, action)
    else:
        # Fallback: try SAC method first (most common)
        try:
            return compute_action_log_prob_sac(algo, obs, action)
        except (AttributeError, TypeError):
            # If SAC method fails, try PPO method
            try:
                return compute_action_log_prob_ppo(algo, obs, action)
            except (AttributeError, TypeError) as e:
                raise ValueError(
                    f"Unsupported algorithm type: {algo_type}. "
                    f"Algorithm class: {type(algo).__name__}. "
                    f"Error: {e}"
                ) from e


def predict_action_sac(
    policy: Any, obs_tensor: th.Tensor, deterministic: bool = True
) -> th.Tensor:
    """Predict action for SAC policy.

    Args:
        policy: SAC policy instance.
        obs_tensor: Observation tensor (batch_size, obs_dim).
        deterministic: Whether to use deterministic action (default: True).

    Returns:
        Action tensor (batch_size, act_dim).
    """
    with th.no_grad():
        mean, log_std, kwargs = policy.actor.get_action_dist_params(obs_tensor)
        actions = policy.actor.action_dist.actions_from_params(
            mean, log_std, deterministic=deterministic, **kwargs
        )
    return actions


def predict_action_ppo(
    policy: Any, obs_tensor: th.Tensor, deterministic: bool = True
) -> th.Tensor:
    """Predict action for PPO policy.

    Args:
        policy: PPO policy instance.
        obs_tensor: Observation tensor (batch_size, obs_dim).
        deterministic: Whether to use deterministic action (default: True).

    Returns:
        Action tensor (batch_size, act_dim).
    """
    with th.no_grad():
        # PPO uses policy.action_dist directly
        features = policy.extract_features(obs_tensor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        mean_actions = policy.action_net(latent_pi)
        
        if deterministic:
            # For deterministic, use mean action
            actions = mean_actions
        else:
            # For stochastic, sample from distribution
            if hasattr(policy, 'log_std'):
                log_std = policy.log_std
            else:
                log_std = getattr(policy.action_dist, 'log_std', None)
                if log_std is None:
                    log_std = th.zeros_like(mean_actions)
            dist = policy.action_dist.proba_distribution(mean_actions, log_std)
            actions = dist.sample()
    return actions


def predict_eval_action(
    algo: BaseAlgorithm, obs_tensor: th.Tensor, deterministic: bool = True
) -> th.Tensor:
    """Predict evaluation policy action (supports multiple algorithms).

    Args:
        algo: Stable-Baselines3 algorithm instance.
        obs_tensor: Observation tensor (batch_size, obs_dim).
        deterministic: Whether to use deterministic action (default: True).

    Returns:
        Action tensor (batch_size, act_dim).

    Raises:
        ValueError: If algorithm type is not supported.
    """
    algo_type = detect_algorithm_type(algo)
    policy = algo.policy
    
    # HER typically wraps SAC or DDPG
    if algo_type == "her":
        return predict_action_sac(policy, obs_tensor, deterministic)
    elif algo_type == "sac":
        return predict_action_sac(policy, obs_tensor, deterministic)
    elif algo_type == "ppo":
        return predict_action_ppo(policy, obs_tensor, deterministic)
    else:
        # Fallback: try SAC method first
        try:
            return predict_action_sac(policy, obs_tensor, deterministic)
        except (AttributeError, TypeError):
            # If SAC method fails, try PPO method
            try:
                return predict_action_ppo(policy, obs_tensor, deterministic)
            except (AttributeError, TypeError) as e:
                raise ValueError(
                    f"Unsupported algorithm type: {algo_type}. "
                    f"Algorithm class: {type(algo).__name__}. "
                    f"Error: {e}"
                ) from e


def predict_eval_action_batch_dict(
    algo: BaseAlgorithm, obs_batch_dict: Dict[str, th.Tensor], deterministic: bool = True
) -> th.Tensor:
    """Predict evaluation policy action from batch dictionary observations.

    Args:
        algo: Stable-Baselines3 algorithm instance.
        obs_batch_dict: Batch dictionary of observations {key: (batch_size, dim)}.
        deterministic: Whether to use deterministic action (default: True).

    Returns:
        Action tensor (batch_size, act_dim).
    """
    policy = algo.policy
    device = policy.device
    
    # Move to device
    obs_batch_dict = {k: v.to(device) for k, v in obs_batch_dict.items()}
    
    # Predict actions based on algorithm type
    # Policy's get_action_dist_params expects dict format for dict observation spaces
    algo_type = detect_algorithm_type(algo)
    with th.no_grad():
        if algo_type in ["sac", "her"]:
            # For SAC, get_action_dist_params expects dict and will handle preprocessing internally
            mean, log_std, kwargs = policy.actor.get_action_dist_params(obs_batch_dict)
            actions = policy.actor.action_dist.actions_from_params(
                mean, log_std, deterministic=deterministic, **kwargs
            )
        elif algo_type == "ppo":
            # For PPO, we need to preprocess and extract features first
            from stable_baselines3.common.preprocessing import preprocess_obs
            obs_tensor = preprocess_obs(
                obs_batch_dict,
                policy.observation_space,
                normalize_images=policy.normalize_images
            )
            if isinstance(obs_tensor, dict):
                obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
            else:
                obs_tensor = obs_tensor.to(device)
            
            if hasattr(policy, 'mlp_extractor'):
                features = policy.extract_features(obs_tensor, policy.features_extractor)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                mean_actions = policy.action_net(latent_pi)
                if deterministic:
                    actions = mean_actions
                else:
                    if hasattr(policy, 'log_std'):
                        log_std = policy.log_std
                    else:
                        log_std = getattr(policy.action_dist, 'log_std', th.zeros_like(mean_actions))
                    dist = policy.action_dist.proba_distribution(mean_actions, log_std)
                    actions = dist.sample()
            else:
                raise ValueError("PPO policy missing mlp_extractor")
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")
    
    return actions


def compute_action_log_prob_batch_dict(
    algo: BaseAlgorithm, obs_batch_dict: Dict[str, th.Tensor], action_tensor: th.Tensor
) -> th.Tensor:
    """Compute action log-probability from batch dictionary observations.

    Args:
        algo: Stable-Baselines3 algorithm instance.
        obs_batch_dict: Batch dictionary of observations {key: (batch_size, dim)}.
        action_tensor: Action tensor (batch_size, act_dim).

    Returns:
        Log-probability tensor (batch_size,).
    """
    policy = algo.policy
    device = policy.device
    
    # Move to device
    obs_batch_dict = {k: v.to(device) for k, v in obs_batch_dict.items()}
    action_tensor = action_tensor.to(device)
    
    # Compute log-prob based on algorithm type
    algo_type = detect_algorithm_type(algo)
    with th.no_grad():
        if algo_type in ["sac", "her"]:
            # For SAC, get_action_dist_params expects dict and will handle preprocessing internally
            mean, log_std, kwargs = policy.actor.get_action_dist_params(obs_batch_dict)
            dist = policy.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
            log_prob = dist.log_prob(action_tensor)
        elif algo_type == "ppo":
            # For PPO, we need to preprocess and extract features first
            from stable_baselines3.common.preprocessing import preprocess_obs
            obs_tensor = preprocess_obs(
                obs_batch_dict,
                policy.observation_space,
                normalize_images=policy.normalize_images
            )
            if isinstance(obs_tensor, dict):
                obs_tensor = {k: v.to(device) for k, v in obs_tensor.items()}
            else:
                obs_tensor = obs_tensor.to(device)
            
            if hasattr(policy, 'mlp_extractor'):
                features = policy.extract_features(obs_tensor, policy.features_extractor)
                latent_pi = policy.mlp_extractor.forward_actor(features)
                mean_actions = policy.action_net(latent_pi)
                if hasattr(policy, 'log_std'):
                    log_std = policy.log_std
                else:
                    log_std = getattr(policy.action_dist, 'log_std', th.zeros_like(mean_actions))
                dist = policy.action_dist.proba_distribution(mean_actions, log_std)
                log_prob = dist.log_prob(action_tensor)
            else:
                raise ValueError("PPO policy missing mlp_extractor")
        else:
            raise ValueError(f"Unsupported algorithm type: {algo_type}")
    
    return log_prob


def compute_action_log_prob_batch_sac(
    algo: BaseAlgorithm, obs_tensor: th.Tensor, action_tensor: th.Tensor
) -> th.Tensor:
    """Compute action log-probability for SAC algorithm (batch version).

    Args:
        algo: SAC algorithm instance.
        obs_tensor: Observation tensor (batch_size, obs_dim).
        action_tensor: Action tensor (batch_size, act_dim).

    Returns:
        Log-probability tensor (batch_size,).
    """
    policy = algo.policy
    with th.no_grad():
        mean, log_std, kwargs = policy.actor.get_action_dist_params(obs_tensor)
        dist = policy.actor.action_dist.proba_distribution(mean, log_std, **kwargs)
        log_prob = dist.log_prob(action_tensor)
    return log_prob


def compute_action_log_prob_batch_ppo(
    algo: BaseAlgorithm, obs_tensor: th.Tensor, action_tensor: th.Tensor
) -> th.Tensor:
    """Compute action log-probability for PPO algorithm (batch version).

    Args:
        algo: PPO algorithm instance.
        obs_tensor: Observation tensor (batch_size, obs_dim).
        action_tensor: Action tensor (batch_size, act_dim).

    Returns:
        Log-probability tensor (batch_size,).
    """
    policy = algo.policy
    with th.no_grad():
        # PPO uses policy.action_dist directly
        features = policy.extract_features(obs_tensor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        mean_actions = policy.action_net(latent_pi)
        
        # PPO uses a learnable log_std or fixed log_std
        if hasattr(policy, 'log_std'):
            log_std = policy.log_std
        else:
            # Fallback: try to get from action_dist
            log_std = getattr(policy.action_dist, 'log_std', None)
            if log_std is None:
                # Default: use a small value
                log_std = th.zeros_like(mean_actions)
        
        dist = policy.action_dist.proba_distribution(mean_actions, log_std)
        log_prob = dist.log_prob(action_tensor)
    return log_prob


def compute_action_log_prob_batch(
    algo: BaseAlgorithm, obs_tensor: th.Tensor, action_tensor: th.Tensor
) -> th.Tensor:
    """Compute action log-probability under an SB3 policy (batch version).

    For continuous actions, computes :math:`\\log \\pi(a|s)` where :math:`\\pi` is the
    policy network. Supports SAC, PPO, and HER (which wraps SAC or DDPG).

    Args:
        algo: Stable-Baselines3 algorithm with a continuous actor.
        obs_tensor: Observation tensor (batch_size, obs_dim).
        action_tensor: Action tensor (batch_size, act_dim).

    Returns:
        Log-probability tensor (batch_size,).

    Raises:
        ValueError: If algorithm type is not supported.
    """
    algo_type = detect_algorithm_type(algo)
    
    # HER typically wraps SAC or DDPG, so we use the underlying algorithm
    if algo_type == "her":
        return compute_action_log_prob_batch_sac(algo, obs_tensor, action_tensor)
    elif algo_type == "sac":
        return compute_action_log_prob_batch_sac(algo, obs_tensor, action_tensor)
    elif algo_type == "ppo":
        return compute_action_log_prob_batch_ppo(algo, obs_tensor, action_tensor)
    else:
        # Fallback: try SAC method first (most common)
        try:
            return compute_action_log_prob_batch_sac(algo, obs_tensor, action_tensor)
        except (AttributeError, TypeError):
            # If SAC method fails, try PPO method
            try:
                return compute_action_log_prob_batch_ppo(algo, obs_tensor, action_tensor)
            except (AttributeError, TypeError) as e:
                raise ValueError(
                    f"Unsupported algorithm type: {algo_type}. "
                    f"Algorithm class: {type(algo).__name__}. "
                    f"Error: {e}"
                ) from e
