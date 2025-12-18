"""Fitted Q Evaluation (FQE) for goal-conditioned continuous control."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

import torch as th
import torch.nn as nn
from stable_baselines3.common.base_class import BaseAlgorithm
from torch.utils.data import DataLoader, TensorDataset

from .logged_dataset import LoggedDataset
from .algorithm_adapter import predict_eval_action


class FQEQNetwork(nn.Module):
    """Q-network for Fitted Q Evaluation with optional goal-conditioned optimization.

    Supports two modes:
    1. Simple mode (default): Concatenates observation and action, then passes through MLP.
    2. Goal-conditioned mode: Separately processes observation state and goal, then fuses them.

    Args:
        obs_dim: Dimension of flattened observation (includes state + goals if flattened).
        act_dim: Dimension of action space.
        hidden_sizes: Hidden layer sizes (default: (256, 256)).
        obs_state_dim: Optional dimension of observation state (without goals).
            If provided, enables goal-conditioned mode.
        goal_dim: Optional dimension of goal (desired_goal and achieved_goal, typically same).
            Required if obs_state_dim is provided.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: Sequence[int] = (256, 256),
        obs_state_dim: Optional[int] = None,
        goal_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.use_goal_conditioned = obs_state_dim is not None and goal_dim is not None

        if self.use_goal_conditioned:
            # Goal-conditioned mode: separate processing for state and goal
            if obs_state_dim + 2 * goal_dim != obs_dim:
                raise ValueError(
                    f"obs_state_dim ({obs_state_dim}) + 2 * goal_dim ({2 * goal_dim}) "
                    f"must equal obs_dim ({obs_dim})"
                )

            # Store dimensions for forward pass
            self._obs_state_dim = obs_state_dim
            self._goal_dim = goal_dim

            # State encoder
            state_layers: list[nn.Module] = []
            in_dim = obs_state_dim
            for h in hidden_sizes:
                state_layers.append(nn.Linear(in_dim, h))
                state_layers.append(nn.ReLU())
                in_dim = h
            self.state_encoder = nn.Sequential(*state_layers)

            # Goal encoder (processes desired_goal and achieved_goal together)
            goal_layers: list[nn.Module] = []
            in_dim = 2 * goal_dim  # desired_goal + achieved_goal
            for h in hidden_sizes:
                goal_layers.append(nn.Linear(in_dim, h))
                goal_layers.append(nn.ReLU())
                in_dim = h
            self.goal_encoder = nn.Sequential(*goal_layers)

            # Fusion layer: combines state and goal features, then adds action
            fusion_layers: list[nn.Module] = []
            in_dim = hidden_sizes[-1] * 2 + act_dim  # state_feat + goal_feat + action
            for h in hidden_sizes:
                fusion_layers.append(nn.Linear(in_dim, h))
                fusion_layers.append(nn.ReLU())
                in_dim = h
            fusion_layers.append(nn.Linear(in_dim, 1))
            self.fusion_net = nn.Sequential(*fusion_layers)
        else:
            # Simple mode: concatenate obs and action, then MLP
            layers: list[nn.Module] = []
            in_dim = obs_dim + act_dim
            for h in hidden_sizes:
                layers.append(nn.Linear(in_dim, h))
                layers.append(nn.ReLU())
                in_dim = h
            layers.append(nn.Linear(in_dim, 1))
            self.net = nn.Sequential(*layers)

    def forward(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        """Forward pass: Q(s, a).

        Args:
            obs: Observation tensor (batch_size, obs_dim).
                In goal-conditioned mode, obs_dim = obs_state_dim + 2 * goal_dim,
                where the format is [state, desired_goal, achieved_goal].
            act: Action tensor (batch_size, act_dim).

        Returns:
            Q-values (batch_size,).
        """
        if self.use_goal_conditioned:
            # Split observation into state and goals
            # obs format: [state, desired_goal, achieved_goal]
            obs_state = obs[:, :self._obs_state_dim]
            goal_part = obs[:, self._obs_state_dim:]
            goal_desired = goal_part[:, :self._goal_dim]
            goal_achieved = goal_part[:, self._goal_dim:]
            goal_combined = th.cat([goal_desired, goal_achieved], dim=-1)

            state_feat = self.state_encoder(obs_state)
            goal_feat = self.goal_encoder(goal_combined)
            fused = th.cat([state_feat, goal_feat, act], dim=-1)
            return self.fusion_net(fused).squeeze(-1)
        else:
            # Simple mode
            x = th.cat([obs, act], dim=-1)
            return self.net(x).squeeze(-1)


class FQETrainer:
    r"""Fitted Q Evaluation trainer for continuous goal-conditioned policies.

    FQE is an off-policy evaluation method that approximates a Q function
    :math:`Q_\theta(s, a)` for the evaluation policy :math:`\pi_\phi(s)`.

    The FQE loss is:

    .. math::

        L(\theta) = \mathbb{E}_{(s_t, a_t, r_{t+1}, s_{t+1}) \sim D}
            \left[ \left( Q_\theta(s_t, a_t) - r_{t+1}
                - \gamma Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1})) \right)^2 \right]

    Plain text formula: L(θ) = E[(Q_θ(s_t, a_t) - r_{t+1} - γ Q_θ'(s_{t+1}, π_φ(s_{t+1})))^2]
    where the expectation is over transitions (s_t, a_t, r_{t+1}, s_{t+1}) from dataset D.

    where :math:`D` is the logged dataset, :math:`\theta'` is the target network
    parameters (soft-updated with :math:`\tau`), and :math:`\pi_\phi(s_{t+1})`
    is the deterministic action from the evaluation policy.

    The trained Q function in FQE estimates evaluation metrics more accurately
    than the Q function learned during policy training.

    References:
        * `Le et al., Batch Policy Learning under Constraints.
          <https://arxiv.org/abs/1903.08738>`_

    Args:
        obs_dim: Dimension of flattened observation.
        act_dim: Dimension of action space.
        eval_algo: Evaluation policy (e.g., SAC checkpoint).
        gamma: Discount factor (default: 0.99).
        tau: Soft update coefficient for target network (default: 0.005).
        lr: Learning rate for Q-network optimizer (default: 3e-4).
        hidden_sizes: Hidden layer sizes for Q-network (default: (256, 256)).
        obs_state_dim: Optional dimension of observation state (without goals).
            If provided with goal_dim, enables goal-conditioned network optimization.
        goal_dim: Optional dimension of goal (desired_goal and achieved_goal, typically same).
            Required if obs_state_dim is provided.
        device: PyTorch device (default: uses eval_algo.device).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        eval_algo: BaseAlgorithm,
        gamma: float = 0.99,
        tau: float = 0.005,
        lr: float = 3e-4,
        hidden_sizes: Sequence[int] = (256, 256),
        obs_state_dim: Optional[int] = None,
        goal_dim: Optional[int] = None,
        device: Optional[str] = None,
    ) -> None:
        # Determine device: use provided device, or eval_algo.device, or default to cuda if available
        if device is not None:
            self.device = th.device(device)
        else:
            algo_device = getattr(eval_algo, 'device', None)
            if algo_device is not None:
                self.device = th.device(algo_device)
            else:
                # Default to cuda if available, otherwise cpu
                self.device = th.device('cuda' if th.cuda.is_available() else 'cpu')
        self.eval_algo = eval_algo
        self.gamma = gamma
        self.tau = tau
        self.q = FQEQNetwork(
            obs_dim, act_dim, hidden_sizes, obs_state_dim=obs_state_dim, goal_dim=goal_dim
        ).to(self.device)
        self.q_target = FQEQNetwork(
            obs_dim, act_dim, hidden_sizes, obs_state_dim=obs_state_dim, goal_dim=goal_dim
        ).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.optim = th.optim.Adam(self.q.parameters(), lr=lr)

    def _predict_eval_action(self, next_obs: th.Tensor) -> th.Tensor:
        """Compute deterministic evaluation policy action.

        Supports multiple algorithms (SAC, PPO, HER, etc.) via algorithm adapter.

        Args:
            next_obs: Next state tensor (batch_size, obs_dim), should be on self.device.

        Returns:
            Deterministic actions from evaluation policy (batch_size, act_dim), on self.device.
        """
        policy = self.eval_algo.policy
        # Convert to policy device for computation, then back to self.device
        next_obs_policy = next_obs.to(policy.device)
        # Use algorithm adapter to support multiple algorithm types
        actions = predict_eval_action(self.eval_algo, next_obs_policy, deterministic=True)
        return actions.to(self.device)

    def fit(
        self,
        dataset: LoggedDataset,
        batch_size: int = 256,
        n_epochs: int = 500,
        shuffle: bool = True,
        logger: Optional[callable] = None,
    ) -> None:
        """Train FQE Q-network on logged dataset.

        Minimizes the FQE loss:

        .. math::

            L(\theta) = \mathbb{E}_{(s_t, a_t, r_{t+1}, s_{t+1}) \sim D}
                \left[ \left( Q_\theta(s_t, a_t) - y_t \right)^2 \right]

        Plain text: L(θ) = E[(Q_θ(s_t, a_t) - y_t)^2]

        where the target is:

        .. math::

            y_t = r_{t+1} + \gamma (1 - \text{done}_t) Q_{\theta'}(s_{t+1}, \pi_\phi(s_{t+1}))

        Plain text: y_t = r_{t+1} + γ (1 - done_t) Q_θ'(s_{t+1}, π_φ(s_{t+1}))

        Args:
            dataset: Logged dataset from behavior policy.
            batch_size: Batch size for training (default: 256).
            n_epochs: Number of training epochs (default: 500).
            shuffle: Whether to shuffle data each epoch (default: True).
            logger: Optional callback(epoch, loss) for logging (default: None).
        """
        obs = th.as_tensor(dataset.obs_flat, device=self.device)
        act = th.as_tensor(dataset.actions, device=self.device)
        rew = th.as_tensor(dataset.rewards, device=self.device).float()
        next_obs = th.as_tensor(dataset.next_obs_flat, device=self.device)
        done = th.as_tensor(dataset.dones, device=self.device).float()

        if dataset.eval_action_next is not None:
            eval_act_next_tensor = th.as_tensor(dataset.eval_action_next, device=self.device)
            data = TensorDataset(obs, act, rew, next_obs, done, eval_act_next_tensor)
            use_cached_eval = True
        else:
            data = TensorDataset(obs, act, rew, next_obs, done)
            use_cached_eval = False
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle)

        for epoch in range(1, n_epochs + 1):
            epoch_loss = 0.0
            n_batches = 0
            for batch in loader:
                # DataLoader returns tensors on CPU, move to device
                if use_cached_eval:
                    obs_b, act_b, rew_b, next_obs_b, done_b, eval_act_next_b = [t.to(self.device) for t in batch]
                    a_next = eval_act_next_b
                else:
                    obs_b, act_b, rew_b, next_obs_b, done_b = [t.to(self.device) for t in batch]
                    a_next = self._predict_eval_action(next_obs_b)

                with th.no_grad():
                    target_q = self.q_target(next_obs_b, a_next)
                    y = rew_b + self.gamma * (1.0 - done_b) * target_q

                q_pred = self.q(obs_b, act_b)
                loss = th.mean((q_pred - y) ** 2)

                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                with th.no_grad():
                    for p, p_targ in zip(self.q.parameters(), self.q_target.parameters()):
                        p_targ.data.mul_(1.0 - self.tau).add_(self.tau * p.data)

                epoch_loss += loss.item()
                n_batches += 1

            if logger is not None:
                logger(epoch=epoch, loss=epoch_loss / max(n_batches, 1))

    def predict_value(self, obs: th.Tensor, act: th.Tensor) -> th.Tensor:
        """Predict Q-value: Q(s, a).

        Args:
            obs: Observation tensor (batch_size, obs_dim), should be on self.device.
            act: Action tensor (batch_size, act_dim), should be on self.device.

        Returns:
            Q-values (batch_size,).
        """
        return self.q(obs, act)

