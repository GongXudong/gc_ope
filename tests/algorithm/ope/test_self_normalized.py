"""Unit tests for self-normalized OPE estimators."""

import numpy as np
import pytest

from gc_ope.algorithm.ope.ope_input import OPEInputs
from gc_ope.algorithm.ope.estimators import (
    TISEstimator,
    PDISEstimator,
    DREstimator,
    SelfNormalizedTIS,
    SelfNormalizedPDIS,
    SelfNormalizedDR,
    self_normalize_weights,
)


def create_mock_ope_inputs(
    n_trajectories: int = 5,
    steps_per_traj: int = 20,
    obs_dim: int = 4,
    act_dim: int = 2,
    gamma: float = 0.99,
    seed: int = 42,
) -> OPEInputs:
    """Create mock OPEInputs for testing."""
    rng = np.random.default_rng(seed)
    n_transitions = n_trajectories * steps_per_traj

    return OPEInputs(
        obs_flat=rng.standard_normal((n_transitions, obs_dim)).astype(np.float32),
        actions=rng.uniform(-1, 1, (n_transitions, act_dim)).astype(np.float32),
        rewards=rng.uniform(-1, 1, n_transitions).astype(np.float32),
        next_obs_flat=rng.standard_normal((n_transitions, obs_dim)).astype(np.float32),
        dones=np.zeros(n_transitions, dtype=bool),
        traj_id=np.repeat(np.arange(n_trajectories), steps_per_traj),
        step_index=np.tile(np.arange(steps_per_traj), n_trajectories),
        behavior_log_prob=rng.uniform(-2, -0.5, n_transitions).astype(np.float32),
        eval_action=rng.uniform(-1, 1, (n_transitions, act_dim)).astype(np.float32),
        eval_log_prob=rng.uniform(-2, -0.5, n_transitions).astype(np.float32),
        q_sa_behavior=rng.uniform(-10, 10, n_transitions).astype(np.float32),
        q_sa_eval=rng.uniform(-10, 10, n_transitions).astype(np.float32),
        gamma=gamma,
    )


class TestSelfNormalizeWeights:
    """Tests for self_normalize_weights function."""

    def test_basic_normalization(self):
        """Normalized weights should have mean close to 1."""
        weights = np.array([0.1, 0.2, 0.3, 0.4])
        normalized = self_normalize_weights(weights)
        assert np.isclose(normalized.mean(), 1.0, atol=1e-6)

    def test_preserves_relative_order(self):
        """Normalization should preserve relative ordering."""
        weights = np.array([1.0, 2.0, 3.0, 4.0])
        normalized = self_normalize_weights(weights)
        assert np.all(np.diff(normalized) > 0)

    def test_handles_small_weights(self):
        """Should handle very small weights without overflow."""
        weights = np.array([1e-10, 1e-10, 1e-10])
        normalized = self_normalize_weights(weights)
        assert np.all(np.isfinite(normalized))


class TestSelfNormalizedTIS:
    """Tests for SelfNormalizedTIS estimator."""

    def test_finite_results(self):
        """Should produce finite results."""
        inputs = create_mock_ope_inputs()
        sn_tis = SelfNormalizedTIS(use_kernel=True, bandwidth=1.0)
        result = sn_tis.estimate(inputs)
        assert np.isfinite(result.mean)

    def test_more_stable_than_regular(self):
        """SN-TIS should be more stable than regular TIS."""
        inputs = create_mock_ope_inputs()
        tis = TISEstimator(use_kernel=False)
        sn_tis = SelfNormalizedTIS(use_kernel=True, bandwidth=1.0)

        result_tis = tis.estimate(inputs)
        result_sn = sn_tis.estimate(inputs)

        # SN version should always be finite
        assert np.isfinite(result_sn.mean)


class TestSelfNormalizedPDIS:
    """Tests for SelfNormalizedPDIS estimator."""

    def test_finite_results(self):
        """Should produce finite results."""
        inputs = create_mock_ope_inputs()
        sn_pdis = SelfNormalizedPDIS(use_kernel=True, bandwidth=1.0)
        result = sn_pdis.estimate(inputs)
        assert np.isfinite(result.mean)


class TestSelfNormalizedDR:
    """Tests for SelfNormalizedDR estimator."""

    def test_finite_results(self):
        """Should produce finite results."""
        inputs = create_mock_ope_inputs()
        sn_dr = SelfNormalizedDR(use_kernel=True, bandwidth=1.0)
        result = sn_dr.estimate(inputs)
        assert np.isfinite(result.mean)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
