"""Integration tests for OPE estimators.

Tests numerical stability, kernel vs non-kernel comparison, and edge cases.
"""

import numpy as np
import pytest

from gc_ope.algorithm.ope.ope_input import OPEInputs
from gc_ope.algorithm.ope.estimators import (
    dm_estimate,
    tis_estimate,
    tis_estimate_kernel,
    pdis_estimate,
    pdis_estimate_kernel,
    dr_estimate,
    dr_estimate_kernel,
    tis_compute_trajectory_values,
    pdis_compute_trajectory_values,
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


class TestNumericalStability:
    """Tests for numerical stability of estimators."""

    def test_dm_always_finite(self):
        """DM should always produce finite results."""
        inputs = create_mock_ope_inputs()
        result = dm_estimate(inputs)
        assert np.isfinite(result.mean), "DM mean should be finite"
        assert np.isfinite(result.ci_lower), "DM ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "DM ci_upper should be finite"

    def test_tis_kernel_finite(self):
        """TIS kernel version should produce finite results."""
        inputs = create_mock_ope_inputs()
        result = tis_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)
        assert np.isfinite(result.mean), "TIS kernel mean should be finite"
        assert np.isfinite(result.ci_lower), "TIS kernel ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "TIS kernel ci_upper should be finite"

    def test_pdis_kernel_finite(self):
        """PDIS kernel version should produce finite results."""
        inputs = create_mock_ope_inputs()
        result = pdis_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)
        assert np.isfinite(result.mean), "PDIS kernel mean should be finite"
        assert np.isfinite(result.ci_lower), "PDIS kernel ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "PDIS kernel ci_upper should be finite"

    def test_dr_kernel_finite(self):
        """DR kernel version should produce finite results."""
        inputs = create_mock_ope_inputs()
        result = dr_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)
        assert np.isfinite(result.mean), "DR kernel mean should be finite"
        assert np.isfinite(result.ci_lower), "DR kernel ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "DR kernel ci_upper should be finite"


class TestKernelVsNonKernel:
    """Tests comparing kernel and non-kernel versions."""

    def test_tis_kernel_more_stable(self):
        """TIS kernel should be at least as stable as non-kernel."""
        inputs = create_mock_ope_inputs()
        result_kernel = tis_estimate_kernel(inputs, kernel="gaussian", bandwidth=10.0)
        result_non_kernel = tis_estimate(inputs)

        # Kernel version should always be finite
        assert np.isfinite(result_kernel.mean)
        # If non-kernel is finite, kernel should also be finite
        if np.isfinite(result_non_kernel.mean):
            assert np.isfinite(result_kernel.mean)

    def test_pdis_kernel_more_stable(self):
        """PDIS kernel should be at least as stable as non-kernel."""
        inputs = create_mock_ope_inputs()
        result_kernel = pdis_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)
        result_non_kernel = pdis_estimate(inputs)

        assert np.isfinite(result_kernel.mean)
        if np.isfinite(result_non_kernel.mean):
            assert np.isfinite(result_kernel.mean)

    def test_dr_kernel_more_stable(self):
        """DR kernel should be at least as stable as non-kernel."""
        inputs = create_mock_ope_inputs()
        result_kernel = dr_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)
        result_non_kernel = dr_estimate(inputs)

        assert np.isfinite(result_kernel.mean)
        if np.isfinite(result_non_kernel.mean):
            assert np.isfinite(result_kernel.mean)


class TestPDISvsTIS:
    """Tests comparing PDIS and TIS estimators."""

    def test_pdis_trajectory_values_finite(self):
        """PDIS trajectory values should be finite."""
        inputs = create_mock_ope_inputs()
        traj_vals = pdis_compute_trajectory_values(inputs)
        assert np.all(np.isfinite(traj_vals)), "All PDIS trajectory values should be finite"

    def test_tis_trajectory_values_finite(self):
        """TIS trajectory values should be finite (with clipping)."""
        inputs = create_mock_ope_inputs()
        traj_vals = tis_compute_trajectory_values(inputs)
        assert np.all(np.isfinite(traj_vals)), "All TIS trajectory values should be finite"


class TestConfidenceIntervals:
    """Tests for confidence interval computation."""

    def test_ci_ordering(self):
        """CI bounds should be properly ordered."""
        inputs = create_mock_ope_inputs()
        result = dm_estimate(inputs)
        assert result.ci_lower <= result.mean <= result.ci_upper

    @pytest.mark.parametrize("ci_method", ["bootstrap", "normal", "t_test"])
    def test_all_ci_methods(self, ci_method):
        """All CI methods should work."""
        inputs = create_mock_ope_inputs()
        result = dm_estimate(inputs, ci_method=ci_method)
        assert np.isfinite(result.mean)
        assert np.isfinite(result.ci_lower)
        assert np.isfinite(result.ci_upper)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_trajectory(self):
        """Should handle single trajectory."""
        inputs = create_mock_ope_inputs(n_trajectories=1, steps_per_traj=10)
        result = dm_estimate(inputs)
        assert np.isfinite(result.mean)

    def test_short_trajectory(self):
        """Should handle very short trajectories."""
        inputs = create_mock_ope_inputs(n_trajectories=5, steps_per_traj=2)
        result = dm_estimate(inputs)
        assert np.isfinite(result.mean)

    def test_zero_rewards(self):
        """Should handle zero rewards."""
        inputs = create_mock_ope_inputs()
        inputs.rewards[:] = 0.0
        result = tis_estimate_kernel(inputs, kernel="gaussian", bandwidth=1.0)
        assert np.isfinite(result.mean)
        assert result.mean == pytest.approx(0.0, abs=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
