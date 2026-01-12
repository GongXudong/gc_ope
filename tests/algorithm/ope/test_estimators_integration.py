"""Integration tests for OPE estimators.

Tests numerical stability, kernel vs non-kernel comparison, and edge cases.
"""

import numpy as np
import pytest

from gc_ope.algorithm.ope.ope_input import OPEInputs
from gc_ope.algorithm.ope.estimators import (
    DMEstimator,
    TISEstimator,
    PDISEstimator,
    DREstimator,
    SelfNormalizedTIS,
    SelfNormalizedPDIS,
    SelfNormalizedDR,
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
        dm = DMEstimator()
        result = dm.estimate(inputs)
        assert np.isfinite(result.mean), "DM mean should be finite"
        assert np.isfinite(result.ci_lower), "DM ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "DM ci_upper should be finite"

    def test_tis_kernel_finite(self):
        """TIS kernel version should produce finite results."""
        inputs = create_mock_ope_inputs()
        tis = TISEstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = tis.estimate(inputs)
        assert np.isfinite(result.mean), "TIS kernel mean should be finite"
        assert np.isfinite(result.ci_lower), "TIS kernel ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "TIS kernel ci_upper should be finite"

    def test_pdis_kernel_finite(self):
        """PDIS kernel version should produce finite results."""
        inputs = create_mock_ope_inputs()
        pdis = PDISEstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = pdis.estimate(inputs)
        assert np.isfinite(result.mean), "PDIS kernel mean should be finite"
        assert np.isfinite(result.ci_lower), "PDIS kernel ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "PDIS kernel ci_upper should be finite"

    def test_dr_kernel_finite(self):
        """DR kernel version should produce finite results."""
        inputs = create_mock_ope_inputs()
        dr = DREstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = dr.estimate(inputs)
        assert np.isfinite(result.mean), "DR kernel mean should be finite"
        assert np.isfinite(result.ci_lower), "DR kernel ci_lower should be finite"
        assert np.isfinite(result.ci_upper), "DR kernel ci_upper should be finite"

    def test_self_normalized_tis_finite(self):
        """Self-normalized TIS should produce finite results."""
        inputs = create_mock_ope_inputs()
        sn_tis = SelfNormalizedTIS(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = sn_tis.estimate(inputs)
        assert np.isfinite(result.mean), "SN-TIS mean should be finite"

    def test_self_normalized_pdis_finite(self):
        """Self-normalized PDIS should produce finite results."""
        inputs = create_mock_ope_inputs()
        sn_pdis = SelfNormalizedPDIS(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = sn_pdis.estimate(inputs)
        assert np.isfinite(result.mean), "SN-PDIS mean should be finite"

    def test_self_normalized_dr_finite(self):
        """Self-normalized DR should produce finite results."""
        inputs = create_mock_ope_inputs()
        sn_dr = SelfNormalizedDR(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = sn_dr.estimate(inputs)
        assert np.isfinite(result.mean), "SN-DR mean should be finite"


class TestKernelVsNonKernel:
    """Tests comparing kernel and non-kernel versions."""

    def test_tis_kernel_more_stable(self):
        """TIS kernel should be at least as stable as non-kernel."""
        inputs = create_mock_ope_inputs()
        tis_kernel = TISEstimator(use_kernel=True, kernel="gaussian", bandwidth=10.0)
        tis_non_kernel = TISEstimator(use_kernel=False)

        result_kernel = tis_kernel.estimate(inputs)
        result_non_kernel = tis_non_kernel.estimate(inputs)

        # Kernel version should always be finite
        assert np.isfinite(result_kernel.mean)
        # If non-kernel is finite, kernel should also be finite
        if np.isfinite(result_non_kernel.mean):
            assert np.isfinite(result_kernel.mean)

    def test_pdis_kernel_more_stable(self):
        """PDIS kernel should be at least as stable as non-kernel."""
        inputs = create_mock_ope_inputs()
        pdis_kernel = PDISEstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        pdis_non_kernel = PDISEstimator(use_kernel=False)

        result_kernel = pdis_kernel.estimate(inputs)
        result_non_kernel = pdis_non_kernel.estimate(inputs)

        assert np.isfinite(result_kernel.mean)
        if np.isfinite(result_non_kernel.mean):
            assert np.isfinite(result_kernel.mean)

    def test_dr_kernel_more_stable(self):
        """DR kernel should be at least as stable as non-kernel."""
        inputs = create_mock_ope_inputs()
        dr_kernel = DREstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        dr_non_kernel = DREstimator(use_kernel=False)

        result_kernel = dr_kernel.estimate(inputs)
        result_non_kernel = dr_non_kernel.estimate(inputs)

        assert np.isfinite(result_kernel.mean)
        if np.isfinite(result_non_kernel.mean):
            assert np.isfinite(result_kernel.mean)


class TestPDISvsTIS:
    """Tests comparing PDIS and TIS estimators."""

    def test_pdis_trajectory_values_finite(self):
        """PDIS trajectory values should be finite."""
        inputs = create_mock_ope_inputs()
        pdis = PDISEstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        traj_vals = pdis.compute_trajectory_values(inputs)
        assert np.all(np.isfinite(traj_vals)), "All PDIS trajectory values should be finite"

    def test_tis_trajectory_values_finite(self):
        """TIS trajectory values should be finite (with clipping)."""
        inputs = create_mock_ope_inputs()
        tis = TISEstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        traj_vals = tis.compute_trajectory_values(inputs)
        assert np.all(np.isfinite(traj_vals)), "All TIS trajectory values should be finite"


class TestConfidenceIntervals:
    """Tests for confidence interval computation."""

    def test_ci_ordering(self):
        """CI bounds should be properly ordered."""
        inputs = create_mock_ope_inputs()
        dm = DMEstimator()
        result = dm.estimate(inputs)
        assert result.ci_lower <= result.mean <= result.ci_upper

    @pytest.mark.parametrize("ci_method", ["bootstrap", "normal", "t_test"])
    def test_all_ci_methods(self, ci_method):
        """All CI methods should work."""
        inputs = create_mock_ope_inputs()
        dm = DMEstimator()
        result = dm.estimate(inputs, ci_method=ci_method)
        assert np.isfinite(result.mean)
        assert np.isfinite(result.ci_lower)
        assert np.isfinite(result.ci_upper)


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_trajectory(self):
        """Should handle single trajectory."""
        inputs = create_mock_ope_inputs(n_trajectories=1, steps_per_traj=10)
        dm = DMEstimator()
        result = dm.estimate(inputs)
        assert np.isfinite(result.mean)

    def test_short_trajectory(self):
        """Should handle very short trajectories."""
        inputs = create_mock_ope_inputs(n_trajectories=5, steps_per_traj=2)
        dm = DMEstimator()
        result = dm.estimate(inputs)
        assert np.isfinite(result.mean)

    def test_zero_rewards(self):
        """Should handle zero rewards."""
        inputs = create_mock_ope_inputs()
        inputs.rewards[:] = 0.0
        tis = TISEstimator(use_kernel=True, kernel="gaussian", bandwidth=1.0)
        result = tis.estimate(inputs)
        assert np.isfinite(result.mean)
        assert result.mean == pytest.approx(0.0, abs=1e-6)

    def test_auto_bandwidth(self):
        """Should handle automatic bandwidth selection."""
        inputs = create_mock_ope_inputs()
        tis = TISEstimator(use_kernel=True, kernel="gaussian", bandwidth="auto")
        result = tis.estimate(inputs)
        assert np.isfinite(result.mean)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
