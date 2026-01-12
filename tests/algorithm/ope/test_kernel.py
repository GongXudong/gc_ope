"""Unit tests for kernel functions and kernel-based OPE estimators."""

import numpy as np
import pytest

from gc_ope.algorithm.ope.kernel_utils import (
    l2_distance,
    gaussian_kernel,
    epanechnikov_kernel,
    get_kernel,
)


class TestL2Distance:
    """Tests for l2_distance function."""

    def test_identical_vectors(self):
        """Distance between identical vectors should be 0."""
        x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        y = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        dist = l2_distance(x, y)
        np.testing.assert_array_almost_equal(dist, [0.0, 0.0])

    def test_known_distance(self):
        """Test with known distance values."""
        x = np.array([[0.0, 0.0], [3.0, 0.0]])
        y = np.array([[3.0, 4.0], [0.0, 4.0]])
        dist = l2_distance(x, y)
        # ||[0,0] - [3,4]||^2 = 9 + 16 = 25
        # ||[3,0] - [0,4]||^2 = 9 + 16 = 25
        np.testing.assert_array_almost_equal(dist, [25.0, 25.0])

    def test_single_dimension(self):
        """Test with single dimension."""
        x = np.array([[1.0], [5.0]])
        y = np.array([[3.0], [2.0]])
        dist = l2_distance(x, y)
        np.testing.assert_array_almost_equal(dist, [4.0, 9.0])


class TestGaussianKernel:
    """Tests for gaussian_kernel function."""

    def test_identical_actions(self):
        """Kernel value should be maximum when actions are identical."""
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([[1.0, 2.0], [3.0, 4.0]])
        k = gaussian_kernel(x, y, bandwidth=1.0)
        expected = 1.0 / np.sqrt(2 * np.pi)
        np.testing.assert_array_almost_equal(k, [expected, expected])

    def test_output_bounded(self):
        """Kernel output should be bounded and positive."""
        np.random.seed(42)
        x = np.random.randn(100, 4)
        y = np.random.randn(100, 4)
        k = gaussian_kernel(x, y, bandwidth=1.0)
        assert np.all(k > 0), "Kernel values should be positive"
        assert np.all(k <= 1.0 / np.sqrt(2 * np.pi)), "Kernel values should be bounded"

    def test_bandwidth_effect(self):
        """Larger bandwidth should give higher kernel values for distant points."""
        x = np.array([[0.0, 0.0]])
        y = np.array([[1.0, 1.0]])
        k_small = gaussian_kernel(x, y, bandwidth=0.5)
        k_large = gaussian_kernel(x, y, bandwidth=2.0)
        assert k_large > k_small, "Larger bandwidth should give higher values"


class TestEpanechnikovKernel:
    """Tests for epanechnikov_kernel function."""

    def test_identical_actions(self):
        """Kernel value should be maximum when actions are identical."""
        x = np.array([[1.0, 2.0]])
        y = np.array([[1.0, 2.0]])
        k = epanechnikov_kernel(x, y, bandwidth=1.0)
        expected = 0.75  # 0.75 * (1 - 0) / 1
        np.testing.assert_array_almost_equal(k, [expected])

    def test_output_bounded(self):
        """Kernel output should be bounded and non-negative."""
        np.random.seed(42)
        x = np.random.randn(100, 4)
        y = np.random.randn(100, 4)
        k = epanechnikov_kernel(x, y, bandwidth=1.0)
        assert np.all(k >= 0), "Kernel values should be non-negative"


class TestGetKernel:
    """Tests for get_kernel function."""

    def test_get_gaussian(self):
        """Should return gaussian kernel function."""
        fn = get_kernel("gaussian")
        assert fn == gaussian_kernel

    def test_get_epanechnikov(self):
        """Should return epanechnikov kernel function."""
        fn = get_kernel("epanechnikov")
        assert fn == epanechnikov_kernel

    def test_invalid_kernel(self):
        """Should raise ValueError for invalid kernel name."""
        with pytest.raises(ValueError, match="Unsupported kernel"):
            get_kernel("invalid_kernel")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
