"""Unit tests for bandwidth selection functions."""

import numpy as np
import pytest

from gc_ope.algorithm.ope.bandwidth_selection import (
    silverman_bandwidth,
    scott_bandwidth,
    median_bandwidth,
    select_bandwidth,
)


class TestSilvermanBandwidth:
    """Tests for Silverman's rule of thumb."""

    def test_basic_functionality(self):
        """Should return a positive bandwidth."""
        data = np.random.randn(100, 2)
        h = silverman_bandwidth(data)
        assert h > 0, "Bandwidth should be positive"

    def test_1d_data(self):
        """Should handle 1D data."""
        data = np.random.randn(100)
        h = silverman_bandwidth(data)
        assert h > 0

    def test_larger_data_smaller_bandwidth(self):
        """More data should generally lead to smaller bandwidth."""
        np.random.seed(42)
        data_small = np.random.randn(50, 2)
        data_large = np.random.randn(500, 2)
        h_small = silverman_bandwidth(data_small)
        h_large = silverman_bandwidth(data_large)
        assert h_large < h_small


class TestScottBandwidth:
    """Tests for Scott's rule."""

    def test_basic_functionality(self):
        """Should return a positive bandwidth."""
        data = np.random.randn(100, 2)
        h = scott_bandwidth(data)
        assert h > 0

    def test_1d_data(self):
        """Should handle 1D data."""
        data = np.random.randn(100)
        h = scott_bandwidth(data)
        assert h > 0


class TestMedianBandwidth:
    """Tests for median heuristic."""

    def test_basic_functionality(self):
        """Should return a positive bandwidth."""
        data = np.random.randn(100, 2)
        h = median_bandwidth(data)
        assert h > 0

    def test_subsampling(self):
        """Should handle large data with subsampling."""
        data = np.random.randn(2000, 2)
        h = median_bandwidth(data, subsample=500)
        assert h > 0


class TestSelectBandwidth:
    """Tests for select_bandwidth function."""

    def test_silverman_method(self):
        """Should use Silverman's rule."""
        data = np.random.randn(100, 2)
        h = select_bandwidth(data, method="silverman")
        assert h > 0

    def test_scott_method(self):
        """Should use Scott's rule."""
        data = np.random.randn(100, 2)
        h = select_bandwidth(data, method="scott")
        assert h > 0

    def test_median_method(self):
        """Should use median heuristic."""
        data = np.random.randn(100, 2)
        h = select_bandwidth(data, method="median")
        assert h > 0

    def test_invalid_method(self):
        """Should raise error for invalid method."""
        data = np.random.randn(100, 2)
        with pytest.raises(ValueError, match="Unknown method"):
            select_bandwidth(data, method="invalid")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
