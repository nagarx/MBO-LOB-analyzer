"""Tests for core.statistics."""

import numpy as np
import pytest

from rawlobanalyzer.core.statistics import (
    WelfordAccumulator,
    coefficient_of_variation,
    distribution_summary,
    var_cvar,
)


class TestWelfordAccumulator:
    def test_single_value(self):
        acc = WelfordAccumulator()
        acc.update(5.0)
        assert acc.count == 1
        assert acc.mean == 5.0
        assert acc.variance == 0.0

    def test_two_values(self):
        acc = WelfordAccumulator()
        acc.update(2.0)
        acc.update(4.0)
        assert acc.count == 2
        assert acc.mean == pytest.approx(3.0)
        assert acc.sample_variance == pytest.approx(2.0)

    def test_batch_matches_sequential(self):
        rng = np.random.default_rng(42)
        data = rng.normal(0, 1, 1000)

        seq = WelfordAccumulator()
        for x in data:
            seq.update(float(x))

        batch = WelfordAccumulator()
        batch.update_batch(data)

        assert batch.count == seq.count
        assert batch.mean == pytest.approx(seq.mean, rel=1e-10)
        assert batch.variance == pytest.approx(seq.variance, rel=1e-6)

    def test_incremental_batch(self):
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 2.0, 500)

        acc = WelfordAccumulator()
        acc.update_batch(data[:200])
        acc.update_batch(data[200:])

        expected_mean = float(np.mean(data))
        expected_var = float(np.var(data))

        assert acc.count == 500
        assert acc.mean == pytest.approx(expected_mean, rel=1e-10)
        assert acc.variance == pytest.approx(expected_var, rel=1e-6)

    def test_empty_batch(self):
        acc = WelfordAccumulator()
        acc.update_batch(np.array([]))
        assert acc.count == 0

    def test_nan_batch_excluded(self):
        """NaN values in a batch should not inflate the count."""
        acc = WelfordAccumulator()
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        acc.update_batch(data)
        assert acc.count == 4
        assert acc.mean == pytest.approx(3.0)

    def test_all_nan_batch(self):
        """Batch of all NaN values is a no-op."""
        acc = WelfordAccumulator()
        acc.update(10.0)
        acc.update_batch(np.array([np.nan, np.nan]))
        assert acc.count == 1
        assert acc.mean == pytest.approx(10.0)


class TestDistributionSummary:
    def test_basic(self):
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        ds = distribution_summary(data)
        assert ds.count == 5
        assert ds.mean == pytest.approx(3.0)
        assert ds.min == 1.0
        assert ds.max == 5.0
        assert "p50" in ds.percentiles
        assert ds.percentiles["p50"] == pytest.approx(3.0)

    def test_empty(self):
        ds = distribution_summary(np.array([]))
        assert ds.count == 0
        assert np.isnan(ds.mean)

    def test_nan_handling(self):
        data = np.array([1.0, np.nan, 3.0, np.nan, 5.0])
        ds = distribution_summary(data, nan_policy="omit")
        assert ds.count == 3
        assert ds.mean == pytest.approx(3.0)

    def test_serialization(self):
        data = np.array([1.0, 2.0, 3.0])
        ds = distribution_summary(data)
        d = ds.to_dict()
        assert isinstance(d, dict)
        assert d["count"] == 3


class TestVarCvar:
    def test_normal_distribution(self):
        rng = np.random.default_rng(42)
        returns = rng.normal(0, 0.01, 10000)
        var, cvar = var_cvar(returns, alpha=0.05)
        assert var < 0
        assert cvar < var  # CVaR is more extreme than VaR

    def test_insufficient_data(self):
        var, cvar = var_cvar(np.array([0.01, -0.01]), alpha=0.05)
        assert np.isnan(var)


class TestCoefficientOfVariation:
    def test_basic(self):
        data = np.array([10.0, 10.5, 9.5, 10.2])
        cv = coefficient_of_variation(data)
        assert 0.0 < cv < 1.0

    def test_zero_mean(self):
        data = np.array([-1.0, 1.0])
        cv = coefficient_of_variation(data)
        assert np.isnan(cv)
