"""Tests for core.price_utils."""

import numpy as np
import pytest

from rawlobanalyzer.core.price_utils import (
    basis_points,
    log_returns,
    nanodollars_to_usd,
    spread_in_ticks,
    usd_to_nanodollars,
)


class TestNanodollarConversion:
    def test_basic_conversion(self):
        prices = np.array([118_000_000_000, 118_010_000_000], dtype=np.int64)
        usd = nanodollars_to_usd(prices)
        np.testing.assert_allclose(usd, [118.0, 118.01])

    def test_roundtrip(self):
        original = np.array([118_000_000_000, 0, 150_500_000_000], dtype=np.int64)
        usd = nanodollars_to_usd(original)
        back = usd_to_nanodollars(usd)
        np.testing.assert_array_equal(back, original)

    def test_zero_price(self):
        prices = np.array([0], dtype=np.int64)
        usd = nanodollars_to_usd(prices)
        assert usd[0] == 0.0


class TestLogReturns:
    def test_basic_returns(self):
        prices = np.array([100.0, 101.0, 100.0])
        rets = log_returns(prices)
        assert len(rets) == 2
        np.testing.assert_allclose(rets[0], np.log(101.0 / 100.0), rtol=1e-12)

    def test_zero_price_produces_nan(self):
        prices = np.array([100.0, 0.0, 100.0])
        rets = log_returns(prices)
        assert np.isnan(rets[0])
        assert np.isnan(rets[1])

    def test_single_price(self):
        prices = np.array([100.0])
        rets = log_returns(prices)
        assert len(rets) == 0

    def test_inf_price_produces_nan(self):
        """Inf prices must be excluded, not propagated (P1-C fix)."""
        prices = np.array([100.0, np.inf, 100.0])
        rets = log_returns(prices)
        assert np.isnan(rets[0]), "return from 100 -> Inf should be NaN"
        assert np.isnan(rets[1]), "return from Inf -> 100 should be NaN"

    def test_negative_inf_price_produces_nan(self):
        prices = np.array([100.0, -np.inf, 100.0])
        rets = log_returns(prices)
        assert np.isnan(rets[0])
        assert np.isnan(rets[1])


class TestSpreadInTicks:
    def test_one_cent_is_one_tick(self):
        spreads = np.array([0.01, 0.02, 0.05])
        ticks = spread_in_ticks(spreads)
        np.testing.assert_allclose(ticks, [1.0, 2.0, 5.0])


class TestBasisPoints:
    def test_basic(self):
        value = np.array([0.01])
        ref = np.array([100.0])
        bps = basis_points(value, ref)
        np.testing.assert_allclose(bps, [1.0])

    def test_zero_reference_is_nan(self):
        value = np.array([1.0])
        ref = np.array([0.0])
        bps = basis_points(value, ref)
        assert np.isnan(bps[0])
