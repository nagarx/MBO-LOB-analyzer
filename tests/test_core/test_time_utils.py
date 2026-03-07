"""Tests for core.time_utils."""

import numpy as np
import pytest

from rawlobanalyzer.core.time_utils import (
    compute_inter_event_times_ns,
    ns_to_hours_since_midnight_utc,
    ns_to_seconds_since_midnight_utc,
    rth_grid_edges_ns,
    rth_mask_utc,
    time_regime,
    utc_offset_for_date,
)
from rawlobanalyzer.core.constants import NS_PER_HOUR, NS_PER_SECOND


def _make_ts(hour_utc: float) -> np.ndarray:
    """Create a timestamp array for a single time point at the given UTC hour on an arbitrary day."""
    day_start_ns = 1_738_540_800_000_000_000  # 2025-02-03 00:00:00 UTC
    return np.array([day_start_ns + int(hour_utc * NS_PER_HOUR)], dtype=np.int64)


class TestSecondsAndHours:
    def test_midnight(self):
        ts = _make_ts(0.0)
        secs = ns_to_seconds_since_midnight_utc(ts)
        np.testing.assert_allclose(secs, [0.0], atol=1e-6)

    def test_noon(self):
        ts = _make_ts(12.0)
        hours = ns_to_hours_since_midnight_utc(ts)
        np.testing.assert_allclose(hours, [12.0], atol=1e-6)


class TestRthMask:
    def test_during_rth_est(self):
        # 10:00 AM ET = 15:00 UTC (EST, offset -5)
        ts = _make_ts(15.0)
        mask = rth_mask_utc(ts, utc_offset_hours=-5)
        assert mask[0] is np.bool_(True)

    def test_before_rth_est(self):
        # 9:00 AM ET = 14:00 UTC
        ts = _make_ts(14.0)
        mask = rth_mask_utc(ts, utc_offset_hours=-5)
        assert mask[0] is np.bool_(False)

    def test_after_rth_est(self):
        # 4:30 PM ET = 21:30 UTC
        ts = _make_ts(21.5)
        mask = rth_mask_utc(ts, utc_offset_hours=-5)
        assert mask[0] is np.bool_(False)


class TestTimeRegime:
    def test_premarket(self):
        # 9:00 AM ET = 14:00 UTC
        ts = _make_ts(14.0)
        regimes = time_regime(ts, utc_offset_hours=-5)
        assert regimes[0] == 0  # pre-market

    def test_open_auction(self):
        # 9:32 AM ET = 14:32 UTC = 14.533 hours
        ts = _make_ts(14.533)
        regimes = time_regime(ts, utc_offset_hours=-5)
        assert regimes[0] == 1  # open-auction

    def test_morning(self):
        # 10:30 AM ET = 15:30 UTC
        ts = _make_ts(15.5)
        regimes = time_regime(ts, utc_offset_hours=-5)
        assert regimes[0] == 2  # morning

    def test_close_auction(self):
        # 3:50 PM ET = 20:50 UTC = 20.833 hours
        ts = _make_ts(20.833)
        regimes = time_regime(ts, utc_offset_hours=-5)
        assert regimes[0] == 5  # close-auction

    def test_after_hours(self):
        # 5:00 PM ET = 22:00 UTC
        ts = _make_ts(22.0)
        regimes = time_regime(ts, utc_offset_hours=-5)
        assert regimes[0] == 6  # after-hours


class TestRthGrid:
    _DAY_EPOCH = 1_738_540_800_000_000_000  # 2025-02-03 00:00:00 UTC

    def test_1s_grid_covers_full_rth(self):
        grid = rth_grid_edges_ns(self._DAY_EPOCH, NS_PER_SECOND, utc_offset_hours=-5)
        rth_open_ns = self._DAY_EPOCH + int(14.5 * NS_PER_HOUR)  # 9:30 ET = 14:30 UTC
        rth_close_ns = self._DAY_EPOCH + int(21.0 * NS_PER_HOUR)  # 16:00 ET = 21:00 UTC
        assert grid[0] <= rth_open_ns
        assert grid[-1] >= rth_close_ns

    def test_deterministic_across_calls(self):
        g1 = rth_grid_edges_ns(self._DAY_EPOCH, NS_PER_SECOND, utc_offset_hours=-5)
        g2 = rth_grid_edges_ns(self._DAY_EPOCH, NS_PER_SECOND, utc_offset_hours=-5)
        np.testing.assert_array_equal(g1, g2)

    def test_uniform_spacing(self):
        res = 5 * NS_PER_SECOND
        grid = rth_grid_edges_ns(self._DAY_EPOCH, res, utc_offset_hours=-5)
        diffs = np.diff(grid)
        np.testing.assert_array_equal(diffs, res)

    def test_5m_bin_count(self):
        res = 300 * NS_PER_SECOND  # 5 minutes
        grid = rth_grid_edges_ns(self._DAY_EPOCH, res, utc_offset_hours=-5)
        n_bins = len(grid) - 1
        assert n_bins == 78, f"6.5h / 5m = 78 bins, got {n_bins}"


class TestUtcOffsetForDate:
    def test_est_winter(self):
        assert utc_offset_for_date("2025-02-03") == -5

    def test_edt_summer(self):
        assert utc_offset_for_date("2025-07-01") == -4

    def test_dst_start_boundary_2025(self):
        assert utc_offset_for_date("2025-03-08") == -5
        assert utc_offset_for_date("2025-03-09") == -4

    def test_dst_end_boundary_2025(self):
        assert utc_offset_for_date("2025-11-01") == -4
        assert utc_offset_for_date("2025-11-02") == -5

    def test_new_years_2026(self):
        assert utc_offset_for_date("2026-01-07") == -5

    def test_cross_year_consistency(self):
        assert utc_offset_for_date("2024-06-15") == -4
        assert utc_offset_for_date("2024-12-25") == -5


class TestInterEventTimes:
    def test_basic(self):
        ts = np.array([100, 200, 350], dtype=np.int64)
        iets = compute_inter_event_times_ns(ts)
        np.testing.assert_array_equal(iets, [100, 150])

    def test_single_event(self):
        ts = np.array([100], dtype=np.int64)
        iets = compute_inter_event_times_ns(ts)
        assert len(iets) == 0
