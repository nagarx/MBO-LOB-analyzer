"""Tests for core.resampler."""

import numpy as np
import pytest

from rawlobanalyzer.core.resampler import resample, resample_to_grid
from rawlobanalyzer.core.constants import NS_PER_SECOND


class TestResample:
    def test_count(self):
        ts = np.array([0, 100, 200, 500, 600, 1100], dtype=np.int64)
        values = np.zeros(len(ts))
        result = resample(ts, values, resolution_ns=500, agg="count", label="test")
        assert result.label == "test"
        populated = result.counts[result.counts > 0]
        assert int(np.sum(populated)) == 6

    def test_mean(self):
        ts = np.array([0, 0, 0], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0])
        result = resample(ts, values, resolution_ns=1000, agg="mean")
        non_nan = result.values[~np.isnan(result.values)]
        assert len(non_nan) >= 1
        np.testing.assert_allclose(non_nan[0], 2.0)

    def test_sum(self):
        ts = np.array([0, 0, 0], dtype=np.int64)
        values = np.array([10.0, 20.0, 30.0])
        result = resample(ts, values, resolution_ns=1000, agg="sum")
        populated = result.values[result.counts > 0]
        np.testing.assert_allclose(populated[0], 60.0)

    def test_last(self):
        ts = np.array([0, 100, 200], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0])
        result = resample(ts, values, resolution_ns=1000, agg="last")
        populated = result.values[~np.isnan(result.values)]
        assert populated[0] == 3.0

    def test_first(self):
        ts = np.array([0, 100, 200], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0])
        result = resample(ts, values, resolution_ns=1000, agg="first")
        populated = result.values[~np.isnan(result.values)]
        assert populated[0] == 1.0

    def test_ohlc(self):
        ts = np.array([0, 100, 200], dtype=np.int64)
        values = np.array([10.0, 15.0, 12.0])
        result = resample(ts, values, resolution_ns=1000, agg="ohlc")
        ohlc = result.values
        populated = ohlc[result.counts > 0]
        assert populated[0, 0] == 10.0   # open
        assert populated[0, 1] == 15.0   # high
        assert populated[0, 2] == 10.0   # low
        assert populated[0, 3] == 12.0   # close

    def test_empty_input(self):
        result = resample(np.array([], dtype=np.int64), np.array([]), resolution_ns=1000)
        assert len(result.values) == 0
        assert len(result.counts) == 0

    def test_multiple_bins(self):
        ts = np.array([0, 500, 1000, 1500], dtype=np.int64)
        values = np.array([1.0, 2.0, 3.0, 4.0])
        result = resample(ts, values, resolution_ns=1000, agg="mean")
        populated = result.values[~np.isnan(result.values)]
        assert len(populated) == 2
        np.testing.assert_allclose(populated[0], 1.5)
        np.testing.assert_allclose(populated[1], 3.5)


class TestResampleMultiBin:
    """Multi-bin correctness for modes that use the sorted cumsum trick."""

    _TS = np.array([0, 100, 200, 1000, 1100, 1200, 1300, 2500], dtype=np.int64)
    _VALS = np.array([1.0, 5.0, 3.0, 10.0, 20.0, 15.0, 25.0, 99.0])
    _RES = 1000

    def test_last_multiple_bins(self):
        r = resample(self._TS, self._VALS, self._RES, agg="last")
        filled = r.values[~np.isnan(r.values)]
        assert len(filled) == 3, f"Expected 3 filled bins, got {len(filled)}"
        assert filled[0] == 3.0, "Bin 0 last should be 3.0 (ts=200)"
        assert filled[1] == 25.0, "Bin 1 last should be 25.0 (ts=1300)"
        assert filled[2] == 99.0, "Bin 2 last should be 99.0 (ts=2500)"

    def test_first_multiple_bins(self):
        r = resample(self._TS, self._VALS, self._RES, agg="first")
        filled = r.values[~np.isnan(r.values)]
        assert len(filled) == 3
        assert filled[0] == 1.0, "Bin 0 first should be 1.0 (ts=0)"
        assert filled[1] == 10.0, "Bin 1 first should be 10.0 (ts=1000)"
        assert filled[2] == 99.0, "Bin 2 first should be 99.0 (ts=2500)"

    def test_ohlc_multiple_bins(self):
        r = resample(self._TS, self._VALS, self._RES, agg="ohlc")
        ohlc = r.values[r.counts > 0]
        assert ohlc.shape[0] == 3

        np.testing.assert_equal(ohlc[0, 0], 1.0)   # open
        np.testing.assert_equal(ohlc[0, 1], 5.0)    # high
        np.testing.assert_equal(ohlc[0, 2], 1.0)    # low
        np.testing.assert_equal(ohlc[0, 3], 3.0)    # close

        np.testing.assert_equal(ohlc[1, 0], 10.0)   # open
        np.testing.assert_equal(ohlc[1, 1], 25.0)   # high
        np.testing.assert_equal(ohlc[1, 2], 10.0)   # low
        np.testing.assert_equal(ohlc[1, 3], 25.0)   # close

        np.testing.assert_equal(ohlc[2, 0], 99.0)   # single event
        np.testing.assert_equal(ohlc[2, 3], 99.0)

    def test_median_multiple_bins(self):
        r = resample(self._TS, self._VALS, self._RES, agg="median")
        filled = r.values[~np.isnan(r.values)]
        assert len(filled) == 3
        np.testing.assert_allclose(filled[0], 3.0)   # median([1,5,3]) = 3
        np.testing.assert_allclose(filled[1], 17.5)   # median([10,20,15,25]) = 17.5
        np.testing.assert_allclose(filled[2], 99.0)


class TestResampleEdgeCases:
    """Boundary conditions and structural edge cases."""

    def test_boundary_timestamps(self):
        """Timestamps landing exactly on bin edges."""
        ts = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        vals = np.array([10.0, 20.0, 30.0, 40.0])
        r = resample(ts, vals, resolution_ns=1000, agg="last")
        filled = r.values[~np.isnan(r.values)]
        assert filled[0] == 10.0, "ts=0 should be in bin 0"
        assert filled[1] == 20.0, "ts=1000 should be in bin 1"
        assert filled[2] == 30.0, "ts=2000 should be in bin 2"
        assert filled[3] == 40.0, "ts=3000 should be in bin 3"

    def test_single_event_per_bin(self):
        """One event per bin: first == last == open == close == value."""
        ts = np.array([500, 1500, 2500], dtype=np.int64)
        vals = np.array([7.0, 8.0, 9.0])

        for mode in ("first", "last"):
            r = resample(ts, vals, resolution_ns=1000, agg=mode)
            filled = r.values[~np.isnan(r.values)]
            np.testing.assert_array_equal(filled, [7.0, 8.0, 9.0])

        r_ohlc = resample(ts, vals, resolution_ns=1000, agg="ohlc")
        ohlc = r_ohlc.values[r_ohlc.counts > 0]
        for col in range(4):
            np.testing.assert_array_equal(ohlc[:, col], [7.0, 8.0, 9.0])

    def test_empty_bins_between_filled(self):
        """Sparse data with gaps should produce NaN in empty bins."""
        ts = np.array([0, 5000], dtype=np.int64)
        vals = np.array([1.0, 2.0])
        r = resample(ts, vals, resolution_ns=1000, agg="last")

        filled_count = int(np.sum(~np.isnan(r.values)))
        nan_count = int(np.sum(np.isnan(r.values)))
        assert filled_count == 2
        assert nan_count >= 3, "Should have NaN gaps between bins 0 and 5"

    def test_all_same_timestamp(self):
        """All events at t=0 land in one bin."""
        ts = np.zeros(100, dtype=np.int64)
        vals = np.arange(100, dtype=np.float64)
        r_last = resample(ts, vals, resolution_ns=1000, agg="last")
        r_first = resample(ts, vals, resolution_ns=1000, agg="first")

        filled_last = r_last.values[~np.isnan(r_last.values)]
        filled_first = r_first.values[~np.isnan(r_first.values)]
        assert len(filled_last) == 1
        assert filled_last[0] == 99.0, "last of 0..99 is 99"
        assert filled_first[0] == 0.0, "first of 0..99 is 0"

    def test_large_scale_golden(self):
        """100K events: vectorized output matches a naive reference implementation."""
        rng = np.random.default_rng(42)
        n = 100_000
        ts = np.sort(rng.integers(0, 10_000_000, n, dtype=np.int64))
        vals = rng.normal(100.0, 5.0, n)
        res_ns = 1_000_000

        result = resample(ts, vals, res_ns, agg="last")

        bin_start = (int(ts[0]) // res_ns) * res_ns
        n_bins = len(result.counts)
        ref = np.full(n_bins, np.nan, dtype=np.float64)
        for i in range(n - 1, -1, -1):
            b = int((int(ts[i]) - bin_start) // res_ns)
            b = min(b, n_bins - 1)
            if np.isnan(ref[b]):
                ref[b] = float(vals[i])

        np.testing.assert_array_equal(result.values, ref)


class TestResampleToGrid:
    """Tests for resample_to_grid with pre-computed bin edges."""

    def test_sum_matches_resample_on_aligned_data(self):
        """When grid edges match data-driven edges, results are identical."""
        ts = np.array([0, 100, 200, 1000, 1100], dtype=np.int64)
        vals = np.array([1.0, 2.0, 3.0, 10.0, 20.0])
        grid = np.arange(0, 3000, 1000, dtype=np.int64)
        r_grid = resample_to_grid(ts, vals, grid, agg="sum")
        r_data = resample(ts, vals, 1000, agg="sum")
        assert r_grid.counts[0] == r_data.counts[0]
        np.testing.assert_allclose(r_grid.values[0], r_data.values[0])

    def test_last_on_grid(self):
        grid = np.array([0, 1000, 2000, 3000], dtype=np.int64)
        ts = np.array([100, 500, 1500, 2800], dtype=np.int64)
        vals = np.array([1.0, 2.0, 3.0, 4.0])
        r = resample_to_grid(ts, vals, grid, agg="last")
        assert r.values[0] == 2.0
        assert r.values[1] == 3.0
        assert r.values[2] == 4.0

    def test_events_outside_grid_clipped(self):
        """Events before or after grid are clipped to first/last bin."""
        grid = np.array([1000, 2000, 3000], dtype=np.int64)
        ts = np.array([500, 1500, 3500], dtype=np.int64)
        vals = np.array([10.0, 20.0, 30.0])
        r = resample_to_grid(ts, vals, grid, agg="sum")
        assert r.values[0] == 30.0  # 500 clipped to bin 0, 1500 in bin 0
        assert r.values[1] == 30.0  # 3500 clipped to bin 1

    def test_empty_input(self):
        grid = np.array([0, 1000, 2000], dtype=np.int64)
        r = resample_to_grid(
            np.array([], dtype=np.int64),
            np.array([], dtype=np.float64),
            grid, agg="sum",
        )
        assert len(r.values) == 2
        np.testing.assert_array_equal(r.counts, [0, 0])

    def test_grid_edges_preserved(self):
        grid = np.array([100, 200, 300, 400], dtype=np.int64)
        ts = np.array([150], dtype=np.int64)
        vals = np.array([5.0])
        r = resample_to_grid(ts, vals, grid, agg="sum")
        np.testing.assert_array_equal(r.bin_edges_ns, grid)
