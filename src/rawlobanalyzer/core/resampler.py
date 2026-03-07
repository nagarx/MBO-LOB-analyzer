"""Time-based resampling of nanosecond event data into fixed-width bins.

Provides the multi-granularity engine that powers all timescale-dependent
analysis. Bins are defined by ``TimescaleConfig`` and support multiple
aggregation modes (mean, sum, last, OHLC, count).

**Invariant**: ``timestamps_ns`` must be sorted (monotonically non-decreasing).
All LOB/MBO data from ``MBO-LOB-reconstructor`` satisfies this by construction.

**Performance**: Bin assignment uses O(N) integer division for uniform-width
bins. Aggregation modes ``last``, ``first``, ``ohlc`` exploit the sorted
invariant via cumulative counts for O(N) total work without Python-level
loops over individual events.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rawlobanalyzer.core.constants import NS_PER_SECOND

AggMode = Literal["mean", "sum", "last", "first", "count", "ohlc", "median"]


@dataclass(frozen=True)
class ResampledSeries:
    """Result of resampling a time series into fixed-width bins.

    Attributes:
        bin_edges_ns: Left edges of each bin in nanoseconds (length ``n_bins + 1``).
        values: Aggregated values per bin (length ``n_bins``).
            For ``ohlc`` mode, shape is ``(n_bins, 4)`` [open, high, low, close].
        counts: Number of raw events per bin (length ``n_bins``).
        label: Human-readable timescale label (e.g. ``"1s"``, ``"5m"``).
    """

    bin_edges_ns: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    label: str


def _compute_segments(
    counts: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive segment boundaries from per-bin counts.

    Requires that the original data was sorted by time (and hence by bin
    index). The cumulative count boundaries then map directly to array
    positions of the first and last element in each bin.

    Returns:
        ``(cum_counts, filled_mask, filled_bins, seg_starts, seg_ends)``
        where ``seg_starts[i]``/``seg_ends[i]`` index into the original
        values array for ``filled_bins[i]``.
    """
    cum_counts = np.cumsum(counts)
    filled_mask = counts > 0
    filled_bins = np.where(filled_mask)[0]
    seg_starts = (cum_counts - counts)[filled_mask]
    seg_ends = cum_counts[filled_mask]
    return cum_counts, filled_mask, filled_bins, seg_starts, seg_ends


def _aggregate_into_bins(
    bin_indices: np.ndarray,
    values: np.ndarray,
    n_bins: int,
    edges: np.ndarray,
    agg: AggMode,
    label: str,
) -> ResampledSeries:
    """Shared aggregation logic for both ``resample`` and ``resample_to_grid``.

    Expects ``bin_indices`` already computed and clipped to ``[0, n_bins-1]``.
    """
    counts = np.bincount(bin_indices, minlength=n_bins)[:n_bins]

    if agg == "count":
        return ResampledSeries(
            bin_edges_ns=edges,
            values=counts.astype(np.float64),
            counts=counts,
            label=label,
        )

    vals_f64 = values.astype(np.float64)

    if agg == "sum":
        sums = np.bincount(bin_indices, weights=vals_f64, minlength=n_bins)[:n_bins]
        return ResampledSeries(
            bin_edges_ns=edges, values=sums, counts=counts, label=label,
        )

    if agg == "mean":
        sums = np.bincount(bin_indices, weights=vals_f64, minlength=n_bins)[:n_bins]
        safe_counts = np.where(counts > 0, counts, 1).astype(np.float64)
        means = sums / safe_counts
        means[counts == 0] = np.nan
        return ResampledSeries(
            bin_edges_ns=edges, values=means, counts=counts, label=label,
        )

    if agg == "last":
        result = np.full(n_bins, np.nan, dtype=np.float64)
        _, _, filled_bins, _, seg_ends = _compute_segments(counts)
        result[filled_bins] = vals_f64[seg_ends - 1]
        return ResampledSeries(
            bin_edges_ns=edges, values=result, counts=counts, label=label,
        )

    if agg == "first":
        result = np.full(n_bins, np.nan, dtype=np.float64)
        _, _, filled_bins, seg_starts, _ = _compute_segments(counts)
        result[filled_bins] = vals_f64[seg_starts]
        return ResampledSeries(
            bin_edges_ns=edges, values=result, counts=counts, label=label,
        )

    if agg == "median":
        result = np.full(n_bins, np.nan, dtype=np.float64)
        _, _, filled_bins, seg_starts, seg_ends = _compute_segments(counts)
        for i, b in enumerate(filled_bins):
            result[b] = float(np.median(vals_f64[seg_starts[i]:seg_ends[i]]))
        return ResampledSeries(
            bin_edges_ns=edges, values=result, counts=counts, label=label,
        )

    if agg == "ohlc":
        ohlc = np.full((n_bins, 4), np.nan, dtype=np.float64)
        cum_counts, filled_mask, filled_bins, seg_starts, seg_ends = (
            _compute_segments(counts)
        )
        ohlc[filled_bins, 0] = vals_f64[seg_starts]
        ohlc[filled_bins, 3] = vals_f64[seg_ends - 1]

        all_seg_starts = np.empty(n_bins, dtype=np.int64)
        all_seg_starts[0] = 0
        all_seg_starts[1:] = cum_counts[:-1]
        all_seg_starts = np.clip(all_seg_starts, 0, len(vals_f64) - 1)
        highs = np.maximum.reduceat(vals_f64, all_seg_starts)
        lows = np.minimum.reduceat(vals_f64, all_seg_starts)
        ohlc[filled_bins, 1] = highs[filled_mask]
        ohlc[filled_bins, 2] = lows[filled_mask]

        return ResampledSeries(
            bin_edges_ns=edges, values=ohlc, counts=counts, label=label,
        )

    raise ValueError(f"Unknown aggregation mode: {agg!r}")


def resample(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    resolution_ns: int,
    *,
    agg: AggMode = "mean",
    label: str = "",
) -> ResampledSeries:
    """Resample event data into fixed-width time bins.

    Complexity:
        - Bin assignment: O(N) via integer division.
        - ``count``, ``sum``, ``mean``: O(N) via ``np.bincount``.
        - ``last``, ``first``: O(N) via cumulative counts (no Python loop).
        - ``ohlc``: O(N) via cumulative counts + ``np.ufunc.reduceat``.
        - ``median``: O(N) total (Python loop over filled bins, not events).

    Args:
        timestamps_ns: Sorted int64 nanosecond timestamps (monotonically
            non-decreasing).
        values: Values to aggregate (same length as ``timestamps_ns``).
            For ``"count"`` mode, values are ignored (pass timestamps).
        resolution_ns: Bin width in nanoseconds.
        agg: Aggregation mode.
        label: Human-readable label for the timescale.

    Returns:
        ``ResampledSeries`` with aggregated bins.
    """
    if len(timestamps_ns) == 0:
        return ResampledSeries(
            bin_edges_ns=np.array([], dtype=np.int64),
            values=np.array([], dtype=np.float64),
            counts=np.array([], dtype=np.int64),
            label=label,
        )

    t_min = int(timestamps_ns[0])
    t_max = int(timestamps_ns[-1])

    bin_start = (t_min // resolution_ns) * resolution_ns
    bin_end = ((t_max // resolution_ns) + 1) * resolution_ns + resolution_ns

    edges = np.arange(bin_start, bin_end, resolution_ns, dtype=np.int64)
    n_bins = len(edges) - 1

    if n_bins <= 0:
        return ResampledSeries(
            bin_edges_ns=edges,
            values=np.array([], dtype=np.float64),
            counts=np.array([], dtype=np.int64),
            label=label,
        )

    bin_indices = ((timestamps_ns - bin_start) // resolution_ns).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return _aggregate_into_bins(bin_indices, values, n_bins, edges, agg, label)


def resample_to_grid(
    timestamps_ns: np.ndarray,
    values: np.ndarray,
    grid_edges_ns: np.ndarray,
    *,
    agg: AggMode = "sum",
    label: str = "",
) -> ResampledSeries:
    """Resample event data into bins defined by a pre-computed grid.

    Unlike ``resample()`` which derives bin edges from the data itself,
    this function uses externally supplied edges.  This guarantees that
    two different data sources (e.g. MBO and LOB) resampled onto the
    same grid will produce perfectly aligned bins, enabling correct
    cross-series correlation (OFI vs returns, OFI vs spread changes).

    Events outside ``[grid_edges_ns[0], grid_edges_ns[-1])`` are clipped
    into the first or last bin respectively.

    Args:
        timestamps_ns: Sorted int64 nanosecond timestamps.
        values: Values to aggregate (same length as ``timestamps_ns``).
        grid_edges_ns: Pre-computed bin edges (length ``n_bins + 1``).
            Must be sorted and uniformly spaced.
        agg: Aggregation mode.
        label: Human-readable label for the timescale.

    Returns:
        ``ResampledSeries`` with bins aligned to ``grid_edges_ns``.
    """
    n_bins = len(grid_edges_ns) - 1
    if n_bins <= 0 or len(timestamps_ns) == 0:
        return ResampledSeries(
            bin_edges_ns=grid_edges_ns,
            values=np.array([], dtype=np.float64) if n_bins <= 0
            else np.zeros(n_bins, dtype=np.float64),
            counts=np.array([], dtype=np.int64) if n_bins <= 0
            else np.zeros(n_bins, dtype=np.int64),
            label=label,
        )

    resolution_ns = int(grid_edges_ns[1] - grid_edges_ns[0])
    bin_start = int(grid_edges_ns[0])

    bin_indices = ((timestamps_ns - bin_start) // resolution_ns).astype(np.int64)
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    return _aggregate_into_bins(bin_indices, values, n_bins, grid_edges_ns, agg, label)
