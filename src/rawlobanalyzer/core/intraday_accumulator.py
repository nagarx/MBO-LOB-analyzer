"""Streaming intraday curve accumulator.

Accumulates per-bin statistics (mean, variance) over multiple trading
days with O(n_bins) memory, regardless of the number of days processed.
Used by volatility and spread analyzers for normalized intraday curves.
"""

from __future__ import annotations

from typing import Any

import numpy as np


class IntradayCurveAccumulator:
    """Fixed-bin streaming accumulator for intraday curves.

    Maintains running sum, sum-of-squares, and count per bin, allowing
    incremental computation of mean and standard deviation across days.

    Args:
        n_bins: Number of time bins across the trading day.
    """

    __slots__ = ("_n_bins", "_sum", "_sum_sq", "_counts")

    def __init__(self, n_bins: int) -> None:
        self._n_bins = n_bins
        self._sum = np.zeros(n_bins, dtype=np.float64)
        self._sum_sq = np.zeros(n_bins, dtype=np.float64)
        self._counts = np.zeros(n_bins, dtype=np.int64)

    def add(self, bin_indices: np.ndarray, values: np.ndarray) -> None:
        """Accumulate values into the specified bins.

        Args:
            bin_indices: Integer array of bin indices (must be < n_bins).
            values: Float array of values, same length as bin_indices.
        """
        for i, bin_idx in enumerate(bin_indices):
            if bin_idx < self._n_bins:
                self._sum[bin_idx] += values[i]
                self._sum_sq[bin_idx] += values[i] ** 2
                self._counts[bin_idx] += 1

    @property
    def n_bins(self) -> int:
        return self._n_bins

    def finalize(
        self,
        bin_width: int,
        *,
        mean_key: str = "mean",
        std_key: str = "std",
    ) -> dict[str, Any]:
        """Compute mean and sample std per bin.

        Args:
            bin_width: Width of each bin in the time-axis unit (e.g. minutes).
            mean_key: Key name for the mean array in the output.
            std_key: Key name for the std array in the output.

        Returns:
            Dict with ``minutes``, mean, std, and ``n_days`` per bin.
            Non-finite values are replaced with ``None``.
        """
        active = self._counts > 0
        if not np.any(active):
            return {"minutes": [], mean_key: [], std_key: [], "n_days": []}

        minutes = (np.arange(self._n_bins) * bin_width).tolist()
        safe_counts = np.where(active, self._counts, 1)
        means = np.where(active, self._sum / safe_counts, np.nan)

        multi = self._counts > 1
        safe_counts_m = np.where(multi, self._counts, 2)
        variance = np.where(
            multi,
            (self._sum_sq - self._sum ** 2 / safe_counts_m) / (safe_counts_m - 1),
            0.0,
        )
        stds = np.sqrt(np.maximum(variance, 0.0))

        return {
            "minutes": minutes,
            mean_key: [float(v) if np.isfinite(v) else None for v in means],
            std_key: [float(v) if np.isfinite(v) else None for v in stds],
            "n_days": self._counts.tolist(),
        }
