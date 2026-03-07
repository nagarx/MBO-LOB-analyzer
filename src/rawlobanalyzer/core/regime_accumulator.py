"""Streaming per-regime accumulator for sum/sum_of_squares/count.

Provides O(1)-per-regime memory for computing mean and standard
deviation of values partitioned by volatility regime labels (e.g.
open, morning, midday, afternoon, close).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from rawlobanalyzer.core.constants import EPS
from rawlobanalyzer.core.time_utils import REGIME_LABELS


class _RegimeBucket:
    """Single-regime streaming statistics."""

    __slots__ = ("sum", "sum_of_squares", "count")

    def __init__(self) -> None:
        self.sum = 0.0
        self.sum_of_squares = 0.0
        self.count = 0


class RegimeStreamingAccumulator:
    """Accumulates streaming mean/std per time-regime.

    Usage::

        acc = RegimeStreamingAccumulator()
        # in process_day:
        acc.add(regime_int, values_array)
        # in finalize:
        result = acc.finalize(min_count=50)
    """

    __slots__ = ("_buckets",)

    def __init__(self) -> None:
        self._buckets: dict[int, _RegimeBucket] = {}

    def add(self, regime: int, values: np.ndarray) -> None:
        """Add a chunk of finite values for *regime*."""
        clean = values[np.isfinite(values)]
        n = len(clean)
        if n == 0:
            return
        if regime not in self._buckets:
            self._buckets[regime] = _RegimeBucket()
        b = self._buckets[regime]
        b.sum += float(np.sum(clean))
        b.sum_of_squares += float(np.sum(clean ** 2))
        b.count += n

    def finalize(self, *, min_count: int = 2) -> dict[str, dict[str, Any]]:
        """Return ``{regime_label: {"mean": ..., "std": ..., "n": ...}}``.

        Regimes with fewer than *min_count* observations are omitted.
        """
        result: dict[str, dict[str, Any]] = {}
        for regime, b in self._buckets.items():
            if b.count < min_count:
                continue
            mean = b.sum / b.count
            variance = (b.sum_of_squares / b.count) - mean ** 2
            std = float(np.sqrt(max(variance * b.count / (b.count - 1), 0.0)))
            label = REGIME_LABELS.get(regime, f"regime_{regime}")
            result[label] = {
                "mean": mean,
                "std": std,
                "n": b.count,
            }
        return result

    def get_bucket(self, regime: int) -> _RegimeBucket | None:
        """Expose raw bucket for analyzers needing custom finalize logic."""
        return self._buckets.get(regime)

    def items(self) -> list[tuple[int, _RegimeBucket]]:
        """Iterate over (regime_int, bucket) pairs."""
        return list(self._buckets.items())
