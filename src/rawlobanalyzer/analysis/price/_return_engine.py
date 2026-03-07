"""Shared multi-scale return computation engine for the price/ domain.

Single source of truth for computing log returns at configurable timescales
from raw LOB data. All price-domain analyzers consume ``DayReturns`` produced
by ``compute_day_returns()`` rather than independently resampling.

Price unit: mid_prices arrive in USD (float64) from ``DayData.mid_prices``.
Time unit: timestamps are int64 nanoseconds since epoch (UTC).
Return unit: natural log returns (dimensionless).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.config.timescale_config import TimescaleConfig
from rawlobanalyzer.core.constants import NS_PER_SECOND
from rawlobanalyzer.core.price_utils import log_returns
from rawlobanalyzer.core.resampler import resample, resample_to_grid
from rawlobanalyzer.core.time_utils import rth_grid_edges_ns, rth_mask_utc, seconds_to_label
from rawlobanalyzer.io.loader import DayData


@dataclass(frozen=True)
class ScaledReturns:
    """Log returns resampled at a single timescale.

    Attributes:
        label: Timescale label (e.g. ``"1s"``, ``"5m"``).
        returns: Log returns between consecutive bin closes. Length = n_bins - 1
            where only bins with data contribute.
        bin_timestamps_ns: Left-edge timestamps for bins that had data.
        n_bins_total: Total number of bins (including empty).
        n_bins_filled: Number of bins with at least one event.
    """

    label: str
    returns: np.ndarray
    bin_timestamps_ns: np.ndarray
    n_bins_total: int
    n_bins_filled: int


@dataclass
class DayReturns:
    """All return series for a single trading day.

    Produced by ``compute_day_returns()`` and consumed by all price-domain
    analyzers.

    Attributes:
        date: Trading date ``YYYY-MM-DD``.
        tick_returns: Raw tick-by-tick log returns (no resampling).
        tick_timestamps_ns: Timestamps for tick returns (mid-point of each pair).
        scaled: Per-timescale returns, keyed by label (e.g. ``"1s"``).
        rth_mask: Boolean mask over original LOB rows for RTH events.
        open_price: First valid mid-price during RTH.
        close_price: Last valid mid-price during RTH.
        n_valid_prices: Number of finite, positive mid-prices used.
    """

    date: str
    tick_returns: np.ndarray
    tick_timestamps_ns: np.ndarray
    scaled: dict[str, ScaledReturns] = field(default_factory=dict)
    rth_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    open_price: float = np.nan
    close_price: float = np.nan
    n_valid_prices: int = 0


def compute_day_returns(
    day: DayData,
    config: AnalysisConfig,
    *,
    extra_scales_seconds: tuple[float, ...] | None = None,
    day_epoch_ns: int = 0,
) -> DayReturns:
    """Compute multi-scale log returns from one day of raw LOB data.

    Args:
        day: One trading day's LOB data (must have ``mid_price``, ``timestamp_ns``).
        config: Analysis configuration (timescales, trading hours).
        extra_scales_seconds: Additional sampling scales in seconds beyond
            those in ``config.timescales`` (used by signature-plot analyzers).
        day_epoch_ns: Midnight UTC of the trading day (nanoseconds since epoch).
            When non-zero and a timescale uses ``trading_hours_only``, returns
            are resampled onto a canonical RTH grid so that bins align with
            those produced by ``compute_day_flow()``.

    Returns:
        ``DayReturns`` with tick-by-tick and per-timescale returns.
    """
    ts_ns = day.lob_timestamps_ns
    mids = day.mid_prices

    valid = np.isfinite(mids) & (mids > 0)
    ts_valid = ts_ns[valid]
    mids_valid = mids[valid]

    rth = rth_mask_utc(ts_ns, utc_offset_hours=config.trading_hours.utc_offset_hours)

    open_price = np.nan
    close_price = np.nan
    rth_valid = rth[valid]
    if np.any(rth_valid):
        rth_mids = mids_valid[rth_valid]
        open_price = float(rth_mids[0])
        close_price = float(rth_mids[-1])

    n_valid = int(np.sum(valid))

    if n_valid < 2:
        return DayReturns(
            date=day.date,
            tick_returns=np.array([], dtype=np.float64),
            tick_timestamps_ns=np.array([], dtype=np.int64),
            rth_mask=rth,
            open_price=open_price,
            close_price=close_price,
            n_valid_prices=n_valid,
        )

    tick_ret = log_returns(mids_valid)
    tick_ts = ts_valid[1:]

    scales: list[TimescaleConfig] = list(config.timescales)
    if extra_scales_seconds:
        existing_ns = {tc.resolution_ns for tc in scales}
        for s in extra_scales_seconds:
            res_ns = int(s * NS_PER_SECOND)
            if res_ns not in existing_ns:
                label = seconds_to_label(s)
                scales.append(TimescaleConfig(
                    resolution_ns=res_ns,
                    label=label,
                    trading_hours_only=True,
                ))

    scaled: dict[str, ScaledReturns] = {}
    for tc in scales:
        if tc.trading_hours_only:
            mask = rth_valid
            ts_use = ts_valid[mask]
            mids_use = mids_valid[mask]
        else:
            ts_use = ts_valid
            mids_use = mids_valid

        if len(ts_use) < 2:
            scaled[tc.label] = ScaledReturns(
                label=tc.label,
                returns=np.array([], dtype=np.float64),
                bin_timestamps_ns=np.array([], dtype=np.int64),
                n_bins_total=0,
                n_bins_filled=0,
            )
            continue

        use_grid = day_epoch_ns > 0 and tc.trading_hours_only
        if use_grid:
            grid = rth_grid_edges_ns(
                day_epoch_ns, tc.resolution_ns,
                utc_offset_hours=config.trading_hours.utc_offset_hours,
            )
            resampled = resample_to_grid(
                ts_use, mids_use, grid,
                agg="last", label=tc.label,
            )
        else:
            resampled = resample(
                ts_use, mids_use, tc.resolution_ns,
                agg="last", label=tc.label,
            )

        filled_mask = resampled.counts > 0
        close_prices = resampled.values[filled_mask]
        bin_edges = resampled.bin_edges_ns[:-1][filled_mask]

        if len(close_prices) < 2:
            scaled[tc.label] = ScaledReturns(
                label=tc.label,
                returns=np.array([], dtype=np.float64),
                bin_timestamps_ns=bin_edges,
                n_bins_total=len(resampled.counts),
                n_bins_filled=int(np.sum(filled_mask)),
            )
            continue

        rets = log_returns(close_prices)
        scaled[tc.label] = ScaledReturns(
            label=tc.label,
            returns=rets,
            bin_timestamps_ns=bin_edges[1:],
            n_bins_total=len(resampled.counts),
            n_bins_filled=int(np.sum(filled_mask)),
        )

    return DayReturns(
        date=day.date,
        tick_returns=tick_ret,
        tick_timestamps_ns=tick_ts,
        scaled=scaled,
        rth_mask=rth,
        open_price=open_price,
        close_price=close_price,
        n_valid_prices=n_valid,
    )


