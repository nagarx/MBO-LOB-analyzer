"""Shared multi-scale spread computation engine for the spread/ domain.

Single source of truth for computing spread statistics at configurable timescales
from raw LOB data. Spread-domain analyzers consume ``DaySpreads`` produced by
``compute_day_spreads()`` rather than independently resampling.

Spread unit: spreads arrive in USD (float64) from ``DayData.spreads``.
Time unit: timestamps are int64 nanoseconds since epoch (UTC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.config.timescale_config import TimescaleConfig
from rawlobanalyzer.core.constants import BPS_FACTOR, NS_PER_SECOND
from rawlobanalyzer.core.resampler import resample
from rawlobanalyzer.core.time_utils import rth_mask_utc, seconds_to_label
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.io.schema import ACTION_TRADE


@dataclass(frozen=True)
class ScaledSpreads:
    """Spread statistics resampled at a single timescale.

    Attributes:
        label: Timescale label (e.g. ``"1s"``, ``"5m"``).
        mean_spreads_usd: Mean spread per time bin (length n_bins_total).
            Empty bins are NaN.
        median_spreads_usd: Median spread per time bin (length n_bins_total).
            Empty bins are NaN.
        bin_edges_ns: Left edges of each bin in nanoseconds.
        counts: Number of raw events per bin.
        n_bins_total: Total number of bins (including empty).
        n_bins_filled: Number of bins with at least one event.
    """

    label: str
    mean_spreads_usd: np.ndarray
    median_spreads_usd: np.ndarray
    bin_edges_ns: np.ndarray
    counts: np.ndarray
    n_bins_total: int
    n_bins_filled: int


@dataclass
class DaySpreads:
    """All spread series for a single trading day.

    Produced by ``compute_day_spreads()`` and consumed by spread-domain
    analyzers.

    Attributes:
        date: Trading date ``YYYY-MM-DD``.
        tick_spreads_usd: Raw tick-by-tick spreads in USD.
        tick_spreads_bps: Raw tick-by-tick spreads in basis points.
        tick_timestamps_ns: Timestamps aligned with tick spreads.
        scaled: Per-timescale spread statistics, keyed by label (e.g. ``"1s"``).
        rth_mask: Boolean mask over original LOB rows for RTH events.
        trade_mask: Boolean mask over valid rows for trade-triggering events.
        n_valid: Number of valid spread observations.
    """

    date: str
    tick_spreads_usd: np.ndarray
    tick_spreads_bps: np.ndarray
    tick_timestamps_ns: np.ndarray
    scaled: dict[str, ScaledSpreads] = field(default_factory=dict)
    rth_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    trade_mask: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.bool_))
    n_valid: int = 0


def compute_day_spreads(
    day: DayData,
    config: AnalysisConfig,
    *,
    extra_scales_seconds: tuple[float, ...] | None = None,
) -> DaySpreads:
    """Compute multi-scale spread statistics from one day of raw LOB data.

    Args:
        day: One trading day's LOB data (must have ``spread``, ``timestamp_ns``).
        config: Analysis configuration (timescales, trading hours).
        extra_scales_seconds: Additional sampling scales in seconds beyond
            those in ``config.timescales``.

    Returns:
        ``DaySpreads`` with tick-by-tick and per-timescale spread statistics.
    """
    ts_ns = day.lob_timestamps_ns
    spreads = day.spreads

    valid = np.isfinite(spreads) & (spreads > 0)
    ts_valid = ts_ns[valid]
    spreads_valid = spreads[valid].astype(np.float64)

    rth = rth_mask_utc(ts_ns, utc_offset_hours=config.trading_hours.utc_offset_hours)
    rth_valid = rth[valid]

    trade_mask_full = np.zeros(len(ts_ns), dtype=np.bool_)
    if "triggering_action" in day.lob.column_names:
        actions = day.lob.column("triggering_action").to_numpy()
        trade_mask_full = (actions == ACTION_TRADE) & valid
    trade_mask_valid = trade_mask_full[valid]

    spread_bps = np.full(len(spreads_valid), np.nan, dtype=np.float64)
    if "spread_bps" in day.lob.column_names:
        bps_col = day.lob.column("spread_bps").to_numpy()
        valid_bps = valid & np.isfinite(bps_col)
        spread_bps = np.where(valid_bps[valid], bps_col[valid], np.nan)
    elif "mid_price" in day.lob.column_names:
        mids = day.lob.column("mid_price").to_numpy()
        mids_valid = mids[valid]
        valid_mid = np.isfinite(mids_valid) & (mids_valid > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            safe_mid = np.where(valid_mid, mids_valid, np.nan)
            spread_bps = np.where(
                valid_mid,
                spreads_valid / safe_mid * BPS_FACTOR,
                np.nan,
            )

    n_valid = int(np.sum(valid))

    if n_valid == 0:
        return DaySpreads(
            date=day.date,
            tick_spreads_usd=np.array([], dtype=np.float64),
            tick_spreads_bps=np.array([], dtype=np.float64),
            tick_timestamps_ns=np.array([], dtype=np.int64),
            rth_mask=rth,
            trade_mask=np.array([], dtype=np.bool_),
            n_valid=0,
        )

    scales: list[TimescaleConfig] = list(config.timescales)
    if extra_scales_seconds:
        existing_ns = {tc.resolution_ns for tc in scales}
        for s in extra_scales_seconds:
            res_ns = int(s * NS_PER_SECOND)
            if res_ns not in existing_ns:
                label = seconds_to_label(s)
                scales.append(
                    TimescaleConfig(
                        resolution_ns=res_ns,
                        label=label,
                        trading_hours_only=True,
                    )
                )

    scaled: dict[str, ScaledSpreads] = {}
    for tc in scales:
        if tc.trading_hours_only:
            mask = rth_valid
            ts_use = ts_valid[mask]
            spreads_use = spreads_valid[mask]
        else:
            ts_use = ts_valid
            spreads_use = spreads_valid

        if len(ts_use) == 0:
            scaled[tc.label] = ScaledSpreads(
                label=tc.label,
                mean_spreads_usd=np.array([], dtype=np.float64),
                median_spreads_usd=np.array([], dtype=np.float64),
                bin_edges_ns=np.array([], dtype=np.int64),
                counts=np.array([], dtype=np.int64),
                n_bins_total=0,
                n_bins_filled=0,
            )
            continue

        res_mean = resample(
            ts_use, spreads_use, tc.resolution_ns,
            agg="mean", label=tc.label,
        )
        res_median = resample(
            ts_use, spreads_use, tc.resolution_ns,
            agg="median", label=tc.label,
        )

        scaled[tc.label] = ScaledSpreads(
            label=tc.label,
            mean_spreads_usd=res_mean.values,
            median_spreads_usd=res_median.values,
            bin_edges_ns=res_mean.bin_edges_ns,
            counts=res_mean.counts,
            n_bins_total=len(res_mean.counts),
            n_bins_filled=int(np.sum(res_mean.counts > 0)),
        )

    return DaySpreads(
        date=day.date,
        tick_spreads_usd=spreads_valid,
        tick_spreads_bps=spread_bps,
        tick_timestamps_ns=ts_valid,
        scaled=scaled,
        rth_mask=rth,
        trade_mask=trade_mask_valid,
        n_valid=n_valid,
    )


