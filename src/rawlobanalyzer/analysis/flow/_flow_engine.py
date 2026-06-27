"""Shared flow computation engine for the flow/ domain.

Single source of truth for extracting trades, computing OFI (Cont, Kukanov &
Stoikov 2014), and resampling at configurable timescales from raw MBO + LOB data.
Flow-domain analyzers consume ``DayFlow`` produced by ``compute_day_flow()``
rather than independently processing MBO events.

**MBO Trade Pairing (Critical)**:
Databento MBO data emits *two* events per trade: one for the aggressor
(incoming/taker, ``order_id=0``) and one for the passive (resting/maker,
``order_id!=0``).  The MBO-LOB-reconstructor exports both without
deduplication, so naive processing double-counts every trade.  This engine
filters to **aggressor-only** events (``order_id == 0``) for trade extraction
and OFI trade contribution.  The ``side`` field on aggressor events gives the
true aggressor direction (``SIDE_BID`` = buyer-initiated, ``SIDE_ASK`` =
seller-initiated).  ``SIDE_NONE`` aggressors are non-attributable system
trades with unknown direction.

OFI formula:
    OFI_t = sum_i(sign_i * size_i)
    where sign is +1 for buy-side pressure (bid add, ask cancel, buyer-initiated
    trade) and -1 for sell-side pressure (ask add, bid cancel, seller-initiated
    trade). Events at BBO only (add/cancel) or aggressor-only (trade).

Price unit: nanodollars (int64) in MBO, USD (float64) in LOB derived columns.
Time unit: timestamps are int64 nanoseconds since epoch (UTC).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EPS, NANODOLLARS_PER_DOLLAR, NS_PER_SECOND
from rawlobanalyzer.core.resampler import resample, resample_to_grid
from rawlobanalyzer.core.time_utils import rth_grid_edges_ns, rth_mask_utc, seconds_to_label
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.io.schema import (
    ACTION_ADD,
    ACTION_CANCEL,
    ACTION_TRADE,
    SIDE_ASK,
    SIDE_BID,
    SIDE_NONE,
)


@dataclass(frozen=True)
class ScaledOFI:
    """OFI resampled at a single timescale.

    Attributes:
        label: Timescale label (e.g. ``"1s"``, ``"5m"``).
        net_ofi: Net OFI per time bin (sum of signed OFI values).
        normalized_ofi: ``net_ofi / std(net_ofi)`` for the day. NaN where
            std is zero. Makes OFI comparable across days and stocks.
        bin_timestamps_ns: Left edges of each bin in nanoseconds.
        counts: Number of OFI events per bin.
        n_bins_total: Total number of bins (including empty).
        n_bins_filled: Number of bins with at least one event.
    """

    label: str
    net_ofi: np.ndarray
    normalized_ofi: np.ndarray
    bin_timestamps_ns: np.ndarray
    counts: np.ndarray
    n_bins_total: int
    n_bins_filled: int


@dataclass
class DayFlow:
    """All flow data for a single trading day.

    Produced by ``compute_day_flow()`` and consumed by all flow-domain analyzers.
    Trade arrays contain **aggressor-only** events (``order_id == 0`` in MBO),
    so each physical trade is represented exactly once.

    Attributes:
        date: Trading date ``YYYY-MM-DD``.
        trade_timestamps_ns: Timestamps of aggressor-side MBO Trade events.
        trade_prices_usd: Trade prices in USD.
        trade_sizes: Trade sizes in shares.
        trade_sides: int8 array -- ``SIDE_BID`` (66) for buyer-initiated,
            ``SIDE_ASK`` (65) for seller-initiated, ``SIDE_NONE`` (78) for
            non-attributable system trades.
        trade_mid_before: LOB mid-price just before each trade (USD).
        trade_spread_before: LOB spread just before each trade (USD).
        ofi_timestamps_ns: Timestamps for OFI-contributing events.
        ofi_values: Signed OFI per event (positive = buy pressure).
        ofi_add_values: OFI from Add events only (bid add = +, ask add = -).
        ofi_cancel_values: OFI from Cancel events only (ask cancel = +, bid cancel = -).
        ofi_trade_values: OFI from aggressor Trade events only (buyer = +, seller = -).
            Invariant: ``ofi_values == ofi_add_values + ofi_cancel_values + ofi_trade_values``.
        scaled_ofi: Per-timescale resampled OFI, keyed by label.
        rth_mask_trades: Boolean RTH mask over trades.
        directional_mask: Boolean mask -- True where trade has a known
            aggressor side (SIDE_BID or SIDE_ASK), False for SIDE_NONE.
            Use this to exclude non-directional trades from cumulative
            delta, aggressor ratio, and trade imbalance calculations.
        n_trades: Total number of aggressor-side trades (unique trades).
        n_ofi_events: Total number of OFI-contributing events.
    """

    date: str
    trade_timestamps_ns: np.ndarray
    trade_prices_usd: np.ndarray
    trade_sizes: np.ndarray
    trade_sides: np.ndarray
    trade_mid_before: np.ndarray
    trade_spread_before: np.ndarray
    ofi_timestamps_ns: np.ndarray
    ofi_values: np.ndarray
    ofi_add_values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    ofi_cancel_values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    ofi_trade_values: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.float64)
    )
    scaled_ofi: dict[str, ScaledOFI] = field(default_factory=dict)
    rth_mask_trades: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    directional_mask: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=np.bool_)
    )
    n_trades: int = 0
    n_ofi_events: int = 0


def _empty_day_flow(date: str) -> DayFlow:
    """Return an empty DayFlow for days with no usable data."""
    _empty_f64 = np.array([], dtype=np.float64)
    return DayFlow(
        date=date,
        trade_timestamps_ns=np.array([], dtype=np.int64),
        trade_prices_usd=_empty_f64,
        trade_sizes=np.array([], dtype=np.uint32),
        trade_sides=np.array([], dtype=np.int8),
        trade_mid_before=_empty_f64,
        trade_spread_before=_empty_f64,
        ofi_timestamps_ns=np.array([], dtype=np.int64),
        ofi_values=_empty_f64,
        ofi_add_values=_empty_f64,
        ofi_cancel_values=_empty_f64,
        ofi_trade_values=_empty_f64,
    )


def compute_day_flow(
    day: DayData,
    config: AnalysisConfig,
    *,
    day_epoch_ns: int = 0,
) -> DayFlow:
    """Compute trade extraction, OFI, and multi-scale OFI from one day's data.

    Fully vectorized: uses a single ``np.searchsorted`` call to align all MBO
    events with the LOB state, then boolean indexing for BBO comparison.

    Args:
        day: One trading day's LOB + MBO data.
        config: Analysis configuration (timescales, trading hours, flow thresholds).
        day_epoch_ns: Midnight UTC of the trading day (nanoseconds since epoch).
            When non-zero, OFI is resampled onto a canonical RTH grid so that
            bins align with those produced by ``compute_day_returns()``.

    Returns:
        ``DayFlow`` with trade data, signed OFI, and per-timescale OFI.
    """
    if day.mbo is None or day.lob is None:
        return _empty_day_flow(day.date)

    mbo = day.mbo
    lob = day.lob

    required_mbo = {"timestamp_ns", "order_id", "action", "side", "price", "size"}
    required_lob = {"timestamp_ns", "best_bid", "best_ask"}
    if not required_mbo.issubset(set(mbo.column_names)):
        return _empty_day_flow(day.date)
    if not required_lob.issubset(set(lob.column_names)):
        return _empty_day_flow(day.date)

    mbo_ts = mbo.column("timestamp_ns").to_numpy()
    mbo_order_id = mbo.column("order_id").to_numpy()
    mbo_action = mbo.column("action").to_numpy()
    mbo_side = mbo.column("side").to_numpy()
    mbo_price = mbo.column("price").to_numpy()
    mbo_size = mbo.column("size").to_numpy()

    lob_ts = lob.column("timestamp_ns").to_numpy()
    lob_best_bid = lob.column("best_bid").to_numpy()
    lob_best_ask = lob.column("best_ask").to_numpy()

    has_mid = "mid_price" in lob.column_names
    has_spread = "spread" in lob.column_names
    lob_mid = lob.column("mid_price").to_numpy() if has_mid else None
    lob_spread = lob.column("spread").to_numpy() if has_spread else None

    if len(mbo_ts) == 0 or len(lob_ts) == 0:
        return _empty_day_flow(day.date)

    # --- Vectorized MBO-LOB alignment ---
    # For each MBO event, find the LOB snapshot just before it.
    # searchsorted(side="right") - 1 gives the last LOB snapshot <= mbo_ts.
    # Events before the first LOB snapshot get lob_idx == -1; these MUST be
    # excluded (not clipped to 0) to avoid lookahead into future snapshots.
    lob_idx = np.searchsorted(lob_ts, mbo_ts, side="right") - 1
    valid_alignment = lob_idx >= 0
    lob_idx_safe = np.where(valid_alignment, lob_idx, 0)

    aligned_best_bid = lob_best_bid[lob_idx_safe]
    aligned_best_ask = lob_best_ask[lob_idx_safe]

    # --- Trade extraction (aggressor-only) ---
    # Databento MBO emits two events per trade: aggressor (order_id=0) and
    # passive (order_id!=0).  Keep only the aggressor side to avoid
    # double-counting trades, volumes, and OFI trade contributions.
    is_aggressor = mbo_order_id == 0
    trade_mask = (mbo_action == ACTION_TRADE) & is_aggressor
    trade_valid = trade_mask & valid_alignment
    n_trades = int(np.sum(trade_valid))

    trade_ts = mbo_ts[trade_valid]
    trade_prices_nd = mbo_price[trade_valid]
    trade_prices_usd = trade_prices_nd.astype(np.float64) / NANODOLLARS_PER_DOLLAR
    trade_sizes = mbo_size[trade_valid]
    trade_sides = mbo_side[trade_valid].astype(np.int8)

    trade_mid_before = np.full(n_trades, np.nan, dtype=np.float64)
    trade_spread_before = np.full(n_trades, np.nan, dtype=np.float64)

    if lob_mid is not None:
        trade_mid_before = lob_mid[lob_idx_safe[trade_valid]]
    else:
        bid_usd = aligned_best_bid[trade_valid].astype(np.float64) / NANODOLLARS_PER_DOLLAR
        ask_usd = aligned_best_ask[trade_valid].astype(np.float64) / NANODOLLARS_PER_DOLLAR
        trade_mid_before = (bid_usd + ask_usd) / 2.0

    if lob_spread is not None:
        trade_spread_before = lob_spread[lob_idx_safe[trade_valid]]
    else:
        bid_usd = aligned_best_bid[trade_valid].astype(np.float64) / NANODOLLARS_PER_DOLLAR
        ask_usd = aligned_best_ask[trade_valid].astype(np.float64) / NANODOLLARS_PER_DOLLAR
        trade_spread_before = ask_usd - bid_usd

    rth_mask_trades = rth_mask_utc(
        trade_ts,
        utc_offset_hours=config.trading_hours.utc_offset_hours,
    ) if n_trades > 0 else np.array([], dtype=np.bool_)

    directional_mask = (trade_sides != SIDE_NONE) if n_trades > 0 else np.array([], dtype=np.bool_)

    # --- OFI computation (Cont, Kukanov & Stoikov 2014) ---
    # Events contributing to OFI: Add/Cancel at BBO, aggressor-only Trade.
    # Using ``is_aggressor`` for trades prevents the passive-side event from
    # cancelling the aggressor's OFI contribution (they have opposite signs
    # but represent the same physical trade).
    is_add = mbo_action == ACTION_ADD
    is_cancel = mbo_action == ACTION_CANCEL
    is_agg_trade = (mbo_action == ACTION_TRADE) & is_aggressor
    is_bid = mbo_side == SIDE_BID
    is_ask = mbo_side == SIDE_ASK

    at_best_bid = (mbo_price == aligned_best_bid) & valid_alignment
    at_best_ask = (mbo_price == aligned_best_ask) & valid_alignment

    ofi_size = mbo_size.astype(np.float64)

    # Component-level signed arrays for decomposition
    ofi_add_signed = np.zeros(len(mbo_ts), dtype=np.float64)
    ofi_cancel_signed = np.zeros(len(mbo_ts), dtype=np.float64)
    ofi_trade_signed = np.zeros(len(mbo_ts), dtype=np.float64)

    # Buy pressure: bid add, ask cancel, buyer-initiated aggressor trade
    ofi_add_signed[is_add & is_bid & at_best_bid] = 1.0
    ofi_cancel_signed[is_cancel & is_ask & at_best_ask] = 1.0
    ofi_trade_signed[is_agg_trade & is_bid & valid_alignment] = 1.0

    # Sell pressure: ask add, bid cancel, seller-initiated aggressor trade
    ofi_add_signed[is_add & is_ask & at_best_ask] = -1.0
    ofi_cancel_signed[is_cancel & is_bid & at_best_bid] = -1.0
    ofi_trade_signed[is_agg_trade & is_ask & valid_alignment] = -1.0

    ofi_signed = ofi_add_signed + ofi_cancel_signed + ofi_trade_signed

    ofi_contributing = ofi_signed != 0.0
    ofi_timestamps = mbo_ts[ofi_contributing]
    ofi_values = (ofi_signed * ofi_size)[ofi_contributing]
    ofi_add_values = (ofi_add_signed * ofi_size)[ofi_contributing]
    ofi_cancel_values = (ofi_cancel_signed * ofi_size)[ofi_contributing]
    ofi_trade_values = (ofi_trade_signed * ofi_size)[ofi_contributing]
    n_ofi_events = int(np.sum(ofi_contributing))

    # --- Multi-scale OFI resampling ---
    flow_cfg = config.thresholds.flow
    utc_off = config.trading_hours.utc_offset_hours
    scaled_ofi: dict[str, ScaledOFI] = {}

    if n_ofi_events > 0:
        rth_ofi = rth_mask_utc(ofi_timestamps, utc_offset_hours=utc_off)
        ofi_ts_rth = ofi_timestamps[rth_ofi]
        ofi_vals_rth = ofi_values[rth_ofi]

        if len(ofi_ts_rth) > 0:
            for scale_s in flow_cfg.ofi_timescales_seconds:
                res_ns = int(scale_s * NS_PER_SECOND)
                label = seconds_to_label(scale_s)

                if day_epoch_ns > 0:
                    grid = rth_grid_edges_ns(
                        day_epoch_ns, res_ns, utc_offset_hours=utc_off,
                    )
                    resampled = resample_to_grid(
                        ofi_ts_rth, ofi_vals_rth, grid,
                        agg="sum", label=label,
                    )
                else:
                    resampled = resample(
                        ofi_ts_rth, ofi_vals_rth, res_ns,
                        agg="sum", label=label,
                    )

                net = resampled.values
                ofi_std = np.nanstd(net)
                if np.isfinite(ofi_std) and ofi_std > EPS:
                    normed = net / ofi_std
                else:
                    normed = np.full_like(net, np.nan)

                scaled_ofi[label] = ScaledOFI(
                    label=label,
                    net_ofi=net,
                    normalized_ofi=normed,
                    bin_timestamps_ns=resampled.bin_edges_ns[:-1]
                    if len(resampled.bin_edges_ns) > 0
                    else np.array([], dtype=np.int64),
                    counts=resampled.counts,
                    n_bins_total=len(resampled.counts),
                    n_bins_filled=int(np.sum(resampled.counts > 0)),
                )

    return DayFlow(
        date=day.date,
        trade_timestamps_ns=trade_ts,
        trade_prices_usd=trade_prices_usd,
        trade_sizes=trade_sizes,
        trade_sides=trade_sides,
        trade_mid_before=trade_mid_before,
        trade_spread_before=trade_spread_before,
        ofi_timestamps_ns=ofi_timestamps,
        ofi_values=ofi_values,
        ofi_add_values=ofi_add_values,
        ofi_cancel_values=ofi_cancel_values,
        ofi_trade_values=ofi_trade_values,
        scaled_ofi=scaled_ofi,
        rth_mask_trades=rth_mask_trades,
        directional_mask=directional_mask,
        n_trades=n_trades,
        n_ofi_events=n_ofi_events,
    )


