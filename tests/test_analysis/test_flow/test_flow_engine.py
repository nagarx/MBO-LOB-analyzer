"""Tests for the shared flow engine (_flow_engine.py).

Validates trade extraction, OFI computation, vectorized MBO-LOB alignment,
and multi-scale OFI resampling against synthetic data with known outcomes.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rawlobanalyzer.analysis.flow._flow_engine import (
    DayFlow,
    ScaledOFI,
    compute_day_flow,
    _empty_day_flow,
)
from rawlobanalyzer.config.analysis_config import AnalysisConfig, FlowThresholds
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.io.schema import (
    ACTION_ADD,
    ACTION_CANCEL,
    ACTION_TRADE,
    SIDE_ASK,
    SIDE_BID,
    SIDE_NONE,
)

NANODOLLARS_PER_DOLLAR = 1_000_000_000


def _make_day(
    mbo_actions: np.ndarray,
    mbo_sides: np.ndarray,
    mbo_prices_nd: np.ndarray,
    mbo_sizes: np.ndarray,
    lob_best_bid_nd: np.ndarray,
    lob_best_ask_nd: np.ndarray,
    *,
    mbo_order_ids: np.ndarray | None = None,
    n_lob: int | None = None,
    ts_start: int = 1_738_594_800_000_000_000,
    ts_step: int = 1_000_000,
) -> DayData:
    """Build a minimal DayData from arrays for testing.

    When ``mbo_order_ids`` is not provided, trade events (ACTION_TRADE)
    automatically receive ``order_id=0`` (aggressor convention) and
    non-trade events receive sequential IDs starting at 1.  This matches
    the real Databento MBO structure where the aggressor side of each
    trade has ``order_id=0``.
    """
    n_mbo = len(mbo_actions)
    if n_lob is None:
        n_lob = n_mbo

    if mbo_order_ids is None:
        mbo_order_ids = np.where(
            mbo_actions == ACTION_TRADE,
            np.uint64(0),
            np.arange(1, n_mbo + 1, dtype=np.uint64),
        )

    mbo_ts = np.arange(ts_start, ts_start + n_mbo * ts_step, ts_step, dtype=np.int64)
    lob_ts = np.arange(ts_start, ts_start + n_lob * ts_step, ts_step, dtype=np.int64)

    mid_nd = (lob_best_bid_nd.astype(np.float64) + lob_best_ask_nd.astype(np.float64)) / 2.0
    mid_usd = mid_nd / NANODOLLARS_PER_DOLLAR
    spread_usd = (lob_best_ask_nd.astype(np.float64) - lob_best_bid_nd.astype(np.float64)) / NANODOLLARS_PER_DOLLAR

    lob = pa.table({
        "timestamp_ns": pa.array(lob_ts, type=pa.int64()),
        "best_bid": pa.array(lob_best_bid_nd, type=pa.int64()),
        "best_ask": pa.array(lob_best_ask_nd, type=pa.int64()),
        "mid_price": pa.array(mid_usd, type=pa.float64()),
        "spread": pa.array(spread_usd, type=pa.float64()),
    })
    mbo = pa.table({
        "timestamp_ns": pa.array(mbo_ts, type=pa.int64()),
        "order_id": pa.array(mbo_order_ids.astype(np.uint64)),
        "action": pa.array(mbo_actions.astype(np.uint8)),
        "side": pa.array(mbo_sides.astype(np.uint8)),
        "price": pa.array(mbo_prices_nd, type=pa.int64()),
        "size": pa.array(mbo_sizes.astype(np.uint32)),
    })

    metadata = {
        b"schema_version": b"1.0",
        b"source": b"mbo-lob-reconstructor",
        b"symbol": b"TEST",
        b"date": b"2025-02-03",
    }
    lob = lob.replace_schema_metadata(metadata)
    mbo = mbo.replace_schema_metadata(metadata)

    return DayData(date="2025-02-03", symbol="TEST", lob=lob, mbo=mbo)


def _default_config(**overrides) -> AnalysisConfig:
    flow_kw = overrides.pop("flow", {})
    return AnalysisConfig(
        data_dir=Path("/tmp/test"),
        symbol="TEST",
        thresholds=AnalysisConfig.__dataclass_fields__["thresholds"].default_factory(),
        **overrides,
    )


class TestEmptyDayFlow:
    def test_empty_day_flow_fields(self):
        df = _empty_day_flow("2025-01-01")
        assert df.date == "2025-01-01"
        assert df.n_trades == 0
        assert df.n_ofi_events == 0
        assert len(df.trade_timestamps_ns) == 0
        assert len(df.ofi_values) == 0

    def test_no_mbo_returns_empty(self):
        lob = pa.table({
            "timestamp_ns": pa.array([1], type=pa.int64()),
            "best_bid": pa.array([100_000_000_000], type=pa.int64()),
            "best_ask": pa.array([100_010_000_000], type=pa.int64()),
        })
        day = DayData(date="2025-02-03", symbol="TEST", lob=lob, mbo=None)
        config = _default_config()
        result = compute_day_flow(day, config)
        assert result.n_trades == 0
        assert result.n_ofi_events == 0


class TestTradeExtraction:
    def test_single_trade(self):
        """Single buyer-initiated trade at the ask."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_TRADE]),
            mbo_sides=np.array([SIDE_BID]),
            mbo_prices_nd=np.array([100_010_000_000], dtype=np.int64),
            mbo_sizes=np.array([100]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 1
        np.testing.assert_allclose(result.trade_prices_usd, [100.01], atol=1e-6)
        np.testing.assert_array_equal(result.trade_sizes, [100])
        assert result.trade_sides[0] == SIDE_BID

    def test_multiple_trades(self):
        """Mix of buyer and seller-initiated trades."""
        n = 5
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_ADD, ACTION_TRADE, ACTION_CANCEL, ACTION_TRADE, ACTION_ADD])
        sides = np.array([SIDE_BID, SIDE_BID, SIDE_ASK, SIDE_ASK, SIDE_BID])
        prices = np.array([100_000_000_000, 100_010_000_000, 100_010_000_000,
                          100_000_000_000, 100_000_000_000], dtype=np.int64)
        sizes = np.array([200, 150, 300, 250, 100], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 2
        np.testing.assert_array_equal(result.trade_sizes, [150, 250])

    def test_mid_before_trade(self):
        """Mid-price before trade is computed from LOB state."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_020_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_TRADE]),
            mbo_sides=np.array([SIDE_BID]),
            mbo_prices_nd=np.array([100_020_000_000], dtype=np.int64),
            mbo_sizes=np.array([50]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        expected_mid = (100.0 + 100.02) / 2.0
        np.testing.assert_allclose(result.trade_mid_before, [expected_mid], atol=1e-6)


class TestOFIComputation:
    def test_bid_add_positive_ofi(self):
        """Add at best bid = positive OFI (buy pressure)."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_ADD]),
            mbo_sides=np.array([SIDE_BID]),
            mbo_prices_nd=np.array([100_000_000_000], dtype=np.int64),
            mbo_sizes=np.array([500]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_ofi_events == 1
        assert result.ofi_values[0] == 500.0

    def test_ask_add_negative_ofi(self):
        """Add at best ask = negative OFI (sell pressure)."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_ADD]),
            mbo_sides=np.array([SIDE_ASK]),
            mbo_prices_nd=np.array([100_010_000_000], dtype=np.int64),
            mbo_sizes=np.array([300]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_ofi_events == 1
        assert result.ofi_values[0] == -300.0

    def test_bid_cancel_negative_ofi(self):
        """Cancel at best bid = negative OFI (buy withdrawal)."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_CANCEL]),
            mbo_sides=np.array([SIDE_BID]),
            mbo_prices_nd=np.array([100_000_000_000], dtype=np.int64),
            mbo_sizes=np.array([200]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_ofi_events == 1
        assert result.ofi_values[0] == -200.0

    def test_ask_cancel_positive_ofi(self):
        """Cancel at best ask = positive OFI (sell withdrawal = buy pressure)."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_CANCEL]),
            mbo_sides=np.array([SIDE_ASK]),
            mbo_prices_nd=np.array([100_010_000_000], dtype=np.int64),
            mbo_sizes=np.array([400]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_ofi_events == 1
        assert result.ofi_values[0] == 400.0

    def test_add_away_from_bbo_no_ofi(self):
        """Add away from BBO does not contribute to OFI."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_ADD]),
            mbo_sides=np.array([SIDE_BID]),
            mbo_prices_nd=np.array([99_990_000_000], dtype=np.int64),
            mbo_sizes=np.array([1000]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_ofi_events == 0

    def test_mixed_ofi_sequence(self):
        """Sequence of events with known net OFI."""
        n = 4
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_ADD, ACTION_ADD, ACTION_CANCEL, ACTION_TRADE])
        sides = np.array([SIDE_BID, SIDE_ASK, SIDE_BID, SIDE_BID])
        prices = np.array([100_000_000_000, 100_010_000_000,
                          100_000_000_000, 100_010_000_000], dtype=np.int64)
        sizes = np.array([100, 200, 50, 300], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        # bid add +100, ask add -200, bid cancel -50, buyer trade +300
        expected_ofi = np.array([100.0, -200.0, -50.0, 300.0])
        assert result.n_ofi_events == 4
        np.testing.assert_array_equal(result.ofi_values, expected_ofi)
        assert np.sum(result.ofi_values) == 150.0


class TestScaledOFI:
    def test_ofi_resampled_at_configured_scales(self):
        """OFI is resampled at all configured timescales."""
        n = 10
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.full(n, ACTION_ADD, dtype=np.uint8)
        sides = np.full(n, SIDE_BID, dtype=np.uint8)
        prices = np.full(n, 100_000_000_000, dtype=np.int64)
        sizes = np.full(n, 100, dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        expected_labels = {"1s", "5s", "10s", "30s", "1m", "5m"}
        assert set(result.scaled_ofi.keys()) == expected_labels

        for label, sofi in result.scaled_ofi.items():
            assert isinstance(sofi, ScaledOFI)
            assert sofi.label == label
            total_ofi = np.nansum(sofi.net_ofi)
            np.testing.assert_allclose(total_ofi, 1000.0, atol=1e-6)


class TestRTHMask:
    def test_rth_mask_applied_to_trades(self):
        """RTH mask is computed for trade timestamps."""
        bid = np.array([100_000_000_000], dtype=np.int64)
        ask = np.array([100_010_000_000], dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_TRADE]),
            mbo_sides=np.array([SIDE_BID]),
            mbo_prices_nd=np.array([100_010_000_000], dtype=np.int64),
            mbo_sizes=np.array([100]),
            lob_best_bid_nd=bid,
            lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert len(result.rth_mask_trades) == 1
        assert result.rth_mask_trades.dtype == np.bool_


class TestEdgeCases:
    def test_no_trades_in_mbo(self):
        """MBO with only adds/cancels produces OFI but no trades."""
        n = 3
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_ADD, ACTION_CANCEL, ACTION_ADD])
        sides = np.array([SIDE_BID, SIDE_ASK, SIDE_ASK])
        prices = np.array([100_000_000_000, 100_010_000_000,
                          100_010_000_000], dtype=np.int64)
        sizes = np.array([100, 200, 300], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 0
        assert result.n_ofi_events == 3
        assert len(result.trade_timestamps_ns) == 0

    def test_large_dataset_vectorized(self):
        """Verify vectorized alignment handles large arrays without error."""
        n = 50_000
        rng = np.random.default_rng(42)
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = rng.choice([ACTION_ADD, ACTION_CANCEL, ACTION_TRADE], size=n).astype(np.uint8)
        sides = rng.choice([SIDE_BID, SIDE_ASK], size=n).astype(np.uint8)
        prices = rng.choice([100_000_000_000, 100_010_000_000], size=n).astype(np.int64)
        sizes = rng.integers(1, 1000, size=n).astype(np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == int(np.sum(actions == ACTION_TRADE))
        assert result.n_ofi_events > 0
        assert len(result.scaled_ofi) > 0


class TestSideNoneHandling:
    """SIDE_NONE trades must be included in total trade count but excluded
    from directional metrics (cumulative delta, aggressor ratio, imbalance)."""

    def test_side_none_counted_as_trade(self):
        """SIDE_NONE trades appear in n_trades and trade arrays."""
        n = 3
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_TRADE, ACTION_TRADE, ACTION_TRADE])
        sides = np.array([SIDE_BID, SIDE_ASK, SIDE_NONE])
        prices = np.full(n, 100_005_000_000, dtype=np.int64)
        sizes = np.array([100, 200, 300], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 3
        assert len(result.trade_sizes) == 3
        np.testing.assert_array_equal(result.trade_sizes, [100, 200, 300])

    def test_directional_mask_excludes_side_none(self):
        """directional_mask is False for SIDE_NONE trades."""
        n = 4
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_TRADE, ACTION_TRADE, ACTION_TRADE, ACTION_TRADE])
        sides = np.array([SIDE_BID, SIDE_NONE, SIDE_ASK, SIDE_NONE])
        prices = np.full(n, 100_005_000_000, dtype=np.int64)
        sizes = np.array([100, 500, 200, 700], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 4
        np.testing.assert_array_equal(
            result.directional_mask, [True, False, True, False],
        )

    def test_side_none_excluded_from_ofi(self):
        """SIDE_NONE trades do not contribute to OFI (neither bid nor ask)."""
        n = 2
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_TRADE, ACTION_TRADE])
        sides = np.array([SIDE_NONE, SIDE_NONE])
        prices = np.full(n, 100_005_000_000, dtype=np.int64)
        sizes = np.array([1000, 2000], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 2
        assert result.n_ofi_events == 0

    def test_mixed_sides_directional_mask(self):
        """Realistic mix of buyer, seller, and system trades."""
        n = 6
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_TRADE] * 6)
        sides = np.array([SIDE_BID, SIDE_ASK, SIDE_NONE, SIDE_BID, SIDE_NONE, SIDE_ASK])
        prices = np.full(n, 100_005_000_000, dtype=np.int64)
        sizes = np.array([100, 200, 5000, 300, 8000, 400], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        assert result.n_trades == 6
        dir_sizes = result.trade_sizes[result.directional_mask]
        np.testing.assert_array_equal(dir_sizes, [100, 200, 300, 400])

        dir_sides = result.trade_sides[result.directional_mask]
        buyer_vol = np.sum(dir_sizes[dir_sides == SIDE_BID].astype(float))
        seller_vol = np.sum(dir_sizes[dir_sides == SIDE_ASK].astype(float))
        expected_buyer_frac = buyer_vol / (buyer_vol + seller_vol)
        np.testing.assert_allclose(expected_buyer_frac, 400 / 1000, atol=1e-6)


class TestOFIDecomposition:
    """OFI component decomposition: add/cancel/trade sum to total OFI."""

    def test_components_sum_to_total(self):
        """ofi_add + ofi_cancel + ofi_trade == ofi_values (invariant)."""
        n = 4
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = np.array([ACTION_ADD, ACTION_ADD, ACTION_CANCEL, ACTION_TRADE])
        sides = np.array([SIDE_BID, SIDE_ASK, SIDE_BID, SIDE_BID])
        prices = np.array([100_000_000_000, 100_010_000_000,
                          100_000_000_000, 100_010_000_000], dtype=np.int64)
        sizes = np.array([100, 200, 50, 300], dtype=np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        total = result.ofi_add_values + result.ofi_cancel_values + result.ofi_trade_values
        np.testing.assert_array_almost_equal(total, result.ofi_values)

    def test_add_only_decomposition(self):
        """Only add events: ofi_add == ofi_values, cancel/trade == 0."""
        n = 2
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_ADD, ACTION_ADD]),
            mbo_sides=np.array([SIDE_BID, SIDE_ASK]),
            mbo_prices_nd=np.array([100_000_000_000, 100_010_000_000], dtype=np.int64),
            mbo_sizes=np.array([100, 200], dtype=np.uint32),
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        np.testing.assert_array_equal(result.ofi_add_values, result.ofi_values)
        np.testing.assert_array_equal(result.ofi_cancel_values, np.zeros(2))
        np.testing.assert_array_equal(result.ofi_trade_values, np.zeros(2))

    def test_cancel_only_decomposition(self):
        """Only cancel events: ofi_cancel == ofi_values, add/trade == 0."""
        n = 2
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        day = _make_day(
            mbo_actions=np.array([ACTION_CANCEL, ACTION_CANCEL]),
            mbo_sides=np.array([SIDE_ASK, SIDE_BID]),
            mbo_prices_nd=np.array([100_010_000_000, 100_000_000_000], dtype=np.int64),
            mbo_sizes=np.array([400, 200], dtype=np.uint32),
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        np.testing.assert_array_equal(result.ofi_cancel_values, result.ofi_values)
        np.testing.assert_array_equal(result.ofi_add_values, np.zeros(2))
        np.testing.assert_array_equal(result.ofi_trade_values, np.zeros(2))

    def test_empty_day_has_empty_components(self):
        """Empty DayFlow has zero-length component arrays."""
        df = _empty_day_flow("2025-01-01")
        assert len(df.ofi_add_values) == 0
        assert len(df.ofi_cancel_values) == 0
        assert len(df.ofi_trade_values) == 0

    def test_large_dataset_invariant(self):
        """Component decomposition invariant holds for large random data."""
        n = 50_000
        rng = np.random.default_rng(99)
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        actions = rng.choice([ACTION_ADD, ACTION_CANCEL, ACTION_TRADE], size=n).astype(np.uint8)
        sides = rng.choice([SIDE_BID, SIDE_ASK], size=n).astype(np.uint8)
        prices = rng.choice([100_000_000_000, 100_010_000_000], size=n).astype(np.int64)
        sizes = rng.integers(1, 1000, size=n).astype(np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        total = result.ofi_add_values + result.ofi_cancel_values + result.ofi_trade_values
        np.testing.assert_array_almost_equal(total, result.ofi_values)


class TestNormalizedOFI:
    """Normalized OFI in ScaledOFI."""

    def test_normalized_ofi_has_unit_std(self):
        """Normalized OFI should have std ~1 (for filled bins)."""
        n = 10
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        rng = np.random.default_rng(7)
        actions = np.full(n, ACTION_ADD, dtype=np.uint8)
        sides = rng.choice([SIDE_BID, SIDE_ASK], size=n).astype(np.uint8)
        prices_list = []
        for s in sides:
            if s == SIDE_BID:
                prices_list.append(100_000_000_000)
            else:
                prices_list.append(100_010_000_000)
        prices = np.array(prices_list, dtype=np.int64)
        sizes = rng.integers(50, 500, size=n).astype(np.uint32)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        for label, sofi in result.scaled_ofi.items():
            assert hasattr(sofi, "normalized_ofi")
            valid = sofi.normalized_ofi[np.isfinite(sofi.normalized_ofi)]
            if len(valid) > 1:
                np.testing.assert_allclose(np.std(valid), 1.0, atol=0.1)

    def test_constant_ofi_produces_nan_normalized(self):
        """When all OFI values are identical (std=0), normalized should be NaN."""
        n = 5
        bid = np.full(n, 100_000_000_000, dtype=np.int64)
        ask = np.full(n, 100_010_000_000, dtype=np.int64)

        day = _make_day(
            mbo_actions=np.full(n, ACTION_ADD, dtype=np.uint8),
            mbo_sides=np.full(n, SIDE_BID, dtype=np.uint8),
            mbo_prices_nd=np.full(n, 100_000_000_000, dtype=np.int64),
            mbo_sizes=np.full(n, 100, dtype=np.uint32),
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
        )
        config = _default_config()
        result = compute_day_flow(day, config)

        for label, sofi in result.scaled_ofi.items():
            filled_norm = sofi.normalized_ofi[sofi.counts > 0]
            if len(filled_norm) > 0:
                all_same = np.all(sofi.net_ofi[sofi.counts > 0] == sofi.net_ofi[sofi.counts > 0][0])
                if all_same:
                    assert np.all(np.isnan(filled_norm))


class TestMBOTradeDeduplication:
    """Verify that paired MBO trade events (aggressor + passive) are
    correctly deduplicated so each physical trade is counted once."""

    @staticmethod
    def _paired_day(
        n_buys: int,
        n_sells: int,
        *,
        n_side_none: int = 0,
        buy_size: int = 100,
        sell_size: int = 200,
    ) -> tuple[DayData, "AnalysisConfig"]:
        """Build a DayData with properly paired aggressor+passive trades.

        Each buy trade produces two MBO rows:
            aggressor: order_id=0, side=BID, price=best_ask
            passive:   order_id=X, side=ASK, price=best_ask
        Each sell trade likewise produces two rows with opposite sides.
        SIDE_NONE trades produce a single aggressor row (no passive).
        """
        rows = []
        oid_counter = 1000
        bid_nd = 100_000_000_000
        ask_nd = 100_010_000_000

        for _ in range(n_buys):
            rows.append((0, ACTION_TRADE, SIDE_BID, ask_nd, buy_size))
            rows.append((oid_counter, ACTION_TRADE, SIDE_ASK, ask_nd, buy_size))
            oid_counter += 1

        for _ in range(n_sells):
            rows.append((0, ACTION_TRADE, SIDE_ASK, bid_nd, sell_size))
            rows.append((oid_counter, ACTION_TRADE, SIDE_BID, bid_nd, sell_size))
            oid_counter += 1

        for _ in range(n_side_none):
            rows.append((0, ACTION_TRADE, SIDE_NONE, bid_nd + 5_000_000, 500))

        n_mbo = len(rows)
        order_ids = np.array([r[0] for r in rows], dtype=np.uint64)
        actions = np.array([r[1] for r in rows], dtype=np.uint8)
        sides = np.array([r[2] for r in rows], dtype=np.uint8)
        prices = np.array([r[3] for r in rows], dtype=np.int64)
        sizes = np.array([r[4] for r in rows], dtype=np.uint32)

        bid = np.full(n_mbo, bid_nd, dtype=np.int64)
        ask = np.full(n_mbo, ask_nd, dtype=np.int64)

        day = _make_day(
            mbo_actions=actions, mbo_sides=sides,
            mbo_prices_nd=prices, mbo_sizes=sizes,
            lob_best_bid_nd=bid, lob_best_ask_nd=ask,
            mbo_order_ids=order_ids,
        )
        config = _default_config()
        return day, config

    def test_paired_trades_counted_once(self):
        """5 buy + 3 sell trades -> n_trades = 8 (not 16)."""
        day, config = self._paired_day(n_buys=5, n_sells=3)
        result = compute_day_flow(day, config)
        assert result.n_trades == 8

    def test_cumulative_delta_nonzero(self):
        """Buy-dominated flow should produce positive cumulative delta."""
        day, config = self._paired_day(n_buys=10, n_sells=2, buy_size=100, sell_size=100)
        result = compute_day_flow(day, config)

        dir_mask = result.directional_mask
        signed = result.trade_sizes[dir_mask].astype(np.float64).copy()
        signed[result.trade_sides[dir_mask] != SIDE_BID] *= -1.0
        delta = float(np.sum(signed))
        assert delta > 0, f"Expected positive delta, got {delta}"
        assert delta == (10 - 2) * 100

    def test_ofi_trade_component_nonzero(self):
        """Trade OFI component should be non-zero for directional trades."""
        day, config = self._paired_day(n_buys=5, n_sells=2, buy_size=100, sell_size=100)
        result = compute_day_flow(day, config)

        trade_ofi_sum = np.sum(result.ofi_trade_values)
        assert trade_ofi_sum != 0.0, "Trade OFI component should not be zero"
        assert trade_ofi_sum > 0, "Buy-dominated flow => positive trade OFI"

    def test_aggressor_ratio_correct(self):
        """6 buys + 4 sells -> buyer fraction = 0.6."""
        day, config = self._paired_day(n_buys=6, n_sells=4, buy_size=100, sell_size=100)
        result = compute_day_flow(day, config)

        dir_mask = result.directional_mask
        buyer_vol = np.sum(
            result.trade_sizes[dir_mask & (result.trade_sides == SIDE_BID)].astype(float)
        )
        seller_vol = np.sum(
            result.trade_sizes[dir_mask & (result.trade_sides == SIDE_ASK)].astype(float)
        )
        buyer_frac = buyer_vol / (buyer_vol + seller_vol)
        np.testing.assert_allclose(buyer_frac, 0.6, atol=1e-6)

    def test_passive_events_excluded(self):
        """Passive-side trade events (order_id != 0) must not appear in DayFlow."""
        day, config = self._paired_day(n_buys=3, n_sells=2)
        result = compute_day_flow(day, config)

        assert result.n_trades == 5
        assert len(result.trade_sizes) == 5
        assert len(result.trade_sides) == 5

    def test_side_none_aggressors_still_included(self):
        """SIDE_NONE aggressors (order_id=0) are included in trade arrays
        but excluded from directional metrics via directional_mask."""
        day, config = self._paired_day(n_buys=2, n_sells=1, n_side_none=3)
        result = compute_day_flow(day, config)

        assert result.n_trades == 6
        assert np.sum(result.directional_mask) == 3
        assert np.sum(~result.directional_mask) == 3

    def test_ofi_decomposition_with_paired_trades(self):
        """Component decomposition invariant holds with paired trades."""
        day, config = self._paired_day(n_buys=5, n_sells=5, buy_size=100, sell_size=100)
        result = compute_day_flow(day, config)

        total = result.ofi_add_values + result.ofi_cancel_values + result.ofi_trade_values
        np.testing.assert_array_almost_equal(total, result.ofi_values)


class TestLookaheadPrevention:
    """Verify that MBO events before the first LOB snapshot are excluded,
    not silently mapped to the first snapshot (P1-A fix).
    """

    def test_early_trade_excluded(self):
        """A trade that arrives BEFORE the first LOB snapshot must not be
        counted in trade arrays, as it would create a lookahead bug.
        """
        bid = 100 * NANODOLLARS_PER_DOLLAR
        ask = 101 * NANODOLLARS_PER_DOLLAR

        n_mbo = 5
        n_lob = 3
        mbo_actions = np.array(
            [ACTION_TRADE, ACTION_ADD, ACTION_ADD, ACTION_TRADE, ACTION_TRADE],
            dtype=np.uint8,
        )
        mbo_sides = np.full(n_mbo, SIDE_BID, dtype=np.uint8)
        mbo_prices = np.full(n_mbo, bid, dtype=np.int64)
        mbo_sizes = np.full(n_mbo, 100, dtype=np.uint32)

        lob_bids = np.full(n_lob, bid, dtype=np.int64)
        lob_asks = np.full(n_lob, ask, dtype=np.int64)

        ts_start = 1_000_000_000
        ts_step = 1_000_000

        mbo_ts = np.array([
            ts_start,
            ts_start + ts_step,
            ts_start + 2 * ts_step,
            ts_start + 3 * ts_step,
            ts_start + 4 * ts_step,
        ], dtype=np.int64)

        lob_ts = np.array([
            ts_start + 2 * ts_step,
            ts_start + 3 * ts_step,
            ts_start + 4 * ts_step,
        ], dtype=np.int64)

        mid_usd = np.full(n_lob, 100.5, dtype=np.float64)
        spread_usd = np.full(n_lob, 1.0, dtype=np.float64)

        mbo_order_ids = np.where(
            mbo_actions == ACTION_TRADE,
            np.uint64(0),
            np.arange(1, n_mbo + 1, dtype=np.uint64),
        )

        lob = pa.table({
            "timestamp_ns": pa.array(lob_ts, type=pa.int64()),
            "best_bid": pa.array(lob_bids, type=pa.int64()),
            "best_ask": pa.array(lob_asks, type=pa.int64()),
            "mid_price": pa.array(mid_usd, type=pa.float64()),
            "spread": pa.array(spread_usd, type=pa.float64()),
        })
        mbo = pa.table({
            "timestamp_ns": pa.array(mbo_ts, type=pa.int64()),
            "order_id": pa.array(mbo_order_ids),
            "action": pa.array(mbo_actions),
            "side": pa.array(mbo_sides),
            "price": pa.array(mbo_prices, type=pa.int64()),
            "size": pa.array(mbo_sizes.astype(np.uint32)),
        })

        metadata = {
            b"schema_version": b"1.0",
            b"source": b"mbo-lob-reconstructor",
            b"symbol": b"TEST",
            b"date": b"2025-02-03",
        }
        lob = lob.replace_schema_metadata(metadata)
        mbo = mbo.replace_schema_metadata(metadata)

        day = DayData(date="2025-02-03", symbol="TEST", lob=lob, mbo=mbo)
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")

        result = compute_day_flow(day, config)

        assert result.n_trades == 2, (
            f"Expected 2 trades (events at ts+3, ts+4 that have LOB data), "
            f"got {result.n_trades}. The early trade at ts+0 should be excluded."
        )
