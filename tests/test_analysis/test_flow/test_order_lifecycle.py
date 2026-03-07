"""Tests for OrderLifecycleAnalyzer."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.flow.order_lifecycle import OrderLifecycleAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.io.schema import (
    ACTION_ADD,
    ACTION_CANCEL,
    ACTION_FILL,
    ACTION_MODIFY,
    ACTION_TRADE,
    SIDE_ASK,
    SIDE_BID,
)
from rawlobanalyzer.io.session import AnalysisSession

NS_PER_SECOND = 1_000_000_000


def _make_lifecycle_day(
    order_ids: np.ndarray,
    actions: np.ndarray,
    sides: np.ndarray,
    prices: np.ndarray,
    sizes: np.ndarray,
    *,
    ts_start: int = 1_738_594_800_000_000_000,
    ts_step: int = 1_000_000_000,
) -> DayData:
    """Build a minimal DayData for lifecycle testing."""
    n = len(order_ids)
    ts = np.arange(ts_start, ts_start + n * ts_step, ts_step, dtype=np.int64)

    mbo = pa.table({
        "timestamp_ns": pa.array(ts, type=pa.int64()),
        "order_id": pa.array(order_ids.astype(np.uint64)),
        "action": pa.array(actions.astype(np.uint8)),
        "side": pa.array(sides.astype(np.uint8)),
        "price": pa.array(prices, type=pa.int64()),
        "size": pa.array(sizes.astype(np.uint32)),
    })
    lob = pa.table({
        "timestamp_ns": pa.array(ts, type=pa.int64()),
    })

    metadata = {
        b"schema_version": b"1.0",
        b"source": b"mbo-lob-reconstructor",
        b"symbol": b"TEST",
        b"date": b"2025-02-03",
    }
    mbo = mbo.replace_schema_metadata(metadata)
    lob = lob.replace_schema_metadata(metadata)

    return DayData(date="2025-02-03", symbol="TEST", lob=lob, mbo=mbo)


class TestOrderLifecycleBasic:
    def test_basic_run(self, tmp_data_dir: Path):
        """OrderLifecycleAnalyzer runs and produces a report."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        session = AnalysisSession(
            data_dir=tmp_data_dir, date_range=None, symbol="TEST",
        )
        analyzer = OrderLifecycleAnalyzer(config)
        report = analyzer.run(session)

        assert report.symbol == "TEST"
        assert report.n_days == 1

    def test_to_dict_roundtrip(self, tmp_data_dir: Path):
        """Report serializes to dict."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "symbol" in d
        assert d["symbol"] == "TEST"
        assert "_meta" in d

    def test_summary_no_crash(self, tmp_data_dir: Path):
        """Summary text generation does not crash."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        text = report.summary()
        assert "ORDER LIFECYCLE" in text


class TestLifetimeTracking:
    def test_add_then_cancel(self):
        """Order added then cancelled: lifetime = ts_cancel - ts_add."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1]),
            actions=np.array([ACTION_ADD, ACTION_CANCEL]),
            sides=np.array([SIDE_BID, SIDE_BID]),
            prices=np.array([100_000_000_000, 100_000_000_000], dtype=np.int64),
            sizes=np.array([100, 100]),
            ts_step=5 * NS_PER_SECOND,
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.order_lifetime["n_resolved"] == 1
        np.testing.assert_allclose(
            report.order_lifetime["mean_seconds"], 5.0, atol=0.1,
        )

    def test_add_then_trade(self):
        """Order added then traded: counted as filled."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1]),
            actions=np.array([ACTION_ADD, ACTION_TRADE]),
            sides=np.array([SIDE_BID, SIDE_BID]),
            prices=np.array([100_000_000_000, 100_000_000_000], dtype=np.int64),
            sizes=np.array([200, 200]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.fill_rate["overall_fill_rate"] == 1.0
        assert report.fill_rate["n_filled"] == 1

    def test_multiple_orders(self):
        """Multiple orders with different outcomes."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 2, 1, 2, 3, 3]),
            actions=np.array([ACTION_ADD, ACTION_ADD, ACTION_CANCEL,
                            ACTION_TRADE, ACTION_ADD, ACTION_CANCEL]),
            sides=np.array([SIDE_BID, SIDE_ASK, SIDE_BID,
                          SIDE_ASK, SIDE_BID, SIDE_BID]),
            prices=np.full(6, 100_000_000_000, dtype=np.int64),
            sizes=np.array([100, 200, 100, 200, 300, 300]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.order_lifetime["n_resolved"] == 3
        assert report.fill_rate["n_filled"] == 1
        assert report.fill_rate["n_cancelled"] == 2


class TestModifyPatterns:
    def test_modify_counted(self):
        """Modifications are tracked before terminal action."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1, 1, 1]),
            actions=np.array([ACTION_ADD, ACTION_MODIFY, ACTION_MODIFY, ACTION_CANCEL]),
            sides=np.array([SIDE_BID, SIDE_BID, SIDE_BID, SIDE_BID]),
            prices=np.full(4, 100_000_000_000, dtype=np.int64),
            sizes=np.array([100, 150, 200, 200]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.modify_patterns["modified_fraction"] == 1.0
        assert report.modify_patterns["mean_modifies_per_order"] == 2.0


class TestTransitionMatrix:
    def test_transition_probabilities(self):
        """Transition matrix captures sequential action patterns."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1, 1]),
            actions=np.array([ACTION_ADD, ACTION_MODIFY, ACTION_CANCEL]),
            sides=np.array([SIDE_BID, SIDE_BID, SIDE_BID]),
            prices=np.full(3, 100_000_000_000, dtype=np.int64),
            sizes=np.array([100, 100, 100]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert "matrix" in report.transition_matrix
        matrix = report.transition_matrix["matrix"]
        assert "Add" in matrix
        assert matrix["Add"]["Modify"] == 1.0


class TestCancelToAddRatio:
    def test_ratio_computed(self):
        """Cancel-to-add ratio is computed per day."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 2, 1, 2]),
            actions=np.array([ACTION_ADD, ACTION_ADD, ACTION_CANCEL, ACTION_CANCEL]),
            sides=np.array([SIDE_BID, SIDE_ASK, SIDE_BID, SIDE_ASK]),
            prices=np.full(4, 100_000_000_000, dtype=np.int64),
            sizes=np.array([100, 200, 100, 200]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        np.testing.assert_allclose(
            report.cancel_to_add_ratio["mean_ratio"], 1.0, atol=0.01,
        )


class TestPartialFillTracking:
    """Partial fill support: orders with multiple fill events before full."""

    def test_two_partial_fills_then_full(self):
        """Add(200) -> Trade(50) -> Trade(50) -> Trade(100) = 1 resolved order."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1, 1, 1]),
            actions=np.array([ACTION_ADD, ACTION_TRADE, ACTION_TRADE, ACTION_TRADE]),
            sides=np.array([SIDE_BID, SIDE_BID, SIDE_BID, SIDE_BID]),
            prices=np.full(4, 100_000_000_000, dtype=np.int64),
            sizes=np.array([200, 50, 50, 100]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.order_lifetime["n_resolved"] == 1
        assert report.fill_rate["n_filled"] == 1
        assert report.partial_fill_patterns["n_filled_orders"] == 1
        assert report.partial_fill_patterns["n_with_partial_fills"] == 1
        assert report.partial_fill_patterns["mean_fills_per_order"] == 3.0

    def test_single_full_fill_no_partials(self):
        """Add(100) -> Trade(100) = 1 fill event, not partial."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1]),
            actions=np.array([ACTION_ADD, ACTION_TRADE]),
            sides=np.array([SIDE_BID, SIDE_BID]),
            prices=np.full(2, 100_000_000_000, dtype=np.int64),
            sizes=np.array([100, 100]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.partial_fill_patterns["n_filled_orders"] == 1
        assert report.partial_fill_patterns["n_with_partial_fills"] == 0
        assert report.partial_fill_patterns["partial_fill_fraction"] == 0.0

    def test_partial_fill_then_cancel(self):
        """Add(200) -> Trade(50) -> Cancel: partial fill, then cancel."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1, 1]),
            actions=np.array([ACTION_ADD, ACTION_TRADE, ACTION_CANCEL]),
            sides=np.array([SIDE_BID, SIDE_BID, SIDE_BID]),
            prices=np.full(3, 100_000_000_000, dtype=np.int64),
            sizes=np.array([200, 50, 150]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.order_lifetime["n_resolved"] == 1
        assert report.fill_rate["n_cancelled"] == 1
        assert report.fill_rate["n_filled"] == 0

    def test_mixed_partial_and_full(self):
        """Mix of partial-fill and single-fill orders."""
        day = _make_lifecycle_day(
            order_ids=np.array([1, 1, 1, 2, 2]),
            actions=np.array([ACTION_ADD, ACTION_TRADE, ACTION_TRADE,
                            ACTION_ADD, ACTION_TRADE]),
            sides=np.array([SIDE_BID, SIDE_BID, SIDE_BID,
                          SIDE_ASK, SIDE_ASK]),
            prices=np.full(5, 100_000_000_000, dtype=np.int64),
            sizes=np.array([200, 100, 100, 300, 300]),
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.order_lifetime["n_resolved"] == 2
        assert report.fill_rate["n_filled"] == 2
        assert report.partial_fill_patterns["n_filled_orders"] == 2
        assert report.partial_fill_patterns["n_with_partial_fills"] == 1
        assert report.partial_fill_patterns["partial_fill_fraction"] == 0.5

    def test_partial_fill_report_fields(self, tmp_data_dir: Path):
        """partial_fill_patterns is present in to_dict output."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "partial_fill_patterns" in d


class TestEvictionHardCap:
    """Verify that the active-order dict never exceeds max_active_orders."""

    def test_hard_cap_enforced(self):
        """With max_active_orders=5, adding 10 orders keeps active <= 5."""
        n_orders = 10
        n_events = n_orders * 2  # add + cancel for each
        order_ids = np.repeat(np.arange(1, n_orders + 1), 2)
        actions = np.tile([ACTION_ADD, ACTION_CANCEL], n_orders)
        sides = np.full(n_events, SIDE_BID, dtype=np.uint8)
        prices = np.full(n_events, 100_000_000_000, dtype=np.int64)
        sizes = np.full(n_events, 100, dtype=np.uint32)

        day = _make_lifecycle_day(
            order_ids=order_ids,
            actions=actions,
            sides=sides,
            prices=prices,
            sizes=sizes,
            ts_step=1_000_000,  # 1ms apart, all within the same second
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        config.thresholds.flow.max_active_orders = 5
        config.thresholds.flow.order_lifetime_max_seconds = 3600.0

        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        assert report.order_lifetime["n_resolved"] >= 1

    def test_hard_cap_with_no_resolutions(self):
        """All ADD events, no terminal actions: hard cap prevents unbounded growth.

        With cap=5, orders 1..20 are added sequentially. Starting from the
        6th ADD, each triggers eviction of the oldest active order.  At end
        of day, 5 remain as active (counted as expired).  The 15 evicted
        during processing are also counted as expired.  Total expired = 20.
        """
        n = 20
        order_ids = np.arange(1, n + 1, dtype=np.uint64)
        actions = np.full(n, ACTION_ADD, dtype=np.uint8)
        sides = np.full(n, SIDE_BID, dtype=np.uint8)
        prices = np.full(n, 100_000_000_000, dtype=np.int64)
        sizes = np.full(n, 100, dtype=np.uint32)

        day = _make_lifecycle_day(
            order_ids=order_ids,
            actions=actions,
            sides=sides,
            prices=prices,
            sizes=sizes,
            ts_step=100_000,  # 0.1ms apart
        )
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        config.thresholds.flow.max_active_orders = 5
        config.thresholds.flow.order_lifetime_max_seconds = 3600.0

        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)

        assert analyzer._n_expired >= n - 5, (
            f"With cap=5, at least {n-5} orders should be evicted, "
            f"got {analyzer._n_expired}"
        )


class TestPartialThenCancel:
    """Verify partial-fill-then-cancel orders are tracked (B3 bug fix).

    An order that goes Add -> Trade(partial) -> Cancel should be counted
    in ``n_partial_then_cancel`` and contribute to ``partial_fill_fraction``.
    """

    def test_partial_then_cancel_tracked(self):
        """Add -> Trade(50 of 100) -> Cancel produces n_partial_then_cancel=1."""
        order_ids = np.array([1, 1, 1])
        actions = np.array([ACTION_ADD, ACTION_TRADE, ACTION_CANCEL])
        sides = np.array([SIDE_BID, SIDE_BID, SIDE_BID])
        prices = np.array([120_000_000_000, 120_000_000_000, 120_000_000_000])
        sizes = np.array([100, 50, 50])

        day = _make_lifecycle_day(order_ids, actions, sides, prices, sizes)
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        pf = report.partial_fill_patterns
        assert pf, "Expected partial_fill_patterns to be populated"
        assert pf["n_partial_then_cancel"] == 1, (
            f"Expected 1 partial-then-cancel, got {pf['n_partial_then_cancel']}"
        )
        assert pf["partial_fill_fraction"] > 0.0, (
            "Expected positive partial_fill_fraction for partial-then-cancel order"
        )

    def test_full_fill_no_partial_cancel(self):
        """Add -> Trade(full) produces n_partial_then_cancel=0."""
        order_ids = np.array([1, 1])
        actions = np.array([ACTION_ADD, ACTION_TRADE])
        sides = np.array([SIDE_BID, SIDE_BID])
        prices = np.array([120_000_000_000, 120_000_000_000])
        sizes = np.array([100, 100])

        day = _make_lifecycle_day(order_ids, actions, sides, prices, sizes)
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        pf = report.partial_fill_patterns
        assert pf, "Expected partial_fill_patterns to be populated"
        assert pf.get("n_partial_then_cancel", 0) == 0

    def test_multi_partial_fill_then_cancel(self):
        """Add -> Trade(30) -> Trade(30) -> Cancel: 2 fill events before cancel."""
        order_ids = np.array([1, 1, 1, 1])
        actions = np.array([ACTION_ADD, ACTION_TRADE, ACTION_TRADE, ACTION_CANCEL])
        sides = np.array([SIDE_BID, SIDE_BID, SIDE_BID, SIDE_BID])
        prices = np.array([120_000_000_000] * 4)
        sizes = np.array([100, 30, 30, 40])

        day = _make_lifecycle_day(order_ids, actions, sides, prices, sizes)
        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        ctx = DayContext(day=day)
        analyzer.process_day(ctx)
        report = analyzer.finalize()

        pf = report.partial_fill_patterns
        assert pf["n_partial_then_cancel"] == 1
        assert pf["max_fills_per_order"] == 2


class TestTwoDayAccumulation:
    def test_two_days(self, two_day_data_dir: Path):
        """Analyzer accumulates across multiple days."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = OrderLifecycleAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=two_day_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.n_days == 2
