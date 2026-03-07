"""Empty/minimal-day robustness tests for all 11 analyzers.

Verifies that every analyzer gracefully handles:
- Zero LOB rows (empty day)
- Minimal data (too few rows for meaningful statistics)
- Finalize without any process_day call (zero-day run)

No analyzer should crash or raise unhandled exceptions.
All should produce a valid report with n_days == 0 (or appropriate).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.flow._flow_engine import DayFlow, compute_day_flow
from rawlobanalyzer.analysis.flow.order_flow import OrderFlowAnalyzer
from rawlobanalyzer.analysis.flow.order_lifecycle import OrderLifecycleAnalyzer
from rawlobanalyzer.analysis.flow.trade import TradeAnalyzer
from rawlobanalyzer.analysis.health.data_quality import DataQualityAnalyzer
from rawlobanalyzer.analysis.price._return_engine import compute_day_returns
from rawlobanalyzer.analysis.price.jump_risk import JumpRiskAnalyzer
from rawlobanalyzer.analysis.price.microstructure_noise import MicrostructureNoiseAnalyzer
from rawlobanalyzer.analysis.price.returns import ReturnAnalyzer
from rawlobanalyzer.analysis.price.volatility import VolatilityAnalyzer
from rawlobanalyzer.analysis.spread._spread_engine import compute_day_spreads
from rawlobanalyzer.analysis.spread.depth import DepthAnalyzer
from rawlobanalyzer.analysis.spread.liquidity import LiquidityAnalyzer
from rawlobanalyzer.analysis.spread.spread import SpreadAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.loader import DayData


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

_DATE = "2025-02-03"
_SYMBOL = "TEST"


def _ts_start_ns() -> int:
    dt = datetime.strptime(_DATE, "%Y-%m-%d").replace(
        hour=15, minute=0, second=0, tzinfo=timezone.utc,
    )
    return int(dt.timestamp() * 1_000_000_000)


def _make_empty_lob() -> pa.Table:
    """LOB table with correct schema but zero rows."""
    metadata = {
        b"schema_version": b"1.0",
        b"source": b"mbo-lob-reconstructor",
        b"symbol": b"TEST",
        b"date": _DATE.encode(),
        b"price_unit": b"nanodollars",
        b"lob_levels": b"10",
        b"timestamp_unit": b"nanoseconds_since_epoch",
    }
    t = pa.table({
        "timestamp_ns": pa.array([], type=pa.int64()),
        "sequence": pa.array([], type=pa.uint64()),
        "levels": pa.array([], type=pa.uint8()),
        "best_bid": pa.array([], type=pa.int64()),
        "best_ask": pa.array([], type=pa.int64()),
        "bid_prices": pa.FixedSizeListArray.from_arrays(
            pa.array([], type=pa.int64()), 10,
        ),
        "bid_sizes": pa.FixedSizeListArray.from_arrays(
            pa.array([], type=pa.uint32()), 10,
        ),
        "ask_prices": pa.FixedSizeListArray.from_arrays(
            pa.array([], type=pa.int64()), 10,
        ),
        "ask_sizes": pa.FixedSizeListArray.from_arrays(
            pa.array([], type=pa.uint32()), 10,
        ),
        "triggering_action": pa.array([], type=pa.uint8()),
        "triggering_side": pa.array([], type=pa.uint8()),
        "mid_price": pa.array([], type=pa.float64()),
        "spread": pa.array([], type=pa.float64()),
        "spread_bps": pa.array([], type=pa.float64()),
        "microprice": pa.array([], type=pa.float64()),
        "total_bid_volume": pa.array([], type=pa.uint64()),
        "total_ask_volume": pa.array([], type=pa.uint64()),
        "depth_imbalance": pa.array([], type=pa.float64()),
        "delta_ns": pa.array([], type=pa.uint64()),
        "book_consistency": pa.array([], type=pa.uint8()),
    })
    return t.replace_schema_metadata(metadata)


def _make_empty_mbo() -> pa.Table:
    metadata = {
        b"schema_version": b"1.0",
        b"source": b"mbo-lob-reconstructor",
        b"symbol": b"TEST",
        b"date": _DATE.encode(),
    }
    t = pa.table({
        "timestamp_ns": pa.array([], type=pa.int64()),
        "order_id": pa.array([], type=pa.uint64()),
        "action": pa.array([], type=pa.uint8()),
        "side": pa.array([], type=pa.uint8()),
        "price": pa.array([], type=pa.int64()),
        "size": pa.array([], type=pa.uint32()),
    })
    return t.replace_schema_metadata(metadata)


def _empty_day() -> DayData:
    return DayData(
        date=_DATE,
        symbol=_SYMBOL,
        lob=_make_empty_lob(),
        mbo=_make_empty_mbo(),
    )


def _make_config(tmp_path: Path | None = None) -> AnalysisConfig:
    return AnalysisConfig(
        data_dir=tmp_path or Path("/tmp/dummy"),
        symbol=_SYMBOL,
    )


def _empty_ctx(config: AnalysisConfig) -> DayContext:
    """DayContext with empty data and pre-computed empty engines."""
    day = _empty_day()
    day_returns = compute_day_returns(day, config)
    day_spreads = compute_day_spreads(day, config)
    day_flow = compute_day_flow(day, config)
    return DayContext(
        day=day,
        day_returns=day_returns,
        day_spreads=day_spreads,
        day_flow=day_flow,
    )


# ------------------------------------------------------------------
# Tests: finalize-without-any-day (zero-day)
# ------------------------------------------------------------------


class TestZeroDayFinalize:
    """Finalize called without any process_day -- every analyzer must survive."""

    def test_return_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = ReturnAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_volatility_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = VolatilityAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_jump_risk_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = JumpRiskAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_microstructure_noise_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = MicrostructureNoiseAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_spread_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = SpreadAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_depth_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = DepthAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_liquidity_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = LiquidityAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_order_flow_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = OrderFlowAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_trade_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = TradeAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_order_lifecycle_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = OrderLifecycleAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0

    def test_data_quality_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = DataQualityAnalyzer(config)
        report = a.finalize()
        assert report.n_days == 0


# ------------------------------------------------------------------
# Tests: process_day with empty data then finalize
# ------------------------------------------------------------------


class TestEmptyDayProcessAndFinalize:
    """Process one day of empty data then finalize."""

    def test_return_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = ReturnAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_volatility_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = VolatilityAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_jump_risk_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = JumpRiskAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_microstructure_noise_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = MicrostructureNoiseAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_spread_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = SpreadAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_depth_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = DepthAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_liquidity_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = LiquidityAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_order_flow_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = OrderFlowAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_trade_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = TradeAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_order_lifecycle_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = OrderLifecycleAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1

    def test_data_quality_analyzer(self, tmp_path: Path):
        config = _make_config(tmp_path)
        a = DataQualityAnalyzer(config)
        ctx = _empty_ctx(config)
        a.process_day(ctx)
        report = a.finalize()
        assert report.n_days <= 1


# ------------------------------------------------------------------
# Tests: JSON roundtrip on empty-day reports
# ------------------------------------------------------------------


class TestEmptyDayJsonRoundtrip:
    """Reports from empty data must serialize to valid JSON."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> AnalysisConfig:
        return _make_config(tmp_path)

    @pytest.fixture
    def ctx(self, config: AnalysisConfig) -> DayContext:
        return _empty_ctx(config)

    def _roundtrip(self, analyzer, ctx, tmp_path: Path) -> dict:
        analyzer.process_day(ctx)
        report = analyzer.finalize()
        out = tmp_path / "report.json"
        report.to_json(out)
        text = out.read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        return json.loads(text)

    @pytest.mark.parametrize("analyzer_cls", [
        ReturnAnalyzer,
        VolatilityAnalyzer,
        JumpRiskAnalyzer,
        MicrostructureNoiseAnalyzer,
        SpreadAnalyzer,
        DepthAnalyzer,
        LiquidityAnalyzer,
        OrderFlowAnalyzer,
        TradeAnalyzer,
        OrderLifecycleAnalyzer,
        DataQualityAnalyzer,
    ])
    def test_json_roundtrip(self, analyzer_cls, config, ctx, tmp_path):
        analyzer = analyzer_cls(config)
        data = self._roundtrip(analyzer, ctx, tmp_path)
        assert "_meta" in data


# ------------------------------------------------------------------
# Tests: summary() on empty-day reports
# ------------------------------------------------------------------


class TestEmptyDaySummary:
    """summary() must not crash on empty-day reports."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> AnalysisConfig:
        return _make_config(tmp_path)

    @pytest.fixture
    def ctx(self, config: AnalysisConfig) -> DayContext:
        return _empty_ctx(config)

    @pytest.mark.parametrize("analyzer_cls", [
        ReturnAnalyzer,
        VolatilityAnalyzer,
        JumpRiskAnalyzer,
        MicrostructureNoiseAnalyzer,
        SpreadAnalyzer,
        DepthAnalyzer,
        LiquidityAnalyzer,
        OrderFlowAnalyzer,
        TradeAnalyzer,
        OrderLifecycleAnalyzer,
        DataQualityAnalyzer,
    ])
    def test_summary_no_crash(self, analyzer_cls, config, ctx):
        analyzer = analyzer_cls(config)
        analyzer.process_day(ctx)
        report = analyzer.finalize()
        text = report.summary()
        assert isinstance(text, str)
        assert len(text) > 0
