"""Tests for TradeAnalyzer."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.analysis.flow._flow_engine import DayFlow
from rawlobanalyzer.analysis.flow.trade import TradeAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import NANODOLLARS_PER_DOLLAR, PRICE_LEVEL_TOLERANCE_USD
from rawlobanalyzer.io.schema import SIDE_BID
from rawlobanalyzer.io.session import AnalysisSession


class TestTradeAnalyzer:
    def test_basic_run(self, tmp_data_dir: Path):
        """TradeAnalyzer runs and produces a report."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        session = AnalysisSession(
            data_dir=tmp_data_dir, date_range=None, symbol="TEST",
        )
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(session)

        assert report.symbol == "TEST"
        assert report.n_days == 1

    def test_trade_size_distribution(self, tmp_data_dir: Path):
        """Trade size distribution is populated with standard stats."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.trade_size_distribution:
            assert "mean" in report.trade_size_distribution
            assert "count" in report.trade_size_distribution
            assert report.trade_size_distribution["count"] > 0

    def test_trade_through_rate(self, tmp_data_dir: Path):
        """Trade-through rate is between 0 and 1."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.trade_through:
            rate = report.trade_through.get("overall_rate", 0)
            assert 0.0 <= rate <= 1.0

    def test_trade_clustering(self, tmp_data_dir: Path):
        """Trade clustering metrics are populated."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.trade_clustering:
            assert "mean_inter_trade_seconds" in report.trade_clustering
            assert "cluster_fraction" in report.trade_clustering

    def test_vwap_trajectory(self, tmp_data_dir: Path):
        """VWAP trajectory has daily records."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.vwap_trajectory:
            assert "daily_records" in report.vwap_trajectory

    def test_trade_price_level(self, tmp_data_dir: Path):
        """Trade price level fractions sum to ~1."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.trade_price_level:
            total = (
                report.trade_price_level.get("at_bid_fraction", 0)
                + report.trade_price_level.get("at_ask_fraction", 0)
                + report.trade_price_level.get("inside_spread_fraction", 0)
                + report.trade_price_level.get("outside_fraction", 0)
            )
            np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_to_dict_roundtrip(self, tmp_data_dir: Path):
        """Report serializes to dict."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
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
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        text = report.summary()
        assert "TRADE ANALYSIS" in text

    def test_two_day_accumulation(self, two_day_data_dir: Path):
        """Analyzer accumulates across multiple days."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=two_day_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.n_days == 2


class TestDirectionalSizeDistribution:
    def test_directional_sizes_field_exists(self, tmp_data_dir: Path):
        """Report includes directional_size_distribution field."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "directional_size_distribution" in d

    def test_directional_sizes_has_buyer_seller(self, tmp_data_dir: Path):
        """Directional sizes have buyer and seller entries."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        dsd = report.directional_size_distribution
        if dsd:
            for side in ("buyer", "seller"):
                if side in dsd:
                    assert "mean" in dsd[side]
                    assert "count" in dsd[side]

    def test_directional_sizes_summary_no_crash(self, tmp_data_dir: Path):
        """Summary with directional sizes does not crash."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        text = report.summary()
        assert isinstance(text, str)


class TestPriceLevelClassification:
    """Targeted tests for trade price-level classification (B1 bug fix).

    Verifies that trades at bid/ask are correctly classified despite float
    rounding from the nanodollar->USD conversion, and that trades inside a
    multi-tick spread are classified as 'inside_spread'.
    """

    @staticmethod
    def _make_flow(
        trade_prices_usd: np.ndarray,
        mid_before: np.ndarray,
        spread_before: np.ndarray,
    ) -> DayFlow:
        """Build a minimal DayFlow with the given trade/BBO arrays."""
        n = len(trade_prices_usd)
        ts = np.arange(n, dtype=np.int64) * 1_000_000
        return DayFlow(
            date="2025-02-03",
            trade_timestamps_ns=ts,
            trade_prices_usd=trade_prices_usd.astype(np.float64),
            trade_sizes=np.full(n, 100, dtype=np.uint32),
            trade_sides=np.full(n, SIDE_BID, dtype=np.int8),
            trade_mid_before=mid_before.astype(np.float64),
            trade_spread_before=spread_before.astype(np.float64),
            ofi_timestamps_ns=np.array([], dtype=np.int64),
            ofi_values=np.array([], dtype=np.float64),
            rth_mask_trades=np.ones(n, dtype=np.bool_),
            directional_mask=np.ones(n, dtype=np.bool_),
            n_trades=n,
        )

    def test_trade_at_bid_exact(self, tmp_data_dir: Path):
        """Trade exactly at bid_price classifies as 'at_bid'."""
        mid = np.array([120.005])
        spread = np.array([0.01])
        bid = mid - spread / 2.0  # 120.00
        flow = self._make_flow(bid, mid, spread)

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        analyzer._compute_price_level(flow)

        assert analyzer._price_level_counts["at_bid"] == 1
        assert analyzer._price_level_counts["inside_spread"] == 0

    def test_trade_at_ask_exact(self, tmp_data_dir: Path):
        """Trade exactly at ask_price classifies as 'at_ask'."""
        mid = np.array([120.005])
        spread = np.array([0.01])
        ask = mid + spread / 2.0  # 120.01
        flow = self._make_flow(ask, mid, spread)

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        analyzer._compute_price_level(flow)

        assert analyzer._price_level_counts["at_ask"] == 1
        assert analyzer._price_level_counts["inside_spread"] == 0

    def test_trade_at_bid_via_nanodollar_roundtrip(self, tmp_data_dir: Path):
        """Trade price from nanodollar->USD roundtrip still classifies as 'at_bid'.

        This is the exact scenario that triggered the original bug: int64
        nanodollar prices converted to float64 USD introduce ~1e-10 rounding.
        """
        bid_nd = 120_000_000_000
        ask_nd = 120_010_000_000
        mid_usd = (bid_nd + ask_nd) / (2.0 * NANODOLLARS_PER_DOLLAR)
        spread_usd = (ask_nd - bid_nd) / NANODOLLARS_PER_DOLLAR
        trade_price_usd = np.float64(bid_nd) / NANODOLLARS_PER_DOLLAR

        flow = self._make_flow(
            np.array([trade_price_usd]),
            np.array([mid_usd]),
            np.array([spread_usd]),
        )

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        analyzer._compute_price_level(flow)

        assert analyzer._price_level_counts["at_bid"] == 1, (
            f"Expected at_bid=1 but got counts={analyzer._price_level_counts}"
        )
        assert analyzer._price_level_counts["inside_spread"] == 0

    def test_trade_inside_2tick_spread(self, tmp_data_dir: Path):
        """Trade at mid of a 2-tick spread classifies as 'inside_spread'."""
        mid = np.array([120.01])
        spread = np.array([0.02])  # 2-tick spread
        flow = self._make_flow(mid, mid, spread)

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        analyzer._compute_price_level(flow)

        assert analyzer._price_level_counts["inside_spread"] == 1
        assert analyzer._price_level_counts["at_bid"] == 0
        assert analyzer._price_level_counts["at_ask"] == 0

    def test_trade_outside_spread(self, tmp_data_dir: Path):
        """Trade beyond the ask classifies as 'outside'."""
        mid = np.array([120.005])
        spread = np.array([0.01])
        outside_price = np.array([120.02])
        flow = self._make_flow(outside_price, mid, spread)

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = TradeAnalyzer(config)
        analyzer._compute_price_level(flow)

        assert analyzer._price_level_counts["outside"] == 1
