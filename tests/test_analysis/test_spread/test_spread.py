"""Tests for SpreadAnalyzer."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.analysis.spread.spread import SpreadAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestSpreadAnalyzer:
    def test_basic_run(self, tmp_data_dir: Path):
        """SpreadAnalyzer runs and produces a report."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        session = AnalysisSession(
            data_dir=tmp_data_dir,
            date_range=None,
            symbol="TEST",
        )
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(session)

        assert report.symbol == "TEST"
        assert report.n_days == 1
        assert report.tick_distribution
        assert "mean_usd" in report.tick_distribution
        assert report.tick_distribution["count"] > 0
        assert report.tick_distribution.get("fraction_1tick") is not None

    def test_regime_spreads(self, tmp_data_dir: Path):
        """Regime-conditional spreads are populated."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.regime_spreads
        for regime, stats in report.regime_spreads.items():
            assert "mean_usd" in stats
            assert "n" in stats
            assert stats["n"] > 0

    def test_trade_conditional(self, tmp_data_dir: Path):
        """Trade-conditional spread section is populated when trades exist."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.trade_conditional:
            assert "mean_at_trades_usd" in report.trade_conditional
            assert "mean_all_usd" in report.trade_conditional
            assert "ratio" in report.trade_conditional

    def test_width_classification(self, tmp_data_dir: Path):
        """Width classification fractions sum to 1."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.width_classification
        fractions = report.width_classification.get("fractions", {})
        if fractions:
            total = sum(fractions.values())
            assert abs(total - 1.0) < 0.01

    def test_to_dict_roundtrip(self, tmp_data_dir: Path):
        """Report serializes to dict and back."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "symbol" in d
        assert d["symbol"] == "TEST"
        assert "tick_distribution" in d
        assert "_meta" in d

    def test_summary_string(self, tmp_data_dir: Path):
        """Summary produces non-empty string."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        summary = report.summary()
        assert isinstance(summary, str)
        assert "SPREAD ANALYSIS REPORT" in summary
        assert "Symbol" in summary


class TestFraction1TickConsistency:
    """Verify fraction_1tick and width_classification 1-tick agree (B4 fix)."""

    def test_fraction_1tick_equals_width_class(self, tmp_data_dir: Path):
        """tick_distribution.fraction_1tick == width_classification.fractions['1-tick']."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = SpreadAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        tick_dist = report.tick_distribution
        wc = report.width_classification
        if tick_dist and wc and "fractions" in wc:
            f1_tick = tick_dist.get("fraction_1tick")
            f1_width = wc["fractions"].get("1-tick")
            assert f1_tick is not None and f1_width is not None
            assert f1_tick == f1_width, (
                f"fraction_1tick ({f1_tick}) must equal "
                f"width_classification 1-tick ({f1_width})"
            )
