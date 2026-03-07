"""Tests for OrderFlowAnalyzer."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.analysis.flow.order_flow import OrderFlowAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestOrderFlowAnalyzer:
    def test_basic_run(self, tmp_data_dir: Path):
        """OrderFlowAnalyzer runs and produces a report."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        session = AnalysisSession(
            data_dir=tmp_data_dir, date_range=None, symbol="TEST",
        )
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(session)

        assert report.symbol == "TEST"
        assert report.n_days == 1

    def test_ofi_distribution_populated(self, tmp_data_dir: Path):
        """OFI distribution is computed at configured timescales."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.ofi_distribution:
            for label, stats in report.ofi_distribution.items():
                assert "mean" in stats
                assert "std" in stats
                assert "count" in stats

    def test_cumulative_delta(self, tmp_data_dir: Path):
        """Cumulative delta has daily records."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.cumulative_delta:
            assert "mean_eod_delta" in report.cumulative_delta
            assert "daily_deltas" in report.cumulative_delta

    def test_aggressor_ratio(self, tmp_data_dir: Path):
        """Aggressor ratio is between 0 and 1."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.aggressor_ratio and "mean_buyer_fraction" in report.aggressor_ratio:
            frac = report.aggressor_ratio["mean_buyer_fraction"]
            assert 0.0 <= frac <= 1.0

    def test_to_dict_roundtrip(self, tmp_data_dir: Path):
        """Report serializes to dict."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "symbol" in d
        assert d["symbol"] == "TEST"
        assert "_meta" in d
        assert "ofi_distribution" in d
        assert "cumulative_delta" in d

    def test_summary_no_crash(self, tmp_data_dir: Path):
        """Summary text generation does not crash."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        text = report.summary()
        assert "ORDER FLOW" in text

    def test_two_day_accumulation(self, two_day_data_dir: Path):
        """Analyzer accumulates across multiple days."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=two_day_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.n_days == 2
        if report.cumulative_delta and "daily_deltas" in report.cumulative_delta:
            assert len(report.cumulative_delta["daily_deltas"]) == 2


class TestOFISpreadCorrelation:
    def test_ofi_spread_corr_field_exists(self, tmp_data_dir: Path):
        """Report includes ofi_spread_correlation field."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "ofi_spread_correlation" in d

    def test_ofi_spread_corr_structure(self, tmp_data_dir: Path):
        """OFI-spread correlation entries have expected keys."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        for label, entry in report.ofi_spread_correlation.items():
            if isinstance(entry, dict):
                assert "peak_lag" in entry
                assert "peak_corr" in entry
                assert "n_days" in entry


class TestOFIComponents:
    def test_ofi_components_field_exists(self, tmp_data_dir: Path):
        """Report includes ofi_components field."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "ofi_components" in d

    def test_ofi_components_fractions_sum_to_one(self, tmp_data_dir: Path):
        """OFI component fractions (add + cancel + trade) sum to 1."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        overall = report.ofi_components.get("overall", {})
        if overall:
            total = (
                overall.get("add_fraction", 0)
                + overall.get("cancel_fraction", 0)
                + overall.get("trade_fraction", 0)
            )
            np.testing.assert_allclose(total, 1.0, atol=0.01)

    def test_ofi_components_summary_no_crash(self, tmp_data_dir: Path):
        """Summary with OFI components does not crash."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = OrderFlowAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        text = report.summary()
        assert isinstance(text, str)
