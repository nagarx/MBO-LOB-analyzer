"""Tests for DepthAnalyzer."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.analysis.spread.depth import DepthAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestDepthAnalyzer:
    def test_basic_run(self, tmp_data_dir: Path):
        """DepthAnalyzer runs and produces a report."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        session = AnalysisSession(
            data_dir=tmp_data_dir,
            date_range=None,
            symbol="TEST",
        )
        analyzer = DepthAnalyzer(config)
        report = analyzer.run(session)

        assert report.symbol == "TEST"
        assert report.n_days == 1
        assert report.depth_profile
        assert "bid_avg" in report.depth_profile
        assert "ask_avg" in report.depth_profile
        assert len(report.depth_profile["bid_avg"]) == 10

    def test_imbalance_distribution(self, tmp_data_dir: Path):
        """Imbalance distribution is populated when depth_imbalance exists."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = DepthAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.imbalance_distribution:
            assert "mean" in report.imbalance_distribution

    def test_top_concentration(self, tmp_data_dir: Path):
        """Top-of-book concentration is populated."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = DepthAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.top_concentration:
            assert "bid_l1_fraction" in report.top_concentration
            assert "ask_l1_fraction" in report.top_concentration

    def test_regime_depth(self, tmp_data_dir: Path):
        """Regime-conditional depth is populated."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = DepthAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.regime_depth
        for regime, stats in report.regime_depth.items():
            assert "bid_l1" in stats
            assert "ask_l1" in stats

    def test_to_dict_roundtrip(self, tmp_data_dir: Path):
        """Report serializes to dict."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = DepthAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "symbol" in d
        assert d["symbol"] == "TEST"
        assert "depth_profile" in d
        assert "_meta" in d
