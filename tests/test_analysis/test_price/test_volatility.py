"""Tests for VolatilityAnalyzer: signature, intraday curve, vol-of-vol, etc."""

from pathlib import Path

import json
import numpy as np
import pytest

from rawlobanalyzer.analysis.price.volatility import (
    VolatilityAnalyzer,
    VolatilityReport,
    _ljung_box,
)
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestLjungBox:
    def test_significant_autocorrelation(self):
        """Strong ACF should produce low p-value."""
        acf_vals = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
        q, p = _ljung_box(acf_vals, n_obs=1000)
        assert q > 0
        assert p < 0.001

    def test_no_autocorrelation(self):
        """Near-zero ACF should produce high p-value."""
        acf_vals = np.array([0.01, -0.02, 0.005, -0.01])
        q, p = _ljung_box(acf_vals, n_obs=5000)
        assert p > 0.05

    def test_insufficient_data(self):
        q, p = _ljung_box(np.array([]), n_obs=100)
        assert np.isnan(q)
        assert np.isnan(p)


class TestVolatilityAnalyzer:
    def test_end_to_end_single_day(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")

        report = analyzer.run(session)

        assert isinstance(report, VolatilityReport)
        assert report.n_days == 1
        assert report.symbol == "TEST"

    def test_signature_plot_populated(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        sig = report.signature_plot
        assert "scales" in sig
        assert "mean_annualized_vol" in sig
        if sig["scales"]:
            assert len(sig["scales"]) == len(sig["mean_annualized_vol"])
            for vol in sig["mean_annualized_vol"]:
                assert vol >= 0, "Annualized vol must be non-negative"

    def test_regime_rv_non_negative(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        for regime, stats in report.regime_rv.items():
            assert stats["rv"] >= 0, f"RV must be non-negative for {regime}"
            assert stats["annualized_vol"] >= 0

    def test_two_day_overnight(self, two_day_data_dir: Path):
        """Overnight decomposition needs 2+ days."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert report.n_days == 2
        oi = report.overnight_intraday
        if oi.get("n_days", 0) >= 1:
            if np.isfinite(oi.get("overnight_fraction", np.nan)):
                assert 0 <= oi["overnight_fraction"] <= 1
                assert 0 <= oi["intraday_fraction"] <= 1

    def test_vol_of_vol_with_two_days(self, two_day_data_dir: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        vov = report.vol_of_vol
        if vov:
            assert vov["count"] >= 2
            assert vov["mean"] >= 0
            assert "cv" in vov

    def test_weekly_patterns(self, two_day_data_dir: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        wp = report.weekly_patterns
        assert isinstance(wp, dict)

    def test_arch_effects(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        arch = report.arch_effects
        if arch:
            assert "squared_return_acf" in arch
            assert "ljung_box_q" in arch

    def test_json_roundtrip(self, two_day_data_dir: Path, tmp_path: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        json_path = tmp_path / "volatility.json"
        report.to_json(json_path)
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["n_days"] == 2
        assert "signature_plot" in data
        assert "intraday_curve" in data
        assert "vol_of_vol" in data
        assert "overnight_intraday" in data
        assert "_meta" in data

    def test_summary_readable(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        summary = report.summary()
        assert "VOLATILITY ANALYSIS" in summary
        assert "TEST" in summary

    def test_intraday_curve_structure(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = VolatilityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        ic = report.intraday_curve
        assert "minutes" in ic
        assert "mean_normalized_var" in ic
        assert "n_days" in ic
        assert len(ic["minutes"]) == len(ic["mean_normalized_var"])
