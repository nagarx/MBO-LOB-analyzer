"""Tests for ReturnAnalyzer: distribution, tails, ACF, regime, weekday."""

from pathlib import Path

import json
import numpy as np
import pytest

from rawlobanalyzer.analysis.price.returns import (
    ReturnAnalyzer,
    ReturnReport,
    _acf,
    _hill_estimator,
    _qq_data,
)
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestHillEstimator:
    def test_pareto_tail(self):
        """Hill estimator on Pareto(alpha=2) should recover alpha ~ 2."""
        rng = np.random.default_rng(99)
        pareto = rng.pareto(2.0, size=10_000)
        alpha = _hill_estimator(pareto, tail_fraction=0.05)
        assert 1.5 < alpha < 3.0, f"Expected alpha near 2, got {alpha}"

    def test_too_few_points(self):
        assert np.isnan(_hill_estimator(np.array([1.0, 2.0]), 0.05))

    def test_all_nan(self):
        assert np.isnan(_hill_estimator(np.array([np.nan, np.nan, np.nan]), 0.05))

    def test_constant_data(self):
        result = _hill_estimator(np.ones(100), 0.05)
        assert np.isnan(result)


class TestACF:
    def test_white_noise(self):
        """ACF of white noise should be near zero at all lags."""
        rng = np.random.default_rng(42)
        wn = rng.normal(0, 1, 5000)
        acf = _acf(wn, 10)
        assert len(acf) == 10
        assert np.all(np.abs(acf) < 0.1), "White noise ACF should be near zero"

    def test_strong_ar1(self):
        """ACF of AR(1) with rho=0.9 should show strong lag-1 autocorrelation."""
        rng = np.random.default_rng(42)
        n = 5000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.normal()
        acf = _acf(x, 5)
        assert acf[0] > 0.8, f"AR(1) lag-1 ACF should be near 0.9, got {acf[0]}"

    def test_short_series(self):
        acf = _acf(np.array([1.0, 2.0]), 5)
        assert len(acf) == 1

    def test_constant_series(self):
        acf = _acf(np.ones(100), 5)
        assert np.all(acf == 0.0)

    def test_fft_matches_direct_computation(self):
        """FFT-based ACF must match the direct dot-product formula exactly."""
        rng = np.random.default_rng(77)
        series = rng.normal(0, 1, 2000)
        max_lag = 50

        fft_acf = _acf(series, max_lag)

        centered = series - np.mean(series)
        var = float(np.sum(centered**2))
        direct_acf = np.array([
            float(np.sum(centered[:-lag] * centered[lag:])) / var
            for lag in range(1, max_lag + 1)
        ])

        np.testing.assert_allclose(
            fft_acf, direct_acf, rtol=1e-10,
            err_msg="FFT ACF must match direct dot-product computation",
        )

    def test_fft_large_lag(self):
        """FFT ACF should handle large max_lag efficiently."""
        rng = np.random.default_rng(42)
        n = 10_000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.95 * x[i - 1] + rng.normal()

        acf = _acf(x, 500)
        assert len(acf) == 500
        assert acf[0] > 0.9, "Lag-1 should be near 0.95"
        assert acf[-1] < acf[0], "ACF should decay"


class TestQQData:
    def test_normal_data(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, 5000)
        qq = _qq_data(normal, n_points=50)
        assert len(qq["theoretical"]) == 50
        assert len(qq["empirical"]) == 50
        correlation = np.corrcoef(qq["theoretical"], qq["empirical"])[0, 1]
        assert correlation > 0.99, "Normal data QQ should be nearly linear"

    def test_too_few_points(self):
        qq = _qq_data(np.array([1.0, 2.0]), n_points=50)
        assert len(qq["theoretical"]) == 0


class TestReturnAnalyzer:
    def test_end_to_end(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")

        report = analyzer.run(session)

        assert isinstance(report, ReturnReport)
        assert report.n_days == 1
        assert report.symbol == "TEST"

    def test_tick_stats_populated(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert "distribution" in report.tick_stats
        dist = report.tick_stats["distribution"]
        assert dist["count"] > 0
        assert np.isfinite(dist["mean"])
        assert np.isfinite(dist["std"])
        assert dist["std"] > 0

    def test_timescale_stats(self, two_day_data_dir: Path):
        """Use two-day fixture which has enough events for multi-second bins."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert len(report.timescale_stats) > 0
        populated = {l: ts for l, ts in report.timescale_stats.items() if ts}
        assert len(populated) > 0, "At least one timescale should have data"
        for label, ts in populated.items():
            assert "distribution" in ts
            assert "risk" in ts
            assert "tails" in ts
            assert "autocorrelation" in ts

    def test_risk_metrics(self, two_day_data_dir: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        for label, ts in report.timescale_stats.items():
            if not ts:
                continue
            risk = ts["risk"]
            if risk.get("var_5pct") is not None and np.isfinite(risk["var_5pct"]):
                assert risk["var_5pct"] <= 0, "VaR should be negative (left tail)"
                assert risk["cvar_5pct"] <= risk["var_5pct"], "CVaR <= VaR"

    def test_regime_stats(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert len(report.regime_stats) > 0

    def test_json_roundtrip(self, tmp_data_dir: Path, tmp_path: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        json_path = tmp_path / "returns.json"
        report.to_json(json_path)
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["n_days"] == 1
        assert "_meta" in data
        assert data["_meta"]["schema_version"] == "1.0"

    def test_summary_readable(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        summary = report.summary()
        assert "RETURN ANALYSIS" in summary
        assert "TEST" in summary

    def test_two_day(self, two_day_data_dir: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = ReturnAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert report.n_days == 2
        assert len(report.weekday_stats) >= 1
