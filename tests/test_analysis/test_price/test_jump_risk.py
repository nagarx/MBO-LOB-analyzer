"""Tests for JumpRiskAnalyzer: bipower variation, BNS test, jump characterization."""

from pathlib import Path

import json
import numpy as np
import pytest

from rawlobanalyzer.analysis.price.jump_risk import (
    JumpRiskAnalyzer,
    JumpRiskReport,
    _bipower_variation,
    _bns_test_statistic,
    _tripower_quarticity,
)
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestBipowerVariation:
    def test_formula(self):
        """Hand-calculated BV for a small return series.

        r = [0.01, -0.02, 0.03]
        BV = (pi/2) * (|r2|*|r1| + |r3|*|r2|)
           = (pi/2) * (0.02*0.01 + 0.03*0.02)
           = (pi/2) * (0.0002 + 0.0006)
           = (pi/2) * 0.0008
        """
        r = np.array([0.01, -0.02, 0.03])
        bv = _bipower_variation(r)
        expected = (np.pi / 2) * (0.02 * 0.01 + 0.03 * 0.02)
        np.testing.assert_allclose(bv, expected, rtol=1e-10)

    def test_single_return(self):
        assert np.isnan(_bipower_variation(np.array([0.01])))

    def test_bv_leq_rv(self):
        """BV should be <= RV in expectation for continuous processes."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 5000)
        rv = float(np.sum(rets**2))
        bv = _bipower_variation(rets)
        assert bv <= rv * 1.2, "BV should be close to or below RV for no-jump data"

    def test_non_negative(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 100)
        bv = _bipower_variation(rets)
        assert bv >= 0


class TestTripowerQuarticity:
    def test_basic_computation(self):
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 1000)
        tpq = _tripower_quarticity(rets)
        assert np.isfinite(tpq)
        assert tpq > 0

    def test_too_few_returns(self):
        assert np.isnan(_tripower_quarticity(np.array([0.01, -0.02])))


class TestBNSTestStatistic:
    def test_no_jumps(self):
        """For continuous process, Z should be small (not significant)."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.01, 5000)
        rv = float(np.sum(rets**2))
        bv = _bipower_variation(rets)
        tpq = _tripower_quarticity(rets)
        z = _bns_test_statistic(rv, bv, tpq, len(rets))
        assert np.isfinite(z)
        assert abs(z) < 5, f"Z should be small for no-jump data, got {z}"

    def test_with_jumps(self):
        """Injecting many large jumps should make RV >> BV, so Z > 0."""
        rng = np.random.default_rng(42)
        rets = rng.normal(0, 0.0005, 5000)
        for i in range(100, 5000, 200):
            rets[i] = rng.choice([-1, 1]) * 0.05
        rv = float(np.sum(rets**2))
        bv = _bipower_variation(rets)
        tpq = _tripower_quarticity(rets)
        z = _bns_test_statistic(rv, bv, tpq, len(rets))
        assert rv > bv, "RV should exceed BV when jumps are present"
        assert z > 0, f"Z should be positive with jumps, got {z}"


class TestJumpRiskAnalyzer:
    def test_end_to_end(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = JumpRiskAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")

        report = analyzer.run(session)

        assert isinstance(report, JumpRiskReport)
        assert report.n_days == 0
        assert report.symbol == "TEST"

    def test_daily_records(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = JumpRiskAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        if report.daily_records:
            rec = report.daily_records[0]
            assert "date" in rec
            assert "rv" in rec
            assert "bv" in rec
            assert "jv" in rec
            assert rec["rv"] >= 0
            assert rec["jv"] >= 0
            assert rec["jv"] <= rec["rv"] + 1e-10

    def test_jump_fraction_bounded(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = JumpRiskAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        for rec in report.daily_records:
            assert 0 <= rec["jump_fraction"] <= 1.0 + 1e-10

    def test_two_day(self, two_day_data_dir: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = JumpRiskAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert report.n_days == 2

    def test_json_roundtrip(self, tmp_data_dir: Path, tmp_path: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = JumpRiskAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        json_path = tmp_path / "jumps.json"
        report.to_json(json_path)
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["n_days"] == 0
        assert "_meta" in data
        assert "daily_records" in data

    def test_summary_readable(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = JumpRiskAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        summary = report.summary()
        assert "JUMP RISK" in summary
