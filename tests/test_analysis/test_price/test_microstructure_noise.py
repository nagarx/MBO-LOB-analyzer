"""Tests for MicrostructureNoiseAnalyzer: noise signature, Roll model, optimal freq."""

from pathlib import Path

import json
import numpy as np
import pytest

from rawlobanalyzer.analysis.price.microstructure_noise import (
    MicrostructureNoiseAnalyzer,
    MicrostructureNoiseReport,
    _round_to_nice_frequency,
)
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestRoundToNiceFrequency:
    def test_exact_values(self):
        assert _round_to_nice_frequency(1.0) == 1.0
        assert _round_to_nice_frequency(5.0) == 5.0
        assert _round_to_nice_frequency(60.0) == 60.0

    def test_rounding(self):
        assert _round_to_nice_frequency(0.4) == 0.5
        assert _round_to_nice_frequency(4.0) == 5.0
        assert _round_to_nice_frequency(12.0) == 10.0
        assert _round_to_nice_frequency(50.0) == 60.0


class TestMicrostructureNoiseAnalyzer:
    def test_end_to_end(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")

        report = analyzer.run(session)

        assert isinstance(report, MicrostructureNoiseReport)
        assert report.n_days == 1
        assert report.symbol == "TEST"

    def test_noise_signature_structure(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        ns = report.noise_signature
        assert "scales" in ns
        assert "mean_rv" in ns
        assert "std_rv" in ns
        if ns["scales"]:
            assert len(ns["scales"]) == len(ns["mean_rv"])
            for rv in ns["mean_rv"]:
                assert rv >= 0, "RV must be non-negative"

    def test_noise_variance_non_negative(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        nv = report.noise_variance
        if nv:
            assert nv.get("mean", 0) >= 0, "Noise variance must be non-negative"

    def test_daily_records(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        if report.daily_records:
            rec = report.daily_records[0]
            assert "date" in rec
            assert "noise_var" in rec
            assert "n_ticks" in rec
            assert rec["noise_var"] >= 0
            assert rec["n_ticks"] > 0

    def test_roll_spread(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        bb = report.bid_ask_bounce
        if bb:
            assert "mean_roll_spread_usd" in bb
            assert bb["mean_roll_spread_usd"] >= 0

    def test_two_day(self, two_day_data_dir: Path):
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(two_day_data_dir, symbol="TEST")
        report = analyzer.run(session)

        assert report.n_days == 2

    def test_json_roundtrip(self, tmp_data_dir: Path, tmp_path: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        json_path = tmp_path / "noise.json"
        report.to_json(json_path)
        assert json_path.exists()

        data = json.loads(json_path.read_text())
        assert data["n_days"] == 1
        assert "_meta" in data
        assert "noise_signature" in data
        assert "bid_ask_bounce" in data

    def test_summary_readable(self, tmp_data_dir: Path):
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = MicrostructureNoiseAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        summary = report.summary()
        assert "MICROSTRUCTURE NOISE" in summary


class TestRegistration:
    def test_all_price_analyzers_registered(self):
        from rawlobanalyzer.analysis.registry import list_analyzers
        names = list_analyzers()
        assert "ReturnAnalyzer" in names
        assert "VolatilityAnalyzer" in names
        assert "JumpRiskAnalyzer" in names
        assert "MicrostructureNoiseAnalyzer" in names
