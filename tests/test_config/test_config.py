"""Tests for config module."""

from pathlib import Path

import pytest

from rawlobanalyzer.config.analysis_config import AnalysisConfig, StatisticalThresholds
from rawlobanalyzer.config.timescale_config import (
    DEFAULT_TIMESCALES,
    TimescaleConfig,
    TradingHours,
)
from rawlobanalyzer.config.profile_loader import (
    ProfileLoadError,
    apply_profile_config,
    load_profile,
)
from rawlobanalyzer.core.constants import NS_PER_MINUTE, NS_PER_SECOND


class TestTimescaleConfig:
    def test_seconds(self):
        ts = TimescaleConfig.seconds(5)
        assert ts.resolution_ns == 5 * NS_PER_SECOND
        assert ts.label == "5s"
        assert ts.trading_hours_only is True

    def test_minutes(self):
        ts = TimescaleConfig.minutes(15)
        assert ts.resolution_ns == 15 * NS_PER_MINUTE
        assert ts.label == "15m"

    def test_from_label(self):
        ts = TimescaleConfig.from_label("30s")
        assert ts.resolution_ns == 30 * NS_PER_SECOND
        assert ts.label == "30s"

    def test_from_label_minutes(self):
        ts = TimescaleConfig.from_label("5m")
        assert ts.resolution_ns == 5 * NS_PER_MINUTE

    def test_from_label_invalid(self):
        with pytest.raises(ValueError, match="Cannot parse"):
            TimescaleConfig.from_label("???")

    def test_defaults_exist(self):
        assert len(DEFAULT_TIMESCALES) == 7


class TestTradingHours:
    def test_us_equity(self):
        th = TradingHours.us_equity()
        assert th.rth_open_utc_h == 14.5
        assert th.rth_close_utc_h == 21.0
        assert th.utc_offset_hours == -5

    def test_from_label(self):
        th = TradingHours.from_label("us_equity_rth")
        assert th.utc_offset_hours == -5

    def test_from_label_invalid(self):
        with pytest.raises(ValueError, match="Unknown trading hours"):
            TradingHours.from_label("tokyo_exchange")


class TestAnalysisConfig:
    def test_defaults(self, tmp_path: Path):
        config = AnalysisConfig(data_dir=tmp_path)
        assert config.symbol == "UNKNOWN"
        assert len(config.timescales) == 7
        assert config.thresholds.significance_alpha == 0.05

    def test_to_dict(self, tmp_path: Path):
        config = AnalysisConfig(data_dir=tmp_path, symbol="NVDA")
        d = config.to_dict()
        assert d["symbol"] == "NVDA"
        assert isinstance(d["timescales"], list)


class TestStatisticalThresholds:
    def test_strict(self):
        t = StatisticalThresholds.strict()
        assert t.significance_alpha == 0.01

    def test_lenient(self):
        t = StatisticalThresholds.lenient()
        assert t.significance_alpha == 0.10


class TestTimescaleConfigEdges:
    """Edge-case tests for TimescaleConfig.from_label (P1-D fix)."""

    def test_zero_seconds_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TimescaleConfig.from_label("0s")

    def test_zero_minutes_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TimescaleConfig.from_label("0m")

    def test_negative_seconds_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TimescaleConfig.from_label("-1s")

    def test_daily_2d_parses_correctly(self):
        tc = TimescaleConfig.from_label("2d")
        assert tc.resolution_ns == 2 * 24 * 3600 * NS_PER_SECOND

    def test_daily_1d_default(self):
        tc = TimescaleConfig.from_label("d")
        assert tc.resolution_ns == 24 * 3600 * NS_PER_SECOND

    def test_valid_labels_parse(self):
        for label in ("1s", "5m", "1h", "1d"):
            tc = TimescaleConfig.from_label(label)
            assert tc.resolution_ns > 0


class TestProfileLoader:
    def test_load_quick_profile(self):
        path = Path(__file__).parent.parent.parent / "configs" / "profiles" / "quick.yaml"
        if not path.exists():
            pytest.skip("Profile file not found")
        profile = load_profile(path)
        assert profile.name == "quick"
        assert len(profile.phases) == 1
        assert "DataQualityAnalyzer" in profile.all_analyzer_names

    def test_missing_file(self, tmp_path: Path):
        with pytest.raises(ProfileLoadError, match="not found"):
            load_profile(tmp_path / "nonexistent.yaml")

    def test_apply_overrides(self, tmp_path: Path):
        base = AnalysisConfig(data_dir=tmp_path, symbol="NVDA")
        path = Path(__file__).parent.parent.parent / "configs" / "profiles" / "quick.yaml"
        if not path.exists():
            pytest.skip("Profile file not found")
        profile = load_profile(path)
        updated = apply_profile_config(base, profile)
        assert len(updated.timescales) == 2
        assert updated.symbol == "NVDA"
