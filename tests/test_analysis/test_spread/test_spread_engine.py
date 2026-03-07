"""Tests for _spread_engine.py: multi-scale spread computation."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.analysis.spread._spread_engine import (
    DaySpreads,
    ScaledSpreads,
    compute_day_spreads,
)
from rawlobanalyzer.core.time_utils import seconds_to_label
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.config.timescale_config import TimescaleConfig
from rawlobanalyzer.io.loader import DayData, ParquetDayLoader


class TestSecondsToLabel:
    def test_sub_second(self):
        assert seconds_to_label(0.1) == "100ms"
        assert seconds_to_label(0.5) == "500ms"

    def test_seconds(self):
        assert seconds_to_label(1.0) == "1s"
        assert seconds_to_label(5.0) == "5s"
        assert seconds_to_label(30.0) == "30s"

    def test_minutes(self):
        assert seconds_to_label(60.0) == "1m"
        assert seconds_to_label(300.0) == "5m"


class TestComputeDaySpreads:
    def test_basic_spreads(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(
            data_dir=tmp_data_dir,
            symbol="TEST",
            timescales=[TimescaleConfig.seconds(1)],
        )

        ds = compute_day_spreads(day, config)

        assert isinstance(ds, DaySpreads)
        assert ds.date == "2025-02-03"
        assert len(ds.tick_spreads_usd) > 0
        assert len(ds.tick_spreads_usd) == ds.n_valid
        assert np.all(np.isfinite(ds.tick_spreads_usd))
        assert np.all(ds.tick_spreads_usd > 0)

    def test_scaled_spreads_present(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(
            data_dir=tmp_data_dir,
            symbol="TEST",
            timescales=[TimescaleConfig.seconds(1), TimescaleConfig.minutes(1)],
        )

        ds = compute_day_spreads(day, config)

        assert "1s" in ds.scaled
        assert "1m" in ds.scaled
        assert isinstance(ds.scaled["1s"], ScaledSpreads)
        assert ds.scaled["1s"].label == "1s"
        assert ds.scaled["1s"].mean_spreads_usd.shape == ds.scaled["1s"].median_spreads_usd.shape

    def test_extra_scales(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(
            data_dir=tmp_data_dir,
            symbol="TEST",
            timescales=[TimescaleConfig.seconds(1)],
        )

        ds = compute_day_spreads(day, config, extra_scales_seconds=(0.5, 2.0))

        assert "500ms" in ds.scaled
        assert "2s" in ds.scaled

    def test_tick_spreads_match_raw(self, tmp_data_dir: Path):
        """Verify tick spreads match raw spread column (valid rows only)."""
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        ds = compute_day_spreads(day, config)

        spreads = day.spreads
        valid = np.isfinite(spreads) & (spreads > 0)
        expected = spreads[valid].astype(np.float64)

        np.testing.assert_allclose(ds.tick_spreads_usd, expected, rtol=1e-12)

    def test_trade_mask_shape(self, tmp_data_dir: Path):
        """Trade mask has same length as tick spreads."""
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        ds = compute_day_spreads(day, config)

        assert len(ds.trade_mask) == len(ds.tick_spreads_usd)
        assert ds.trade_mask.dtype == np.bool_

    def test_empty_day(self, tmp_path: Path):
        """Day with no LOB data raises ValueError."""
        day = DayData(date="2025-01-01", symbol="EMPTY")
        config = AnalysisConfig(data_dir=tmp_path, symbol="EMPTY")

        with pytest.raises(ValueError):
            compute_day_spreads(day, config)
