"""Tests for _return_engine.py: multi-scale return computation."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.analysis.price._return_engine import (
    DayReturns,
    ScaledReturns,
    compute_day_returns,
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


class TestComputeDayReturns:
    def test_basic_returns(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(
            data_dir=tmp_data_dir,
            symbol="TEST",
            timescales=[TimescaleConfig.seconds(1)],
        )

        dr = compute_day_returns(day, config)

        assert isinstance(dr, DayReturns)
        assert dr.date == "2025-02-03"
        assert len(dr.tick_returns) > 0
        assert len(dr.tick_returns) == dr.n_valid_prices - 1
        assert np.all(np.isfinite(dr.tick_returns))

    def test_scaled_returns_present(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(
            data_dir=tmp_data_dir,
            symbol="TEST",
            timescales=[TimescaleConfig.seconds(1), TimescaleConfig.minutes(1)],
        )

        dr = compute_day_returns(day, config)

        assert "1s" in dr.scaled
        assert "1m" in dr.scaled
        assert isinstance(dr.scaled["1s"], ScaledReturns)
        assert dr.scaled["1s"].label == "1s"

    def test_extra_scales(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(
            data_dir=tmp_data_dir,
            symbol="TEST",
            timescales=[TimescaleConfig.seconds(1)],
        )

        dr = compute_day_returns(day, config, extra_scales_seconds=(0.5, 2.0))

        assert "500ms" in dr.scaled
        assert "2s" in dr.scaled

    def test_open_close_prices(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        dr = compute_day_returns(day, config)

        assert np.isfinite(dr.open_price)
        assert np.isfinite(dr.close_price)
        assert dr.open_price > 100
        assert dr.close_price > 100

    def test_tick_returns_are_log_returns(self, tmp_data_dir: Path):
        """Verify tick returns match the log-return formula."""
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03")

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        dr = compute_day_returns(day, config)

        mids = day.mid_prices
        valid = np.isfinite(mids) & (mids > 0)
        mids_v = mids[valid]
        expected = np.diff(np.log(mids_v))

        np.testing.assert_allclose(dr.tick_returns, expected, rtol=1e-12)

    def test_empty_day(self, tmp_path: Path):
        """Day with no data produces empty returns."""
        day = DayData(date="2025-01-01", symbol="EMPTY")

        config = AnalysisConfig(data_dir=tmp_path, symbol="EMPTY")

        with pytest.raises(ValueError):
            compute_day_returns(day, config)
