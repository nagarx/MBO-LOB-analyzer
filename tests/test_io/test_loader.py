"""Tests for io.loader and io.session."""

from pathlib import Path

import numpy as np
import pytest

from rawlobanalyzer.io.loader import DayData, ParquetDayLoader, extract_date_from_filename
from rawlobanalyzer.io.session import AnalysisSession


class TestExtractDate:
    def test_standard_format(self):
        assert extract_date_from_filename(Path("2025-02-03_lob_snapshots.parquet")) == "2025-02-03"

    def test_no_date(self):
        assert extract_date_from_filename(Path("some_file.parquet")) is None


class TestParquetDayLoader:
    def test_discover_dates(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        dates = loader.discover_dates()
        assert dates == ["2025-02-03"]

    def test_load_lob_only(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03", need_lob=True, need_mbo=False)
        assert day.lob is not None
        assert day.mbo is None
        assert day.n_lob_rows == 1000
        assert day.symbol == "TEST"

    def test_load_both(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03", need_lob=True, need_mbo=True)
        assert day.lob is not None
        assert day.mbo is not None
        assert day.n_mbo_rows == 1000

    def test_column_projection(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day(
            "2025-02-03",
            need_lob=True,
            lob_columns=["timestamp_ns", "mid_price"],
        )
        assert set(day.lob.column_names) == {"timestamp_ns", "mid_price"}

    def test_missing_file(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_day("2099-01-01")


class TestDayData:
    def test_cached_properties(self, tmp_data_dir: Path):
        loader = ParquetDayLoader(tmp_data_dir)
        day = loader.load_day("2025-02-03", need_lob=True, need_mbo=True)

        ts = day.lob_timestamps_ns
        assert len(ts) == 1000
        assert ts.dtype == np.int64

        mids = day.mid_prices
        assert len(mids) == 1000
        assert mids.dtype == np.float64
        assert np.all(np.isfinite(mids))

    def test_no_lob_raises(self):
        day = DayData(date="2025-01-01", symbol="TEST", lob=None, mbo=None)
        with pytest.raises(ValueError, match="LOB data not loaded"):
            _ = day.lob_timestamps_ns


class TestAnalysisSession:
    def test_iter_days(self, tmp_data_dir: Path):
        session = AnalysisSession(tmp_data_dir)
        assert session.n_days == 1
        assert session.dates == ["2025-02-03"]

        days = list(session.iter_days())
        assert len(days) == 1
        assert days[0].date == "2025-02-03"

    def test_symbol_override(self, tmp_data_dir: Path):
        session = AnalysisSession(tmp_data_dir, symbol="NVDA")
        days = list(session.iter_days())
        assert days[0].symbol == "NVDA"

    def test_date_range_filter(self, tmp_data_dir: Path):
        session = AnalysisSession(
            tmp_data_dir,
            date_range=("2025-02-04", "2025-02-05"),
        )
        assert session.n_days == 0
