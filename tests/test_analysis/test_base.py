"""Tests for analysis framework: base, registry, orchestrator."""

from pathlib import Path
from typing import Any, ClassVar

import pytest

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.registry import (
    ANALYZER_REGISTRY,
    get_analyzer,
    list_analyzers,
    register_analyzer,
)
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.reports.base_report import BaseReport


class _DummyReport(BaseReport):
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __init__(self, value: int):
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "_meta": self._meta_dict()}

    def summary(self) -> str:
        return f"DummyReport(value={self.value})"


class TestRegistry:
    def test_data_quality_registered(self):
        names = list_analyzers()
        assert "DataQualityAnalyzer" in names

    def test_get_analyzer(self):
        cls = get_analyzer("DataQualityAnalyzer")
        assert cls.name == "DataQualityAnalyzer"

    def test_get_unknown(self):
        with pytest.raises(KeyError, match="Unknown analyzer"):
            get_analyzer("NonExistentAnalyzer")


class TestBaseReport:
    def test_to_dict(self):
        report = _DummyReport(42)
        d = report.to_dict()
        assert d["value"] == 42
        assert "_meta" in d
        assert d["_meta"]["schema_version"] == "1.0"

    def test_summary(self):
        report = _DummyReport(42)
        assert "42" in report.summary()

    def test_to_json(self, tmp_path: Path):
        report = _DummyReport(99)
        json_path = tmp_path / "test.json"
        report.to_json(json_path)
        assert json_path.exists()
        import json
        data = json.loads(json_path.read_text())
        assert data["value"] == 99


class TestDataQualityAnalyzer:
    def test_end_to_end(self, tmp_data_dir: Path):
        from rawlobanalyzer.analysis.health.data_quality import DataQualityAnalyzer
        from rawlobanalyzer.io.session import AnalysisSession

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = DataQualityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")

        report = analyzer.run(session)

        assert report.n_days == 1
        assert report.total_lob_rows == 1000
        assert report.total_mbo_rows == 1000
        assert len(report.day_stats) == 1

        ds = report.day_stats[0]
        assert ds.date == "2025-02-03"
        assert ds.n_lob_rows == 1000
        assert ds.system_msg_count == ds.action_counts.get("Trade", 0)  # aggressor trades have order_id=0
        assert ds.spread_mean_usd > 0
        assert ds.mid_price_open_usd > 100

        summary = report.summary()
        assert "DATA QUALITY" in summary
        assert "2025-02-03" in summary

    def test_json_roundtrip(self, tmp_data_dir: Path, tmp_path: Path):
        from rawlobanalyzer.analysis.health.data_quality import DataQualityAnalyzer
        from rawlobanalyzer.io.session import AnalysisSession
        import json

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = DataQualityAnalyzer(config)
        session = AnalysisSession(tmp_data_dir, symbol="TEST")
        report = analyzer.run(session)

        json_path = tmp_path / "quality.json"
        report.to_json(json_path)

        data = json.loads(json_path.read_text())
        assert data["n_days"] == 1
        assert data["total_lob_rows"] == 1000
        assert "_meta" in data
