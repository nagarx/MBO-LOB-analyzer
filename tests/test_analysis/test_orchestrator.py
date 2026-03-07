"""Tests for BatchOrchestrator: DayContext creation, scale merging, caching."""

from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import numpy as np
import pytest

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.orchestrator import BatchOrchestrator
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.reports.base_report import BaseReport


class _SimpleReport(BaseReport):
    SCHEMA_VERSION: ClassVar[str] = "1.0"

    def __init__(self, days: int):
        self.days = days

    def to_dict(self) -> dict[str, Any]:
        return {"days": self.days, "_meta": self._meta_dict()}

    def summary(self) -> str:
        return f"SimpleReport(days={self.days})"


class _CountingAnalyzer(BaseAnalyzer[_SimpleReport]):
    """Analyzer that counts days and records what it receives."""

    name: ClassVar[str] = "CountingAnalyzer"
    description: ClassVar[str] = "Test counter"
    lob_columns: ClassVar[list[str] | None] = ["timestamp_ns", "mid_price"]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = False

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self.ctx_list: list[DayContext] = []

    def process_day(self, ctx: DayContext) -> None:
        self.ctx_list.append(ctx)

    def finalize(self) -> _SimpleReport:
        return _SimpleReport(days=len(self.ctx_list))


class _ReturnsAnalyzer(BaseAnalyzer[_SimpleReport]):
    """Analyzer that needs returns."""

    name: ClassVar[str] = "ReturnsTestAnalyzer"
    description: ClassVar[str] = "Test returns consumer"
    lob_columns: ClassVar[list[str] | None] = ["timestamp_ns", "mid_price"]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self.ctx_list: list[DayContext] = []

    def process_day(self, ctx: DayContext) -> None:
        self.ctx_list.append(ctx)

    def finalize(self) -> _SimpleReport:
        return _SimpleReport(days=len(self.ctx_list))


class _ExtraScaleAnalyzer(BaseAnalyzer[_SimpleReport]):
    """Analyzer that declares extra scales."""

    name: ClassVar[str] = "ExtraScaleAnalyzer"
    description: ClassVar[str] = "Test extra scales"
    lob_columns: ClassVar[list[str] | None] = ["timestamp_ns", "mid_price"]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self.ctx_list: list[DayContext] = []

    def get_extra_scales(self) -> tuple[float, ...] | None:
        return (0.1, 0.5, 2.0)

    def process_day(self, ctx: DayContext) -> None:
        self.ctx_list.append(ctx)

    def finalize(self) -> _SimpleReport:
        return _SimpleReport(days=len(self.ctx_list))


class _SpreadsAnalyzer(BaseAnalyzer[_SimpleReport]):
    """Analyzer that needs spreads."""

    name: ClassVar[str] = "SpreadsTestAnalyzer"
    description: ClassVar[str] = "Test spreads consumer"
    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "spread", "spread_bps", "triggering_action",
    ]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = False
    needs_spreads: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self.ctx_list: list[DayContext] = []

    def process_day(self, ctx: DayContext) -> None:
        self.ctx_list.append(ctx)

    def finalize(self) -> _SimpleReport:
        return _SimpleReport(days=len(self.ctx_list))


class TestBatchOrchestrator:
    def test_no_returns_analyzer(self, tmp_data_dir: Path):
        """Analyzer without needs_returns should get day_returns=None."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = _CountingAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [analyzer])

        results = orchestrator.run()

        assert "CountingAnalyzer" in results
        assert results["CountingAnalyzer"].days == 1
        assert len(analyzer.ctx_list) == 1
        assert analyzer.ctx_list[0].day_returns is None

    def test_returns_computed_when_needed(self, tmp_data_dir: Path):
        """Analyzer with needs_returns=True should get a populated DayReturns."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = _ReturnsAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [analyzer])

        results = orchestrator.run()

        assert len(analyzer.ctx_list) == 1
        dr = analyzer.ctx_list[0].day_returns
        assert dr is not None
        assert dr.date == "2025-02-03"
        assert len(dr.tick_returns) > 0

    def test_shared_day_returns(self, tmp_data_dir: Path):
        """Two analyzers needing returns should receive the same DayReturns object."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        a1 = _ReturnsAnalyzer(config)
        a2 = _ExtraScaleAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [a1, a2])

        orchestrator.run()

        assert len(a1.ctx_list) == 1
        assert len(a2.ctx_list) == 1
        assert a1.ctx_list[0].day_returns is a2.ctx_list[0].day_returns

    def test_extra_scales_merged(self, tmp_data_dir: Path):
        """Extra scales from the analyzer should appear in DayReturns."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = _ExtraScaleAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [analyzer])

        orchestrator.run()

        dr = analyzer.ctx_list[0].day_returns
        assert dr is not None
        assert "100ms" in dr.scaled
        assert "500ms" in dr.scaled
        assert "2s" in dr.scaled

    def test_compute_called_once_per_day(self, two_day_data_dir: Path):
        """DayReturns should be computed exactly once per day, not per analyzer."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        a1 = _ReturnsAnalyzer(config)
        a2 = _ExtraScaleAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [a1, a2])

        with patch(
            "rawlobanalyzer.analysis.orchestrator.compute_day_returns",
            wraps=__import__(
                "rawlobanalyzer.analysis.price._return_engine",
                fromlist=["compute_day_returns"],
            ).compute_day_returns,
        ) as mock_cdr:
            orchestrator.run()
            assert mock_cdr.call_count == 2, (
                f"Expected 2 calls (1 per day), got {mock_cdr.call_count}"
            )

    def test_mixed_analyzers(self, tmp_data_dir: Path):
        """Mix of returns and non-returns analyzers should all work."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        counter = _CountingAnalyzer(config)
        returner = _ReturnsAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [counter, returner])

        results = orchestrator.run()

        assert counter.ctx_list[0].day_returns is not None
        assert returner.ctx_list[0].day_returns is not None
        assert counter.ctx_list[0].day_returns is returner.ctx_list[0].day_returns

    def test_merge_extra_scales_union(self):
        """Scale merging should produce sorted unique union."""
        from pathlib import Path

        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        a1 = _ExtraScaleAnalyzer(config)  # (0.1, 0.5, 2.0)

        class _AnotherScaleAnalyzer(_ExtraScaleAnalyzer):
            name: ClassVar[str] = "AnotherScale"

            def get_extra_scales(self) -> tuple[float, ...] | None:
                return (0.5, 5.0, 10.0)

        a2 = _AnotherScaleAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [a1, a2])
        merged = orchestrator._merge_extra_scales()

        assert merged == (0.1, 0.5, 2.0, 5.0, 10.0)

    def test_merge_extra_scales_none(self):
        """When no analyzer has extra scales, result should be None."""
        from pathlib import Path

        config = AnalysisConfig(data_dir=Path("/tmp"), symbol="TEST")
        a = _CountingAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [a])
        merged = orchestrator._merge_extra_scales()
        assert merged is None

    def test_no_spreads_analyzer(self, tmp_data_dir: Path):
        """Analyzer without needs_spreads should get day_spreads=None."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = _CountingAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [analyzer])

        orchestrator.run()

        assert len(analyzer.ctx_list) == 1
        assert analyzer.ctx_list[0].day_spreads is None

    def test_spreads_computed_when_needed(self, tmp_data_dir: Path):
        """Analyzer with needs_spreads=True should get a populated DaySpreads."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = _SpreadsAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [analyzer])

        orchestrator.run()

        assert len(analyzer.ctx_list) == 1
        ds = analyzer.ctx_list[0].day_spreads
        assert ds is not None
        assert ds.date == "2025-02-03"
        assert len(ds.tick_spreads_usd) > 0

    def test_shared_day_spreads(self, tmp_data_dir: Path):
        """Two analyzers needing spreads should receive the same DaySpreads."""

        class _SpreadsAnalyzer2(_SpreadsAnalyzer):
            name: ClassVar[str] = "SpreadsAnalyzer2"

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        a1 = _SpreadsAnalyzer(config)
        a2 = _SpreadsAnalyzer2(config)
        orchestrator = BatchOrchestrator(config, [a1, a2])

        orchestrator.run()

        assert len(a1.ctx_list) == 1
        assert len(a2.ctx_list) == 1
        assert a1.ctx_list[0].day_spreads is a2.ctx_list[0].day_spreads

    def test_compute_day_spreads_once_per_day(self, two_day_data_dir: Path):
        """DaySpreads should be computed exactly once per day."""
        config = AnalysisConfig(data_dir=two_day_data_dir, symbol="TEST")
        analyzer = _SpreadsAnalyzer(config)
        orchestrator = BatchOrchestrator(config, [analyzer])

        with patch(
            "rawlobanalyzer.analysis.orchestrator.compute_day_spreads",
            wraps=__import__(
                "rawlobanalyzer.analysis.spread._spread_engine",
                fromlist=["compute_day_spreads"],
            ).compute_day_spreads,
        ) as mock_cds:
            orchestrator.run()
            assert mock_cds.call_count == 2, (
                f"Expected 2 calls (1 per day), got {mock_cds.call_count}"
            )
