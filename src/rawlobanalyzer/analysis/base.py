"""Base analyzer protocol for all MBO-LOB-analyzer implementations.

Every analyzer inherits from ``BaseAnalyzer`` and implements the
``process_day`` / ``finalize`` streaming contract. This guarantees
bounded memory usage regardless of dataset size.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar

from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession
from rawlobanalyzer.reports.base_report import BaseReport

ReportT = TypeVar("ReportT", bound=BaseReport)


class BaseAnalyzer(ABC, Generic[ReportT]):
    """Abstract base class for all analyzers.

    Subclasses must define:
        - ``name``: Human-readable analyzer name.
        - ``description``: Brief description of what the analyzer computes.
        - ``process_day(ctx)``: Accumulate statistics from one trading day.
        - ``finalize()``: Produce the final report from accumulated state.

    Subclasses should declare:
        - ``lob_columns``: LOB columns needed (``None`` = all, ``[]`` = none).
        - ``mbo_columns``: MBO columns needed (``None`` = not needed).
        - ``needs_mbo``: Whether MBO data is required.
        - ``needs_returns``: Whether pre-computed ``DayReturns`` are required.
          If ``True``, the orchestrator guarantees ``ctx.day_returns`` is set.
        - ``needs_spreads``: Whether pre-computed ``DaySpreads`` are required.
          If ``True``, the orchestrator guarantees ``ctx.day_spreads`` is set.
        - ``needs_flow``: Whether pre-computed ``DayFlow`` is required.
          If ``True``, the orchestrator guarantees ``ctx.day_flow`` is set.

    The ``run()`` method provides the default streaming loop. The orchestrator
    calls ``process_day`` / ``finalize`` directly for single-pass batch mode.
    """

    name: ClassVar[str] = "BaseAnalyzer"
    description: ClassVar[str] = ""

    lob_columns: ClassVar[list[str] | None] = None
    mbo_columns: ClassVar[list[str] | None] = None
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = False
    needs_spreads: ClassVar[bool] = False
    needs_flow: ClassVar[bool] = False

    def __init__(self, config: AnalysisConfig) -> None:
        self.config = config

    @abstractmethod
    def process_day(self, ctx: DayContext) -> None:
        """Accumulate statistics from one trading day.

        Called once per day by the orchestrator or ``run()`` loop.
        Must not store the full ``DayData`` -- extract and aggregate only.

        Args:
            ctx: Per-day context containing the raw data and optionally
                pre-computed returns (if ``needs_returns`` is ``True``).
        """
        ...

    @abstractmethod
    def finalize(self) -> ReportT:
        """Compute and return the final report from accumulated state.

        Called once after all days have been processed.
        """
        ...

    def get_extra_scales(self) -> tuple[float, ...] | None:
        """Return additional sampling scales in seconds needed by this analyzer.

        The orchestrator merges scales from all analyzers into a single superset
        so ``compute_day_returns`` is called once per day. Override this in
        analyzers that need scales beyond ``config.timescales`` (e.g. the
        volatility signature plot).

        Returns:
            Tuple of extra scale values in seconds, or ``None`` (default).
        """
        return None

    def run(self, session: AnalysisSession) -> ReportT:
        """Execute the analyzer over a full session (default streaming loop).

        Iterates day-by-day with column projection based on the analyzer's
        declared requirements, then finalizes. When running standalone (not
        via the orchestrator), shared artifacts are computed per-day if needed.
        """
        from rawlobanalyzer.analysis.price._return_engine import compute_day_returns

        day_spreads_fn = None
        if self.needs_spreads:
            from rawlobanalyzer.analysis.spread._spread_engine import (
                compute_day_spreads,
            )
            day_spreads_fn = compute_day_spreads

        day_flow_fn = None
        if self.needs_flow:
            from rawlobanalyzer.analysis.flow._flow_engine import (
                compute_day_flow,
            )
            day_flow_fn = compute_day_flow

        for day in session.iter_days(
            need_lob=True,
            need_mbo=self.needs_mbo,
            lob_columns=self.lob_columns,
            mbo_columns=self.mbo_columns,
        ):
            day_returns = None
            if self.needs_returns:
                day_returns = compute_day_returns(
                    day, self.config,
                    extra_scales_seconds=self.get_extra_scales(),
                )
            day_spreads = None
            if day_spreads_fn is not None:
                day_spreads = day_spreads_fn(
                    day, self.config,
                    extra_scales_seconds=self.get_extra_scales(),
                )
            day_flow = None
            if day_flow_fn is not None:
                day_flow = day_flow_fn(day, self.config)
            ctx = DayContext(
                day=day, day_returns=day_returns,
                day_spreads=day_spreads, day_flow=day_flow,
            )
            self.process_day(ctx)
        return self.finalize()
