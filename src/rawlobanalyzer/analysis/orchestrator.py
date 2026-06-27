"""BatchOrchestrator: run multiple analyzers in a single pass over the data.

Instead of each analyzer independently iterating over all days (K*N passes),
the orchestrator iterates once and fans each day out to all analyzers (1 pass).
Shared expensive computations (e.g. ``DayReturns``) are computed once per day
and cached in a ``DayContext`` that all analyzers receive.
"""

from __future__ import annotations

import gc
import logging
import pickle
import resource
import time
from pathlib import Path
from typing import Any

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.flow._flow_engine import compute_day_flow
from rawlobanalyzer.analysis.price._return_engine import compute_day_returns
from rawlobanalyzer.analysis.spread._spread_engine import compute_day_spreads
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EMA_PROGRESS_ALPHA, MIN_ELAPSED_SECONDS, NS_PER_DAY
from rawlobanalyzer.core.time_utils import utc_offset_for_date
from rawlobanalyzer.io.session import AnalysisSession
from rawlobanalyzer.reports.base_report import BaseReport

logger = logging.getLogger(__name__)


def _peak_memory_mb() -> float:
    """Return peak RSS in MB (macOS/Linux)."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # macOS reports in bytes, Linux in KB
    import sys
    if sys.platform == "darwin":
        return ru.ru_maxrss / (1024 * 1024)
    return ru.ru_maxrss / 1024


def _format_eta(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


class BatchOrchestrator:
    """Run a list of analyzers in a single pass over the dataset.

    The orchestrator computes the union of all column requirements,
    loads each day once with that superset, and dispatches to all
    analyzers. Shared intermediate results (``DayReturns``, ``DaySpreads``,
    ``DayFlow``) are computed once per day when at least one analyzer
    needs them.

    Args:
        config: Analysis configuration.
        analyzers: List of initialized analyzer instances.
    """

    def __init__(self, config: AnalysisConfig, analyzers: list[BaseAnalyzer]) -> None:
        self.config = config
        self.analyzers = analyzers

    def run(self) -> dict[str, BaseReport]:
        """Execute all analyzers in a single data pass.

        Returns:
            Dict mapping analyzer name to its finalized report.
        """
        session = AnalysisSession(
            data_dir=self.config.data_dir,
            date_range=self.config.date_range,
            dates_list=self.config.dates_list,
            symbol=self.config.symbol,
        )

        lob_columns = self._merge_columns("lob_columns")
        mbo_columns = self._merge_columns("mbo_columns")
        need_mbo = any(a.needs_mbo for a in self.analyzers)
        any_need_returns = any(a.needs_returns for a in self.analyzers)
        any_need_spreads = any(a.needs_spreads for a in self.analyzers)
        any_need_flow = any(a.needs_flow for a in self.analyzers)
        merged_extra_scales = self._merge_extra_scales() if any_need_returns else None
        merged_spread_scales = (
            self._merge_extra_scales() if any_need_spreads else None
        )

        if any_need_flow and any_need_returns:
            ofi_scales = self.config.thresholds.flow.ofi_timescales_seconds
            extra = set(ofi_scales)
            if merged_extra_scales:
                extra.update(merged_extra_scales)
            merged_extra_scales = tuple(sorted(extra))

        n_days = session.n_days
        logger.info(
            "BatchOrchestrator: %d analyzer(s), %d day(s), need_mbo=%s, "
            "need_returns=%s, need_spreads=%s, need_flow=%s, extra_scales=%s",
            len(self.analyzers),
            n_days,
            need_mbo,
            any_need_returns,
            any_need_spreads,
            any_need_flow,
            len(merged_extra_scales) if merged_extra_scales else 0,
        )

        checkpoint_dir = self.config.checkpoint_dir
        completed_dates: set[str] = set()
        if self.config.resume and checkpoint_dir and checkpoint_dir.is_dir():
            completed_dates = self._restore_checkpoint(checkpoint_dir)
            if completed_dates:
                logger.info(
                    "Resumed from checkpoint: %d days already processed, skipping",
                    len(completed_dates),
                )

        t0 = time.monotonic()
        ema_day_time: float = 0.0
        total_rows: int = 0

        for i, day in enumerate(
            session.iter_days(
                need_lob=True,
                need_mbo=need_mbo,
                lob_columns=lob_columns,
                mbo_columns=mbo_columns if need_mbo else None,
            )
        ):
            if day.date in completed_dates:
                logger.info("  [%d/%d] %s: SKIPPED (checkpoint)", i + 1, n_days, day.date)
                continue

            day_t0 = time.monotonic()

            ts0 = day.lob_timestamps_ns[0] if day.n_lob_rows > 0 else 0
            day_epoch_ns = int(ts0 - (ts0 % NS_PER_DAY)) if ts0 > 0 else 0

            day_utc_offset = utc_offset_for_date(day.date)
            day_config = self._day_config(day_utc_offset)

            day_returns = None
            if any_need_returns:
                day_returns = compute_day_returns(
                    day, day_config,
                    extra_scales_seconds=merged_extra_scales,
                    day_epoch_ns=day_epoch_ns,
                )
            day_spreads = None
            if any_need_spreads:
                day_spreads = compute_day_spreads(
                    day, day_config,
                    extra_scales_seconds=merged_spread_scales,
                )
            day_flow = None
            if any_need_flow:
                day_flow = compute_day_flow(
                    day, day_config, day_epoch_ns=day_epoch_ns,
                )
            ctx = DayContext(
                day=day, day_epoch_ns=day_epoch_ns,
                utc_offset_hours=day_utc_offset,
                day_returns=day_returns,
                day_spreads=day_spreads, day_flow=day_flow,
            )

            for analyzer in self.analyzers:
                analyzer.process_day(ctx)

            day_elapsed = time.monotonic() - day_t0
            day_rows = day.n_lob_rows + day.n_mbo_rows
            total_rows += day_rows
            ema_day_time = (
                day_elapsed if ema_day_time == 0.0
                else (EMA_PROGRESS_ALPHA * day_elapsed + (1 - EMA_PROGRESS_ALPHA) * ema_day_time)
            )
            remaining = n_days - (i + 1)
            eta = ema_day_time * remaining
            rows_per_sec = day_rows / max(day_elapsed, MIN_ELAPSED_SECONDS)
            mem_mb = _peak_memory_mb()

            logger.info(
                "  [%d/%d] %s  %.1fs  %s rows/s  ETA %s  peak %s MB  (LOB=%s MBO=%s)",
                i + 1, n_days, day.date,
                day_elapsed,
                f"{rows_per_sec:,.0f}",
                _format_eta(eta),
                f"{mem_mb:,.0f}",
                f"{day.n_lob_rows:,}",
                f"{day.n_mbo_rows:,}",
            )

            if checkpoint_dir:
                self._save_checkpoint(checkpoint_dir, day.date)

            del ctx, day_returns, day_spreads, day_flow, day
            gc.collect()

        results: dict[str, BaseReport] = {}
        for analyzer in self.analyzers:
            report = analyzer.finalize()
            results[analyzer.name] = report

        elapsed = time.monotonic() - t0
        logger.info(
            "BatchOrchestrator complete: %.1fs total, %s rows, peak %s MB",
            elapsed, f"{total_rows:,}", f"{_peak_memory_mb():,.0f}",
        )

        return results

    def _save_checkpoint(self, checkpoint_dir: Path, date: str) -> None:
        """Persist analyzer state after processing a day."""
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        try:
            for analyzer in self.analyzers:
                path = checkpoint_dir / f"{analyzer.name}.pkl"
                with open(path, "wb") as f:
                    pickle.dump(analyzer, f, protocol=pickle.HIGHEST_PROTOCOL)
            manifest = checkpoint_dir / "manifest.txt"
            existing = set()
            if manifest.exists():
                existing = set(manifest.read_text().strip().splitlines())
            existing.add(date)
            manifest.write_text("\n".join(sorted(existing)) + "\n")
        except Exception:
            logger.warning("Checkpoint save failed for %s (non-fatal)", date, exc_info=True)

    def _restore_checkpoint(self, checkpoint_dir: Path) -> set[str]:
        """Restore analyzer state from a checkpoint directory."""
        manifest = checkpoint_dir / "manifest.txt"
        if not manifest.exists():
            return set()
        try:
            restored_analyzers: list[BaseAnalyzer] = []
            for analyzer in self.analyzers:
                path = checkpoint_dir / f"{analyzer.name}.pkl"
                if not path.exists():
                    logger.warning("Checkpoint missing for %s, starting fresh", analyzer.name)
                    return set()
                with open(path, "rb") as f:
                    restored_analyzers.append(pickle.load(f))  # noqa: S301
            self.analyzers = restored_analyzers
            dates = set(manifest.read_text().strip().splitlines())
            return dates
        except Exception:
            logger.warning("Checkpoint restore failed, starting fresh", exc_info=True)
            return set()

    def _day_config(self, utc_offset: int) -> AnalysisConfig:
        """Return a config with the trading-hours UTC offset set for one day.

        If the offset already matches the base config, returns the base config
        directly (no allocation).
        """
        if utc_offset == self.config.trading_hours.utc_offset_hours:
            return self.config
        from dataclasses import replace as dc_replace

        from rawlobanalyzer.config.timescale_config import TradingHours

        from rawlobanalyzer.core.time_utils import (
            _EXT_CLOSE_ET_H,
            _EXT_OPEN_ET_H,
            _RTH_CLOSE_ET_H,
            _RTH_OPEN_ET_H,
        )

        adjusted_hours = TradingHours(
            rth_open_utc_h=_RTH_OPEN_ET_H - utc_offset,
            rth_close_utc_h=_RTH_CLOSE_ET_H - utc_offset,
            ext_open_utc_h=_EXT_OPEN_ET_H - utc_offset,
            ext_close_utc_h=_EXT_CLOSE_ET_H - utc_offset,
            label=f"us_equity_{'edt' if utc_offset == -4 else 'est'}",
            utc_offset_hours=utc_offset,
        )
        return dc_replace(self.config, trading_hours=adjusted_hours)

    def _merge_columns(self, attr: str) -> list[str] | None:
        """Compute the union of column requirements across all analyzers.

        If any analyzer requests all columns (``None``), returns ``None``.
        """
        all_cols: set[str] = set()
        for analyzer in self.analyzers:
            cols = getattr(analyzer, attr, None)
            if cols is None:
                return None
            all_cols.update(cols)
        return sorted(all_cols) if all_cols else []

    def _merge_extra_scales(self) -> tuple[float, ...] | None:
        """Compute the union of extra sampling scales across all analyzers.

        Returns:
            Sorted tuple of unique extra scales, or ``None`` if no analyzer
            declared any.
        """
        all_scales: set[float] = set()
        for analyzer in self.analyzers:
            scales = analyzer.get_extra_scales()
            if scales:
                all_scales.update(scales)
        return tuple(sorted(all_scales)) if all_scales else None
