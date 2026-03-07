"""DataQualityAnalyzer: schema validation, gap detection, system message stats.

This is the foundational "health check" analyzer that should always run first.
It validates that the Parquet export is well-formed, identifies data gaps,
and provides a high-level overview of each trading day's characteristics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import NS_PER_SECOND
from rawlobanalyzer.core.price_utils import nanodollars_to_usd
from rawlobanalyzer.core.time_utils import REGIME_LABELS, time_regime
from rawlobanalyzer.io.schema import (
    ACTION_LABELS,
    BOOK_CONSISTENCY_LABELS,
    SIDE_LABELS,
)
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)


@dataclass
class DayStats:
    """Per-day summary statistics accumulated during processing."""

    date: str
    n_lob_rows: int = 0
    n_mbo_rows: int = 0
    first_ts_ns: int = 0
    last_ts_ns: int = 0
    max_gap_ns: int = 0
    mean_inter_event_us: float = 0.0
    median_inter_event_us: float = 0.0

    best_bid_min_usd: float = 0.0
    best_bid_max_usd: float = 0.0
    best_ask_min_usd: float = 0.0
    best_ask_max_usd: float = 0.0
    mid_price_open_usd: float = 0.0
    mid_price_close_usd: float = 0.0

    spread_mean_usd: float = 0.0
    spread_median_usd: float = 0.0
    spread_max_usd: float = 0.0

    action_counts: dict[str, int] = field(default_factory=dict)
    side_counts: dict[str, int] = field(default_factory=dict)
    consistency_counts: dict[str, int] = field(default_factory=dict)
    regime_counts: dict[str, int] = field(default_factory=dict)

    system_msg_count: int = 0
    crossed_book_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "n_lob_rows": self.n_lob_rows,
            "n_mbo_rows": self.n_mbo_rows,
            "first_ts_ns": self.first_ts_ns,
            "last_ts_ns": self.last_ts_ns,
            "max_gap_s": self.max_gap_ns / NS_PER_SECOND,
            "mean_inter_event_us": self.mean_inter_event_us,
            "median_inter_event_us": self.median_inter_event_us,
            "best_bid_range_usd": [self.best_bid_min_usd, self.best_bid_max_usd],
            "best_ask_range_usd": [self.best_ask_min_usd, self.best_ask_max_usd],
            "mid_price_open_usd": self.mid_price_open_usd,
            "mid_price_close_usd": self.mid_price_close_usd,
            "spread_mean_usd": self.spread_mean_usd,
            "spread_median_usd": self.spread_median_usd,
            "spread_max_usd": self.spread_max_usd,
            "action_counts": self.action_counts,
            "side_counts": self.side_counts,
            "consistency_counts": self.consistency_counts,
            "regime_counts": self.regime_counts,
            "system_msg_count": self.system_msg_count,
            "crossed_book_count": self.crossed_book_count,
        }


@dataclass
class DataQualityReport(BaseReport):
    """Report from the DataQualityAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    total_lob_rows: int
    total_mbo_rows: int
    day_stats: list[DayStats]
    missing_dates: list[str]
    issues: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "total_lob_rows": self.total_lob_rows,
            "total_mbo_rows": self.total_mbo_rows,
            "missing_dates": self.missing_dates,
            "issues": self.issues,
            "day_stats": [d.to_dict() for d in self.day_stats],
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("DATA QUALITY REPORT"))
        lines.append(format_kv({
            "Symbol": self.symbol,
            "Days analyzed": self.n_days,
            "Total LOB rows": self.total_lob_rows,
            "Total MBO rows": self.total_mbo_rows,
        }))

        if self.missing_dates:
            lines.append(format_subsection("MISSING DATES"))
            for d in self.missing_dates:
                lines.append(f"      {d}")

        if self.issues:
            lines.append(format_subsection("ISSUES"))
            for issue in self.issues:
                lines.append(f"      ! {issue}")
        else:
            lines.append("\n    No issues detected.")

        lines.append(format_subsection("PER-DAY SUMMARY"))
        headers = ["Date", "LOB Rows", "MBO Rows", "MaxGap(s)", "MedIET(us)",
                    "Open($)", "Close($)", "AvgSprd($)"]
        rows = []
        for d in self.day_stats:
            rows.append([
                d.date,
                d.n_lob_rows,
                d.n_mbo_rows,
                d.max_gap_ns / NS_PER_SECOND,
                d.median_inter_event_us,
                d.mid_price_open_usd,
                d.mid_price_close_usd,
                d.spread_mean_usd,
            ])
        lines.append(format_table(headers, rows))

        if self.day_stats:
            d0 = self.day_stats[0]
            lines.append(format_subsection(f"ACTION DISTRIBUTION ({d0.date})"))
            for action, count in sorted(d0.action_counts.items()):
                lines.append(f"      {action:>8}: {count:>12,}")

            lines.append(format_subsection(f"BOOK CONSISTENCY ({d0.date})"))
            for state, count in sorted(d0.consistency_counts.items()):
                lines.append(f"      {state:>8}: {count:>12,}")

            lines.append(format_subsection(f"TIME REGIME DISTRIBUTION ({d0.date})"))
            for regime, count in sorted(d0.regime_counts.items()):
                lines.append(f"      {regime:>16}: {count:>12,}")

        return "\n".join(lines)


@register_analyzer
class DataQualityAnalyzer(BaseAnalyzer[DataQualityReport]):
    """Validates data integrity and provides a high-level overview.

    Checks:
    - Schema and metadata validation (via the loader)
    - Per-day row counts, timestamp ranges, gaps
    - System message counts (order_id=0)
    - Book consistency distribution
    - Price range sanity
    - Intraday regime distribution
    """

    name: ClassVar[str] = "DataQualityAnalyzer"
    description: ClassVar[str] = "Data integrity validation and overview statistics"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "sequence", "best_bid", "best_ask",
        "mid_price", "spread", "book_consistency",
        "triggering_action", "triggering_side",
        "total_bid_volume", "total_ask_volume",
    ]
    mbo_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "order_id", "action", "side",
    ]
    needs_mbo: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._day_stats: list[DayStats] = []
        self._issues: list[str] = []
        self._symbol: str = config.symbol

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        self._symbol = day.symbol
        ds = DayStats(date=day.date)

        if day.lob is not None:
            lob = day.lob
            ds.n_lob_rows = lob.num_rows

            ts = lob.column("timestamp_ns").to_numpy()
            if len(ts) > 0:
                ds.first_ts_ns = int(ts[0])
                ds.last_ts_ns = int(ts[-1])

                diffs = np.diff(ts)
                if len(diffs) > 0:
                    ds.max_gap_ns = int(np.max(diffs))
                    positive_diffs = diffs[diffs > 0]
                    if len(positive_diffs) > 0:
                        ds.mean_inter_event_us = float(np.mean(positive_diffs)) / 1e3
                        ds.median_inter_event_us = float(np.median(positive_diffs)) / 1e3

                    reversals = int(np.sum(diffs < 0))
                    if reversals > 0:
                        self._issues.append(
                            f"{day.date}: {reversals:,} timestamp reversals in LOB"
                        )

            if "best_bid" in lob.column_names:
                bids = lob.column("best_bid").to_numpy()
                valid_bids = bids[bids > 0]
                if len(valid_bids) > 0:
                    ds.best_bid_min_usd = float(valid_bids.min()) / 1e9
                    ds.best_bid_max_usd = float(valid_bids.max()) / 1e9

            if "best_ask" in lob.column_names:
                asks = lob.column("best_ask").to_numpy()
                valid_asks = asks[asks > 0]
                if len(valid_asks) > 0:
                    ds.best_ask_min_usd = float(valid_asks.min()) / 1e9
                    ds.best_ask_max_usd = float(valid_asks.max()) / 1e9

            if "mid_price" in lob.column_names:
                mids = lob.column("mid_price").to_numpy()
                valid_mids = mids[np.isfinite(mids)]
                if len(valid_mids) > 0:
                    ds.mid_price_open_usd = float(valid_mids[0])
                    ds.mid_price_close_usd = float(valid_mids[-1])

            if "spread" in lob.column_names:
                spreads = lob.column("spread").to_numpy()
                valid_spreads = spreads[np.isfinite(spreads)]
                if len(valid_spreads) > 0:
                    ds.spread_mean_usd = float(np.mean(valid_spreads))
                    ds.spread_median_usd = float(np.median(valid_spreads))
                    ds.spread_max_usd = float(np.max(valid_spreads))

            if "book_consistency" in lob.column_names:
                consist = lob.column("book_consistency").to_numpy()
                for val, label in BOOK_CONSISTENCY_LABELS.items():
                    cnt = int(np.sum(consist == val))
                    if cnt > 0:
                        ds.consistency_counts[label] = cnt

            if "best_bid" in lob.column_names and "best_ask" in lob.column_names:
                bids_raw = lob.column("best_bid").to_numpy()
                asks_raw = lob.column("best_ask").to_numpy()
                valid_both = (bids_raw > 0) & (asks_raw > 0)
                crossed = int(np.sum((bids_raw >= asks_raw) & valid_both))
                ds.crossed_book_count = crossed
                if crossed > 0:
                    self._issues.append(
                        f"{day.date}: {crossed:,} crossed book states"
                    )

            if len(ts) > 0:
                utc_offset = ctx.utc_offset_hours
                regimes = time_regime(ts, utc_offset_hours=utc_offset)
                for val, label in REGIME_LABELS.items():
                    cnt = int(np.sum(regimes == val))
                    if cnt > 0:
                        ds.regime_counts[label] = cnt

        if day.mbo is not None:
            mbo = day.mbo
            ds.n_mbo_rows = mbo.num_rows

            if "order_id" in mbo.column_names:
                order_ids = mbo.column("order_id").to_numpy()
                ds.system_msg_count = int(np.sum(order_ids == 0))

            if "action" in mbo.column_names:
                actions = mbo.column("action").to_numpy()
                for val, label in ACTION_LABELS.items():
                    cnt = int(np.sum(actions == val))
                    if cnt > 0:
                        ds.action_counts[label] = cnt

            if "side" in mbo.column_names:
                sides = mbo.column("side").to_numpy()
                for val, label in SIDE_LABELS.items():
                    cnt = int(np.sum(sides == val))
                    if cnt > 0:
                        ds.side_counts[label] = cnt

        self._day_stats.append(ds)

    def finalize(self) -> DataQualityReport:
        total_lob = sum(d.n_lob_rows for d in self._day_stats)
        total_mbo = sum(d.n_mbo_rows for d in self._day_stats)

        missing: list[str] = []
        if len(self._day_stats) >= 2:
            from datetime import datetime, timedelta
            dates_set = {d.date for d in self._day_stats}
            first = datetime.strptime(self._day_stats[0].date, "%Y-%m-%d")
            last = datetime.strptime(self._day_stats[-1].date, "%Y-%m-%d")
            current = first
            while current <= last:
                if current.weekday() < 5:
                    date_str = current.strftime("%Y-%m-%d")
                    if date_str not in dates_set:
                        missing.append(date_str)
                current += timedelta(days=1)

        return DataQualityReport(
            symbol=self._symbol,
            n_days=len(self._day_stats),
            total_lob_rows=total_lob,
            total_mbo_rows=total_mbo,
            day_stats=self._day_stats,
            missing_dates=missing,
            issues=self._issues,
        )
