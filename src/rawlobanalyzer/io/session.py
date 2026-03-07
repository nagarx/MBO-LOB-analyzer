"""AnalysisSession: day-by-day streaming over a Parquet export directory.

Provides bounded-memory iteration that yields one ``DayData`` at a time,
with column projection to minimize I/O and memory footprint.
"""

from __future__ import annotations

import gc
import logging
from collections.abc import Iterator
from pathlib import Path

from rawlobanalyzer.io.loader import DayData, ParquetDayLoader

logger = logging.getLogger(__name__)


class AnalysisSession:
    """Stream ``DayData`` objects from a Parquet export directory.

    The session discovers all available dates, optionally filters by a date
    range, and yields one day at a time. After each day, the previous day's
    data is eligible for garbage collection.

    Args:
        data_dir: Directory containing ``{date}_lob_snapshots.parquet`` files.
        date_range: Optional ``(start, end)`` inclusive date filter (``YYYY-MM-DD``).
        symbol: Ticker symbol override. If ``None``, reads from file metadata.
    """

    def __init__(
        self,
        data_dir: Path | str,
        *,
        date_range: tuple[str, str] | None = None,
        dates_list: list[str] | None = None,
        symbol: str | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.date_range = date_range
        self.dates_list = set(dates_list) if dates_list else None
        self.symbol_override = symbol
        self._loader = ParquetDayLoader(self.data_dir)
        self._all_dates = self._loader.discover_dates()
        self._dates = self._filter_dates(self._all_dates)

    def _filter_dates(self, dates: list[str]) -> list[str]:
        filtered = dates
        if self.date_range is not None:
            start, end = self.date_range
            filtered = [d for d in filtered if start <= d <= end]
        if self.dates_list is not None:
            filtered = [d for d in filtered if d in self.dates_list]
        return filtered

    @property
    def dates(self) -> list[str]:
        """Sorted list of trading dates in the session (after filtering)."""
        return list(self._dates)

    @property
    def n_days(self) -> int:
        """Number of trading days in the session."""
        return len(self._dates)

    def iter_days(
        self,
        *,
        need_lob: bool = True,
        need_mbo: bool = False,
        lob_columns: list[str] | None = None,
        mbo_columns: list[str] | None = None,
    ) -> Iterator[DayData]:
        """Yield one ``DayData`` per trading day with bounded memory.

        Args:
            need_lob: Whether to load LOB snapshots.
            need_mbo: Whether to load MBO events.
            lob_columns: Specific LOB columns to load (``None`` = all).
            mbo_columns: Specific MBO columns to load (``None`` = all).

        Yields:
            ``DayData`` objects, one per day, in chronological order.
        """
        for i, date in enumerate(self._dates):
            logger.info(
                "Loading day %d/%d: %s",
                i + 1,
                len(self._dates),
                date,
            )
            day = self._loader.load_day(
                date,
                need_lob=need_lob,
                need_mbo=need_mbo,
                lob_columns=lob_columns,
                mbo_columns=mbo_columns,
                validate=(i == 0),
            )
            if self.symbol_override:
                day = DayData(
                    date=day.date,
                    symbol=self.symbol_override,
                    lob=day.lob,
                    mbo=day.mbo,
                    metadata=day.metadata,
                )
            yield day
            gc.collect()
