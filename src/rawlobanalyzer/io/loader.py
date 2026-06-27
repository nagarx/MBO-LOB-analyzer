"""Parquet day loader with column projection and lazy derived properties.

The loader reads one trading day at a time from Parquet files, using PyArrow's
columnar format for zero-copy reads and memory-efficient column projection.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from rawlobanalyzer.core.constants import NANODOLLARS_PER_DOLLAR
from rawlobanalyzer.io.schema import (
    LOB_ALL_COLUMNS,
    MBO_COLUMNS,
    META_DATE,
    META_SYMBOL,
    SchemaValidationError,
    validate_lob_schema,
    validate_mbo_schema,
    validate_parquet_metadata,
)

_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


def extract_date_from_filename(path: Path) -> str | None:
    """Extract YYYY-MM-DD date from a Parquet filename."""
    m = _DATE_RE.search(path.stem)
    return m.group(1) if m else None


@dataclass
class DayData:
    """One trading day's raw data loaded from Parquet.

    Attributes:
        date: Trading date as ``YYYY-MM-DD``.
        symbol: Ticker symbol (e.g. ``"NVDA"``).
        lob: LOB snapshot table (PyArrow), or ``None`` if not loaded.
        mbo: MBO event table (PyArrow), or ``None`` if not loaded.
        metadata: Raw Parquet file-level metadata dict.
    """

    date: str
    symbol: str
    lob: pa.Table | None = None
    mbo: pa.Table | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def lob_timestamps_ns(self) -> np.ndarray:
        """LOB timestamp_ns as int64 numpy array."""
        if self.lob is None:
            raise ValueError("LOB data not loaded")
        return self.lob.column("timestamp_ns").to_numpy()

    @cached_property
    def mbo_timestamps_ns(self) -> np.ndarray:
        """MBO timestamp_ns as int64 numpy array."""
        if self.mbo is None:
            raise ValueError("MBO data not loaded")
        return self.mbo.column("timestamp_ns").to_numpy()

    @cached_property
    def mid_prices(self) -> np.ndarray:
        """Mid prices in dollars (float64). Requires LOB with ``mid_price`` column."""
        if self.lob is None:
            raise ValueError("LOB data not loaded")
        if "mid_price" not in self.lob.column_names:
            raise ValueError("mid_price column not in LOB table (was --no-derived used?)")
        return self.lob.column("mid_price").to_numpy()

    @cached_property
    def spreads(self) -> np.ndarray:
        """Spreads in dollars (float64). Requires LOB with ``spread`` column."""
        if self.lob is None:
            raise ValueError("LOB data not loaded")
        if "spread" not in self.lob.column_names:
            raise ValueError("spread column not in LOB table")
        return self.lob.column("spread").to_numpy()

    @cached_property
    def best_bids_usd(self) -> np.ndarray:
        """Best bid prices in dollars (float64, nanodollars -> USD)."""
        if self.lob is None:
            raise ValueError("LOB data not loaded")
        raw = self.lob.column("best_bid").to_numpy()
        return raw.astype(np.float64) / NANODOLLARS_PER_DOLLAR

    @cached_property
    def best_asks_usd(self) -> np.ndarray:
        """Best ask prices in dollars (float64, nanodollars -> USD)."""
        if self.lob is None:
            raise ValueError("LOB data not loaded")
        raw = self.lob.column("best_ask").to_numpy()
        return raw.astype(np.float64) / NANODOLLARS_PER_DOLLAR

    @property
    def n_lob_rows(self) -> int:
        """Number of LOB snapshot rows."""
        return self.lob.num_rows if self.lob is not None else 0

    @property
    def n_mbo_rows(self) -> int:
        """Number of MBO event rows."""
        return self.mbo.num_rows if self.mbo is not None else 0


class ParquetDayLoader:
    """Loads one day's LOB and/or MBO Parquet files with column projection.

    Args:
        data_dir: Directory containing ``{date}_lob_snapshots.parquet``
            and ``{date}_mbo_events.parquet`` files.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

    def discover_dates(self) -> list[str]:
        """Find all dates that have LOB snapshot files, sorted ascending."""
        dates: list[str] = []
        for f in sorted(self.data_dir.iterdir()):
            if f.name.endswith("_lob_snapshots.parquet"):
                date = extract_date_from_filename(f)
                if date:
                    dates.append(date)
        return dates

    def lob_path(self, date: str) -> Path:
        return self.data_dir / f"{date}_lob_snapshots.parquet"

    def mbo_path(self, date: str) -> Path:
        return self.data_dir / f"{date}_mbo_events.parquet"

    def load_day(
        self,
        date: str,
        *,
        need_lob: bool = True,
        need_mbo: bool = False,
        lob_columns: list[str] | None = None,
        mbo_columns: list[str] | None = None,
        validate: bool = True,
    ) -> DayData:
        """Load one day's data from Parquet files.

        Args:
            date: Trading date ``YYYY-MM-DD``.
            need_lob: Whether to load LOB snapshots.
            need_mbo: Whether to load MBO events.
            lob_columns: Specific LOB columns to load (``None`` = all available).
            mbo_columns: Specific MBO columns to load (``None`` = all).
            validate: Whether to validate schema on first load.

        Returns:
            ``DayData`` with the requested tables loaded.

        Raises:
            FileNotFoundError: If required Parquet files are missing.
            SchemaValidationError: If schema validation fails.
        """
        symbol = "UNKNOWN"
        file_meta: dict[str, Any] = {}
        lob_table: pa.Table | None = None
        mbo_table: pa.Table | None = None

        if need_lob:
            lob_p = self.lob_path(date)
            if not lob_p.exists():
                raise FileNotFoundError(f"LOB file not found: {lob_p}")

            pf = pq.ParquetFile(lob_p)
            arrow_meta = pf.schema_arrow.metadata

            if validate:
                if arrow_meta:
                    file_meta = validate_parquet_metadata(arrow_meta)
                    symbol = file_meta.get(META_SYMBOL, symbol)

                missing = validate_lob_schema(pf.schema_arrow)
                if missing:
                    raise SchemaValidationError(
                        f"LOB file {lob_p.name} missing columns: {missing}"
                    )

            columns = lob_columns
            if columns is not None:
                available = set(pf.schema_arrow.names)
                columns = [c for c in columns if c in available]

            lob_table = pf.read(columns=columns)

            if not file_meta and arrow_meta:
                file_meta = validate_parquet_metadata(arrow_meta)
                symbol = file_meta.get(META_SYMBOL, symbol)

        if need_mbo:
            mbo_p = self.mbo_path(date)
            if not mbo_p.exists():
                raise FileNotFoundError(f"MBO file not found: {mbo_p}")

            pf_mbo = pq.ParquetFile(mbo_p)
            arrow_meta_mbo = pf_mbo.schema_arrow.metadata

            if validate:
                missing = validate_mbo_schema(pf_mbo.schema_arrow)
                if missing:
                    raise SchemaValidationError(
                        f"MBO file {mbo_p.name} missing columns: {missing}"
                    )

            columns = mbo_columns
            if columns is not None:
                available = set(pf_mbo.schema_arrow.names)
                columns = [c for c in columns if c in available]

            mbo_table = pf_mbo.read(columns=columns)

            if not file_meta and arrow_meta_mbo:
                file_meta = validate_parquet_metadata(arrow_meta_mbo)
                symbol = file_meta.get(META_SYMBOL, symbol)

        return DayData(
            date=date,
            symbol=symbol,
            lob=lob_table,
            mbo=mbo_table,
            metadata=file_meta,
        )
