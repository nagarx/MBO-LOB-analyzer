"""Schema constants and validation for MBO-LOB-reconstructor Parquet exports.

Defines the exact data contract between the Rust exporter and this analyzer.
All column names, types, and metadata keys are centralized here as the single
source of truth -- never hardcode schema details elsewhere.

Schema version: 1.0
Source: mbo-lob-reconstructor
Price unit: nanodollars (int64, divide by 1e9 for USD)
Timestamp unit: nanoseconds since epoch (UTC)
Size unit: shares (uint32)

**MBO Trade Pairing**:
Databento MBO data emits *two* events per physical trade: one for the
aggressor (``order_id=0``) and one for the passive fill (``order_id!=0``).
The MBO-LOB-reconstructor exports both with ``action='T'`` (merging
the original Databento 'T' and 'F' actions).  The flow engine filters
to aggressor-only events (``order_id == 0``) to avoid double-counting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pyarrow as pa

SCHEMA_VERSION = "1.0"
SOURCE_TAG = "mbo-lob-reconstructor"

# --- LOB Snapshot columns ---

LOB_CORE_COLUMNS: list[str] = [
    "timestamp_ns",
    "sequence",
    "levels",
    "best_bid",
    "best_ask",
    "bid_prices",
    "bid_sizes",
    "ask_prices",
    "ask_sizes",
    "delta_ns",
    "triggering_action",
    "triggering_side",
]

LOB_DERIVED_COLUMNS: list[str] = [
    "mid_price",
    "spread",
    "spread_bps",
    "microprice",
    "total_bid_volume",
    "total_ask_volume",
    "depth_imbalance",
    "book_consistency",
]

LOB_ALL_COLUMNS: list[str] = LOB_CORE_COLUMNS + LOB_DERIVED_COLUMNS

# --- MBO Event columns ---

MBO_COLUMNS: list[str] = [
    "timestamp_ns",
    "order_id",
    "action",
    "side",
    "price",
    "size",
]

# --- Expected PyArrow types (for validation) ---

LOB_SCHEMA = pa.schema([
    pa.field("timestamp_ns", pa.int64(), nullable=False),
    pa.field("sequence", pa.uint64(), nullable=False),
    pa.field("levels", pa.uint8(), nullable=False),
    pa.field("best_bid", pa.int64(), nullable=True),
    pa.field("best_ask", pa.int64(), nullable=True),
    pa.field("bid_prices", pa.list_(pa.int64(), 10), nullable=False),
    pa.field("bid_sizes", pa.list_(pa.uint32(), 10), nullable=False),
    pa.field("ask_prices", pa.list_(pa.int64(), 10), nullable=False),
    pa.field("ask_sizes", pa.list_(pa.uint32(), 10), nullable=False),
    pa.field("delta_ns", pa.uint64(), nullable=False),
    pa.field("triggering_action", pa.uint8(), nullable=True),
    pa.field("triggering_side", pa.uint8(), nullable=True),
    pa.field("mid_price", pa.float64(), nullable=True),
    pa.field("spread", pa.float64(), nullable=True),
    pa.field("spread_bps", pa.float64(), nullable=True),
    pa.field("microprice", pa.float64(), nullable=True),
    pa.field("total_bid_volume", pa.uint64(), nullable=False),
    pa.field("total_ask_volume", pa.uint64(), nullable=False),
    pa.field("depth_imbalance", pa.float64(), nullable=True),
    pa.field("book_consistency", pa.uint8(), nullable=False),
])

MBO_SCHEMA = pa.schema([
    pa.field("timestamp_ns", pa.int64(), nullable=True),
    pa.field("order_id", pa.uint64(), nullable=False),
    pa.field("action", pa.uint8(), nullable=False),
    pa.field("side", pa.uint8(), nullable=False),
    pa.field("price", pa.int64(), nullable=False),
    pa.field("size", pa.uint32(), nullable=False),
])

# --- Action / Side enums (byte values from Rust) ---

ACTION_ADD = 65       # b'A'
ACTION_CANCEL = 67    # b'C'
ACTION_MODIFY = 77    # b'M'
ACTION_TRADE = 84     # b'T'
ACTION_FILL = 70      # b'F'
ACTION_CLEAR = 82     # b'R'
ACTION_NONE = 78      # b'N'

SIDE_BID = 66         # b'B'
SIDE_ASK = 65         # b'A'
SIDE_NONE = 78        # b'N'

ACTION_LABELS: dict[int, str] = {
    ACTION_ADD: "Add",
    ACTION_CANCEL: "Cancel",
    ACTION_MODIFY: "Modify",
    ACTION_TRADE: "Trade",
    ACTION_FILL: "Fill",
    ACTION_CLEAR: "Clear",
    ACTION_NONE: "None",
}

SIDE_LABELS: dict[int, str] = {
    SIDE_BID: "Bid",
    SIDE_ASK: "Ask",
    SIDE_NONE: "None",
}

# --- Book consistency values ---

BOOK_VALID = 0
BOOK_EMPTY = 1
BOOK_LOCKED = 2
BOOK_CROSSED = 3

BOOK_CONSISTENCY_LABELS: dict[int, str] = {
    BOOK_VALID: "Valid",
    BOOK_EMPTY: "Empty",
    BOOK_LOCKED: "Locked",
    BOOK_CROSSED: "Crossed",
}

# --- Metadata keys ---

META_SCHEMA_VERSION = "schema_version"
META_SOURCE = "source"
META_SYMBOL = "symbol"
META_DATE = "date"
META_PRICE_UNIT = "price_unit"
META_LOB_LEVELS = "lob_levels"
META_TIMESTAMP_UNIT = "timestamp_unit"


@dataclass(frozen=True)
class ParquetFileInfo:
    """Validated metadata from a Parquet file."""

    path: str
    schema_version: str
    source: str
    symbol: str
    date: str
    num_rows: int
    num_columns: int
    is_lob: bool


class SchemaValidationError(Exception):
    """Raised when a Parquet file does not match the expected schema."""


def validate_parquet_metadata(
    metadata: dict[bytes, bytes] | None,
    *,
    expected_source: str = SOURCE_TAG,
) -> dict[str, str]:
    """Extract and validate file-level metadata from Parquet key-value pairs.

    Args:
        metadata: Raw metadata dict from ``ParquetFile.metadata.metadata``.
        expected_source: Expected value of the ``source`` key.

    Returns:
        Decoded metadata dict (str keys and values).

    Raises:
        SchemaValidationError: If required metadata is missing or mismatched.
    """
    if metadata is None:
        raise SchemaValidationError("Parquet file has no metadata")

    decoded: dict[str, str] = {}
    for k, v in metadata.items():
        key = k.decode("utf-8") if isinstance(k, bytes) else k
        val = v.decode("utf-8") if isinstance(v, bytes) else v
        decoded[key] = val

    source = decoded.get(META_SOURCE, "")
    if source != expected_source:
        raise SchemaValidationError(
            f"Expected source={expected_source!r}, got {source!r}"
        )

    return decoded


def validate_lob_schema(schema: pa.Schema) -> list[str]:
    """Check that a LOB Parquet file's schema has the required columns.

    Returns a list of missing column names (empty if valid).
    """
    actual_names = set(schema.names)
    return [c for c in LOB_CORE_COLUMNS if c not in actual_names]


def validate_mbo_schema(schema: pa.Schema) -> list[str]:
    """Check that an MBO Parquet file's schema has the required columns.

    Returns a list of missing column names (empty if valid).
    """
    actual_names = set(schema.names)
    return [c for c in MBO_COLUMNS if c not in actual_names]
