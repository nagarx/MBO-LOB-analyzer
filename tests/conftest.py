"""Shared fixtures for MBO-LOB-analyzer tests.

Provides synthetic Parquet files that match the MBO-LOB-reconstructor schema.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_day(
    tmp_path: Path,
    date: str,
    n_rows: int,
    rng: np.random.Generator,
    base_price_nd: int = 118_000_000_000,
    ts_start_utc: int | None = None,
    ts_step: int = 1_000_000,
) -> None:
    """Write synthetic LOB + MBO Parquet files for a single day.

    Args:
        tmp_path: Directory to write into.
        date: Date string ``YYYY-MM-DD``.
        n_rows: Number of events.
        rng: Numpy random generator (deterministic seed).
        base_price_nd: Base best-bid price in nanodollars.
        ts_start_utc: First timestamp in nanoseconds since epoch.
            If None, defaults to 15:00 UTC on that date (within US RTH).
        ts_step: Nanoseconds between events.
    """
    if ts_start_utc is None:
        from datetime import datetime, timezone
        dt = datetime.strptime(date, "%Y-%m-%d").replace(
            hour=15, minute=0, second=0, tzinfo=timezone.utc,
        )
        ts_start_utc = int(dt.timestamp() * 1_000_000_000)

    timestamps = np.arange(
        ts_start_utc, ts_start_utc + n_rows * ts_step, ts_step, dtype=np.int64,
    )

    price_walk = np.cumsum(rng.normal(0, 5_000_000, size=n_rows)).astype(np.int64)
    best_bid = np.full(n_rows, base_price_nd, dtype=np.int64) + price_walk
    best_ask = best_bid + rng.integers(8_000_000, 15_000_000, size=n_rows)

    mid_price = (best_bid.astype(np.float64) + best_ask.astype(np.float64)) / 2e9
    spread = (best_ask.astype(np.float64) - best_bid.astype(np.float64)) / 1e9
    spread_bps = spread / np.where(mid_price > 0, mid_price, 1.0) * 10_000
    microprice = mid_price + rng.normal(0, 0.0005, n_rows)

    bid_prices_flat = np.zeros((n_rows, 10), dtype=np.int64)
    ask_prices_flat = np.zeros((n_rows, 10), dtype=np.int64)
    bid_sizes_flat = np.zeros((n_rows, 10), dtype=np.uint32)
    ask_sizes_flat = np.zeros((n_rows, 10), dtype=np.uint32)

    for i in range(10):
        bid_prices_flat[:, i] = best_bid - i * 10_000_000
        ask_prices_flat[:, i] = best_ask + i * 10_000_000
        bid_sizes_flat[:, i] = rng.integers(100, 5000, size=n_rows).astype(np.uint32)
        ask_sizes_flat[:, i] = rng.integers(100, 5000, size=n_rows).astype(np.uint32)

    total_bid_volume = bid_sizes_flat.sum(axis=1).astype(np.uint64)
    total_ask_volume = ask_sizes_flat.sum(axis=1).astype(np.uint64)
    depth_imb = (total_bid_volume.astype(np.float64) - total_ask_volume.astype(np.float64)) / (
        total_bid_volume.astype(np.float64) + total_ask_volume.astype(np.float64)
    )

    lob_table = pa.table({
        "timestamp_ns": pa.array(timestamps, type=pa.int64()),
        "sequence": pa.array(np.arange(1, n_rows + 1, dtype=np.uint64)),
        "levels": pa.array(np.full(n_rows, 10, dtype=np.uint8)),
        "best_bid": pa.array(best_bid, type=pa.int64()),
        "best_ask": pa.array(best_ask, type=pa.int64()),
        "bid_prices": pa.FixedSizeListArray.from_arrays(
            pa.array(bid_prices_flat.ravel(), type=pa.int64()), 10,
        ),
        "bid_sizes": pa.FixedSizeListArray.from_arrays(
            pa.array(bid_sizes_flat.ravel(), type=pa.uint32()), 10,
        ),
        "ask_prices": pa.FixedSizeListArray.from_arrays(
            pa.array(ask_prices_flat.ravel(), type=pa.int64()), 10,
        ),
        "ask_sizes": pa.FixedSizeListArray.from_arrays(
            pa.array(ask_sizes_flat.ravel(), type=pa.uint32()), 10,
        ),
        "delta_ns": pa.array(np.full(n_rows, ts_step, dtype=np.uint64)),
        "triggering_action": pa.array(rng.choice([65, 67, 84], size=n_rows).astype(np.uint8)),
        "triggering_side": pa.array(rng.choice([65, 66], size=n_rows).astype(np.uint8)),
        "mid_price": pa.array(mid_price, type=pa.float64()),
        "spread": pa.array(spread, type=pa.float64()),
        "spread_bps": pa.array(spread_bps, type=pa.float64()),
        "microprice": pa.array(microprice, type=pa.float64()),
        "total_bid_volume": pa.array(total_bid_volume, type=pa.uint64()),
        "total_ask_volume": pa.array(total_ask_volume, type=pa.uint64()),
        "depth_imbalance": pa.array(depth_imb, type=pa.float64()),
        "book_consistency": pa.array(np.zeros(n_rows, dtype=np.uint8)),
    })

    metadata = {
        b"schema_version": b"1.0",
        b"source": b"mbo-lob-reconstructor",
        b"symbol": b"TEST",
        b"date": date.encode(),
        b"price_unit": b"nanodollars",
        b"lob_levels": b"10",
        b"timestamp_unit": b"nanoseconds_since_epoch",
    }
    lob_table = lob_table.replace_schema_metadata(metadata)
    pq.write_table(lob_table, tmp_path / f"{date}_lob_snapshots.parquet")

    actions = rng.choice([65, 67, 84], size=n_rows).astype(np.uint8)
    order_ids = rng.integers(1, 100_000, size=n_rows).astype(np.uint64)
    order_ids[actions == 84] = 0  # trade events get aggressor order_id=0
    mbo_table = pa.table({
        "timestamp_ns": pa.array(timestamps, type=pa.int64()),
        "order_id": pa.array(order_ids, type=pa.uint64()),
        "action": pa.array(actions),
        "side": pa.array(rng.choice([65, 66], size=n_rows).astype(np.uint8)),
        "price": pa.array(best_bid + rng.integers(-1e8, 1e8, size=n_rows), type=pa.int64()),
        "size": pa.array(rng.integers(1, 10000, size=n_rows).astype(np.uint32)),
    })
    mbo_table = mbo_table.replace_schema_metadata(metadata)
    pq.write_table(mbo_table, tmp_path / f"{date}_mbo_events.parquet")


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create a temp directory with synthetic LOB + MBO Parquet files for one day."""
    rng = np.random.default_rng(42)
    _write_day(tmp_path, "2025-02-03", n_rows=1000, rng=rng)
    return tmp_path


@pytest.fixture
def two_day_data_dir(tmp_path: Path) -> Path:
    """Two consecutive trading days with realistic price variance.

    Needed for overnight decomposition and cross-day statistics.
    Day 1: 2025-02-03 (Monday), Day 2: 2025-02-04 (Tuesday).
    Both days have 5000 events spanning ~5 seconds within RTH.
    ts_step=1_000_000 (1ms) gives 5 seconds of data per day.
    """
    rng = np.random.default_rng(123)
    _write_day(tmp_path, "2025-02-03", n_rows=5000, rng=rng, ts_step=1_000_000)
    _write_day(tmp_path, "2025-02-04", n_rows=5000, rng=rng, ts_step=1_000_000)
    return tmp_path
