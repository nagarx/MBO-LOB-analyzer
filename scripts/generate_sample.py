#!/usr/bin/env python3
"""Generate a stratified date sample from an MBO-LOB export directory.

Produces a text file with one date per line, suitable for ``--dates-file``.

Strategy:
  1. Read all available dates from the export directory.
  2. Compute a proxy for daily volume (MBO file size as a fast heuristic).
  3. Stratify by month (proportional representation).
  4. Within each month, stratify by volume quartile.
  5. Ensure all 5 weekdays are represented.
  6. Force-include the top-K and bottom-K days by volume (extremes).
  7. Target ~N total days (default 50).

Usage:
    python scripts/generate_sample.py \\
        --data-dir /path/to/exports/ \\
        --output configs/sampling/nvda_stratified_50.txt \\
        --target 50
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


def discover_dates_with_volume(data_dir: Path) -> list[tuple[str, int]]:
    """Return (date, volume_proxy) pairs sorted by date.

    Uses MBO file size as a fast proxy for daily volume. Falls back to LOB
    file size if MBO is unavailable.
    """
    results: dict[str, int] = {}
    for f in sorted(data_dir.glob("*_lob_snapshots.parquet")):
        date = f.name.split("_")[0]
        mbo_file = data_dir / f"{date}_mbo_events.parquet"
        size = mbo_file.stat().st_size if mbo_file.exists() else f.stat().st_size
        results[date] = size
    return sorted(results.items())


def generate_stratified_sample(
    date_volume: list[tuple[str, int]],
    target: int = 50,
    extreme_k: int = 5,
    seed: int = 42,
) -> list[str]:
    """Select a stratified sample of dates."""
    rng = np.random.default_rng(seed)

    dates = [d for d, _ in date_volume]
    volumes = np.array([v for _, v in date_volume], dtype=np.float64)

    forced: set[str] = set()
    sorted_by_vol = sorted(range(len(dates)), key=lambda i: volumes[i])
    for i in sorted_by_vol[:extreme_k]:
        forced.add(dates[i])
    for i in sorted_by_vol[-extreme_k:]:
        forced.add(dates[i])

    by_month: dict[str, list[int]] = defaultdict(list)
    for i, d in enumerate(dates):
        month_key = d[:7]
        by_month[month_key].append(i)

    remaining_budget = max(target - len(forced), 0)
    total_non_forced = sum(
        len([i for i in indices if dates[i] not in forced])
        for indices in by_month.values()
    )

    selected: set[str] = set(forced)

    for month_key in sorted(by_month.keys()):
        indices = [i for i in by_month[month_key] if dates[i] not in forced]
        if not indices or total_non_forced == 0:
            continue

        month_budget = max(1, round(remaining_budget * len(indices) / total_non_forced))
        month_budget = min(month_budget, len(indices))

        month_vols = volumes[indices]
        quartiles = np.percentile(month_vols, [25, 50, 75])

        quartile_bins: list[list[int]] = [[] for _ in range(4)]
        for idx in indices:
            v = volumes[idx]
            if v <= quartiles[0]:
                quartile_bins[0].append(idx)
            elif v <= quartiles[1]:
                quartile_bins[1].append(idx)
            elif v <= quartiles[2]:
                quartile_bins[2].append(idx)
            else:
                quartile_bins[3].append(idx)

        per_quartile = max(1, month_budget // 4)
        for qbin in quartile_bins:
            if not qbin:
                continue
            n_pick = min(per_quartile, len(qbin))
            picks = rng.choice(qbin, size=n_pick, replace=False)
            for p in picks:
                selected.add(dates[p])

    weekdays_present = set()
    for d in selected:
        weekdays_present.add(datetime.strptime(d, "%Y-%m-%d").weekday())

    for wd in range(5):
        if wd not in weekdays_present:
            candidates = [
                d for d in dates
                if d not in selected and datetime.strptime(d, "%Y-%m-%d").weekday() == wd
            ]
            if candidates:
                selected.add(rng.choice(candidates))

    return sorted(selected)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a stratified date sample for MBO-LOB analysis",
    )
    parser.add_argument(
        "--data-dir", "-d", type=Path, required=True,
        help="Export directory with Parquet files",
    )
    parser.add_argument(
        "--output", "-o", type=Path, required=True,
        help="Output text file (one date per line)",
    )
    parser.add_argument(
        "--target", "-n", type=int, default=50,
        help="Target number of dates (default: 50)",
    )
    parser.add_argument(
        "--extreme-k", type=int, default=5,
        help="Force-include top-K and bottom-K volume days (default: 5)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        print(f"Error: directory not found: {args.data_dir}", file=sys.stderr)
        return 1

    date_volume = discover_dates_with_volume(args.data_dir)
    if not date_volume:
        print(f"Error: no Parquet files found in {args.data_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(date_volume)} trading days in {args.data_dir}")

    sample = generate_stratified_sample(
        date_volume,
        target=args.target,
        extreme_k=args.extreme_k,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(sample) + "\n")

    months = defaultdict(int)
    for d in sample:
        months[d[:7]] += 1

    print(f"Selected {len(sample)} dates (target: {args.target})")
    print(f"Month distribution: {dict(sorted(months.items()))}")
    print(f"Written to: {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
