"""Nanosecond-precision time utilities and trading hours detection.

All timestamps from MBO-LOB-reconstructor are int64 nanoseconds since epoch (UTC).
US equity regular trading hours are 9:30 - 16:00 ET (14:30 - 21:00 UTC, non-DST).
Extended hours: 4:00 - 20:00 ET (9:00 - 1:00+1 UTC).

DST transitions are handled explicitly: US Eastern observes DST from the second
Sunday in March to the first Sunday in November.
"""

from __future__ import annotations

from datetime import date, timedelta
from functools import lru_cache

import numpy as np

from rawlobanalyzer.core.constants import NS_PER_DAY, NS_PER_HOUR, NS_PER_SECOND

# UTC offsets for US Eastern
_ET_OFFSET_STANDARD_H: int = -5  # EST
_ET_OFFSET_DST_H: int = -4       # EDT

# Regular trading hours in ET (fractional hours)
_RTH_OPEN_ET_H: float = 9.5     # 9:30 AM ET
_RTH_CLOSE_ET_H: float = 16.0   # 4:00 PM ET

# Extended hours in ET
_EXT_OPEN_ET_H: float = 4.0     # 4:00 AM ET
_EXT_CLOSE_ET_H: float = 20.0   # 8:00 PM ET


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the *n*-th occurrence of *weekday* (0=Mon, 6=Sun) in *month*."""
    first = date(year, month, 1)
    offset = (weekday - first.weekday()) % 7
    return first + timedelta(days=offset + 7 * (n - 1))


@lru_cache(maxsize=64)
def _dst_range(year: int) -> tuple[date, date]:
    """US Eastern DST boundaries for *year*.

    DST starts: second Sunday in March at 02:00 local.
    DST ends:   first Sunday in November at 02:00 local.

    For trading-day classification we only need the calendar date, not the
    exact 02:00 transition, because RTH (9:30-16:00 ET) is always well
    after the 02:00 switch.
    """
    start = _nth_weekday(year, 3, 6, 2)   # 2nd Sunday in March
    end = _nth_weekday(year, 11, 6, 1)    # 1st Sunday in November
    return start, end


def utc_offset_for_date(date_str: str) -> int:
    """Return the US Eastern UTC offset (-4 for EDT, -5 for EST) for a date.

    Args:
        date_str: ``YYYY-MM-DD`` trading date string.

    Returns:
        -4 during daylight saving time, -5 otherwise.
    """
    d = date.fromisoformat(date_str)
    dst_start, dst_end = _dst_range(d.year)
    if dst_start <= d < dst_end:
        return _ET_OFFSET_DST_H
    return _ET_OFFSET_STANDARD_H


def ns_to_seconds_since_midnight_utc(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert nanosecond timestamps to seconds since midnight UTC.

    Args:
        timestamps_ns: Array of int64 nanoseconds since epoch.

    Returns:
        Array of float64 seconds since midnight UTC.
    """
    seconds_since_epoch = timestamps_ns.astype(np.float64) / NS_PER_SECOND
    return np.mod(seconds_since_epoch, 86400.0)


def ns_to_hours_since_midnight_utc(timestamps_ns: np.ndarray) -> np.ndarray:
    """Convert nanosecond timestamps to fractional hours since midnight UTC.

    Args:
        timestamps_ns: Array of int64 nanoseconds since epoch.

    Returns:
        Array of float64 hours since midnight UTC (0.0 - 24.0).
    """
    return ns_to_seconds_since_midnight_utc(timestamps_ns) / 3600.0


def rth_mask_utc(
    timestamps_ns: np.ndarray,
    *,
    utc_offset_hours: int = _ET_OFFSET_STANDARD_H,
) -> np.ndarray:
    """Boolean mask for Regular Trading Hours (9:30 - 16:00 ET).

    Args:
        timestamps_ns: Array of int64 nanoseconds since epoch.
        utc_offset_hours: UTC offset for the local market timezone.
            Default: -5 (EST). Use -4 for EDT.

    Returns:
        Boolean array -- ``True`` for events during RTH.
    """
    hours_utc = ns_to_hours_since_midnight_utc(timestamps_ns)
    rth_open_utc = _RTH_OPEN_ET_H - utc_offset_hours
    rth_close_utc = _RTH_CLOSE_ET_H - utc_offset_hours
    return (hours_utc >= rth_open_utc) & (hours_utc < rth_close_utc)


def extended_hours_mask_utc(
    timestamps_ns: np.ndarray,
    *,
    utc_offset_hours: int = _ET_OFFSET_STANDARD_H,
) -> np.ndarray:
    """Boolean mask for extended trading hours (4:00 AM - 8:00 PM ET).

    Args:
        timestamps_ns: Array of int64 nanoseconds since epoch.
        utc_offset_hours: UTC offset for the local market timezone.

    Returns:
        Boolean array -- ``True`` for events during extended hours.
    """
    hours_utc = ns_to_hours_since_midnight_utc(timestamps_ns)
    ext_open_utc = _EXT_OPEN_ET_H - utc_offset_hours
    ext_close_utc = _EXT_CLOSE_ET_H - utc_offset_hours

    if ext_close_utc > 24.0:
        return (hours_utc >= ext_open_utc) | (hours_utc < ext_close_utc - 24.0)
    return (hours_utc >= ext_open_utc) & (hours_utc < ext_close_utc)


def time_regime(
    timestamps_ns: np.ndarray,
    *,
    utc_offset_hours: int = _ET_OFFSET_STANDARD_H,
) -> np.ndarray:
    """Classify timestamps into intraday regimes.

    Regimes (int8):
        0 = pre-market (before RTH open)
        1 = open auction (first 5 min of RTH, 9:30-9:35 ET)
        2 = morning (9:35 - 12:00 ET)
        3 = midday (12:00 - 14:00 ET)
        4 = afternoon (14:00 - 15:45 ET)
        5 = close auction (last 15 min of RTH, 15:45-16:00 ET)
        6 = after-hours (after RTH close)

    Args:
        timestamps_ns: Array of int64 nanoseconds since epoch.
        utc_offset_hours: UTC offset for local market timezone.

    Returns:
        Array of int8 regime labels.
    """
    hours_utc = ns_to_hours_since_midnight_utc(timestamps_ns)
    offset = -utc_offset_hours

    regime = np.full(len(timestamps_ns), 6, dtype=np.int8)  # default: after-hours

    rth_open = _RTH_OPEN_ET_H + offset
    open_end = 9.0 + 35.0 / 60.0 + offset          # 9:35 ET
    morning_end = 12.0 + offset                      # 12:00 ET
    midday_end = 14.0 + offset                       # 14:00 ET
    close_start = 15.0 + 45.0 / 60.0 + offset       # 15:45 ET
    rth_close = _RTH_CLOSE_ET_H + offset             # 16:00 ET

    regime[hours_utc < rth_open] = 0
    mask_open = (hours_utc >= rth_open) & (hours_utc < open_end)
    mask_morning = (hours_utc >= open_end) & (hours_utc < morning_end)
    mask_midday = (hours_utc >= morning_end) & (hours_utc < midday_end)
    mask_afternoon = (hours_utc >= midday_end) & (hours_utc < close_start)
    mask_close = (hours_utc >= close_start) & (hours_utc < rth_close)

    regime[mask_open] = 1
    regime[mask_morning] = 2
    regime[mask_midday] = 3
    regime[mask_afternoon] = 4
    regime[mask_close] = 5

    return regime


REGIME_LABELS: dict[int, str] = {
    0: "pre-market",
    1: "open-auction",
    2: "morning",
    3: "midday",
    4: "afternoon",
    5: "close-auction",
    6: "after-hours",
}


def seconds_to_label(s: float) -> str:
    """Convert a float seconds value to a human-readable timescale label.

    Examples: 0.1 -> ``"100ms"``, 5 -> ``"5s"``, 300 -> ``"5m"``.
    """
    if s < 1.0:
        ms = s * 1000
        if ms == int(ms):
            return f"{int(ms)}ms"
        return f"{ms:.0f}ms"
    if s < 60:
        if s == int(s):
            return f"{int(s)}s"
        return f"{s:.1f}s"
    minutes = s / 60
    if minutes == int(minutes):
        return f"{int(minutes)}m"
    return f"{minutes:.1f}m"


def rth_grid_edges_ns(
    day_epoch_ns: int,
    resolution_ns: int,
    *,
    utc_offset_hours: int = _ET_OFFSET_STANDARD_H,
) -> np.ndarray:
    """Canonical RTH bin edges for a given trading day and resolution.

    Returns a deterministic grid from market open to market close, aligned
    to ``resolution_ns`` boundaries.  The grid depends only on the day and
    the resolution, not on the data, so MBO and LOB engines that resample
    onto the same grid will produce perfectly aligned bins.

    Args:
        day_epoch_ns: Midnight UTC of the trading day in nanoseconds since
            epoch.  Obtain from ``DayData`` timestamps:
            ``ts[0] - (ts[0] % NS_PER_DAY)``.
        resolution_ns: Bin width in nanoseconds.
        utc_offset_hours: UTC offset for the local market timezone.
            Default: -5 (EST).

    Returns:
        Array of int64 bin edges (left edges + one trailing right edge).
        Length is ``n_bins + 1`` where ``n_bins = ceil(rth_duration / resolution)``.
    """
    rth_open_utc_h = _RTH_OPEN_ET_H - utc_offset_hours
    rth_close_utc_h = _RTH_CLOSE_ET_H - utc_offset_hours

    open_ns = day_epoch_ns + int(rth_open_utc_h * NS_PER_HOUR)
    close_ns = day_epoch_ns + int(rth_close_utc_h * NS_PER_HOUR)

    grid_start = (open_ns // resolution_ns) * resolution_ns
    grid_end = ((close_ns - 1) // resolution_ns + 1) * resolution_ns + resolution_ns

    return np.arange(grid_start, grid_end, resolution_ns, dtype=np.int64)


def compute_inter_event_times_ns(timestamps_ns: np.ndarray) -> np.ndarray:
    """Compute inter-event times in nanoseconds.

    Args:
        timestamps_ns: Sorted array of int64 nanoseconds since epoch.

    Returns:
        Array of int64 inter-event times, length ``len(timestamps_ns) - 1``.
    """
    return np.diff(timestamps_ns)
