"""Trading calendar utilities: weekday names and date parsing.

Provides a single source of truth for weekday-related logic used across
all analyzers. Extensible for holiday calendars in the future.
"""

from __future__ import annotations

from datetime import datetime

WEEKDAY_NAMES: tuple[str, ...] = (
    "Monday", "Tuesday", "Wednesday", "Thursday", "Friday",
)


def weekday_from_date(date_str: str) -> int:
    """Return 0-based weekday index (Monday=0) from a ``YYYY-MM-DD`` string."""
    return datetime.strptime(date_str, "%Y-%m-%d").weekday()


def weekday_name(date_str: str) -> str:
    """Return human-readable weekday name from a ``YYYY-MM-DD`` string."""
    wd = weekday_from_date(date_str)
    return WEEKDAY_NAMES[wd] if wd < len(WEEKDAY_NAMES) else f"day_{wd}"
