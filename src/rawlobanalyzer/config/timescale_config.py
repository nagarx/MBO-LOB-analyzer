"""Timescale and trading hours configuration.

TimescaleConfig defines a single analysis granularity (e.g. 1 second, 5 minutes).
TradingHours defines the market session boundaries used for filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

from rawlobanalyzer.core.constants import NS_PER_HOUR, NS_PER_MINUTE, NS_PER_SECOND


@dataclass(frozen=True)
class TimescaleConfig:
    """A single analysis timescale (bin width for resampling).

    Attributes:
        resolution_ns: Bin width in nanoseconds.
        label: Human-readable label (e.g. ``"1s"``, ``"5m"``, ``"1h"``).
        trading_hours_only: If ``True``, filter to RTH before resampling.
    """

    resolution_ns: int
    label: str
    trading_hours_only: bool = True

    @classmethod
    def seconds(cls, n: int, *, rth_only: bool = True) -> TimescaleConfig:
        return cls(resolution_ns=n * NS_PER_SECOND, label=f"{n}s", trading_hours_only=rth_only)

    @classmethod
    def minutes(cls, n: int, *, rth_only: bool = True) -> TimescaleConfig:
        return cls(resolution_ns=n * NS_PER_MINUTE, label=f"{n}m", trading_hours_only=rth_only)

    @classmethod
    def hourly(cls, *, rth_only: bool = True) -> TimescaleConfig:
        return cls(resolution_ns=NS_PER_HOUR, label="1h", trading_hours_only=rth_only)

    @classmethod
    def daily(cls) -> TimescaleConfig:
        return cls(resolution_ns=24 * NS_PER_HOUR, label="1d", trading_hours_only=False)

    @classmethod
    def from_label(cls, label: str, *, rth_only: bool = True) -> TimescaleConfig:
        """Parse a human-readable label into a TimescaleConfig.

        Supported formats: ``"1s"``, ``"5s"``, ``"30s"``, ``"1m"``, ``"5m"``,
        ``"15m"``, ``"1h"``, ``"1d"``.

        Raises:
            ValueError: If the label is unparseable or the numeric part is <= 0.
        """
        label = label.strip().lower()
        if label.endswith("s"):
            n = int(label[:-1])
            if n <= 0:
                raise ValueError(f"Timescale must be positive, got {label!r}")
            return cls.seconds(n, rth_only=rth_only)
        elif label.endswith("m"):
            n = int(label[:-1])
            if n <= 0:
                raise ValueError(f"Timescale must be positive, got {label!r}")
            return cls.minutes(n, rth_only=rth_only)
        elif label.endswith("h"):
            n = int(label[:-1])
            if n <= 0:
                raise ValueError(f"Timescale must be positive, got {label!r}")
            return cls(
                resolution_ns=n * NS_PER_HOUR, label=label, trading_hours_only=rth_only,
            )
        elif label.endswith("d"):
            n = int(label[:-1]) if len(label) > 1 else 1
            if n <= 0:
                raise ValueError(f"Timescale must be positive, got {label!r}")
            return cls(
                resolution_ns=n * 24 * NS_PER_HOUR,
                label=label,
                trading_hours_only=False,
            )
        else:
            raise ValueError(
                f"Cannot parse timescale label {label!r}. "
                "Expected format: '1s', '5m', '1h', '1d', etc."
            )


@dataclass(frozen=True)
class TradingHours:
    """Market session boundaries in UTC fractional hours.

    Attributes:
        rth_open_utc_h: Regular trading hours open (UTC fractional hours).
        rth_close_utc_h: Regular trading hours close (UTC fractional hours).
        ext_open_utc_h: Extended hours open (UTC fractional hours).
        ext_close_utc_h: Extended hours close (UTC fractional hours, may wrap past 24).
        label: Human-readable label.
        utc_offset_hours: Local timezone offset from UTC.
    """

    rth_open_utc_h: float
    rth_close_utc_h: float
    ext_open_utc_h: float
    ext_close_utc_h: float
    label: str
    utc_offset_hours: int

    @classmethod
    def us_equity(cls) -> TradingHours:
        """US equity regular hours (EST, UTC-5). 9:30-16:00 ET = 14:30-21:00 UTC."""
        return cls(
            rth_open_utc_h=14.5,
            rth_close_utc_h=21.0,
            ext_open_utc_h=9.0,
            ext_close_utc_h=25.0,   # 1:00 AM next day UTC
            label="us_equity_est",
            utc_offset_hours=-5,
        )

    @classmethod
    def us_equity_dst(cls) -> TradingHours:
        """US equity during daylight saving (EDT, UTC-4). 9:30-16:00 ET = 13:30-20:00 UTC."""
        return cls(
            rth_open_utc_h=13.5,
            rth_close_utc_h=20.0,
            ext_open_utc_h=8.0,
            ext_close_utc_h=24.0,
            label="us_equity_edt",
            utc_offset_hours=-4,
        )

    @classmethod
    def from_label(cls, label: str) -> TradingHours:
        """Resolve a trading hours preset by label."""
        presets = {
            "us_equity": cls.us_equity,
            "us_equity_est": cls.us_equity,
            "us_equity_rth": cls.us_equity,
            "us_equity_dst": cls.us_equity_dst,
            "us_equity_edt": cls.us_equity_dst,
        }
        factory = presets.get(label.lower())
        if factory is None:
            raise ValueError(
                f"Unknown trading hours label: {label!r}. "
                f"Available: {sorted(presets.keys())}"
            )
        return factory()


DEFAULT_TIMESCALES: list[TimescaleConfig] = [
    TimescaleConfig.seconds(1),
    TimescaleConfig.seconds(5),
    TimescaleConfig.seconds(30),
    TimescaleConfig.minutes(1),
    TimescaleConfig.minutes(5),
    TimescaleConfig.minutes(15),
    TimescaleConfig.hourly(),
]
