"""DayContext: per-day shared computation context for the orchestrator.

The orchestrator creates a ``DayContext`` for each trading day, pre-computes
expensive shared artifacts (e.g. ``DayReturns``), and passes the context to
every analyzer. This eliminates redundant computation when multiple analyzers
consume the same intermediate data.

Ownership: the orchestrator creates and owns the context; analyzers read from it.
Lifetime: one context per day, garbage-collected after all analyzers finish.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rawlobanalyzer.analysis.flow._flow_engine import DayFlow
    from rawlobanalyzer.analysis.price._return_engine import DayReturns
    from rawlobanalyzer.analysis.spread._spread_engine import DaySpreads
    from rawlobanalyzer.io.loader import DayData


@dataclass
class DayContext:
    """Per-day shared context passed to every analyzer by the orchestrator.

    Attributes:
        day: The raw LOB/MBO data for this trading day.
        day_epoch_ns: Midnight UTC of the trading day in nanoseconds since
            epoch. Used to build canonical RTH grids for cross-series alignment.
        utc_offset_hours: US Eastern UTC offset for this day (-4 EDT, -5 EST).
            Computed automatically from the date by the orchestrator so that
            all RTH masks, regime classifications, and grid alignments use the
            correct DST-aware offset.
        day_returns: Pre-computed multi-scale returns, or ``None`` if no
            analyzer in the current batch requires them. Analyzers that
            set ``needs_returns = True`` may assume this is not ``None``.
        day_spreads: Pre-computed multi-scale spreads, or ``None`` if no
            analyzer in the current batch requires them. Analyzers that
            set ``needs_spreads = True`` may assume this is not ``None``.
        day_flow: Pre-computed OFI, signed trades, and BBO deltas, or ``None``
            if no analyzer in the current batch requires them. Analyzers that
            set ``needs_flow = True`` may assume this is not ``None``.
    """

    day: DayData
    day_epoch_ns: int = 0
    utc_offset_hours: int = -5
    day_returns: DayReturns | None = None
    day_spreads: DaySpreads | None = None
    day_flow: DayFlow | None = None
