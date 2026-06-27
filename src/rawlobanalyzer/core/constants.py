"""Named constants with citations for MBO-LOB-analyzer.

Every magic number lives here with a clear rationale. Never use raw literals
in analysis code -- import from this module instead.
"""

from __future__ import annotations

# Epsilon for division guards (prevents division by zero in float64 arithmetic).
# Chosen as 10x machine epsilon for float64 to avoid near-zero instability.
EPS: float = 1e-15

# --- Price units ---

NANODOLLARS_PER_DOLLAR: int = 1_000_000_000
"""Conversion factor: 1 USD = 1e9 nanodollars (int64 fixed-point in Parquet)."""

# US equity tick size: $0.01 for stocks >= $1.00 (SEC Rule 612, Reg NMS).
TICK_SIZE_USD: float = 0.01
TICK_SIZE_NANODOLLARS: int = 10_000_000

PRICE_LEVEL_TOLERANCE_USD: float = TICK_SIZE_USD / 2.0
"""Half-tick tolerance for classifying a trade as 'at bid' or 'at ask'.

Nanodollar-to-USD conversion (int64 / 1e9) introduces float rounding of
order ~1e-10, far below this 0.005 threshold.  A trade within half a tick
of the BBO level is economically indistinguishable from being at that level.
"""

# --- Time units ---

NS_PER_SECOND: int = 1_000_000_000
NS_PER_MILLISECOND: int = 1_000_000
NS_PER_MICROSECOND: int = 1_000
NS_PER_MINUTE: int = 60 * NS_PER_SECOND
NS_PER_HOUR: int = 3_600 * NS_PER_SECOND
NS_PER_DAY: int = 86_400 * NS_PER_SECOND

# --- Statistical defaults ---

DEFAULT_SIGNIFICANCE_ALPHA: float = 0.05
"""Default significance level for statistical tests."""

DEFAULT_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)
"""Default percentile breakpoints for distribution summaries."""

# --- Trading calendar ---

TRADING_DAYS_PER_YEAR: int = 252
"""Standard US equity trading days per year (for annualization)."""

TRADING_HOURS_PER_DAY: float = 6.5
"""Regular trading hours for US equities (9:30 AM - 4:00 PM ET)."""

TRADING_SECONDS_PER_DAY: int = int(TRADING_HOURS_PER_DAY * 3600)
"""Seconds in a regular US trading day: 23,400."""

TRADING_MINUTES_PER_DAY: int = int(TRADING_HOURS_PER_DAY * 60)
"""Minutes in a regular US trading day: 390."""

# --- Basis points ---

BPS_FACTOR: int = 10_000
"""Basis-point conversion factor: 1 bps = 1/10,000 = 0.01%."""

# --- Volatility constants ---

ANNUALIZATION_FACTOR: int = TRADING_DAYS_PER_YEAR
"""Alias for TRADING_DAYS_PER_YEAR; used in volatility annualization."""

BIPOWER_MU_1: float = 0.7978845608028654
"""mu_1 = sqrt(2/pi), used in bipower variation.
Barndorff-Nielsen & Shephard (2004), Eq. (4)."""

# --- Operational defaults ---

EMA_PROGRESS_ALPHA: float = 0.3
"""Exponential moving average smoothing factor for progress ETA estimation."""

MIN_ELAPSED_SECONDS: float = 0.001
"""Floor for per-day elapsed time to avoid division by zero in throughput."""

# --- Statistical minimum sample sizes ---

MIN_VAR_CVAR_SAMPLES: int = 10
"""Minimum observations required for VaR/CVaR estimation."""

MIN_HILL_SAMPLES: int = 20
"""Minimum observations for the Hill tail-index estimator."""

QQ_PLOT_POINTS: int = 100
"""Number of quantile points for QQ-plot comparisons."""
