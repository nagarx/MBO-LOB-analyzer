"""AnalysisConfig: top-level configuration for an MBO-LOB-analyzer session.

All thresholds, timescales, and behavioral options are centralized here.
Configs are serializable to YAML/JSON for experiment tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rawlobanalyzer.config.timescale_config import (
    DEFAULT_TIMESCALES,
    TimescaleConfig,
    TradingHours,
)
from rawlobanalyzer.core.constants import DEFAULT_SIGNIFICANCE_ALPHA, TICK_SIZE_USD


def _check_positive(name: str, value: float | int) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value!r}")


def _check_in_range(name: str, value: float, lo: float, hi: float) -> None:
    if not (lo <= value <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}], got {value!r}")


def _check_non_empty(name: str, value: tuple) -> None:
    if len(value) == 0:
        raise ValueError(f"{name} must be non-empty")


def _check_sorted(name: str, value: tuple) -> None:
    for i in range(1, len(value)):
        if value[i] <= value[i - 1]:
            raise ValueError(f"{name} must be strictly increasing, got {value!r}")


@dataclass
class VolatilityThresholds:
    """Configurable parameters for volatility, return, jump, and noise analysis.

    All defaults have academic or empirical justification documented inline.
    """

    signature_scales_seconds: tuple[float, ...] = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0, 300.0)
    """Sampling frequencies in seconds for the volatility signature plot."""

    min_returns_per_bin: int = 10
    """Minimum returns in a time bin for a valid RV estimate."""

    intraday_bin_minutes: int = 1
    """Bin width in minutes for the intraday volatility curve."""

    max_acf_lag: int = 20
    """Maximum lag for volatility autocorrelation."""

    jump_confidence: float = 0.999
    """Confidence level for the BNS jump detection test."""

    jump_threshold_sigma: float = 3.0
    """Sigma multiplier for classifying individual tick returns as jumps."""

    hill_tail_fraction: float = 0.05
    """Fraction of sorted data used for Hill tail index estimation.
    Embrechts, Kluppelberg & Mikosch (1997), Ch. 6."""

    noise_max_scale_seconds: float = 60.0
    """Maximum sampling scale in seconds for microstructure noise analysis."""

    noise_n_scales: int = 20
    """Number of log-spaced scales for the noise signature curve."""

    arch_lags: tuple[int, ...] = (1, 5, 10, 50, 100, 500)
    """Lags for squared-return autocorrelation (ARCH effects)."""

    primary_vol_scale_seconds: float = 5.0
    """Primary scale for vol-of-vol analysis. Selected from signature_scales_seconds."""

    def __post_init__(self) -> None:
        _check_non_empty("signature_scales_seconds", self.signature_scales_seconds)
        _check_sorted("signature_scales_seconds", self.signature_scales_seconds)
        _check_positive("min_returns_per_bin", self.min_returns_per_bin)
        _check_positive("intraday_bin_minutes", self.intraday_bin_minutes)
        _check_positive("max_acf_lag", self.max_acf_lag)
        _check_in_range("jump_confidence", self.jump_confidence, 0.0, 1.0)
        _check_positive("jump_threshold_sigma", self.jump_threshold_sigma)
        _check_in_range("hill_tail_fraction", self.hill_tail_fraction, 0.0, 1.0)
        _check_positive("noise_max_scale_seconds", self.noise_max_scale_seconds)
        _check_positive("noise_n_scales", self.noise_n_scales)
        _check_non_empty("arch_lags", self.arch_lags)
        _check_positive("primary_vol_scale_seconds", self.primary_vol_scale_seconds)
        if self.primary_vol_scale_seconds not in self.signature_scales_seconds:
            raise ValueError(
                f"primary_vol_scale_seconds ({self.primary_vol_scale_seconds}) "
                f"must be one of signature_scales_seconds {self.signature_scales_seconds}"
            )


@dataclass
class SpreadThresholds:
    """Configurable parameters for spread distribution analysis."""

    intraday_bin_minutes: int = 1
    """Bin width in minutes for the intraday spread curve."""

    max_acf_lag: int = 20
    """Maximum lag for spread autocorrelation."""

    wide_spread_ticks: float = 5.0
    """Threshold in ticks for 'wide' spread classification."""

    narrow_spread_ticks: float = 1.5
    """Threshold in ticks for 'narrow' (1-tick) spread classification."""

    width_bucket_ticks: tuple[int, ...] = (1, 2, 5)
    """Tick-width bucket boundaries for spread width classification."""

    tick_size_usd: float = TICK_SIZE_USD
    """US equity tick size in USD (SEC Rule 612, Reg NMS)."""

    def __post_init__(self) -> None:
        _check_positive("intraday_bin_minutes", self.intraday_bin_minutes)
        _check_positive("max_acf_lag", self.max_acf_lag)
        _check_positive("wide_spread_ticks", self.wide_spread_ticks)
        _check_positive("narrow_spread_ticks", self.narrow_spread_ticks)
        if self.narrow_spread_ticks >= self.wide_spread_ticks:
            raise ValueError(
                f"narrow_spread_ticks ({self.narrow_spread_ticks}) must be "
                f"< wide_spread_ticks ({self.wide_spread_ticks})"
            )
        _check_non_empty("width_bucket_ticks", self.width_bucket_ticks)
        _check_sorted("width_bucket_ticks", self.width_bucket_ticks)
        _check_positive("tick_size_usd", self.tick_size_usd)


@dataclass
class DepthThresholds:
    """Configurable parameters for order book depth analysis."""

    n_levels: int = 10
    """Number of LOB levels to analyze."""

    recovery_horizons: tuple[int, ...] = (1, 5, 10, 50)
    """Events after trade for post-trade depth recovery."""

    intraday_bin_minutes: int = 5
    """Bin width in minutes for intraday depth curve (coarser than spread)."""

    def __post_init__(self) -> None:
        _check_positive("n_levels", self.n_levels)
        _check_non_empty("recovery_horizons", self.recovery_horizons)
        _check_sorted("recovery_horizons", self.recovery_horizons)
        _check_positive("intraday_bin_minutes", self.intraday_bin_minutes)


@dataclass
class LiquidityThresholds:
    """Configurable parameters for liquidity and execution cost analysis."""

    realized_spread_horizons_seconds: tuple[float, ...] = (1.0, 5.0, 30.0)
    """Time horizons in seconds for realized spread decomposition."""

    cost_to_trade_sizes: tuple[int, ...] = (100, 500, 1000, 5000, 10000)
    """Order sizes (shares) for cost-to-trade estimation."""

    kyle_lambda_scale_seconds: float = 1.0
    """Sampling scale in seconds for Kyle lambda regression."""

    def __post_init__(self) -> None:
        _check_non_empty("realized_spread_horizons_seconds", self.realized_spread_horizons_seconds)
        _check_sorted("realized_spread_horizons_seconds", self.realized_spread_horizons_seconds)
        _check_non_empty("cost_to_trade_sizes", self.cost_to_trade_sizes)
        _check_sorted("cost_to_trade_sizes", self.cost_to_trade_sizes)
        _check_positive("kyle_lambda_scale_seconds", self.kyle_lambda_scale_seconds)


@dataclass
class FlowThresholds:
    """Configurable parameters for OFI, trade, and order lifecycle analysis.

    Covers three analyzers: OrderFlowAnalyzer, TradeAnalyzer, OrderLifecycleAnalyzer.
    """

    ofi_timescales_seconds: tuple[float, ...] = (1.0, 5.0, 10.0, 30.0, 60.0, 300.0)
    """Timescales in seconds for multi-scale OFI resampling."""

    intraday_bin_minutes: int = 1
    """Bin width in minutes for intraday flow curves (390 bins for RTH)."""

    max_acf_lag: int = 20
    """Maximum lag for OFI autocorrelation."""

    flow_return_max_lag_seconds: float = 60.0
    """Maximum lag in seconds for OFI-return cross-correlation."""

    flow_return_n_lags: int = 12
    """Number of lag bins for OFI-return cross-correlation."""

    large_trade_percentile: float = 95.0
    """Percentile threshold for classifying a trade as 'large'."""

    trade_cluster_gap_seconds: float = 1.0
    """Maximum inter-trade gap in seconds within a trade cluster."""

    order_lifetime_max_seconds: float = 3600.0
    """Maximum tracked order lifetime before eviction (memory bound)."""

    order_lifetime_n_bins: int = 50
    """Number of log-spaced bins for order lifetime histogram."""

    max_active_orders: int = 500_000
    """Cap on simultaneously tracked active orders (memory bound).

    NVDA data shows ~5.3M adds/day but median lifetime is 38ms, so the
    number of *simultaneous* active orders is well below 500K.  LRU
    eviction handles any overflow.  The previous 2M default caused OOM
    on 16 GB machines when running 5-day full-profile analyses.
    """

    def __post_init__(self) -> None:
        _check_non_empty("ofi_timescales_seconds", self.ofi_timescales_seconds)
        _check_sorted("ofi_timescales_seconds", self.ofi_timescales_seconds)
        _check_positive("intraday_bin_minutes", self.intraday_bin_minutes)
        _check_positive("max_acf_lag", self.max_acf_lag)
        _check_positive("flow_return_max_lag_seconds", self.flow_return_max_lag_seconds)
        _check_positive("flow_return_n_lags", self.flow_return_n_lags)
        _check_in_range("large_trade_percentile", self.large_trade_percentile, 0.0, 100.0)
        _check_positive("trade_cluster_gap_seconds", self.trade_cluster_gap_seconds)
        _check_positive("order_lifetime_max_seconds", self.order_lifetime_max_seconds)
        _check_positive("order_lifetime_n_bins", self.order_lifetime_n_bins)
        _check_positive("max_active_orders", self.max_active_orders)


@dataclass
class StatisticalThresholds:
    """Configurable thresholds for statistical tests and classifications.

    All defaults have academic or empirical justification documented inline.
    """

    significance_alpha: float = DEFAULT_SIGNIFICANCE_ALPHA
    """Significance level for hypothesis tests (ADF, KPSS, ARCH, etc.)."""

    use_bonferroni: bool = True
    """Apply Bonferroni correction for multiple comparisons."""

    correlation_weak: float = 0.3
    """Threshold for 'weak' correlation (|r| >= this)."""

    correlation_moderate: float = 0.5
    """Threshold for 'moderate' correlation (|r| >= this)."""

    correlation_strong: float = 0.7
    """Threshold for 'strong' correlation (|r| >= this)."""

    spread_wide_multiplier: float = 3.0
    """Spread is 'wide' if > median * this multiplier."""

    volume_high_multiplier: float = 2.0
    """Volume is 'high' if > daily_mean * this multiplier."""

    large_order_percentile: float = 99.0
    """Orders above this size percentile are classified as 'large'."""

    def __post_init__(self) -> None:
        _check_in_range("significance_alpha", self.significance_alpha, 0.0, 1.0)
        _check_in_range("correlation_weak", self.correlation_weak, 0.0, 1.0)
        _check_in_range("correlation_moderate", self.correlation_moderate, 0.0, 1.0)
        _check_in_range("correlation_strong", self.correlation_strong, 0.0, 1.0)
        if not (self.correlation_weak <= self.correlation_moderate <= self.correlation_strong):
            raise ValueError(
                "Correlation thresholds must be ordered: "
                f"weak ({self.correlation_weak}) <= moderate ({self.correlation_moderate}) "
                f"<= strong ({self.correlation_strong})"
            )
        _check_positive("spread_wide_multiplier", self.spread_wide_multiplier)
        _check_positive("volume_high_multiplier", self.volume_high_multiplier)
        _check_in_range("large_order_percentile", self.large_order_percentile, 0.0, 100.0)

    volatility: VolatilityThresholds = field(default_factory=VolatilityThresholds)
    """Volatility, return, jump, and noise analysis parameters."""

    spread: SpreadThresholds = field(default_factory=SpreadThresholds)
    """Spread distribution analysis parameters."""

    depth: DepthThresholds = field(default_factory=DepthThresholds)
    """Order book depth analysis parameters."""

    liquidity: LiquidityThresholds = field(default_factory=LiquidityThresholds)
    """Liquidity and execution cost analysis parameters."""

    flow: FlowThresholds = field(default_factory=FlowThresholds)
    """OFI, trade, and order lifecycle analysis parameters."""

    @classmethod
    def strict(cls) -> StatisticalThresholds:
        """Conservative thresholds for rigorous analysis."""
        return cls(significance_alpha=0.01, use_bonferroni=True)

    @classmethod
    def lenient(cls) -> StatisticalThresholds:
        """Relaxed thresholds for exploratory analysis."""
        return cls(significance_alpha=0.10, use_bonferroni=False)


@dataclass
class AnalysisConfig:
    """Top-level configuration for an MBO-LOB-analyzer session.

    Attributes:
        data_dir: Directory containing Parquet export files.
        symbol: Ticker symbol (e.g. ``"NVDA"``).
        date_range: Optional inclusive date filter ``(start, end)`` in ``YYYY-MM-DD``.
        timescales: List of analysis granularities.
        trading_hours: Market session definition.
        thresholds: Statistical test thresholds.
        max_rows_per_day: Subsample each day for speed (``None`` = all rows).
        output_dir: Directory for JSON reports and summaries.
        save_json: Write JSON reports to ``output_dir``.
        save_summary: Write text summaries to ``output_dir``.
        verbose: Print progress to stderr.
    """

    data_dir: Path
    symbol: str = "UNKNOWN"
    date_range: tuple[str, str] | None = None
    dates_list: list[str] | None = None

    timescales: list[TimescaleConfig] = field(default_factory=lambda: list(DEFAULT_TIMESCALES))

    trading_hours: TradingHours = field(default_factory=TradingHours.us_equity)

    thresholds: StatisticalThresholds = field(default_factory=StatisticalThresholds)

    max_rows_per_day: int | None = None

    output_dir: Path | None = None
    checkpoint_dir: Path | None = None
    resume: bool = False
    save_json: bool = True
    save_summary: bool = True
    verbose: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize config for JSON/YAML export."""
        return {
            "data_dir": str(self.data_dir),
            "symbol": self.symbol,
            "date_range": list(self.date_range) if self.date_range else None,
            "timescales": [ts.label for ts in self.timescales],
            "trading_hours": self.trading_hours.label,
            "thresholds": {
                "significance_alpha": self.thresholds.significance_alpha,
                "use_bonferroni": self.thresholds.use_bonferroni,
                "volatility": {
                    "signature_scales_seconds": list(self.thresholds.volatility.signature_scales_seconds),
                    "min_returns_per_bin": self.thresholds.volatility.min_returns_per_bin,
                    "intraday_bin_minutes": self.thresholds.volatility.intraday_bin_minutes,
                    "max_acf_lag": self.thresholds.volatility.max_acf_lag,
                    "jump_confidence": self.thresholds.volatility.jump_confidence,
                    "hill_tail_fraction": self.thresholds.volatility.hill_tail_fraction,
                    "noise_max_scale_seconds": self.thresholds.volatility.noise_max_scale_seconds,
                    "noise_n_scales": self.thresholds.volatility.noise_n_scales,
                },
                "spread": {
                    "intraday_bin_minutes": self.thresholds.spread.intraday_bin_minutes,
                    "max_acf_lag": self.thresholds.spread.max_acf_lag,
                    "wide_spread_ticks": self.thresholds.spread.wide_spread_ticks,
                    "tick_size_usd": self.thresholds.spread.tick_size_usd,
                },
                "depth": {
                    "n_levels": self.thresholds.depth.n_levels,
                    "recovery_horizons": list(self.thresholds.depth.recovery_horizons),
                    "intraday_bin_minutes": self.thresholds.depth.intraday_bin_minutes,
                },
                "liquidity": {
                    "realized_spread_horizons_seconds": list(
                        self.thresholds.liquidity.realized_spread_horizons_seconds
                    ),
                    "cost_to_trade_sizes": list(self.thresholds.liquidity.cost_to_trade_sizes),
                    "kyle_lambda_scale_seconds": self.thresholds.liquidity.kyle_lambda_scale_seconds,
                },
                "flow": {
                    "ofi_timescales_seconds": list(self.thresholds.flow.ofi_timescales_seconds),
                    "intraday_bin_minutes": self.thresholds.flow.intraday_bin_minutes,
                    "max_acf_lag": self.thresholds.flow.max_acf_lag,
                    "flow_return_max_lag_seconds": self.thresholds.flow.flow_return_max_lag_seconds,
                    "flow_return_n_lags": self.thresholds.flow.flow_return_n_lags,
                    "large_trade_percentile": self.thresholds.flow.large_trade_percentile,
                    "trade_cluster_gap_seconds": self.thresholds.flow.trade_cluster_gap_seconds,
                    "order_lifetime_max_seconds": self.thresholds.flow.order_lifetime_max_seconds,
                    "order_lifetime_n_bins": self.thresholds.flow.order_lifetime_n_bins,
                    "max_active_orders": self.thresholds.flow.max_active_orders,
                },
            },
            "max_rows_per_day": self.max_rows_per_day,
            "output_dir": str(self.output_dir) if self.output_dir else None,
        }
