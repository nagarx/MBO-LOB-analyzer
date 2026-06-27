"""ReturnAnalyzer: multi-scale return distribution characterization.

Computes per-timescale distribution statistics, tail analysis, autocorrelation,
regime-conditional and day-of-week return patterns from raw LOB mid-prices.

All returns are natural logarithmic: r_t = ln(P_t / P_{t-1}).
Reference: Campbell, Lo, MacKinlay (1997), Ch. 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from rawlobanalyzer.core.calendar import WEEKDAY_NAMES, weekday_from_date
from typing import Any, ClassVar

import numpy as np
from scipy import stats as sp_stats

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.price._return_engine import DayReturns
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EPS, MIN_HILL_SAMPLES, MIN_VAR_CVAR_SAMPLES, QQ_PLOT_POINTS
from rawlobanalyzer.core.statistics import (
    DistributionSummary,
    StreamingDistribution,
    WelfordAccumulator,
    acf as _acf,
    distribution_summary,
    var_cvar,
)
from rawlobanalyzer.core.time_utils import REGIME_LABELS, time_regime
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)



def _hill_estimator(data: np.ndarray, tail_fraction: float) -> float:
    """Hill tail index estimator for the right tail.

    Formula: alpha_hill = (1/k) * sum(ln(X_{n-i+1}) - ln(X_{n-k})) for i=1..k
    where k = floor(n * tail_fraction).

    Embrechts, Kluppelberg & Mikosch (1997), Eq. (6.4).

    Args:
        data: 1D array of positive values (e.g. absolute returns for a tail).
        tail_fraction: Fraction of data to use (top tail_fraction of sorted values).

    Returns:
        Estimated tail index alpha (higher = thinner tail).
        Returns NaN if insufficient data.
    """
    clean = data[np.isfinite(data) & (data > EPS)]
    if len(clean) < MIN_HILL_SAMPLES:
        return np.nan

    k = max(int(len(clean) * tail_fraction), 2)
    sorted_data = np.sort(clean)
    tail = sorted_data[-k:]
    threshold = sorted_data[-(k + 1)] if k < len(sorted_data) else sorted_data[0]

    if threshold <= EPS:
        return np.nan

    log_ratios = np.log(tail) - np.log(threshold)
    mean_log_ratio = float(np.mean(log_ratios))

    if mean_log_ratio <= EPS:
        return np.nan

    return 1.0 / mean_log_ratio



def _qq_data(returns: np.ndarray, n_points: int = QQ_PLOT_POINTS) -> dict[str, list[float]]:
    """Compute QQ-plot data comparing returns to a normal distribution.

    Returns dict with 'theoretical' and 'empirical' quantile arrays.
    """
    clean = returns[np.isfinite(returns)]
    if len(clean) < MIN_VAR_CVAR_SAMPLES:
        return {"theoretical": [], "empirical": []}

    probs = np.linspace(0.5 / n_points, 1.0 - 0.5 / n_points, n_points)
    theoretical = sp_stats.norm.ppf(probs)
    empirical = np.percentile(clean, probs * 100)
    return {
        "theoretical": [float(v) for v in theoretical],
        "empirical": [float(v) for v in empirical],
    }


# ---------------------------------------------------------------------------
# Per-timescale accumulator
# ---------------------------------------------------------------------------

class _ScaleAccumulator:
    """Accumulates return statistics across days for one timescale (streaming).

    Uses O(1) memory per timescale via streaming accumulators instead of
    growing per-day lists.
    """

    def __init__(self, label: str) -> None:
        self.label = label
        self.dist = StreamingDistribution()
        self.daily_var_acc = WelfordAccumulator()

    def add(self, returns: np.ndarray) -> None:
        if len(returns) > 0:
            self.dist.add_batch(returns)
            clean = returns[np.isfinite(returns)]
            dv = float(np.var(clean, ddof=1)) if len(clean) > 1 else 0.0
            self.daily_var_acc.update(dv)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class ReturnReport(BaseReport):
    """Report from the ReturnAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    timescale_stats: dict[str, dict[str, Any]]
    tick_stats: dict[str, Any]
    regime_stats: dict[str, dict[str, Any]]
    weekday_stats: dict[str, dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "timescale_stats": self.timescale_stats,
            "tick_stats": self.tick_stats,
            "regime_stats": self.regime_stats,
            "weekday_stats": self.weekday_stats,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("RETURN ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.tick_stats:
            lines.append(format_subsection("TICK-BY-TICK RETURNS"))
            dist = self.tick_stats.get("distribution", {})
            lines.append(format_kv({
                "Count": dist.get("count", 0),
                "Mean": dist.get("mean", "N/A"),
                "Std": dist.get("std", "N/A"),
                "Skewness": dist.get("skewness", "N/A"),
                "Kurtosis": dist.get("kurtosis", "N/A"),
            }))

        if self.timescale_stats:
            lines.append(format_subsection("PER-TIMESCALE SUMMARY"))
            headers = ["Scale", "Count", "Mean", "Std", "Skew", "Kurt", "VaR5%", "CVaR5%"]
            rows = []
            for label, ts in sorted(self.timescale_stats.items(), key=lambda x: x[0]):
                dist = ts.get("distribution", {})
                risk = ts.get("risk", {})
                rows.append([
                    label,
                    dist.get("count", 0),
                    dist.get("mean", np.nan),
                    dist.get("std", np.nan),
                    dist.get("skewness", np.nan),
                    dist.get("kurtosis", np.nan),
                    risk.get("var_5pct", np.nan),
                    risk.get("cvar_5pct", np.nan),
                ])
            lines.append(format_table(headers, rows))

        if self.regime_stats:
            lines.append(format_subsection("REGIME-CONDITIONAL RETURNS (tick)"))
            headers = ["Regime", "Count", "Mean", "Std", "Skew"]
            rows = []
            for regime, stats in sorted(self.regime_stats.items()):
                d = stats.get("distribution", {})
                rows.append([regime, d.get("count", 0), d.get("mean", np.nan),
                             d.get("std", np.nan), d.get("skewness", np.nan)])
            lines.append(format_table(headers, rows))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@register_analyzer
class ReturnAnalyzer(BaseAnalyzer[ReturnReport]):
    """Multi-scale return distribution analysis from raw LOB mid-prices.

    Computes per-timescale distribution statistics, heavy-tail indicators,
    autocorrelation, regime-conditional distributions, and day-of-week patterns.
    """

    name: ClassVar[str] = "ReturnAnalyzer"
    description: ClassVar[str] = "Return distributions at configurable timescales"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "mid_price", "best_bid", "best_ask",
    ]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._n_days: int = 0
        self._scale_accums: dict[str, _ScaleAccumulator] = {}
        self._tick_dist = StreamingDistribution()
        self._regime_dists: dict[int, StreamingDistribution] = {}
        self._weekday_dists: dict[int, StreamingDistribution] = {}
        self._vol_cfg = config.thresholds.volatility

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        dr = ctx.day_returns
        assert dr is not None, "ReturnAnalyzer requires DayReturns (needs_returns=True)"

        self._symbol = day.symbol
        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        if len(dr.tick_returns) > 0:
            self._tick_dist.add_batch(dr.tick_returns)

        for label, sr in dr.scaled.items():
            if label not in self._scale_accums:
                self._scale_accums[label] = _ScaleAccumulator(label)
            self._scale_accums[label].add(sr.returns)

        if len(dr.tick_returns) > 0:
            ts_ns = day.lob_timestamps_ns
            valid_mask = np.isfinite(day.mid_prices) & (day.mid_prices > 0)
            ts_valid = ts_ns[valid_mask]
            regimes = time_regime(
                ts_valid[1:],
                utc_offset_hours=self._utc_off,
            )
            for regime_val in np.unique(regimes):
                rv = int(regime_val)
                mask = regimes == regime_val
                if rv not in self._regime_dists:
                    self._regime_dists[rv] = StreamingDistribution()
                self._regime_dists[rv].add_batch(dr.tick_returns[mask])

        weekday = weekday_from_date(day.date)
        if len(dr.tick_returns) > 0:
            if weekday not in self._weekday_dists:
                self._weekday_dists[weekday] = StreamingDistribution()
            self._weekday_dists[weekday].add_batch(dr.tick_returns)

    def finalize(self) -> ReturnReport:
        tail_frac = self._vol_cfg.hill_tail_fraction
        max_lag = self._vol_cfg.max_acf_lag

        tick_stats = self._compute_streaming_stats(self._tick_dist, tail_frac, max_lag)

        timescale_stats: dict[str, dict[str, Any]] = {}
        for label, acc in self._scale_accums.items():
            timescale_stats[label] = self._compute_streaming_stats(acc.dist, tail_frac, max_lag)

        regime_stats: dict[str, dict[str, Any]] = {}
        for rv, sd in self._regime_dists.items():
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            dist = sd.distribution_summary()
            regime_stats[label] = {"distribution": dist.to_dict()}

        weekday_stats: dict[str, dict[str, Any]] = {}
        for wd, sd in self._weekday_dists.items():
            name = WEEKDAY_NAMES[wd] if wd < len(WEEKDAY_NAMES) else f"day_{wd}"
            dist = sd.distribution_summary()
            weekday_stats[name] = {"distribution": dist.to_dict()}

        return ReturnReport(
            symbol=self._symbol,
            n_days=self._n_days,
            timescale_stats=timescale_stats,
            tick_stats=tick_stats,
            regime_stats=regime_stats,
            weekday_stats=weekday_stats,
        )

    @staticmethod
    def _compute_streaming_stats(
        sd: StreamingDistribution,
        tail_frac: float,
        max_lag: int,
    ) -> dict[str, Any]:
        """Compute full statistics from a StreamingDistribution's reservoir."""
        if sd.count == 0:
            return {}

        sample = sd.sample()
        dist = sd.distribution_summary()

        var_5, cvar_5 = var_cvar(sample, alpha=0.05)
        var_1, cvar_1 = var_cvar(sample, alpha=0.01)

        neg_rets = -sample[sample < 0]
        pos_rets = sample[sample > 0]

        hill_left = _hill_estimator(neg_rets, tail_frac)
        hill_right = _hill_estimator(pos_rets, tail_frac)

        acf_vals = _acf(sample, max_lag)

        qq = _qq_data(sample)

        return {
            "distribution": dist.to_dict(),
            "risk": {
                "var_5pct": var_5,
                "cvar_5pct": cvar_5,
                "var_1pct": var_1,
                "cvar_1pct": cvar_1,
            },
            "tails": {
                "hill_left": hill_left,
                "hill_right": hill_right,
            },
            "autocorrelation": {
                "lags": list(range(1, len(acf_vals) + 1)),
                "values": [float(v) for v in acf_vals],
            },
            "qq_plot": qq,
        }
