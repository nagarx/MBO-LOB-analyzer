"""JumpRiskAnalyzer: jump detection and characterization from raw LOB data.

Separates continuous (diffusion) price variation from jump variation using
bipower variation, tests for statistically significant jumps, and characterizes
jump size, timing, asymmetry, and clustering.

Core formulas:
    BV = (pi/2) * sum_{i=2}^{n} |r_i| * |r_{i-1}|
    RV = sum_{i=1}^{n} r_i^2
    JV = max(RV - BV, 0)
    Z_BNS = (RV - BV) / sqrt(Theta * max(1/n, TriPV))

References:
    Barndorff-Nielsen & Shephard (2004), "Power and bipower variation with
    stochastic volatility and jumps", Journal of Financial Econometrics, 2(1).
    Barndorff-Nielsen & Shephard (2006), "Econometrics of testing for jumps
    in financial economics using bipower variation", Journal of Financial
    Econometrics, 4(1).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from typing import Any, ClassVar

import numpy as np
from scipy import stats as sp_stats

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.core.calendar import weekday_from_date
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.core.statistics import acf as _acf
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig

logger = logging.getLogger(__name__)
from rawlobanalyzer.core.constants import BIPOWER_MU_1, BPS_FACTOR, EPS
from rawlobanalyzer.core.statistics import StreamingDistribution, WelfordAccumulator, distribution_summary
from rawlobanalyzer.core.time_utils import REGIME_LABELS, time_regime
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)

MU_2_OVER_MU1_SQ = np.pi / 2
"""pi/2 = mu_2 / mu_1^2, scaling constant for bipower variation.
Barndorff-Nielsen & Shephard (2004), Eq. (4)."""

THETA_CONSTANT = (np.pi**2 / 4) + np.pi - 5
"""Theta = pi^2/4 + pi - 5, variance scaling for the BNS test statistic.
Barndorff-Nielsen & Shephard (2006), Eq. (8)."""


def _bipower_variation(returns: np.ndarray) -> float:
    """Bipower variation: BV = (pi/2) * sum(|r_i| * |r_{i-1}|) for i=2..n.

    Barndorff-Nielsen & Shephard (2004), Eq. (4).

    Args:
        returns: 1D array of log returns (length >= 2).

    Returns:
        Bipower variation (non-negative). NaN if insufficient data.
    """
    if len(returns) < 2:
        return np.nan
    abs_r = np.abs(returns)
    return float(MU_2_OVER_MU1_SQ * np.sum(abs_r[1:] * abs_r[:-1]))


def _tripower_quarticity(returns: np.ndarray) -> float:
    """Tripower quarticity estimator for variance of BV.

    TPQ = n * (mu_{4/3})^{-3} * sum(|r_i|^{4/3} * |r_{i-1}|^{4/3} * |r_{i-2}|^{4/3})

    Barndorff-Nielsen & Shephard (2006), Eq. (10).
    mu_{4/3} = 2^{2/3} * Gamma(7/6) / Gamma(1/2).
    """
    n = len(returns)
    if n < 3:
        return np.nan

    from scipy.special import gamma as gamma_func
    mu_4_3 = 2**(2.0 / 3.0) * float(gamma_func(7.0 / 6.0) / np.sqrt(np.pi))
    abs_r = np.abs(returns) ** (4.0 / 3.0)
    tpq_sum = float(np.sum(abs_r[2:] * abs_r[1:-1] * abs_r[:-2]))

    return n * tpq_sum / (mu_4_3**3 * (n - 2))


def _bns_test_statistic(rv: float, bv: float, tpq: float, n: int) -> float:
    """BNS jump test Z-statistic.

    Z = (RV - BV) / sqrt(Theta * max(1/n, TPQ))

    Barndorff-Nielsen & Shephard (2006), Eq. (9).

    Returns:
        Z-statistic. Positive values indicate jump presence.
    """
    if not np.isfinite(tpq) or n < 3:
        return np.nan

    var_estimate = THETA_CONSTANT * max(1.0 / n, tpq)
    if var_estimate <= EPS:
        return np.nan

    return (rv - bv) / np.sqrt(var_estimate)


# ---------------------------------------------------------------------------
# Per-day record
# ---------------------------------------------------------------------------

@dataclass
class _DayJumpRecord:
    date: str
    weekday: int
    rv: float
    bv: float
    tpq: float
    jv: float
    jump_fraction: float
    z_stat: float
    has_jump: bool
    n_returns: int


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class JumpRiskReport(BaseReport):
    """Report from the JumpRiskAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    n_jump_days: int
    daily_records: list[dict[str, Any]]
    jump_fraction_stats: dict[str, Any]
    jump_size_distribution: dict[str, Any]
    jump_timing: dict[str, Any]
    jump_asymmetry: dict[str, Any]
    jump_clustering: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "n_jump_days": self.n_jump_days,
            "daily_records": self.daily_records,
            "jump_fraction_stats": self.jump_fraction_stats,
            "jump_size_distribution": self.jump_size_distribution,
            "jump_timing": self.jump_timing,
            "jump_asymmetry": self.jump_asymmetry,
            "jump_clustering": self.jump_clustering,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("JUMP RISK ANALYSIS REPORT"))
        lines.append(format_kv({
            "Symbol": self.symbol,
            "Days analyzed": self.n_days,
            "Jump days detected": self.n_jump_days,
            "Jump day fraction": f"{self.n_jump_days / max(self.n_days, 1):.1%}",
        }))

        if self.daily_records:
            lines.append(format_subsection("PER-DAY JUMP STATISTICS"))
            headers = ["Date", "RV", "BV", "JV", "Jump%", "Z-stat", "Jump?"]
            rows = []
            for rec in self.daily_records:
                rows.append([
                    rec["date"], rec["rv"], rec["bv"], rec["jv"],
                    rec["jump_fraction"] * 100, rec["z_stat"],
                    "YES" if rec["has_jump"] else "no",
                ])
            lines.append(format_table(headers, rows))

        if self.jump_fraction_stats:
            lines.append(format_subsection("JUMP FRACTION DISTRIBUTION"))
            lines.append(format_kv({
                "Mean JV/RV": self.jump_fraction_stats.get("mean", "N/A"),
                "Median JV/RV": self.jump_fraction_stats.get("percentiles", {}).get("p50", "N/A"),
                "Max JV/RV": self.jump_fraction_stats.get("max", "N/A"),
            }))

        if self.jump_asymmetry:
            lines.append(format_subsection("JUMP ASYMMETRY"))
            lines.append(format_kv({
                "Positive jumps": self.jump_asymmetry.get("n_positive", 0),
                "Negative jumps": self.jump_asymmetry.get("n_negative", 0),
                "Mean pos magnitude": self.jump_asymmetry.get("mean_positive_bps", "N/A"),
                "Mean neg magnitude": self.jump_asymmetry.get("mean_negative_bps", "N/A"),
            }))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@register_analyzer
class JumpRiskAnalyzer(BaseAnalyzer[JumpRiskReport]):
    """Jump detection and characterization from raw LOB mid-prices.

    Uses bipower variation to separate continuous variation from jumps,
    applies the BNS test for significance, and characterizes jump size,
    timing, asymmetry, and clustering.
    """

    name: ClassVar[str] = "JumpRiskAnalyzer"
    description: ClassVar[str] = "Jump detection and risk characterization"

    lob_columns: ClassVar[list[str] | None] = ["timestamp_ns", "mid_price"]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._vol_cfg = config.thresholds.volatility
        self._n_days: int = 0
        self._n_days_skipped: int = 0
        self._records: list[_DayJumpRecord] = []
        self._jump_return_dist = StreamingDistribution()
        self._jump_pos_acc = WelfordAccumulator()
        self._jump_neg_acc = WelfordAccumulator()
        self._jump_regime_counts: dict[int, int] = {}
        self._total_jump_events: int = 0

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        dr = ctx.day_returns
        assert dr is not None, "JumpRiskAnalyzer requires DayReturns (needs_returns=True)"

        self._symbol = day.symbol

        primary_label = self.config.timescales[0].label if self.config.timescales else "1s"
        sr = dr.scaled.get(primary_label)

        if sr is None or len(sr.returns) < 3:
            self._n_days_skipped += 1
            logger.debug("JumpRisk: skipping %s (insufficient returns)", day.date)
            return

        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        returns = sr.returns
        rv = float(np.sum(returns**2))
        bv = _bipower_variation(returns)
        tpq = _tripower_quarticity(returns)

        jv = max(rv - bv, 0.0) if np.isfinite(bv) else 0.0
        jump_frac = jv / rv if rv > EPS else 0.0
        z_stat = _bns_test_statistic(rv, bv, tpq, len(returns))

        z_crit = sp_stats.norm.ppf(self._vol_cfg.jump_confidence)
        has_jump = bool(np.isfinite(z_stat) and z_stat > z_crit)

        weekday = weekday_from_date(day.date)

        self._records.append(_DayJumpRecord(
            date=day.date, weekday=weekday,
            rv=rv, bv=bv, tpq=tpq, jv=jv,
            jump_fraction=jump_frac, z_stat=z_stat,
            has_jump=has_jump, n_returns=len(returns),
        ))

        if has_jump and len(dr.tick_returns) > 0:
            continuous_vol = np.sqrt(bv / len(returns)) if bv > EPS else EPS
            threshold = self._vol_cfg.jump_threshold_sigma * continuous_vol
            tick_abs = np.abs(dr.tick_returns)
            jump_mask = tick_abs > threshold

            if np.any(jump_mask):
                jump_rets = dr.tick_returns[jump_mask]
                self._jump_return_dist.add_batch(jump_rets)
                pos_rets = jump_rets[jump_rets > 0]
                neg_rets = jump_rets[jump_rets < 0]
                if len(pos_rets) > 0:
                    self._jump_pos_acc.update_batch(pos_rets)
                if len(neg_rets) > 0:
                    self._jump_neg_acc.update_batch(neg_rets)

                ts_ns = day.lob_timestamps_ns
                valid = np.isfinite(day.mid_prices) & (day.mid_prices > 0)
                ts_valid = ts_ns[valid]
                if len(ts_valid) > 1:
                    regimes = time_regime(
                        ts_valid[1:],
                        utc_offset_hours=self._utc_off,
                    )
                    if len(regimes) == len(dr.tick_returns):
                        jump_regimes = regimes[jump_mask]
                        for rv_val in np.unique(jump_regimes):
                            rv = int(rv_val)
                            cnt = int(np.sum(jump_regimes == rv_val))
                            self._jump_regime_counts[rv] = self._jump_regime_counts.get(rv, 0) + cnt
                            self._total_jump_events += cnt

    def finalize(self) -> JumpRiskReport:
        n_jump_days = sum(1 for r in self._records if r.has_jump)

        daily_dicts = [
            {
                "date": r.date,
                "rv": r.rv,
                "bv": r.bv,
                "jv": r.jv,
                "jump_fraction": r.jump_fraction,
                "z_stat": r.z_stat,
                "has_jump": r.has_jump,
                "n_returns": r.n_returns,
            }
            for r in self._records
        ]

        jf_arr = np.array([r.jump_fraction for r in self._records]) if self._records else np.array([])
        jf_stats = distribution_summary(jf_arr).to_dict() if len(jf_arr) > 0 else {}

        jump_size_dist: dict[str, Any] = {}
        if self._jump_return_dist.count > 0:
            sample_bps = self._jump_return_dist.sample() * BPS_FACTOR
            dist = distribution_summary(sample_bps)
            jump_size_dist = {
                "unit": "basis_points",
                "n_jumps": self._jump_return_dist.count,
                **dist.to_dict(),
            }

        timing: dict[str, Any] = {}
        if self._total_jump_events > 0:
            regime_counts: dict[str, int] = {}
            for rv, cnt in self._jump_regime_counts.items():
                label = REGIME_LABELS.get(rv, f"regime_{rv}")
                regime_counts[label] = cnt
            timing = {"regime_distribution": regime_counts, "total_jumps": self._total_jump_events}

        asymmetry: dict[str, Any] = {}
        if self._jump_return_dist.count > 0:
            n_pos = self._jump_pos_acc.count
            n_neg = self._jump_neg_acc.count
            asymmetry = {
                "n_positive": n_pos,
                "n_negative": n_neg,
                "mean_positive_bps": self._jump_pos_acc.mean * BPS_FACTOR if n_pos > 0 else np.nan,
                "mean_negative_bps": self._jump_neg_acc.mean * BPS_FACTOR if n_neg > 0 else np.nan,
            }

        clustering: dict[str, Any] = {}
        if len(self._records) >= 3:
            indicators = np.array([1.0 if r.has_jump else 0.0 for r in self._records])
            max_lag = min(self._vol_cfg.max_acf_lag, len(indicators) - 1)
            if max_lag >= 1:
                acf_vals = _acf(indicators, max_lag)
                clustering = {
                    "lags": list(range(1, len(acf_vals) + 1)),
                    "acf_values": [float(v) for v in acf_vals],
                    "jump_rate": float(np.mean(indicators)),
                }

        return JumpRiskReport(
            symbol=self._symbol,
            n_days=self._n_days,
            n_jump_days=n_jump_days,
            daily_records=daily_dicts,
            jump_fraction_stats=jf_stats,
            jump_size_distribution=jump_size_dist,
            jump_timing=timing,
            jump_asymmetry=asymmetry,
            jump_clustering=clustering,
        )
