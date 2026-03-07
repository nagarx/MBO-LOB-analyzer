"""VolatilityAnalyzer: multi-scale realized volatility characterization.

Computes the volatility signature plot, intraday vol curve, vol-of-vol,
volatility autocorrelation/half-life, overnight vs. intraday decomposition,
regime-conditional RV, weekly patterns, and ARCH effects from raw LOB data.

Core formula:
    RV_daily = sum(r_i^2)  for returns r_i at a given sampling scale
    AnnualizedVol = sqrt(RV * 252) * 100  (percentage)

Note: RV already sums all squared returns for the day, so no bins-per-day
multiplier is needed.

References:
    - Andersen, Bollerslev, Diebold & Labys (2003), "Modeling and Forecasting
      Realized Volatility", Econometrica, 71(2).
    - Barndorff-Nielsen & Shephard (2002), "Econometric analysis of realized
      volatility and its use in estimating stochastic volatility models",
      JRSS Series B.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing import Any, ClassVar

import numpy as np
from scipy import stats as sp_stats

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.core.calendar import WEEKDAY_NAMES, weekday_from_date
from rawlobanalyzer.core.intraday_accumulator import IntradayCurveAccumulator
from rawlobanalyzer.core.regime_accumulator import RegimeStreamingAccumulator
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.price._return_engine import DayReturns
from rawlobanalyzer.core.statistics import StreamingDistribution, acf as _acf
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import (
    ANNUALIZATION_FACTOR,
    BPS_FACTOR,
    EPS,
    NS_PER_MINUTE,
    NS_PER_SECOND,
    TRADING_MINUTES_PER_DAY,
)
from rawlobanalyzer.core.price_utils import log_returns
from rawlobanalyzer.core.resampler import resample
from rawlobanalyzer.core.statistics import distribution_summary
from rawlobanalyzer.core.time_utils import REGIME_LABELS, rth_mask_utc, time_regime
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)


def _ljung_box(acf_vals: np.ndarray, n_obs: int) -> tuple[float, float]:
    """Ljung-Box Q statistic for testing autocorrelation significance.

    Q = n*(n+2) * sum_{k=1}^{h} rho_k^2 / (n-k)
    Under H0: Q ~ chi2(h)

    Args:
        acf_vals: ACF values at lags 1..h.
        n_obs: Number of observations in the original series.

    Returns:
        (Q_statistic, p_value). Returns (NaN, NaN) if insufficient data.
    """
    h = len(acf_vals)
    if h == 0 or n_obs < h + 1:
        return (np.nan, np.nan)

    lags = np.arange(1, h + 1, dtype=np.float64)
    denominators = np.maximum(n_obs - lags, 1.0)
    q_stat = float(n_obs * (n_obs + 2) * np.sum(acf_vals**2 / denominators))
    p_val = float(1.0 - sp_stats.chi2.cdf(q_stat, df=h))
    return (q_stat, p_val)


# ---------------------------------------------------------------------------
# Per-day accumulators
# ---------------------------------------------------------------------------

@dataclass
class _DayRVRecord:
    """Per-day realized volatility at one sampling scale."""
    date: str
    weekday: int
    rv: float
    n_returns: int


@dataclass
class _OvernightRecord:
    """Open/close prices for overnight decomposition."""
    date: str
    open_price: float
    close_price: float


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class VolatilityReport(BaseReport):
    """Report from the VolatilityAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    signature_plot: dict[str, Any]
    intraday_curve: dict[str, Any]
    vol_of_vol: dict[str, Any]
    vol_autocorrelation: dict[str, Any]
    overnight_intraday: dict[str, Any]
    regime_rv: dict[str, Any]
    weekly_patterns: dict[str, Any]
    arch_effects: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "signature_plot": self.signature_plot,
            "intraday_curve": self.intraday_curve,
            "vol_of_vol": self.vol_of_vol,
            "vol_autocorrelation": self.vol_autocorrelation,
            "overnight_intraday": self.overnight_intraday,
            "regime_rv": self.regime_rv,
            "weekly_patterns": self.weekly_patterns,
            "arch_effects": self.arch_effects,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("VOLATILITY ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.signature_plot.get("scales"):
            lines.append(format_subsection("VOLATILITY SIGNATURE PLOT"))
            headers = ["Scale(s)", "Mean AnnVol(%)", "Std AnnVol(%)"]
            rows = []
            scales = self.signature_plot["scales"]
            means = self.signature_plot["mean_annualized_vol"]
            stds = self.signature_plot["std_annualized_vol"]
            for s, m, sd in zip(scales, means, stds):
                rows.append([s, m, sd])
            lines.append(format_table(headers, rows))

        if self.vol_of_vol:
            lines.append(format_subsection("VOL-OF-VOL"))
            lines.append(format_kv({
                "Mean daily RV": self.vol_of_vol.get("mean", "N/A"),
                "Std daily RV": self.vol_of_vol.get("std", "N/A"),
                "CV": self.vol_of_vol.get("cv", "N/A"),
                "Skewness": self.vol_of_vol.get("skewness", "N/A"),
                "Kurtosis": self.vol_of_vol.get("kurtosis", "N/A"),
            }))

        if self.vol_autocorrelation.get("half_life_days") is not None:
            lines.append(format_subsection("VOLATILITY PERSISTENCE"))
            lines.append(format_kv({
                "Half-life (days)": self.vol_autocorrelation.get("half_life_days", "N/A"),
                "ACF lag-1": self.vol_autocorrelation.get("acf_values", [None])[0],
            }))

        if self.overnight_intraday.get("n_days", 0) > 0:
            lines.append(format_subsection("OVERNIGHT vs INTRADAY"))
            lines.append(format_kv({
                "Overnight variance fraction": self.overnight_intraday.get("overnight_fraction", "N/A"),
                "Intraday variance fraction": self.overnight_intraday.get("intraday_fraction", "N/A"),
            }))

        if self.regime_rv:
            lines.append(format_subsection("REGIME-CONDITIONAL ANNUALIZED VOL (%)"))
            headers = ["Regime", "AnnVol(%)", "N_returns"]
            rows = []
            for regime, stats in sorted(self.regime_rv.items()):
                rows.append([regime, stats.get("annualized_vol", np.nan),
                             stats.get("n_returns", 0)])
            lines.append(format_table(headers, rows))

        if self.arch_effects:
            lines.append(format_subsection("ARCH EFFECTS"))
            lines.append(format_kv({
                "Ljung-Box Q": self.arch_effects.get("ljung_box_q", "N/A"),
                "p-value": self.arch_effects.get("ljung_box_pvalue", "N/A"),
                "Significant": self.arch_effects.get("significant", "N/A"),
            }))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@register_analyzer
class VolatilityAnalyzer(BaseAnalyzer[VolatilityReport]):
    """Multi-scale realized volatility characterization from raw LOB data.

    Computes signature plot, intraday vol curve, vol-of-vol, persistence,
    overnight/intraday decomposition, regime-conditional RV, weekly patterns,
    and ARCH effects.
    """

    name: ClassVar[str] = "VolatilityAnalyzer"
    description: ClassVar[str] = "Multi-scale realized volatility characterization"

    lob_columns: ClassVar[list[str] | None] = ["timestamp_ns", "mid_price"]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._vol_cfg = config.thresholds.volatility
        self._n_days: int = 0

        self._sig_rv: dict[float, list[_DayRVRecord]] = {}
        for scale in self._vol_cfg.signature_scales_seconds:
            self._sig_rv[scale] = []

        n_bins = TRADING_MINUTES_PER_DAY // max(self._vol_cfg.intraday_bin_minutes, 1)
        self._intraday_acc = IntradayCurveAccumulator(n_bins)

        self._overnight_records: list[_OvernightRecord] = []

        self._regime_acc = RegimeStreamingAccumulator()

        self._weekday_rv: dict[int, list[float]] = {d: [] for d in range(5)}

        self._sq_returns_dist = StreamingDistribution()

    def get_extra_scales(self) -> tuple[float, ...] | None:
        return self._vol_cfg.signature_scales_seconds

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        dr = ctx.day_returns
        assert dr is not None, "VolatilityAnalyzer requires DayReturns (needs_returns=True)"

        self._symbol = day.symbol
        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        weekday = weekday_from_date(day.date)

        for scale in self._vol_cfg.signature_scales_seconds:
            label = self._scale_label(scale)
            sr = dr.scaled.get(label)
            if sr is not None and len(sr.returns) >= self._vol_cfg.min_returns_per_bin:
                rv = float(np.sum(sr.returns**2))
                self._sig_rv[scale].append(_DayRVRecord(
                    date=day.date, weekday=weekday, rv=rv, n_returns=len(sr.returns),
                ))

        self._accumulate_intraday_curve(day, dr)

        if np.isfinite(dr.open_price) and np.isfinite(dr.close_price):
            self._overnight_records.append(_OvernightRecord(
                date=day.date,
                open_price=dr.open_price,
                close_price=dr.close_price,
            ))

        self._accumulate_regime_rv(day, dr)

        primary_label = self.config.timescales[0].label if self.config.timescales else "1s"
        primary_sr = dr.scaled.get(primary_label)
        if primary_sr is not None and len(primary_sr.returns) >= self._vol_cfg.min_returns_per_bin:
            daily_rv = float(np.sum(primary_sr.returns**2))
            if weekday < 5:
                self._weekday_rv[weekday].append(daily_rv)

        if len(dr.tick_returns) > 0:
            self._sq_returns_dist.add_batch(dr.tick_returns**2)

    def finalize(self) -> VolatilityReport:
        return VolatilityReport(
            symbol=self._symbol,
            n_days=self._n_days,
            signature_plot=self._finalize_signature(),
            intraday_curve=self._finalize_intraday_curve(),
            vol_of_vol=self._finalize_vol_of_vol(),
            vol_autocorrelation=self._finalize_vol_acf(),
            overnight_intraday=self._finalize_overnight(),
            regime_rv=self._finalize_regime_rv(),
            weekly_patterns=self._finalize_weekly(),
            arch_effects=self._finalize_arch(),
        )

    # --- Signature plot ---

    def _finalize_signature(self) -> dict[str, Any]:
        scales: list[float] = []
        mean_vols: list[float] = []
        std_vols: list[float] = []

        for scale in sorted(self._sig_rv.keys()):
            records = self._sig_rv[scale]
            if not records:
                continue

            rvs = np.array([r.rv for r in records])
            ann_vols = np.sqrt(rvs * ANNUALIZATION_FACTOR) * 100

            scales.append(scale)
            mean_vols.append(float(np.mean(ann_vols)))
            std_vols.append(float(np.std(ann_vols, ddof=1)) if len(ann_vols) > 1 else 0.0)

        return {
            "scales": scales,
            "mean_annualized_vol": mean_vols,
            "std_annualized_vol": std_vols,
        }

    # --- Intraday volatility curve ---

    def _accumulate_intraday_curve(self, day: DayData, dr: DayReturns) -> None:
        ts_ns = day.lob_timestamps_ns
        mids = day.mid_prices
        utc_off = self._utc_off

        valid = np.isfinite(mids) & (mids > 0)
        rth = rth_mask_utc(ts_ns, utc_offset_hours=utc_off)
        mask = valid & rth

        ts_rth = ts_ns[mask]
        mids_rth = mids[mask]

        if len(mids_rth) < 2:
            return

        bin_min = self._vol_cfg.intraday_bin_minutes
        res_ns = bin_min * NS_PER_MINUTE

        resampled = resample(ts_rth, mids_rth, res_ns, agg="last", label=f"{bin_min}m")
        filled = resampled.counts > 0
        close_prices = resampled.values[filled]

        if len(close_prices) < 2:
            return

        rets = log_returns(close_prices)
        sq_rets = rets**2

        daily_rv = float(np.sum(sq_rets))
        if daily_rv < EPS:
            return

        normalized = sq_rets / daily_rv

        filled_indices = np.where(filled)[0][1:]
        valid_mask = np.arange(len(filled_indices)) < len(normalized)
        bin_idxs = filled_indices[valid_mask]
        vals = normalized[:len(bin_idxs)]
        self._intraday_acc.add(bin_idxs, vals)

    def _finalize_intraday_curve(self) -> dict[str, Any]:
        return self._intraday_acc.finalize(
            self._vol_cfg.intraday_bin_minutes,
            mean_key="mean_normalized_var",
            std_key="std_normalized_var",
        )

    # --- Vol-of-Vol ---

    def _finalize_vol_of_vol(self) -> dict[str, Any]:
        primary_scale = self._primary_vol_scale()

        records = self._sig_rv.get(primary_scale, [])
        if len(records) < 2:
            return {}

        daily_rvs = np.array([r.rv for r in records])
        dist = distribution_summary(daily_rvs)

        mean_rv = float(np.mean(daily_rvs))
        cv = float(np.std(daily_rvs, ddof=1) / mean_rv) if mean_rv > EPS else np.nan

        return {
            **dist.to_dict(),
            "cv": cv,
            "daily_rvs": [float(v) for v in daily_rvs],
            "dates": [r.date for r in records],
        }

    # --- Volatility autocorrelation ---

    def _finalize_vol_acf(self) -> dict[str, Any]:
        primary_scale = self._primary_vol_scale()

        records = self._sig_rv.get(primary_scale, [])
        if len(records) < 3:
            return {"lags": [], "acf_values": [], "half_life_days": None}

        daily_rvs = np.array([r.rv for r in records])
        max_lag = min(self._vol_cfg.max_acf_lag, len(daily_rvs) - 1)
        acf_vals = _acf(daily_rvs, max_lag)

        half_life: int | None = None
        for i, v in enumerate(acf_vals):
            if v < 0.5:
                half_life = i + 1
                break

        return {
            "lags": list(range(1, len(acf_vals) + 1)),
            "acf_values": [float(v) for v in acf_vals],
            "half_life_days": half_life,
        }

    # --- Overnight vs. Intraday ---

    def _finalize_overnight(self) -> dict[str, Any]:
        if len(self._overnight_records) < 2:
            return {"n_days": 0}

        overnight_rets: list[float] = []
        intraday_rets: list[float] = []

        for i in range(1, len(self._overnight_records)):
            prev = self._overnight_records[i - 1]
            curr = self._overnight_records[i]

            if prev.close_price > EPS and curr.open_price > EPS:
                overnight_rets.append(np.log(curr.open_price / prev.close_price))
            if curr.open_price > EPS and curr.close_price > EPS:
                intraday_rets.append(np.log(curr.close_price / curr.open_price))

        if not overnight_rets or not intraday_rets:
            return {"n_days": 0}

        overnight_var = float(np.var(overnight_rets, ddof=1)) if len(overnight_rets) > 1 else 0.0
        intraday_var = float(np.var(intraday_rets, ddof=1)) if len(intraday_rets) > 1 else 0.0
        total_var = overnight_var + intraday_var

        return {
            "overnight_variance": overnight_var,
            "intraday_variance": intraday_var,
            "total_variance": total_var,
            "overnight_fraction": overnight_var / total_var if total_var > EPS else np.nan,
            "intraday_fraction": intraday_var / total_var if total_var > EPS else np.nan,
            "mean_overnight_return_bps": float(np.mean(overnight_rets)) * BPS_FACTOR,
            "mean_intraday_return_bps": float(np.mean(intraday_rets)) * BPS_FACTOR,
            "n_days": len(overnight_rets),
        }

    # --- Regime-conditional RV ---

    def _accumulate_regime_rv(self, day: DayData, dr: DayReturns) -> None:
        if len(dr.tick_returns) == 0:
            return

        ts_ns = day.lob_timestamps_ns
        valid = np.isfinite(day.mid_prices) & (day.mid_prices > 0)
        ts_valid = ts_ns[valid]

        if len(ts_valid) < 2:
            return

        regimes = time_regime(
            ts_valid[1:],
            utc_offset_hours=self._utc_off,
        )
        for rv in np.unique(regimes):
            rv_int = int(rv)
            mask = regimes == rv
            chunk = dr.tick_returns[mask]
            self._regime_acc.add(rv_int, chunk)

    def _finalize_regime_rv(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rv, bucket in self._regime_acc.items():
            n_rets = bucket.count
            if n_rets < self._vol_cfg.min_returns_per_bin:
                continue

            rv_val = bucket.sum_of_squares
            mean_ret = bucket.sum / n_rets
            variance = (bucket.sum_of_squares / n_rets) - mean_ret ** 2
            std_ret = float(np.sqrt(max(variance * n_rets / (n_rets - 1), 0.0))) if n_rets > 1 else 0.0

            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            result[label] = {
                "rv": rv_val,
                "annualized_vol": float(np.sqrt(rv_val / max(1, self._n_days) * ANNUALIZATION_FACTOR) * 100),
                "n_returns": n_rets,
                "mean_return": mean_ret,
                "std_return": std_ret,
            }

        return result

    # --- Weekly patterns ---

    def _finalize_weekly(self) -> dict[str, Any]:
        result: dict[str, Any] = {}

        for wd in range(5):
            rvs = self._weekday_rv.get(wd, [])
            if not rvs:
                continue
            rv_arr = np.array(rvs)
            result[WEEKDAY_NAMES[wd]] = {
                "n_days": len(rvs),
                "mean_rv": float(np.mean(rv_arr)),
                "std_rv": float(np.std(rv_arr, ddof=1)) if len(rvs) > 1 else 0.0,
                "median_rv": float(np.median(rv_arr)),
            }

        return result

    # --- ARCH effects ---

    def _finalize_arch(self) -> dict[str, Any]:
        if self._sq_returns_dist.count < 100:
            return {}

        clean = self._sq_returns_dist.sample()
        if len(clean) < 100:
            return {}

        max_lag_needed = max(self._vol_cfg.arch_lags) if self._vol_cfg.arch_lags else 20
        max_lag_actual = min(max_lag_needed, len(clean) - 1)

        if max_lag_actual < 1:
            return {}

        acf_vals = _acf(clean, max_lag_actual)

        arch_acf: dict[str, float] = {}
        for lag in self._vol_cfg.arch_lags:
            if lag <= len(acf_vals):
                arch_acf[f"lag_{lag}"] = float(acf_vals[lag - 1])

        test_lags = [l for l in self._vol_cfg.arch_lags if l <= len(acf_vals)]
        if test_lags:
            test_acf = np.array([acf_vals[l - 1] for l in test_lags])
            q_stat, p_val = _ljung_box(test_acf, len(clean))
        else:
            q_stat, p_val = np.nan, np.nan

        return {
            "squared_return_acf": arch_acf,
            "ljung_box_q": q_stat,
            "ljung_box_pvalue": p_val,
            "significant": p_val < self.config.thresholds.significance_alpha if np.isfinite(p_val) else None,
            "n_observations": len(clean),
        }

    # --- Helpers ---

    def _primary_vol_scale(self) -> float:
        """Pick the primary volatility scale from config, with fallback."""
        target = self._vol_cfg.primary_vol_scale_seconds
        scales = self._vol_cfg.signature_scales_seconds
        if target in scales:
            return target
        return scales[-1] if scales else target

    @staticmethod
    def _scale_label(scale_seconds: float) -> str:
        """Convert scale in seconds to the label format used by _return_engine."""
        if scale_seconds < 1.0:
            ms = scale_seconds * 1000
            return f"{int(ms)}ms" if ms == int(ms) else f"{ms:.0f}ms"
        if scale_seconds < 60:
            return f"{int(scale_seconds)}s" if scale_seconds == int(scale_seconds) else f"{scale_seconds:.1f}s"
        minutes = scale_seconds / 60
        return f"{int(minutes)}m" if minutes == int(minutes) else f"{minutes:.1f}m"
