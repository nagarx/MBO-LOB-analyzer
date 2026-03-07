"""MicrostructureNoiseAnalyzer: noise floor estimation and optimal sampling.

Quantifies microstructure noise in raw LOB mid-prices using the noise
signature curve, two-scale realized volatility, Roll's bid-ask bounce model,
and determines the optimal sampling frequency for downstream feature extraction.

Core formulas:
    Noise variance:  sigma^2_noise = (RV_fast - RV_slow) / (2 * n_fast)
    Roll spread:     s_roll = 2 * sqrt(-gamma_1)  where gamma_1 = Cov(r_t, r_{t-1})
    Optimal freq:    f* = argmin MSE(RV_f) ~ (noise_var / IQ)^{1/3} * n^{2/3}

References:
    Zhang, Mykland & Ait-Sahalia (2005), "A tale of two time scales:
    Determining integrated volatility with noisy high-frequency data",
    JASA, 100(472).
    Roll (1984), "A simple implicit measure of the effective bid-ask spread
    in an efficient market", Journal of Finance, 39(4).
    Hansen & Lunde (2006), "Realized variance and market microstructure noise",
    Journal of Business & Economic Statistics, 24(2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import BPS_FACTOR, EPS, NS_PER_SECOND, TRADING_SECONDS_PER_DAY
from rawlobanalyzer.core.price_utils import log_returns

logger = logging.getLogger(__name__)
from rawlobanalyzer.core.resampler import resample
from rawlobanalyzer.core.statistics import distribution_summary
from rawlobanalyzer.core.time_utils import rth_mask_utc
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)


# ---------------------------------------------------------------------------
# Per-day accumulators
# ---------------------------------------------------------------------------

@dataclass
class _DayNoiseRecord:
    date: str
    noise_var: float
    snr: float
    roll_spread_usd: float
    observed_spread_usd: float
    optimal_freq_seconds: float
    rv_by_scale: dict[float, float]
    n_ticks: int


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

@dataclass
class MicrostructureNoiseReport(BaseReport):
    """Report from the MicrostructureNoiseAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    noise_signature: dict[str, Any]
    noise_variance: dict[str, Any]
    signal_to_noise: dict[str, Any]
    bid_ask_bounce: dict[str, Any]
    optimal_sampling: dict[str, Any]
    daily_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "noise_signature": self.noise_signature,
            "noise_variance": self.noise_variance,
            "signal_to_noise": self.signal_to_noise,
            "bid_ask_bounce": self.bid_ask_bounce,
            "optimal_sampling": self.optimal_sampling,
            "daily_records": self.daily_records,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("MICROSTRUCTURE NOISE REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.noise_signature.get("scales"):
            lines.append(format_subsection("NOISE SIGNATURE (RV vs scale)"))
            headers = ["Scale(s)", "Mean RV"]
            rows = [[s, rv] for s, rv in
                    zip(self.noise_signature["scales"], self.noise_signature["mean_rv"])]
            lines.append(format_table(headers, rows))

        if self.noise_variance:
            lines.append(format_subsection("NOISE VARIANCE"))
            lines.append(format_kv({
                "Mean noise var": self.noise_variance.get("mean", "N/A"),
                "Mean noise std (bps)": self.noise_variance.get("mean_noise_std_bps", "N/A"),
            }))

        if self.signal_to_noise:
            lines.append(format_subsection("SIGNAL-TO-NOISE RATIO"))
            lines.append(format_kv({
                "Mean SNR": self.signal_to_noise.get("mean", "N/A"),
                "Median SNR": self.signal_to_noise.get("median", "N/A"),
            }))

        if self.bid_ask_bounce:
            lines.append(format_subsection("BID-ASK BOUNCE (Roll Model)"))
            lines.append(format_kv({
                "Mean Roll spread ($)": self.bid_ask_bounce.get("mean_roll_spread_usd", "N/A"),
                "Mean observed spread ($)": self.bid_ask_bounce.get("mean_observed_spread_usd", "N/A"),
                "Roll/Observed ratio": self.bid_ask_bounce.get("roll_to_observed_ratio", "N/A"),
            }))

        if self.optimal_sampling:
            lines.append(format_subsection("OPTIMAL SAMPLING FREQUENCY"))
            lines.append(format_kv({
                "Mean optimal freq (s)": self.optimal_sampling.get("mean_seconds", "N/A"),
                "Median optimal freq (s)": self.optimal_sampling.get("median_seconds", "N/A"),
                "Recommendation": self.optimal_sampling.get("recommendation", "N/A"),
            }))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------

@register_analyzer
class MicrostructureNoiseAnalyzer(BaseAnalyzer[MicrostructureNoiseReport]):
    """Microstructure noise estimation and optimal sampling frequency.

    Quantifies the noise floor of raw LOB mid-prices, estimates the bid-ask
    bounce via Roll's model, and recommends an optimal sampling frequency
    for downstream feature extraction.
    """

    name: ClassVar[str] = "MicrostructureNoiseAnalyzer"
    description: ClassVar[str] = "Microstructure noise estimation and optimal sampling"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "mid_price", "best_bid", "best_ask",
    ]
    needs_mbo: ClassVar[bool] = False

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._vol_cfg = config.thresholds.volatility
        self._n_days: int = 0
        self._n_days_skipped: int = 0
        self._records: list[_DayNoiseRecord] = []

        max_s = self._vol_cfg.noise_max_scale_seconds
        n_scales = self._vol_cfg.noise_n_scales
        self._noise_scales = np.logspace(np.log10(0.1), np.log10(max_s), n_scales)

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        self._symbol = day.symbol

        ts_ns = day.lob_timestamps_ns
        mids = day.mid_prices
        utc_off = ctx.utc_offset_hours

        valid = np.isfinite(mids) & (mids > 0)
        rth = rth_mask_utc(ts_ns, utc_offset_hours=utc_off)
        mask = valid & rth

        ts_rth = ts_ns[mask]
        mids_rth = mids[mask]

        if len(mids_rth) < 20:
            self._n_days_skipped += 1
            logger.debug("MicrostructureNoise: skipping %s (< 20 RTH mids)", day.date)
            return

        rv_by_scale: dict[float, float] = {}
        for scale in self._noise_scales:
            res_ns = int(scale * NS_PER_SECOND)
            resampled = resample(ts_rth, mids_rth, res_ns, agg="last")
            filled = resampled.counts > 0
            closes = resampled.values[filled]
            if len(closes) < 2:
                continue
            rets = log_returns(closes)
            rv_by_scale[float(scale)] = float(np.sum(rets**2))

        if len(rv_by_scale) < 2:
            self._n_days_skipped += 1
            logger.debug("MicrostructureNoise: skipping %s (< 2 RV scales)", day.date)
            return

        self._n_days += 1

        sorted_scales = sorted(rv_by_scale.keys())
        fastest_scale = sorted_scales[0]
        slowest_scale = sorted_scales[-1]
        rv_fast = rv_by_scale[fastest_scale]
        rv_slow = rv_by_scale[slowest_scale]

        res_ns_fast = int(fastest_scale * NS_PER_SECOND)
        resampled_fast = resample(ts_rth, mids_rth, res_ns_fast, agg="last")
        n_fast = int(np.sum(resampled_fast.counts > 0)) - 1

        if n_fast < 1:
            return

        noise_var = max((rv_fast - rv_slow) / (2.0 * n_fast), 0.0)

        true_var = max(rv_slow - 2.0 * n_fast * noise_var, EPS)
        snr = true_var / noise_var if noise_var > EPS else np.inf

        tick_rets = log_returns(mids_rth)
        gamma_1 = self._first_autocovariance(tick_rets)

        roll_spread_usd = 0.0
        if gamma_1 < -EPS:
            mid_level = float(np.mean(mids_rth))
            roll_spread_usd = 2.0 * np.sqrt(-gamma_1) * mid_level

        observed_spread_usd = np.nan
        if day.lob is not None:
            if "best_bid" in day.lob.column_names and "best_ask" in day.lob.column_names:
                bids = day.best_bids_usd[mask]
                asks = day.best_asks_usd[mask]
                if len(bids) > 0:
                    spreads = asks - bids
                    valid_sp = spreads[np.isfinite(spreads) & (spreads > 0)]
                    if len(valid_sp) > 0:
                        observed_spread_usd = float(np.mean(valid_sp))

        opt_freq = self._estimate_optimal_frequency(noise_var, rv_slow, n_fast)

        self._records.append(_DayNoiseRecord(
            date=day.date,
            noise_var=noise_var,
            snr=snr,
            roll_spread_usd=roll_spread_usd,
            observed_spread_usd=observed_spread_usd,
            optimal_freq_seconds=opt_freq,
            rv_by_scale=rv_by_scale,
            n_ticks=len(mids_rth),
        ))

    def finalize(self) -> MicrostructureNoiseReport:
        return MicrostructureNoiseReport(
            symbol=self._symbol,
            n_days=self._n_days,
            noise_signature=self._finalize_signature(),
            noise_variance=self._finalize_noise_var(),
            signal_to_noise=self._finalize_snr(),
            bid_ask_bounce=self._finalize_roll(),
            optimal_sampling=self._finalize_optimal(),
            daily_records=self._finalize_daily(),
        )

    def _finalize_signature(self) -> dict[str, Any]:
        if not self._records:
            return {"scales": [], "mean_rv": [], "std_rv": []}

        all_scales_set: set[float] = set()
        for rec in self._records:
            all_scales_set.update(rec.rv_by_scale.keys())
        sorted_scales = sorted(all_scales_set)

        mean_rvs: list[float] = []
        std_rvs: list[float] = []
        valid_scales: list[float] = []

        for scale in sorted_scales:
            values = [rec.rv_by_scale[scale] for rec in self._records if scale in rec.rv_by_scale]
            if values:
                valid_scales.append(scale)
                mean_rvs.append(float(np.mean(values)))
                std_rvs.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)

        return {"scales": valid_scales, "mean_rv": mean_rvs, "std_rv": std_rvs}

    def _finalize_noise_var(self) -> dict[str, Any]:
        if not self._records:
            return {}
        nvars = np.array([r.noise_var for r in self._records])
        dist = distribution_summary(nvars)
        mean_noise_std = float(np.sqrt(np.mean(nvars))) if np.mean(nvars) > 0 else 0.0
        return {
            **dist.to_dict(),
            "mean_noise_std_bps": mean_noise_std * BPS_FACTOR,
        }

    def _finalize_snr(self) -> dict[str, Any]:
        if not self._records:
            return {}
        snrs = np.array([r.snr for r in self._records])
        finite = snrs[np.isfinite(snrs)]
        if len(finite) == 0:
            return {}
        return {
            "mean": float(np.mean(finite)),
            "median": float(np.median(finite)),
            "min": float(np.min(finite)),
            "max": float(np.max(finite)),
        }

    def _finalize_roll(self) -> dict[str, Any]:
        if not self._records:
            return {}

        rolls = np.array([r.roll_spread_usd for r in self._records])
        obs = np.array([r.observed_spread_usd for r in self._records])
        valid_obs = obs[np.isfinite(obs)]

        mean_roll = float(np.mean(rolls))
        mean_obs = float(np.mean(valid_obs)) if len(valid_obs) > 0 else np.nan
        ratio = mean_roll / mean_obs if np.isfinite(mean_obs) and mean_obs > EPS else np.nan

        return {
            "mean_roll_spread_usd": mean_roll,
            "mean_observed_spread_usd": mean_obs,
            "roll_to_observed_ratio": ratio,
        }

    def _finalize_optimal(self) -> dict[str, Any]:
        if not self._records:
            return {}

        freqs = np.array([r.optimal_freq_seconds for r in self._records])
        valid = freqs[np.isfinite(freqs) & (freqs > 0)]
        if len(valid) == 0:
            return {}

        mean_f = float(np.mean(valid))
        median_f = float(np.median(valid))

        rounded = _round_to_nice_frequency(median_f)
        rec = f"{rounded:.1f}s" if rounded < 60 else f"{rounded / 60:.0f}m"

        return {
            "mean_seconds": mean_f,
            "median_seconds": median_f,
            "min_seconds": float(np.min(valid)),
            "max_seconds": float(np.max(valid)),
            "recommendation": rec,
        }

    def _finalize_daily(self) -> list[dict[str, Any]]:
        return [
            {
                "date": r.date,
                "noise_var": r.noise_var,
                "snr": r.snr if np.isfinite(r.snr) else None,
                "roll_spread_usd": r.roll_spread_usd,
                "observed_spread_usd": r.observed_spread_usd if np.isfinite(r.observed_spread_usd) else None,
                "optimal_freq_seconds": r.optimal_freq_seconds if np.isfinite(r.optimal_freq_seconds) else None,
                "n_ticks": r.n_ticks,
            }
            for r in self._records
        ]

    @staticmethod
    def _first_autocovariance(returns: np.ndarray) -> float:
        """Compute first-order autocovariance of a return series.

        gamma_1 = (1/n) * sum(r_t * r_{t-1}) for t=2..n
        Roll (1984), Eq. (2).
        """
        clean = returns[np.isfinite(returns)]
        if len(clean) < 2:
            return 0.0
        return float(np.mean(clean[1:] * clean[:-1]))

    @staticmethod
    def _estimate_optimal_frequency(
        noise_var: float, rv_slow: float, n_fast: int,
    ) -> float:
        """Estimate optimal sampling frequency in seconds.

        Based on MSE-minimizing frequency from Zhang, Mykland & Ait-Sahalia (2005).
        f* proportional to (noise_var / IQ)^{1/3} * T^{2/3}
        where IQ ~ rv_slow^2 (approximation) and T = trading day duration.

        Returns seconds. NaN if inputs are degenerate.
        """
        if noise_var <= EPS or rv_slow <= EPS or n_fast < 2:
            return np.nan

        iq_approx = rv_slow**2
        if iq_approx <= EPS:
            return np.nan

        ratio = noise_var / iq_approx

        opt_n = (ratio ** (1.0 / 3.0)) * (TRADING_SECONDS_PER_DAY ** (2.0 / 3.0))
        if opt_n <= EPS:
            return np.nan

        return TRADING_SECONDS_PER_DAY / opt_n


def _round_to_nice_frequency(seconds: float) -> float:
    """Round to a 'nice' frequency for human readability."""
    nice_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0]
    diffs = [abs(seconds - v) for v in nice_values]
    return nice_values[diffs.index(min(diffs))]
