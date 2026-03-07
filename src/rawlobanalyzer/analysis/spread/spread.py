"""SpreadAnalyzer: spread distribution and regime characterization.

Computes multi-scale spread distribution, intraday spread curve, regime-conditional
spreads, spread autocorrelation, trade-conditional spread, width classification,
and weekday patterns from raw LOB data.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from rawlobanalyzer.core.calendar import WEEKDAY_NAMES, weekday_from_date
from rawlobanalyzer.core.intraday_accumulator import IntradayCurveAccumulator
from rawlobanalyzer.core.regime_accumulator import RegimeStreamingAccumulator
from typing import Any, ClassVar

import numpy as np

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.core.statistics import StreamingDistribution, WelfordAccumulator, acf as _acf
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.analysis.spread._spread_engine import DaySpreads
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EPS, NS_PER_MINUTE, TRADING_MINUTES_PER_DAY
from rawlobanalyzer.core.price_utils import spread_in_ticks
from rawlobanalyzer.core.resampler import resample
from rawlobanalyzer.core.statistics import distribution_summary
from rawlobanalyzer.core.time_utils import REGIME_LABELS, rth_mask_utc, time_regime
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)



# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class SpreadReport(BaseReport):
    """Report from the SpreadAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    tick_distribution: dict[str, Any]
    timescale_stats: dict[str, Any]
    intraday_curve: dict[str, Any]
    regime_spreads: dict[str, Any]
    spread_acf: dict[str, Any]
    trade_conditional: dict[str, Any]
    width_classification: dict[str, Any]
    weekly_patterns: dict[str, Any]
    daily_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "tick_distribution": self.tick_distribution,
            "timescale_stats": self.timescale_stats,
            "intraday_curve": self.intraday_curve,
            "regime_spreads": self.regime_spreads,
            "spread_acf": self.spread_acf,
            "trade_conditional": self.trade_conditional,
            "width_classification": self.width_classification,
            "weekly_patterns": self.weekly_patterns,
            "daily_records": self.daily_records,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("SPREAD ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.tick_distribution:
            lines.append(format_subsection("TICK-LEVEL SPREAD DISTRIBUTION"))
            lines.append(format_kv({
                "Count": self.tick_distribution.get("count", "N/A"),
                "Mean (USD)": self.tick_distribution.get("mean_usd", "N/A"),
                "Median (USD)": self.tick_distribution.get("median_usd", "N/A"),
                "Std (USD)": self.tick_distribution.get("std_usd", "N/A"),
                "Fraction 1-tick": self.tick_distribution.get("fraction_1tick", "N/A"),
            }))

        if self.regime_spreads:
            lines.append(format_subsection("REGIME-CONDITIONAL SPREADS (USD)"))
            headers = ["Regime", "Mean", "Std", "N"]
            rows = []
            for regime, stats in sorted(self.regime_spreads.items()):
                rows.append([
                    regime,
                    stats.get("mean_usd", np.nan),
                    stats.get("std_usd", np.nan),
                    stats.get("n", 0),
                ])
            lines.append(format_table(headers, rows))

        if self.spread_acf.get("lags"):
            lines.append(format_subsection("SPREAD AUTOCORRELATION"))
            lines.append(format_kv({
                "ACF lag-1": self.spread_acf.get("acf_values", [None])[0] if self.spread_acf.get("acf_values") else None,
            }))

        if self.trade_conditional:
            lines.append(format_subsection("TRADE-CONDITIONAL SPREAD"))
            lines.append(format_kv({
                "Mean at trades (USD)": self.trade_conditional.get("mean_at_trades_usd", "N/A"),
                "Mean all events (USD)": self.trade_conditional.get("mean_all_usd", "N/A"),
                "Ratio": self.trade_conditional.get("ratio", "N/A"),
            }))

        if self.width_classification:
            lines.append(format_subsection("SPREAD WIDTH CLASSIFICATION"))
            lines.append(format_kv(self.width_classification.get("fractions", {})))

        if self.weekly_patterns:
            lines.append(format_subsection("WEEKDAY PATTERNS"))
            headers = ["Day", "Mean (USD)", "N_days"]
            rows = []
            for day_name, stats in sorted(self.weekly_patterns.items()):
                rows.append([day_name, stats.get("mean_usd", np.nan), stats.get("n_days", 0)])
            lines.append(format_table(headers, rows))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class SpreadAnalyzer(BaseAnalyzer[SpreadReport]):
    """Multi-scale spread distribution and regime characterization.

    Computes tick-level and timescale distribution, intraday curve,
    regime-conditional spreads, autocorrelation, trade-conditional spread,
    width classification, and weekday patterns.
    """

    name: ClassVar[str] = "SpreadAnalyzer"
    description: ClassVar[str] = "Spread distribution and regime characterization"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns",
        "spread",
        "spread_bps",
        "triggering_action",
        "mid_price",
    ]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = False
    needs_spreads: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._spread_cfg = config.thresholds.spread
        self._n_days: int = 0

        self._tick_dist_usd = StreamingDistribution()
        self._tick_dist_bps = StreamingDistribution()
        self._tick_dist_ticks = StreamingDistribution()
        self._trade_spread_welford = WelfordAccumulator()
        self._trade_spread_count: int = 0
        self._daily_mean_spreads: list[float] = []
        self._daily_dates: list[str] = []
        self._daily_weekdays: list[int] = []

        self._scaled_spreads: dict[str, list[np.ndarray]] = {}
        for tc in config.timescales:
            self._scaled_spreads[tc.label] = []

        n_bins = TRADING_MINUTES_PER_DAY // max(self._spread_cfg.intraday_bin_minutes, 1)
        self._intraday_acc = IntradayCurveAccumulator(n_bins)

        self._regime_acc = RegimeStreamingAccumulator()
        self._weekday_spreads: dict[int, list[float]] = {d: [] for d in range(5)}
        self._width_counts: dict[str, int] = {"1tick": 0, "2tick": 0, "3-5tick": 0, "5+tick": 0}
        self._resampled_for_acf: deque[np.ndarray] = deque(maxlen=60)
        self._acf_max_days: int = 60

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        ds = ctx.day_spreads
        assert ds is not None, "SpreadAnalyzer requires DaySpreads (needs_spreads=True)"

        self._symbol = day.symbol
        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        weekday = weekday_from_date(day.date)

        spr_usd = ds.tick_spreads_usd
        spr_bps = ds.tick_spreads_bps
        tick_size = self._spread_cfg.tick_size_usd
        spr_ticks = spread_in_ticks(spr_usd, tick_size=tick_size)

        valid_usd = np.isfinite(spr_usd) & (spr_usd > 0)
        if np.any(valid_usd):
            self._tick_dist_usd.add_batch(spr_usd[valid_usd])
            self._tick_dist_ticks.add_batch(spr_ticks[valid_usd])
        if np.any(np.isfinite(spr_bps)):
            valid_bps = np.isfinite(spr_bps) & (spr_bps > 0)
            if np.any(valid_bps):
                self._tick_dist_bps.add_batch(spr_bps[valid_bps])

        if np.any(ds.trade_mask):
            trade_spr = spr_usd[ds.trade_mask]
            valid_trade = trade_spr[np.isfinite(trade_spr) & (trade_spr > 0)]
            self._trade_spread_welford.update_batch(valid_trade)
            self._trade_spread_count += len(valid_trade)

        daily_mean = float(np.mean(spr_usd)) if len(spr_usd) > 0 else np.nan
        if np.isfinite(daily_mean):
            self._daily_mean_spreads.append(daily_mean)
            self._daily_dates.append(day.date)
            self._daily_weekdays.append(weekday)
            if weekday < 5:
                self._weekday_spreads[weekday].append(daily_mean)

        wb = self._spread_cfg.width_bucket_ticks
        for t_val in spr_ticks[valid_usd]:
            t = round(float(t_val))
            if t <= wb[0]:
                self._width_counts["1tick"] += 1
            elif t <= wb[1]:
                self._width_counts["2tick"] += 1
            elif t <= wb[2]:
                self._width_counts["3-5tick"] += 1
            else:
                self._width_counts["5+tick"] += 1

        for label, ss in ds.scaled.items():
            filled = ss.counts > 0
            means = ss.mean_spreads_usd[filled]
            if len(means) > 0:
                self._scaled_spreads.setdefault(label, []).append(means)

        primary = self.config.timescales[0].label if self.config.timescales else "1s"
        if primary in ds.scaled:
            ss = ds.scaled[primary]
            filled = ss.counts > 0
            if np.any(filled):
                vals = ss.mean_spreads_usd[filled]
                self._resampled_for_acf.append(vals)
                if len(self._resampled_for_acf) > self._acf_max_days:
                    self._resampled_for_acf.pop(0)

        self._accumulate_intraday_curve(ds)
        self._accumulate_regime_spreads(ds)

    def _accumulate_intraday_curve(self, ds: DaySpreads) -> None:
        utc_off = self._utc_off
        rth = rth_mask_utc(ds.tick_timestamps_ns, utc_offset_hours=utc_off)

        ts_rth = ds.tick_timestamps_ns[rth]
        spreads_rth = ds.tick_spreads_usd[rth]

        if len(spreads_rth) < 2:
            return

        bin_min = self._spread_cfg.intraday_bin_minutes
        res_ns = bin_min * NS_PER_MINUTE

        resampled = resample(ts_rth, spreads_rth, res_ns, agg="mean", label=f"{bin_min}m")
        filled = resampled.counts > 0
        means = resampled.values[filled]

        if len(means) == 0:
            return

        daily_mean = float(np.mean(means))
        if daily_mean < EPS:
            return

        normalized = means / daily_mean
        filled_indices = np.where(filled)[0]
        self._intraday_acc.add(filled_indices, normalized)

    def _accumulate_regime_spreads(self, ds: DaySpreads) -> None:
        ts_valid = ds.tick_timestamps_ns
        spreads_valid = ds.tick_spreads_usd

        if len(ts_valid) == 0:
            return

        regimes = time_regime(
            ts_valid,
            utc_offset_hours=self._utc_off,
        )
        for rv in np.unique(regimes):
            rv_int = int(rv)
            mask = regimes == rv
            chunk = spreads_valid[mask]
            self._regime_acc.add(rv_int, chunk)

    def finalize(self) -> SpreadReport:
        return SpreadReport(
            symbol=self._symbol,
            n_days=self._n_days,
            tick_distribution=self._finalize_tick_distribution(),
            timescale_stats=self._finalize_timescale_stats(),
            intraday_curve=self._finalize_intraday_curve(),
            regime_spreads=self._finalize_regime_spreads(),
            spread_acf=self._finalize_spread_acf(),
            trade_conditional=self._finalize_trade_conditional(),
            width_classification=self._finalize_width_classification(),
            weekly_patterns=self._finalize_weekly_patterns(),
            daily_records=self._finalize_daily_records(),
        )

    def _finalize_tick_distribution(self) -> dict[str, Any]:
        if self._tick_dist_usd.count == 0:
            return {}

        dist = self._tick_dist_usd.distribution_summary()
        usd_sample = self._tick_dist_usd.sample()
        ticks_sample = self._tick_dist_ticks.sample()
        bps_sample = self._tick_dist_bps.sample()

        width_total = sum(self._width_counts.values())
        fraction_1tick = self._width_counts["1tick"] / max(width_total, 1)

        return {
            **dist.to_dict(),
            "mean_usd": dist.mean,
            "median_usd": float(np.median(usd_sample)) if len(usd_sample) > 0 else np.nan,
            "std_usd": dist.std,
            "fraction_1tick": fraction_1tick,
            "mean_bps": float(np.mean(bps_sample)) if len(bps_sample) > 0 else np.nan,
        }

    def _finalize_timescale_stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label, chunks in self._scaled_spreads.items():
            if not chunks:
                continue
            merged = np.concatenate(chunks)
            clean = merged[np.isfinite(merged) & (merged > 0)]
            if len(clean) < 2:
                continue
            dist = distribution_summary(clean)
            result[label] = {
                **dist.to_dict(),
                "mean_usd": dist.mean,
                "median_usd": float(np.median(clean)),
                "std_usd": dist.std,
            }
        return result

    def _finalize_intraday_curve(self) -> dict[str, Any]:
        return self._intraday_acc.finalize(
            self._spread_cfg.intraday_bin_minutes,
            mean_key="mean_normalized",
            std_key="std_normalized",
        )

    def _finalize_regime_spreads(self) -> dict[str, Any]:
        base = self._regime_acc.finalize(min_count=2)
        return {
            label: {
                "mean_usd": stats["mean"],
                "std_usd": stats["std"],
                "n": stats["n"],
            }
            for label, stats in base.items()
        }

    def _finalize_spread_acf(self) -> dict[str, Any]:
        if not self._resampled_for_acf:
            return {"lags": [], "acf_values": []}

        concatenated = np.concatenate(self._resampled_for_acf)
        clean = concatenated[np.isfinite(concatenated) & (concatenated > 0)]
        if len(clean) < self._spread_cfg.max_acf_lag + 2:
            return {"lags": [], "acf_values": []}

        max_lag = min(self._spread_cfg.max_acf_lag, len(clean) - 1)
        acf_vals = _acf(clean, max_lag)

        return {
            "lags": list(range(1, len(acf_vals) + 1)),
            "acf_values": [float(v) for v in acf_vals],
        }

    def _finalize_trade_conditional(self) -> dict[str, Any]:
        if self._trade_spread_count == 0 or self._tick_dist_usd.count == 0:
            return {}

        mean_trade = self._trade_spread_welford.mean
        mean_all = self._tick_dist_usd.mean
        ratio = mean_trade / mean_all if mean_all > EPS else np.nan

        return {
            "mean_at_trades_usd": mean_trade,
            "mean_all_usd": mean_all,
            "n_trades": self._trade_spread_count,
            "ratio": ratio,
        }

    def _finalize_width_classification(self) -> dict[str, Any]:
        total = sum(self._width_counts.values())
        if total == 0:
            return {}

        fractions = {
            "1-tick": self._width_counts["1tick"] / total,
            "2-tick": self._width_counts["2tick"] / total,
            "3-5 tick": self._width_counts["3-5tick"] / total,
            "5+ tick": self._width_counts["5+tick"] / total,
        }
        return {"fractions": fractions, "total": total}

    def _finalize_weekly_patterns(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for wd in range(5):
            vals = self._weekday_spreads.get(wd, [])
            if not vals:
                continue
            arr = np.array(vals)
            result[WEEKDAY_NAMES[wd]] = {
                "mean_usd": float(np.mean(arr)),
                "n_days": len(vals),
                "std_usd": float(np.std(arr, ddof=1)) if len(vals) > 1 else 0.0,
            }
        return result

    def _finalize_daily_records(self) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for i, date in enumerate(self._daily_dates):
            rec: dict[str, Any] = {"date": date}
            if i < len(self._daily_mean_spreads):
                rec["mean_spread_usd"] = self._daily_mean_spreads[i]
            records.append(rec)
        return records
