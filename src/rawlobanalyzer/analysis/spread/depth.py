"""DepthAnalyzer: 10-level order book depth and resilience characterization.

Computes depth profile, imbalance distribution, level gaps, top-of-book
concentration, regime-conditional depth, intraday depth curve, and
post-trade depth recovery from raw LOB data.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, ClassVar

import numpy as np

import pyarrow as pa

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EPS, NANODOLLARS_PER_DOLLAR, NS_PER_MINUTE, TRADING_MINUTES_PER_DAY
from rawlobanalyzer.core.price_utils import spread_in_ticks
from rawlobanalyzer.core.resampler import resample
from rawlobanalyzer.core.statistics import StreamingDistribution, distribution_summary
from rawlobanalyzer.core.time_utils import REGIME_LABELS, rth_mask_utc, time_regime
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.io.schema import ACTION_TRADE
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)


def _extract_list_column(table: pa.Table, col: str, dtype: type, n_levels: int = 10) -> np.ndarray:
    """Extract fixed-size list column as (n_rows, n_levels) numpy array."""
    arr = table.column(col)
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    vals = arr.values
    if hasattr(vals, "combine_chunks"):
        vals = vals.combine_chunks()
    flat = np.asarray(vals, dtype=dtype)
    n_rows = len(arr)
    return flat.reshape(n_rows, n_levels)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class DepthReport(BaseReport):
    """Report from the DepthAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    depth_profile: dict[str, Any]
    imbalance_distribution: dict[str, Any]
    level_gaps: dict[str, Any]
    top_concentration: dict[str, Any]
    conditional_depth: dict[str, Any]
    intraday_curve: dict[str, Any]
    regime_depth: dict[str, Any]
    post_trade_recovery: dict[str, Any]
    daily_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "depth_profile": self.depth_profile,
            "imbalance_distribution": self.imbalance_distribution,
            "level_gaps": self.level_gaps,
            "top_concentration": self.top_concentration,
            "conditional_depth": self.conditional_depth,
            "intraday_curve": self.intraday_curve,
            "regime_depth": self.regime_depth,
            "post_trade_recovery": self.post_trade_recovery,
            "daily_records": self.daily_records,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("DEPTH ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.depth_profile:
            lines.append(format_subsection("DEPTH PROFILE (avg size per level)"))
            headers = ["Level", "Bid", "Ask"]
            rows = []
            bid_avg = self.depth_profile.get("bid_avg", [])
            ask_avg = self.depth_profile.get("ask_avg", [])
            for k in range(min(len(bid_avg), len(ask_avg))):
                rows.append([k + 1, bid_avg[k] if k < len(bid_avg) else np.nan,
                             ask_avg[k] if k < len(ask_avg) else np.nan])
            lines.append(format_table(headers, rows))

        if self.imbalance_distribution:
            lines.append(format_subsection("DEPTH IMBALANCE"))
            lines.append(format_kv({
                "Mean": self.imbalance_distribution.get("mean", "N/A"),
                "Std": self.imbalance_distribution.get("std", "N/A"),
            }))

        if self.top_concentration:
            lines.append(format_subsection("TOP-OF-BOOK CONCENTRATION"))
            lines.append(format_kv({
                "Bid L1 fraction": self.top_concentration.get("bid_l1_fraction", "N/A"),
                "Ask L1 fraction": self.top_concentration.get("ask_l1_fraction", "N/A"),
            }))

        if self.regime_depth:
            lines.append(format_subsection("REGIME-CONDITIONAL DEPTH"))
            headers = ["Regime", "Bid L1", "Ask L1", "Imbalance"]
            rows = []
            for regime, stats in sorted(self.regime_depth.items()):
                rows.append([
                    regime,
                    stats.get("bid_l1", np.nan),
                    stats.get("ask_l1", np.nan),
                    stats.get("imbalance", np.nan),
                ])
            lines.append(format_table(headers, rows))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class DepthAnalyzer(BaseAnalyzer[DepthReport]):
    """10-level order book depth and resilience characterization.

    Computes depth profile, imbalance, level gaps, top concentration,
    conditional depth, intraday curve, regime depth, and post-trade recovery.
    """

    name: ClassVar[str] = "DepthAnalyzer"
    description: ClassVar[str] = "Order book depth and resilience"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns",
        "bid_prices",
        "bid_sizes",
        "ask_prices",
        "ask_sizes",
        "best_bid",
        "best_ask",
        "spread",
        "depth_imbalance",
        "triggering_action",
    ]
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = False
    needs_spreads: ClassVar[bool] = False

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._depth_cfg = config.thresholds.depth
        self._spread_cfg = config.thresholds.spread
        self._n_days: int = 0

        n_lvl = self._depth_cfg.n_levels
        self._bid_sizes_sum: np.ndarray = np.zeros(n_lvl, dtype=np.float64)
        self._ask_sizes_sum: np.ndarray = np.zeros(n_lvl, dtype=np.float64)
        self._bid_sizes_count: int = 0
        self._ask_sizes_count: int = 0

        self._imbalance_dist = StreamingDistribution()
        self._bid_l1_dist = StreamingDistribution()
        self._ask_l1_dist = StreamingDistribution()

        self._level_gaps_bid_sum: np.ndarray = np.zeros(n_lvl - 1, dtype=np.float64)
        self._level_gaps_bid_count: np.ndarray = np.zeros(n_lvl - 1, dtype=np.int64)
        self._level_gaps_ask_sum: np.ndarray = np.zeros(n_lvl - 1, dtype=np.float64)
        self._level_gaps_ask_count: np.ndarray = np.zeros(n_lvl - 1, dtype=np.int64)

        self._conc_bid_frac_sum: float = 0.0
        self._conc_bid_frac_count: int = 0
        self._conc_ask_frac_sum: float = 0.0
        self._conc_ask_frac_count: int = 0

        self._depth_1tick_bid_sum: float = 0.0
        self._depth_1tick_ask_sum: float = 0.0
        self._depth_1tick_count: int = 0
        self._depth_wide_bid_sum: float = 0.0
        self._depth_wide_ask_sum: float = 0.0
        self._depth_wide_count: int = 0

        n_bins = TRADING_MINUTES_PER_DAY // max(self._depth_cfg.intraday_bin_minutes, 1)
        self._intraday_depth_sum: np.ndarray = np.zeros(n_bins, dtype=np.float64)
        self._intraday_counts: np.ndarray = np.zeros(n_bins, dtype=np.int64)

        self._regime_depth: dict[int, dict[str, float]] = {}

        self._post_trade_depth_sums: np.ndarray | None = None
        self._post_trade_depth_count: int = 0

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        lob = day.lob
        if lob is None:
            return

        self._symbol = day.symbol
        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        n_lvl = self._depth_cfg.n_levels
        bid_sizes = _extract_list_column(lob, "bid_sizes", np.uint32, n_lvl)
        ask_sizes = _extract_list_column(lob, "ask_sizes", np.uint32, n_lvl)
        bid_prices = _extract_list_column(lob, "bid_prices", np.int64, n_lvl)
        ask_prices = _extract_list_column(lob, "ask_prices", np.int64, n_lvl)

        n_rows = bid_sizes.shape[0]

        for k in range(n_lvl):
            self._bid_sizes_sum[k] += np.sum(bid_sizes[:, k])
            self._ask_sizes_sum[k] += np.sum(ask_sizes[:, k])
        self._bid_sizes_count += n_rows
        self._ask_sizes_count += n_rows

        if "depth_imbalance" in lob.column_names:
            imb = lob.column("depth_imbalance").to_numpy()
            valid_imb = np.isfinite(imb) & (np.abs(imb) <= 1.0 + EPS)
            self._imbalance_dist.add_batch(imb[valid_imb])

        bid_l1 = bid_sizes[:, 0].astype(np.float64)
        ask_l1 = ask_sizes[:, 0].astype(np.float64)
        self._bid_l1_dist.add_batch(bid_l1)
        self._ask_l1_dist.add_batch(ask_l1)

        total_bid = np.sum(bid_sizes, axis=1).astype(np.float64)
        total_ask = np.sum(ask_sizes, axis=1).astype(np.float64)
        valid_bid = total_bid > EPS
        valid_ask = total_ask > EPS
        if np.any(valid_bid):
            self._conc_bid_frac_sum += float(np.sum(bid_l1[valid_bid] / total_bid[valid_bid]))
            self._conc_bid_frac_count += int(np.count_nonzero(valid_bid))
        if np.any(valid_ask):
            self._conc_ask_frac_sum += float(np.sum(ask_l1[valid_ask] / total_ask[valid_ask]))
            self._conc_ask_frac_count += int(np.count_nonzero(valid_ask))

        for k in range(n_lvl - 1):
            bp = bid_prices[:, k].astype(np.float64) / NANODOLLARS_PER_DOLLAR
            bp_next = bid_prices[:, k + 1].astype(np.float64) / NANODOLLARS_PER_DOLLAR
            valid = (bid_sizes[:, k] > 0) & (bid_sizes[:, k + 1] > 0)
            gaps = bp[valid] - bp_next[valid]
            self._level_gaps_bid_sum[k] += float(np.sum(gaps))
            self._level_gaps_bid_count[k] += len(gaps)

            ap = ask_prices[:, k].astype(np.float64) / NANODOLLARS_PER_DOLLAR
            ap_next = ask_prices[:, k + 1].astype(np.float64) / NANODOLLARS_PER_DOLLAR
            valid_a = (ask_sizes[:, k] > 0) & (ask_sizes[:, k + 1] > 0)
            gaps_a = ap_next[valid_a] - ap[valid_a]
            self._level_gaps_ask_sum[k] += float(np.sum(gaps_a))
            self._level_gaps_ask_count[k] += len(gaps_a)

        if "spread" in lob.column_names and "triggering_action" in lob.column_names:
            spreads = lob.column("spread").to_numpy()
            spreads_usd = np.where(np.isfinite(spreads), spreads, np.nan)
            ticks = spread_in_ticks(spreads_usd, self._spread_cfg.tick_size_usd)
            one_tick = ticks <= self._spread_cfg.narrow_spread_ticks
            wide = ticks > self._spread_cfg.wide_spread_ticks
            if np.any(one_tick):
                self._depth_1tick_bid_sum += float(np.sum(bid_l1[one_tick]))
                self._depth_1tick_ask_sum += float(np.sum(ask_l1[one_tick]))
                self._depth_1tick_count += int(np.count_nonzero(one_tick))
            if np.any(wide):
                self._depth_wide_bid_sum += float(np.sum(bid_l1[wide]))
                self._depth_wide_ask_sum += float(np.sum(ask_l1[wide]))
                self._depth_wide_count += int(np.count_nonzero(wide))

        self._accumulate_intraday(day, bid_sizes, ask_sizes)
        self._accumulate_regime(day, bid_sizes, ask_sizes)
        self._accumulate_post_trade(lob, bid_sizes, ask_sizes)

    def _accumulate_intraday(
        self,
        day: DayData,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> None:
        ts_ns = day.lob_timestamps_ns
        utc_off = self._utc_off
        rth = rth_mask_utc(ts_ns, utc_offset_hours=utc_off)

        bid_l1 = bid_sizes[:, 0].astype(np.float64)
        ask_l1 = ask_sizes[:, 0].astype(np.float64)
        depth = bid_l1 + ask_l1
        depth_rth = depth[rth]
        ts_rth = ts_ns[rth]

        if len(ts_rth) < 2:
            return

        bin_min = self._depth_cfg.intraday_bin_minutes
        res_ns = bin_min * NS_PER_MINUTE
        resampled = resample(
            ts_rth, depth_rth, res_ns,
            agg="mean", label=f"{bin_min}m",
        )
        filled = resampled.counts > 0
        means = resampled.values[filled]
        filled_indices = np.where(filled)[0]
        n_bins = len(self._intraday_counts)
        for i, bin_idx in enumerate(filled_indices):
            if bin_idx < n_bins and i < len(means):
                self._intraday_depth_sum[bin_idx] += means[i]
                self._intraday_counts[bin_idx] += 1

    def _accumulate_regime(
        self,
        day: DayData,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> None:
        ts_ns = day.lob_timestamps_ns
        has_imb = "depth_imbalance" in day.lob.column_names
        imb = day.lob.column("depth_imbalance").to_numpy() if has_imb else np.zeros(len(ts_ns))

        regimes = time_regime(
            ts_ns,
            utc_offset_hours=self._utc_off,
        )
        bid_l1 = bid_sizes[:, 0].astype(np.float64)
        ask_l1 = ask_sizes[:, 0].astype(np.float64)

        for rv in np.unique(regimes):
            rv_int = int(rv)
            mask = regimes == rv
            if rv_int not in self._regime_depth:
                self._regime_depth[rv_int] = {
                    "bid_sum": 0.0, "ask_sum": 0.0,
                    "imb_sum": 0.0, "imb_count": 0, "count": 0,
                }
            acc = self._regime_depth[rv_int]
            acc["bid_sum"] += float(np.sum(bid_l1[mask]))
            acc["ask_sum"] += float(np.sum(ask_l1[mask]))
            acc["count"] += int(np.count_nonzero(mask))
            if has_imb:
                imb_slice = imb[mask]
                valid_imb = np.isfinite(imb_slice) & (np.abs(imb_slice) <= 1.0 + EPS)
                if np.any(valid_imb):
                    acc["imb_sum"] += float(np.sum(imb_slice[valid_imb]))
                    acc["imb_count"] += int(np.count_nonzero(valid_imb))

    def _accumulate_post_trade(
        self,
        lob: pa.Table,
        bid_sizes: np.ndarray,
        ask_sizes: np.ndarray,
    ) -> None:
        if "triggering_action" not in lob.column_names:
            return

        actions = lob.column("triggering_action").to_numpy()
        trade_idx = np.where(actions == ACTION_TRADE)[0]
        if len(trade_idx) == 0:
            return

        bid_l1 = bid_sizes[:, 0].astype(np.float64)
        ask_l1 = ask_sizes[:, 0].astype(np.float64)
        horizons = self._depth_cfg.recovery_horizons
        n_horizons = len(horizons)

        if self._post_trade_depth_sums is None:
            self._post_trade_depth_sums = np.zeros(n_horizons, dtype=np.float64)

        for idx in trade_idx:
            for hi, h in enumerate(horizons):
                future_idx = min(idx + h, len(bid_l1) - 1)
                self._post_trade_depth_sums[hi] += bid_l1[future_idx] + ask_l1[future_idx]
        self._post_trade_depth_count += len(trade_idx)

    def finalize(self) -> DepthReport:
        return DepthReport(
            symbol=self._symbol,
            n_days=self._n_days,
            depth_profile=self._finalize_depth_profile(),
            imbalance_distribution=self._finalize_imbalance(),
            level_gaps=self._finalize_level_gaps(),
            top_concentration=self._finalize_top_concentration(),
            conditional_depth=self._finalize_conditional_depth(),
            intraday_curve=self._finalize_intraday_curve(),
            regime_depth=self._finalize_regime_depth(),
            post_trade_recovery=self._finalize_post_trade_recovery(),
            daily_records=[],
        )

    def _finalize_depth_profile(self) -> dict[str, Any]:
        if self._bid_sizes_count == 0 or self._ask_sizes_count == 0:
            return {}
        n_lvl = self._depth_cfg.n_levels
        bid_avg = [
            self._bid_sizes_sum[k] / self._bid_sizes_count for k in range(n_lvl)
        ]
        ask_avg = [
            self._ask_sizes_sum[k] / self._ask_sizes_count for k in range(n_lvl)
        ]
        return {"bid_avg": bid_avg, "ask_avg": ask_avg}

    def _finalize_imbalance(self) -> dict[str, Any]:
        if self._imbalance_dist.count < 2:
            return {}
        return self._imbalance_dist.distribution_summary().to_dict()

    def _finalize_level_gaps(self) -> dict[str, Any]:
        n_gaps = self._depth_cfg.n_levels - 1
        bid_gaps = [
            float(self._level_gaps_bid_sum[k] / self._level_gaps_bid_count[k])
            for k in range(n_gaps)
            if self._level_gaps_bid_count[k] > 0
        ]
        ask_gaps = [
            float(self._level_gaps_ask_sum[k] / self._level_gaps_ask_count[k])
            for k in range(n_gaps)
            if self._level_gaps_ask_count[k] > 0
        ]
        return {"bid_gaps_usd": bid_gaps, "ask_gaps_usd": ask_gaps}

    def _finalize_top_concentration(self) -> dict[str, Any]:
        if self._conc_bid_frac_count == 0 or self._conc_ask_frac_count == 0:
            return {}
        bid_frac = self._conc_bid_frac_sum / self._conc_bid_frac_count
        ask_frac = self._conc_ask_frac_sum / self._conc_ask_frac_count
        return {"bid_l1_fraction": bid_frac, "ask_l1_fraction": ask_frac}

    def _finalize_conditional_depth(self) -> dict[str, Any]:
        if self._depth_1tick_count == 0 and self._depth_wide_count == 0:
            return {}
        result: dict[str, Any] = {}
        if self._depth_1tick_count > 0:
            result["mean_bid_l1_1tick"] = self._depth_1tick_bid_sum / self._depth_1tick_count
            result["mean_ask_l1_1tick"] = self._depth_1tick_ask_sum / self._depth_1tick_count
        if self._depth_wide_count > 0:
            result["mean_bid_l1_wide"] = self._depth_wide_bid_sum / self._depth_wide_count
            result["mean_ask_l1_wide"] = self._depth_wide_ask_sum / self._depth_wide_count
        return result

    def _finalize_intraday_curve(self) -> dict[str, Any]:
        active = self._intraday_counts > 0
        if not np.any(active):
            return {"minutes": [], "mean_depth": [], "n_days": []}
        bin_min = self._depth_cfg.intraday_bin_minutes
        minutes = (np.arange(len(self._intraday_counts)) * bin_min).tolist()
        safe_counts = np.where(active, self._intraday_counts, 1)
        means = np.where(active, self._intraday_depth_sum / safe_counts, np.nan)
        return {
            "minutes": minutes,
            "mean_depth": [float(v) if np.isfinite(v) else None for v in means],
            "n_days": self._intraday_counts.tolist(),
        }

    def _finalize_regime_depth(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rv, acc in self._regime_depth.items():
            if acc["count"] == 0:
                continue
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            result[label] = {
                "bid_l1": acc["bid_sum"] / acc["count"],
                "ask_l1": acc["ask_sum"] / acc["count"],
                "imbalance": (acc["imb_sum"] / acc["imb_count"]
                              if acc["imb_count"] > 0 else np.nan),
            }
        return result

    def _finalize_post_trade_recovery(self) -> dict[str, Any]:
        if self._post_trade_depth_count == 0 or self._post_trade_depth_sums is None:
            return {}
        horizons = self._depth_cfg.recovery_horizons
        n = min(len(horizons), len(self._post_trade_depth_sums))
        means = [float(self._post_trade_depth_sums[i] / self._post_trade_depth_count)
                 for i in range(n)]
        return {
            "horizons": list(horizons[:n]),
            "mean_depth_at_horizon": means,
            "n_trades": self._post_trade_depth_count,
        }
