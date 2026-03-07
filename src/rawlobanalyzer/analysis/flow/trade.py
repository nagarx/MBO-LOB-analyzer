"""TradeAnalyzer: trade size, timing, clustering, and execution quality.

Computes trade size distribution, trade-through analysis, trade clustering,
VWAP trajectory, large trade impact, trade rate by regime, and trade price
level analysis from MBO trade events aligned with LOB state.

All heavy computation (trade extraction, MBO-LOB alignment) is in the shared
``_flow_engine.py``; this analyzer consumes ``DayFlow`` from ``DayContext``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

import numpy as np

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.flow._flow_engine import DayFlow
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import BPS_FACTOR, EPS, NS_PER_SECOND, PRICE_LEVEL_TOLERANCE_USD, TRADING_SECONDS_PER_DAY
from rawlobanalyzer.core.statistics import StreamingDistribution, distribution_summary
from rawlobanalyzer.core.time_utils import REGIME_LABELS, time_regime
from rawlobanalyzer.io.schema import SIDE_ASK, SIDE_BID
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
class TradeReport(BaseReport):
    """Report from the TradeAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    trade_size_distribution: dict[str, Any]
    trade_through: dict[str, Any]
    trade_clustering: dict[str, Any]
    vwap_trajectory: dict[str, Any]
    large_trade_impact: dict[str, Any]
    trade_rate_by_regime: dict[str, Any]
    trade_price_level: dict[str, Any]
    directional_size_distribution: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "trade_size_distribution": self.trade_size_distribution,
            "trade_through": self.trade_through,
            "trade_clustering": self.trade_clustering,
            "vwap_trajectory": self.vwap_trajectory,
            "large_trade_impact": self.large_trade_impact,
            "trade_rate_by_regime": self.trade_rate_by_regime,
            "trade_price_level": self.trade_price_level,
            "directional_size_distribution": self.directional_size_distribution,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("TRADE ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.trade_size_distribution:
            lines.append(format_subsection("TRADE SIZE DISTRIBUTION"))
            lines.append(format_kv({
                "Mean size": self.trade_size_distribution.get("mean", "N/A"),
                "Median size": self.trade_size_distribution.get("percentiles", {}).get("p50", "N/A"),
                "Large threshold": self.trade_size_distribution.get("large_threshold", "N/A"),
            }))

        if self.trade_through:
            lines.append(format_subsection("TRADE-THROUGH"))
            lines.append(format_kv({
                "Trade-through rate": self.trade_through.get("overall_rate", "N/A"),
                "N trades analyzed": self.trade_through.get("n_trades", "N/A"),
            }))

        if self.trade_clustering:
            lines.append(format_subsection("TRADE CLUSTERING"))
            lines.append(format_kv({
                "Mean inter-trade (s)": self.trade_clustering.get("mean_inter_trade_seconds", "N/A"),
                "Cluster fraction": self.trade_clustering.get("cluster_fraction", "N/A"),
            }))

        if self.trade_rate_by_regime:
            lines.append(format_subsection("TRADE RATE BY REGIME"))
            headers = ["Regime", "Trades/sec"]
            rows = []
            for regime, stats in sorted(self.trade_rate_by_regime.items()):
                if isinstance(stats, dict):
                    rows.append([regime, stats.get("mean_trades_per_second", np.nan)])
            if rows:
                lines.append(format_table(headers, rows))

        if self.trade_price_level:
            lines.append(format_subsection("TRADE PRICE LEVEL"))
            lines.append(format_kv({
                "At bid": self.trade_price_level.get("at_bid_fraction", "N/A"),
                "At ask": self.trade_price_level.get("at_ask_fraction", "N/A"),
                "Inside spread": self.trade_price_level.get("inside_spread_fraction", "N/A"),
                "Outside spread": self.trade_price_level.get("outside_fraction", "N/A"),
            }))

        if self.directional_size_distribution:
            lines.append(format_subsection("DIRECTIONAL TRADE SIZE"))
            for side in ("buyer", "seller"):
                side_data = self.directional_size_distribution.get(side, {})
                if side_data:
                    lines.append(f"    {side.capitalize()}: "
                                 f"mean={side_data.get('mean', 'N/A'):.1f}, "
                                 f"median={side_data.get('percentiles', {}).get('p50', 'N/A')}")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-day accumulators
# ---------------------------------------------------------------------------


@dataclass
class _DayTradeRecord:
    date: str
    n_trades: int
    mean_size: float
    large_threshold: float
    trade_through_rate: float
    cluster_fraction: float
    mean_inter_trade_s: float


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class TradeAnalyzer(BaseAnalyzer[TradeReport]):
    """Trade execution quality and microstructure analysis.

    Computes trade size distribution, trade-through rates, clustering,
    VWAP trajectory, large trade impact, trade rate by regime, and
    trade price level classification.
    """

    name: ClassVar[str] = "TradeAnalyzer"
    description: ClassVar[str] = "Trade size, timing, clustering, and execution quality"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "mid_price", "best_bid", "best_ask", "spread",
    ]
    mbo_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "order_id", "action", "side", "price", "size",
    ]
    needs_mbo: ClassVar[bool] = True
    needs_returns: ClassVar[bool] = False
    needs_flow: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._n_days: int = 0
        self._n_days_skipped: int = 0
        self._flow_cfg = config.thresholds.flow

        self._size_dist = StreamingDistribution()
        self._buyer_dist = StreamingDistribution()
        self._seller_dist = StreamingDistribution()
        self._day_records: list[_DayTradeRecord] = []

        self._trade_through_counts: list[tuple[int, int]] = []
        self._regime_trade_through: dict[int, list[tuple[int, int]]] = {}

        self._inter_trade_dist = StreamingDistribution()
        self._cluster_fractions: list[float] = []

        self._vwap_daily: list[dict[str, Any]] = []

        self._large_impact_dist = StreamingDistribution()
        self._large_buyer_count: int = 0
        self._large_seller_count: int = 0

        self._regime_trade_counts: dict[int, list[int]] = {}
        self._regime_durations_s: dict[int, list[float]] = {}

        self._price_level_counts: dict[str, int] = {
            "at_bid": 0, "at_ask": 0, "inside_spread": 0, "outside": 0,
        }
        self._price_level_total: int = 0

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        flow = ctx.day_flow
        self._symbol = day.symbol

        if flow is None or flow.n_trades == 0:
            self._n_days_skipped += 1
            logger.debug("Trade: skipping %s (no flow/trades)", day.date)
            return

        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        rth = flow.rth_mask_trades
        has_rth = len(rth) > 0 and np.any(rth)

        sizes = flow.trade_sizes.astype(np.float64)
        self._size_dist.add_batch(sizes)

        dir_mask = flow.directional_mask
        buyer_mask = dir_mask & (flow.trade_sides == SIDE_BID)
        seller_mask = dir_mask & (flow.trade_sides == SIDE_ASK)
        if np.any(buyer_mask):
            self._buyer_dist.add_batch(sizes[buyer_mask])
        if np.any(seller_mask):
            self._seller_dist.add_batch(sizes[seller_mask])

        large_pct = self._flow_cfg.large_trade_percentile
        large_threshold = float(np.percentile(sizes, large_pct)) if len(sizes) > 0 else 0.0

        # --- Trade-through ---
        tt_count, tt_total = self._compute_trade_through(flow)

        # --- Clustering ---
        cluster_frac, mean_inter_s = self._compute_clustering(flow, rth, has_rth)

        # --- VWAP ---
        self._compute_vwap(flow, rth, has_rth)

        # --- Large trade impact ---
        self._compute_large_impact(flow, large_threshold)

        # --- Trade rate by regime ---
        self._compute_regime_rate(flow)

        # --- Price level classification ---
        self._compute_price_level(flow)

        self._day_records.append(_DayTradeRecord(
            date=day.date,
            n_trades=flow.n_trades,
            mean_size=float(np.mean(sizes)),
            large_threshold=large_threshold,
            trade_through_rate=tt_count / max(tt_total, 1),
            cluster_fraction=cluster_frac,
            mean_inter_trade_s=mean_inter_s,
        ))

    def _compute_trade_through(self, flow: DayFlow) -> tuple[int, int]:
        """Fraction of trades executing beyond the pre-trade BBO."""
        prices = flow.trade_prices_usd
        mid = flow.trade_mid_before
        spread = flow.trade_spread_before
        valid = np.isfinite(mid) & np.isfinite(spread) & (spread > 0)

        if not np.any(valid):
            return (0, 0)

        half_spread = spread[valid] / 2.0
        deviation = np.abs(prices[valid] - mid[valid])
        through = deviation > half_spread + EPS
        tt_count = int(np.sum(through))
        tt_total = int(np.sum(valid))
        self._trade_through_counts.append((tt_count, tt_total))

        regimes = time_regime(
            flow.trade_timestamps_ns[valid],
            utc_offset_hours=self._utc_off,
        )
        for rv in range(7):
            mask = regimes == rv
            if np.any(mask):
                if rv not in self._regime_trade_through:
                    self._regime_trade_through[rv] = []
                self._regime_trade_through[rv].append(
                    (int(np.sum(through[mask])), int(np.sum(mask)))
                )

        return (tt_count, tt_total)

    def _compute_clustering(
        self, flow: DayFlow, rth: np.ndarray, has_rth: bool,
    ) -> tuple[float, float]:
        """Inter-trade time distribution and burst detection."""
        ts = flow.trade_timestamps_ns
        if has_rth:
            ts = ts[rth]
        if len(ts) < 2:
            return (0.0, 0.0)

        inter = np.diff(ts).astype(np.float64) / NS_PER_SECOND
        self._inter_trade_dist.add_batch(inter)

        gap_threshold = self._flow_cfg.trade_cluster_gap_seconds
        cluster_frac = float(np.mean(inter < gap_threshold))
        self._cluster_fractions.append(cluster_frac)
        mean_inter = float(np.mean(inter))
        return (cluster_frac, mean_inter)

    def _compute_vwap(
        self, flow: DayFlow, rth: np.ndarray, has_rth: bool,
    ) -> None:
        """Cumulative VWAP path through the day."""
        if has_rth:
            prices = flow.trade_prices_usd[rth]
            sizes = flow.trade_sizes[rth].astype(np.float64)
        else:
            prices = flow.trade_prices_usd
            sizes = flow.trade_sizes.astype(np.float64)

        if len(prices) == 0:
            return

        cum_pv = np.cumsum(prices * sizes)
        cum_vol = np.cumsum(sizes)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap_path = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)

        mid = flow.trade_mid_before
        if has_rth:
            mid = mid[rth]
        valid_mid = np.isfinite(mid) & (mid > 0)
        cum_mid_pv = np.cumsum(np.where(valid_mid, mid * sizes, 0.0))
        with np.errstate(divide="ignore", invalid="ignore"):
            mid_vwap = np.where(cum_vol > 0, cum_mid_pv / cum_vol, np.nan)

        final_vwap = float(vwap_path[-1]) if len(vwap_path) > 0 else np.nan
        final_mid_vwap = float(mid_vwap[-1]) if len(mid_vwap) > 0 else np.nan

        self._vwap_daily.append({
            "date": flow.date,
            "final_vwap": final_vwap,
            "final_mid_vwap": final_mid_vwap,
            "vwap_mid_deviation": final_vwap - final_mid_vwap
            if np.isfinite(final_vwap) and np.isfinite(final_mid_vwap)
            else np.nan,
            "total_volume": float(np.sum(sizes)),
        })

    def _compute_large_impact(self, flow: DayFlow, threshold: float) -> None:
        """Price impact of large trades (directional only for buyer/seller split)."""
        sizes = flow.trade_sizes.astype(np.float64)
        large_mask = sizes >= threshold
        if not np.any(large_mask):
            return

        mid_before = flow.trade_mid_before[large_mask]
        prices = flow.trade_prices_usd[large_mask]
        sides = flow.trade_sides[large_mask]

        valid = np.isfinite(mid_before) & (mid_before > 0)
        if not np.any(valid):
            return

        impact = (prices[valid] - mid_before[valid]) / mid_before[valid] * BPS_FACTOR
        self._large_impact_dist.add_batch(impact)

        dir_sides = sides[valid]
        self._large_buyer_count += int(np.sum(dir_sides == SIDE_BID))
        self._large_seller_count += int(np.sum(dir_sides == SIDE_ASK))

    def _compute_regime_rate(self, flow: DayFlow) -> None:
        """Trades per second in each regime."""
        if flow.n_trades == 0:
            return
        regimes = time_regime(
            flow.trade_timestamps_ns,
            utc_offset_hours=self._utc_off,
        )
        for rv in range(7):
            mask = regimes == rv
            n = int(np.sum(mask))
            if n > 0:
                if rv not in self._regime_trade_counts:
                    self._regime_trade_counts[rv] = []
                self._regime_trade_counts[rv].append(n)

                ts_regime = flow.trade_timestamps_ns[mask]
                duration_s = float(ts_regime[-1] - ts_regime[0]) / NS_PER_SECOND
                if rv not in self._regime_durations_s:
                    self._regime_durations_s[rv] = []
                self._regime_durations_s[rv].append(max(duration_s, 1.0))

    def _compute_price_level(self, flow: DayFlow) -> None:
        """Classify trades relative to pre-trade BBO."""
        prices = flow.trade_prices_usd
        mid = flow.trade_mid_before
        spread = flow.trade_spread_before

        valid = np.isfinite(mid) & np.isfinite(spread) & (spread > 0)
        if not np.any(valid):
            return

        p = prices[valid]
        m = mid[valid]
        half_s = spread[valid] / 2.0
        bid_price = m - half_s
        ask_price = m + half_s

        at_bid = np.abs(p - bid_price) < PRICE_LEVEL_TOLERANCE_USD
        at_ask = np.abs(p - ask_price) < PRICE_LEVEL_TOLERANCE_USD
        inside = (~at_bid) & (~at_ask) & (p > bid_price) & (p < ask_price)
        outside = (~at_bid) & (~at_ask) & (~inside)

        n = int(np.sum(valid))
        self._price_level_counts["at_bid"] += int(np.sum(at_bid))
        self._price_level_counts["at_ask"] += int(np.sum(at_ask))
        self._price_level_counts["inside_spread"] += int(np.sum(inside))
        self._price_level_counts["outside"] += int(np.sum(outside))
        self._price_level_total += n

    def finalize(self) -> TradeReport:
        return TradeReport(
            symbol=self._symbol,
            n_days=self._n_days,
            trade_size_distribution=self._finalize_size_dist(),
            trade_through=self._finalize_trade_through(),
            trade_clustering=self._finalize_clustering(),
            vwap_trajectory=self._finalize_vwap(),
            large_trade_impact=self._finalize_large_impact(),
            trade_rate_by_regime=self._finalize_regime_rate(),
            trade_price_level=self._finalize_price_level(),
            directional_size_distribution=self._finalize_directional_size_dist(),
        )

    def _finalize_size_dist(self) -> dict[str, Any]:
        if self._size_dist.count == 0:
            return {}
        ds = self._size_dist.distribution_summary()
        result = ds.to_dict()
        if self._day_records:
            result["large_threshold"] = float(np.mean(
                [r.large_threshold for r in self._day_records]
            ))
        return result

    def _finalize_trade_through(self) -> dict[str, Any]:
        if not self._trade_through_counts:
            return {}
        total_tt = sum(c for c, _ in self._trade_through_counts)
        total_n = sum(n for _, n in self._trade_through_counts)
        result: dict[str, Any] = {
            "overall_rate": total_tt / max(total_n, 1),
            "n_trades": total_n,
            "n_trade_through": total_tt,
        }
        regime_rates: dict[str, Any] = {}
        for rv, pairs in self._regime_trade_through.items():
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            tt = sum(c for c, _ in pairs)
            n = sum(n for _, n in pairs)
            regime_rates[label] = {"rate": tt / max(n, 1), "n": n}
        result["by_regime"] = regime_rates
        return result

    def _finalize_clustering(self) -> dict[str, Any]:
        if self._inter_trade_dist.count == 0:
            return {}
        ds = self._inter_trade_dist.distribution_summary()
        return {
            "mean_inter_trade_seconds": ds.mean,
            "median_inter_trade_seconds": ds.percentiles.get("p50", np.nan),
            "cluster_fraction": float(np.mean(self._cluster_fractions))
            if self._cluster_fractions else 0.0,
            "gap_threshold_seconds": self._flow_cfg.trade_cluster_gap_seconds,
            "inter_trade_distribution": ds.to_dict(),
        }

    def _finalize_vwap(self) -> dict[str, Any]:
        if not self._vwap_daily:
            return {}
        deviations = [
            d["vwap_mid_deviation"] for d in self._vwap_daily
            if np.isfinite(d.get("vwap_mid_deviation", np.nan))
        ]
        return {
            "daily_records": self._vwap_daily,
            "mean_vwap_mid_deviation": float(np.mean(deviations)) if deviations else np.nan,
            "n_days": len(self._vwap_daily),
        }

    def _finalize_large_impact(self) -> dict[str, Any]:
        if self._large_impact_dist.count == 0:
            return {}
        ds = self._large_impact_dist.distribution_summary()
        total = self._large_buyer_count + self._large_seller_count
        return {
            "mean_impact_bps": ds.mean,
            "median_impact_bps": ds.percentiles.get("p50", np.nan),
            "std_impact_bps": ds.std,
            "n_large_trades": ds.count,
            "buyer_fraction": self._large_buyer_count / max(total, 1),
            "percentile_threshold": self._flow_cfg.large_trade_percentile,
        }

    def _finalize_regime_rate(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rv in range(7):
            if rv not in self._regime_trade_counts:
                continue
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            counts = self._regime_trade_counts[rv]
            durations = self._regime_durations_s.get(rv, [1.0] * len(counts))
            rates = [c / d for c, d in zip(counts, durations)]
            result[label] = {
                "mean_trades_per_second": float(np.mean(rates)),
                "mean_total_trades": float(np.mean(counts)),
                "n_days": len(counts),
            }
        return result

    def _finalize_price_level(self) -> dict[str, Any]:
        total = self._price_level_total
        if total == 0:
            return {}
        return {
            "at_bid_fraction": self._price_level_counts["at_bid"] / total,
            "at_ask_fraction": self._price_level_counts["at_ask"] / total,
            "inside_spread_fraction": self._price_level_counts["inside_spread"] / total,
            "outside_fraction": self._price_level_counts["outside"] / total,
            "total_classified": total,
        }

    def _finalize_directional_size_dist(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self._buyer_dist.count > 0:
            result["buyer"] = self._buyer_dist.distribution_summary().to_dict()
        if self._seller_dist.count > 0:
            result["seller"] = self._seller_dist.distribution_summary().to_dict()
        return result
