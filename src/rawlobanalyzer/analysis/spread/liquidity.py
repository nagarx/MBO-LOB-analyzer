"""LiquidityAnalyzer: execution cost and price impact characterization.

Computes effective spread, volume-weighted spread, microprice deviation,
and regime-conditional execution costs from raw LOB and MBO data.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import Any, ClassVar

import numpy as np

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EPS, NANODOLLARS_PER_DOLLAR
from rawlobanalyzer.core.statistics import StreamingDistribution, WelfordAccumulator
from rawlobanalyzer.core.time_utils import REGIME_LABELS, time_regime
from rawlobanalyzer.io.loader import DayData
from rawlobanalyzer.io.schema import ACTION_TRADE
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
class LiquidityReport(BaseReport):
    """Report from the LiquidityAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    effective_spread: dict[str, Any]
    volume_weighted_spread: dict[str, Any]
    microprice_deviation: dict[str, Any]
    regime_effective_spread: dict[str, Any]
    daily_records: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "effective_spread": self.effective_spread,
            "volume_weighted_spread": self.volume_weighted_spread,
            "microprice_deviation": self.microprice_deviation,
            "regime_effective_spread": self.regime_effective_spread,
            "daily_records": self.daily_records,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("LIQUIDITY ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.effective_spread:
            lines.append(format_subsection("EFFECTIVE SPREAD"))
            lines.append(format_kv({
                "Mean (USD)": self.effective_spread.get("mean_usd", "N/A"),
                "Median (USD)": self.effective_spread.get("median_usd", "N/A"),
                "N trades": self.effective_spread.get("n_trades", "N/A"),
            }))

        if self.volume_weighted_spread:
            lines.append(format_subsection("VOLUME-WEIGHTED SPREAD"))
            lines.append(format_kv({
                "Mean (USD)": self.volume_weighted_spread.get("mean_usd", "N/A"),
            }))

        if self.microprice_deviation:
            lines.append(format_subsection("MICROPRICE DEVIATION"))
            lines.append(format_kv({
                "Mean": self.microprice_deviation.get("mean", "N/A"),
                "Std": self.microprice_deviation.get("std", "N/A"),
            }))

        if self.regime_effective_spread:
            lines.append(format_subsection("REGIME-CONDITIONAL EFFECTIVE SPREAD"))
            headers = ["Regime", "Mean (USD)", "N"]
            rows = []
            for regime, stats in sorted(self.regime_effective_spread.items()):
                rows.append([regime, stats.get("mean_usd", np.nan), stats.get("n", 0)])
            lines.append(format_table(headers, rows))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class LiquidityAnalyzer(BaseAnalyzer[LiquidityReport]):
    """Execution cost and price impact characterization.

    Computes effective spread (MBO+LOB), volume-weighted spread,
    microprice deviation, and regime-conditional execution costs.
    """

    name: ClassVar[str] = "LiquidityAnalyzer"
    description: ClassVar[str] = "Execution cost and price impact"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns",
        "mid_price",
        "microprice",
        "best_bid",
        "best_ask",
        "spread",
        "total_bid_volume",
        "total_ask_volume",
        "triggering_action",
    ]
    mbo_columns: ClassVar[list[str] | None] = [
        "timestamp_ns",
        "order_id",
        "action",
        "side",
        "price",
        "size",
    ]
    needs_mbo: ClassVar[bool] = True
    needs_returns: ClassVar[bool] = False
    needs_spreads: ClassVar[bool] = False

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._n_days: int = 0

        self._eff_spread_dist = StreamingDistribution()
        self._eff_spread_by_regime: dict[int, WelfordAccumulator] = {}
        self._volume_weighted_spreads: list[float] = []
        self._micro_dev_sum: float = 0.0
        self._micro_dev_sum_sq: float = 0.0
        self._micro_dev_count: int = 0
        self._daily_records: list[dict[str, Any]] = []

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        lob = day.lob
        mbo = day.mbo

        self._symbol = day.symbol
        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        if lob is None:
            return

        self._compute_volume_weighted_spread(lob)
        self._compute_microprice_deviation(lob)

        if mbo is not None and "action" in mbo.column_names and "price" in mbo.column_names:
            self._compute_effective_spread(day, lob, mbo)

    def _compute_volume_weighted_spread(self, lob) -> None:
        if "spread" not in lob.column_names or "total_bid_volume" not in lob.column_names:
            return
        spreads = lob.column("spread").to_numpy()
        bid_vol = lob.column("total_bid_volume").to_numpy().astype(np.float64)
        ask_vol = lob.column("total_ask_volume").to_numpy().astype(np.float64)
        total_vol = bid_vol + ask_vol
        valid = np.isfinite(spreads) & (spreads > 0) & (total_vol > 0)
        if np.any(valid):
            vw = np.average(spreads[valid], weights=total_vol[valid])
            self._volume_weighted_spreads.append(float(vw))

    def _compute_microprice_deviation(self, lob) -> None:
        if "microprice" not in lob.column_names or "mid_price" not in lob.column_names:
            return
        micro = lob.column("microprice").to_numpy()
        mid = lob.column("mid_price").to_numpy()
        dev = micro - mid
        valid = np.isfinite(dev)
        if np.any(valid):
            clean = dev[valid]
            self._micro_dev_sum += float(np.sum(clean))
            self._micro_dev_sum_sq += float(np.sum(clean ** 2))
            self._micro_dev_count += len(clean)

    def _compute_effective_spread(self, day: DayData, lob, mbo) -> None:
        lob_ts = lob.column("timestamp_ns").to_numpy()
        mid = lob.column("mid_price").to_numpy()
        mbo_ts = mbo.column("timestamp_ns").to_numpy()
        mbo_action = mbo.column("action").to_numpy()
        mbo_price = mbo.column("price").to_numpy()
        mbo_order_id = mbo.column("order_id").to_numpy()

        # Aggressor-only: Databento MBO emits two events per physical trade
        # (aggressor order_id=0, passive order_id!=0). Match the flow engine
        # convention to avoid double-counting.
        trade_mask = (mbo_action == ACTION_TRADE) & (mbo_order_id == 0)
        trade_indices = np.where(trade_mask)[0]
        if len(trade_indices) == 0:
            return

        trade_prices_usd = mbo_price[trade_mask].astype(np.float64) / NANODOLLARS_PER_DOLLAR
        trade_ts = mbo_ts[trade_mask]

        mid_valid = np.isfinite(mid) & (mid > 0)

        # searchsorted(side="right") - 1 gives the last LOB snapshot with
        # timestamp <= trade time, matching the flow engine convention.
        positions = np.searchsorted(lob_ts, trade_ts, side="right") - 1
        valid_pos = (positions >= 0) & (positions < len(mid)) & mid_valid[np.clip(positions, 0, len(mid) - 1)]
        if not np.any(valid_pos):
            return

        pos_valid = positions[valid_pos]
        eff_spreads = 2.0 * np.abs(trade_prices_usd[valid_pos] - mid[pos_valid])

        self._eff_spread_dist.add_batch(eff_spreads)

        trade_ts_valid = trade_ts[valid_pos]
        regimes = time_regime(trade_ts_valid, utc_offset_hours=self._utc_off)
        for rv_val in np.unique(regimes):
            rv = int(rv_val)
            if rv not in self._eff_spread_by_regime:
                self._eff_spread_by_regime[rv] = WelfordAccumulator()
            mask = regimes == rv_val
            self._eff_spread_by_regime[rv].update_batch(eff_spreads[mask])

        self._daily_records.append({
            "date": day.date,
            "n_trades": len(eff_spreads),
            "mean_effective_spread_usd": float(np.mean(eff_spreads)),
        })

    def finalize(self) -> LiquidityReport:
        return LiquidityReport(
            symbol=self._symbol,
            n_days=self._n_days,
            effective_spread=self._finalize_effective_spread(),
            volume_weighted_spread=self._finalize_volume_weighted_spread(),
            microprice_deviation=self._finalize_microprice_deviation(),
            regime_effective_spread=self._finalize_regime_effective_spread(),
            daily_records=self._daily_records,
        )

    def _finalize_effective_spread(self) -> dict[str, Any]:
        if self._eff_spread_dist.count == 0:
            return {}
        ds = self._eff_spread_dist.distribution_summary()
        return {
            "mean_usd": ds.mean,
            "median_usd": ds.percentiles.get("p50", np.nan),
            "std_usd": ds.std,
            "n_trades": ds.count,
        }

    def _finalize_volume_weighted_spread(self) -> dict[str, Any]:
        if not self._volume_weighted_spreads:
            return {}
        return {
            "mean_usd": float(np.mean(self._volume_weighted_spreads)),
            "daily_values": self._volume_weighted_spreads,
        }

    def _finalize_microprice_deviation(self) -> dict[str, Any]:
        n = self._micro_dev_count
        if n == 0:
            return {}
        mean = self._micro_dev_sum / n
        variance = (self._micro_dev_sum_sq / n) - mean ** 2
        std = float(np.sqrt(max(variance * n / (n - 1), 0.0))) if n > 1 else 0.0
        return {
            "mean": mean,
            "std": std,
            "n": n,
        }

    def _finalize_regime_effective_spread(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rv, acc in self._eff_spread_by_regime.items():
            if acc.count == 0:
                continue
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            result[label] = {
                "mean_usd": acc.mean,
                "std_usd": acc.sample_std,
                "n": acc.count,
            }
        return result
