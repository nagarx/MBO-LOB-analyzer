"""OrderFlowAnalyzer: OFI, cumulative delta, and flow-return prediction.

The highest-value flow analyzer: computes Order Flow Imbalance at multiple
timescales, cumulative delta (signed trade volume), aggressor ratios,
OFI-return cross-correlation (the key predictive signal), OFI autocorrelation,
flow intensity, intraday flow curves, trade imbalance, and weekday patterns.

OFI formula (Cont, Kukanov & Stoikov 2014):
    OFI_t = sum_i(sign_i * size_i)
    where sign is +1 for buy pressure and -1 for sell pressure.

All heavy computation is in the shared ``_flow_engine.py``; this analyzer
consumes ``DayFlow`` from ``DayContext`` and accumulates cross-day statistics.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass
from typing import Any, ClassVar

logger = logging.getLogger(__name__)

import numpy as np
from scipy import stats as sp_stats

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.flow._flow_engine import DayFlow, ScaledOFI
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import (
    EPS,
    NS_PER_MINUTE,
    NS_PER_SECOND,
    TRADING_MINUTES_PER_DAY,
)
from rawlobanalyzer.core.calendar import WEEKDAY_NAMES, weekday_from_date
from rawlobanalyzer.core.resampler import resample
from rawlobanalyzer.core.statistics import StreamingDistribution, acf as _acf, distribution_summary
from rawlobanalyzer.core.time_utils import REGIME_LABELS, rth_mask_utc, seconds_to_label, time_regime
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
class OrderFlowReport(BaseReport):
    """Report from the OrderFlowAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    ofi_distribution: dict[str, Any]
    cumulative_delta: dict[str, Any]
    aggressor_ratio: dict[str, Any]
    ofi_return_correlation: dict[str, Any]
    ofi_spread_correlation: dict[str, Any]
    ofi_components: dict[str, Any]
    ofi_autocorrelation: dict[str, Any]
    flow_intensity: dict[str, Any]
    intraday_flow_curve: dict[str, Any]
    trade_imbalance: dict[str, Any]
    weekday_patterns: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "ofi_distribution": self.ofi_distribution,
            "cumulative_delta": self.cumulative_delta,
            "aggressor_ratio": self.aggressor_ratio,
            "ofi_return_correlation": self.ofi_return_correlation,
            "ofi_spread_correlation": self.ofi_spread_correlation,
            "ofi_components": self.ofi_components,
            "ofi_autocorrelation": self.ofi_autocorrelation,
            "flow_intensity": self.flow_intensity,
            "intraday_flow_curve": self.intraday_flow_curve,
            "trade_imbalance": self.trade_imbalance,
            "weekday_patterns": self.weekday_patterns,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("ORDER FLOW ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.ofi_distribution:
            lines.append(format_subsection("OFI DISTRIBUTION (multi-scale)"))
            headers = ["Scale", "Mean", "Std", "Skew", "Kurt"]
            rows = []
            for label, stats in sorted(self.ofi_distribution.items()):
                if isinstance(stats, dict):
                    rows.append([
                        label,
                        stats.get("mean", np.nan),
                        stats.get("std", np.nan),
                        stats.get("skewness", np.nan),
                        stats.get("kurtosis", np.nan),
                    ])
            if rows:
                lines.append(format_table(headers, rows))

        if self.cumulative_delta:
            lines.append(format_subsection("CUMULATIVE DELTA"))
            lines.append(format_kv({
                "Mean end-of-day delta": self.cumulative_delta.get("mean_eod_delta", "N/A"),
                "Std end-of-day delta": self.cumulative_delta.get("std_eod_delta", "N/A"),
            }))

        if self.aggressor_ratio:
            lines.append(format_subsection("AGGRESSOR RATIO"))
            lines.append(format_kv({
                "Mean buyer fraction": self.aggressor_ratio.get("mean_buyer_fraction", "N/A"),
            }))

        if self.ofi_return_correlation:
            lines.append(format_subsection("OFI-RETURN CORRELATION"))
            for label, corr_data in sorted(self.ofi_return_correlation.items()):
                if isinstance(corr_data, dict) and "peak_lag" in corr_data:
                    lines.append(f"    {label}: peak_lag={corr_data['peak_lag']}, "
                                 f"peak_corr={corr_data.get('peak_corr', 'N/A'):.4f}")

        if self.ofi_spread_correlation:
            lines.append(format_subsection("OFI-SPREAD CORRELATION"))
            for label, corr_data in sorted(self.ofi_spread_correlation.items()):
                if isinstance(corr_data, dict) and "peak_lag" in corr_data:
                    lines.append(f"    {label}: peak_lag={corr_data['peak_lag']}, "
                                 f"peak_corr={corr_data.get('peak_corr', 'N/A'):.4f}")

        if self.ofi_components:
            lines.append(format_subsection("OFI COMPONENT BREAKDOWN"))
            overall = self.ofi_components.get("overall", {})
            if overall:
                lines.append(format_kv({
                    "Add fraction": f"{overall.get('add_fraction', 0):.3f}",
                    "Cancel fraction": f"{overall.get('cancel_fraction', 0):.3f}",
                    "Trade fraction": f"{overall.get('trade_fraction', 0):.3f}",
                }))
            by_regime = self.ofi_components.get("by_regime", {})
            if by_regime:
                headers = ["Regime", "Add frac", "Cancel frac", "Trade frac"]
                rows = []
                for regime, fracs in sorted(by_regime.items()):
                    if isinstance(fracs, dict):
                        rows.append([
                            regime,
                            f"{fracs.get('add_fraction', 0):.3f}",
                            f"{fracs.get('cancel_fraction', 0):.3f}",
                            f"{fracs.get('trade_fraction', 0):.3f}",
                        ])
                if rows:
                    lines.append(format_table(headers, rows))

        if self.flow_intensity:
            lines.append(format_subsection("FLOW INTENSITY BY REGIME"))
            headers = ["Regime", "Mean abs flow"]
            rows = []
            for regime, val in sorted(self.flow_intensity.items()):
                if isinstance(val, dict):
                    rows.append([regime, val.get("mean_abs_flow", np.nan)])
            if rows:
                lines.append(format_table(headers, rows))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-day accumulators
# ---------------------------------------------------------------------------


@dataclass
class _DayOFIRecord:
    date: str
    weekday: int
    eod_delta: float
    buyer_volume: float
    seller_volume: float
    total_abs_ofi: float
    n_trades: int
    n_ofi_events: int


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class OrderFlowAnalyzer(BaseAnalyzer[OrderFlowReport]):
    """Order Flow Imbalance and directional flow analysis.

    Computes OFI at multiple timescales, cumulative delta, aggressor ratios,
    OFI-return cross-correlation, OFI autocorrelation, flow intensity,
    intraday flow curves, trade imbalance, and weekday patterns.
    """

    name: ClassVar[str] = "OrderFlowAnalyzer"
    description: ClassVar[str] = "OFI, cumulative delta, and flow-return prediction"

    lob_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "mid_price", "best_bid", "best_ask", "spread",
    ]
    mbo_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "order_id", "action", "side", "price", "size",
    ]
    needs_mbo: ClassVar[bool] = True
    needs_returns: ClassVar[bool] = True
    needs_spreads: ClassVar[bool] = True
    needs_flow: ClassVar[bool] = True

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._n_days: int = 0
        self._n_days_skipped: int = 0
        self._flow_cfg = config.thresholds.flow

        self._day_records: list[_DayOFIRecord] = []

        self._ofi_per_scale: dict[str, StreamingDistribution] = {}
        self._norm_ofi_per_scale: dict[str, StreamingDistribution] = {}
        self._ofi_return_corr: dict[str, list[np.ndarray]] = {}
        self._ofi_spread_corr: dict[str, list[np.ndarray]] = {}
        self._ofi_acf_per_scale: dict[str, list[np.ndarray]] = {}

        self._regime_abs_flow: dict[int, list[float]] = {}
        self._intraday_ofi_bins: list[np.ndarray] = []

        self._trade_imbalance_per_scale: dict[str, list[np.ndarray]] = {}

        self._ofi_component_per_day: list[dict[str, float]] = []

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        flow = ctx.day_flow
        returns = ctx.day_returns
        self._symbol = day.symbol

        if flow is None or flow.n_trades == 0:
            self._n_days_skipped += 1
            logger.debug("OrderFlow: skipping %s (no flow/trades)", day.date)
            return

        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        rth_trades = flow.rth_mask_trades
        weekday = weekday_from_date(day.date)

        # --- Cumulative delta (exclude SIDE_NONE trades) ---
        dir_mask = flow.directional_mask
        dir_sizes = flow.trade_sizes[dir_mask].astype(np.float64).copy()
        dir_sides = flow.trade_sides[dir_mask]
        dir_sizes[dir_sides != SIDE_BID] *= -1.0
        if len(rth_trades) > 0 and np.any(rth_trades):
            dir_rth = rth_trades[dir_mask]
            rth_signed = dir_sizes[dir_rth] if np.any(dir_rth) else dir_sizes
            eod_delta = float(np.sum(rth_signed))
        else:
            eod_delta = float(np.sum(dir_sizes))

        # --- Aggressor ratio (exclude SIDE_NONE trades) ---
        buyer_vol = float(np.sum(flow.trade_sizes[dir_mask & (flow.trade_sides == SIDE_BID)].astype(np.float64)))
        seller_vol = float(np.sum(flow.trade_sizes[dir_mask & (flow.trade_sides == SIDE_ASK)].astype(np.float64)))

        # --- Flow intensity by regime ---
        if len(flow.ofi_timestamps_ns) > 0:
            regimes = time_regime(
                flow.ofi_timestamps_ns,
                utc_offset_hours=self._utc_off,
            )
            abs_ofi = np.abs(flow.ofi_values)
            for rv in range(7):
                mask = regimes == rv
                if np.any(mask):
                    if rv not in self._regime_abs_flow:
                        self._regime_abs_flow[rv] = []
                    self._regime_abs_flow[rv].append(float(np.sum(abs_ofi[mask])))

        total_abs_ofi = float(np.sum(np.abs(flow.ofi_values)))

        self._day_records.append(_DayOFIRecord(
            date=day.date, weekday=weekday, eod_delta=eod_delta,
            buyer_volume=buyer_vol, seller_volume=seller_vol,
            total_abs_ofi=total_abs_ofi,
            n_trades=flow.n_trades, n_ofi_events=flow.n_ofi_events,
        ))

        # --- OFI component decomposition ---
        if flow.n_ofi_events > 0:
            abs_add = float(np.sum(np.abs(flow.ofi_add_values)))
            abs_cancel = float(np.sum(np.abs(flow.ofi_cancel_values)))
            abs_trade = float(np.sum(np.abs(flow.ofi_trade_values)))
            total_comp = abs_add + abs_cancel + abs_trade

            comp_record: dict[str, float] = {
                "abs_add": abs_add, "abs_cancel": abs_cancel, "abs_trade": abs_trade,
            }

            if len(flow.ofi_timestamps_ns) > 0:
                regimes = time_regime(
                    flow.ofi_timestamps_ns,
                    utc_offset_hours=self._utc_off,
                )
                for rv in range(7):
                    rmask = regimes == rv
                    if np.any(rmask):
                        comp_record[f"abs_add_{rv}"] = float(np.sum(np.abs(flow.ofi_add_values[rmask])))
                        comp_record[f"abs_cancel_{rv}"] = float(np.sum(np.abs(flow.ofi_cancel_values[rmask])))
                        comp_record[f"abs_trade_{rv}"] = float(np.sum(np.abs(flow.ofi_trade_values[rmask])))

            self._ofi_component_per_day.append(comp_record)

        # --- Per-scale OFI distribution + ACF ---
        for label, sofi in flow.scaled_ofi.items():
            filled = sofi.net_ofi[sofi.counts > 0]
            if len(filled) > 0:
                if label not in self._ofi_per_scale:
                    self._ofi_per_scale[label] = StreamingDistribution()
                self._ofi_per_scale[label].add_batch(filled)

            norm_filled = sofi.normalized_ofi[sofi.counts > 0]
            valid_norm = norm_filled[np.isfinite(norm_filled)]
            if len(valid_norm) > 0:
                if label not in self._norm_ofi_per_scale:
                    self._norm_ofi_per_scale[label] = StreamingDistribution()
                self._norm_ofi_per_scale[label].add_batch(valid_norm)

            if len(filled) > self._flow_cfg.max_acf_lag + 1:
                acf = _acf(filled, self._flow_cfg.max_acf_lag)
                if label not in self._ofi_acf_per_scale:
                    self._ofi_acf_per_scale[label] = []
                self._ofi_acf_per_scale[label].append(acf)

        # --- OFI-return cross-correlation ---
        if returns is not None:
            self._compute_ofi_return_corr(flow, returns)

        # --- OFI-spread cross-correlation ---
        spreads = ctx.day_spreads
        if spreads is not None:
            self._compute_ofi_spread_corr(flow, spreads)

        # --- Intraday flow curve (1-min bins) ---
        self._compute_intraday_curve(flow)

        # --- Trade imbalance per scale ---
        self._compute_trade_imbalance(flow)

    def _compute_ofi_return_corr(self, flow: DayFlow, returns) -> None:
        """Cross-correlate normalized OFI with binned returns at multiple lags.

        Both OFI and returns are resampled onto a canonical RTH grid (when
        ``day_epoch_ns`` is provided to the engines), so their bin timestamps
        are identical. We align by bin timestamp to guarantee that OFI in bin
        ``t`` is paired with the return *from* bin ``t`` to bin ``t+1``.
        """
        cfg = self._flow_cfg
        for label, sofi in flow.scaled_ofi.items():
            if label not in returns.scaled:
                continue
            sr = returns.scaled[label]

            ofi_bins = sofi.bin_timestamps_ns
            ret_bins = sr.bin_timestamps_ns

            if len(ofi_bins) == 0 or len(ret_bins) == 0:
                continue

            shared, ofi_idx, ret_idx = np.intersect1d(
                ofi_bins, ret_bins, return_indices=True,
            )
            if len(shared) < cfg.flow_return_n_lags + 2:
                continue

            ofi_use = sofi.normalized_ofi[ofi_idx]
            ret_use = sr.returns[ret_idx]

            finite = np.isfinite(ofi_use) & np.isfinite(ret_use)
            ofi_use = np.where(finite, ofi_use, 0.0)
            ret_use = np.where(finite, ret_use, 0.0)

            n = len(ofi_use)
            corrs = np.full(cfg.flow_return_n_lags, np.nan)
            for lag in range(cfg.flow_return_n_lags):
                if lag == 0:
                    corrs[lag] = _safe_corr(ofi_use, ret_use)
                elif lag < n:
                    corrs[lag] = _safe_corr(ofi_use[:-lag], ret_use[lag:])

            if label not in self._ofi_return_corr:
                self._ofi_return_corr[label] = []
            self._ofi_return_corr[label].append(corrs)

    def _compute_ofi_spread_corr(self, flow: DayFlow, spreads) -> None:
        """Cross-correlate normalized OFI with binned spread changes at multiple lags.

        Alignment uses bin timestamps, same approach as OFI-return correlation.
        """
        cfg = self._flow_cfg
        for label, sofi in flow.scaled_ofi.items():
            if label not in spreads.scaled:
                continue
            ss = spreads.scaled[label]

            ofi_bins = sofi.bin_timestamps_ns
            # bin_edges_ns has n_bins+1 elements (left edge of each bin +
            # right edge of last bin).  Use only left edges so indices
            # align with mean_spreads_usd which has n_bins elements.
            spread_bins = ss.bin_edges_ns[:-1] if len(ss.bin_edges_ns) > 0 else ss.bin_edges_ns
            spread_vals = ss.mean_spreads_usd

            if len(ofi_bins) == 0 or len(spread_bins) == 0:
                continue

            shared, ofi_idx, sp_idx = np.intersect1d(
                ofi_bins, spread_bins, return_indices=True,
            )
            if len(shared) < cfg.flow_return_n_lags + 3:
                continue

            sp_aligned = spread_vals[sp_idx]
            valid = np.isfinite(sp_aligned)
            if np.sum(valid) < cfg.flow_return_n_lags + 3:
                continue

            ofi_valid = sofi.normalized_ofi[ofi_idx][valid]
            sp_valid = sp_aligned[valid]
            spread_changes = np.diff(sp_valid)
            ofi_trimmed = ofi_valid[:-1]

            n = len(ofi_trimmed)
            if n < cfg.flow_return_n_lags + 2:
                continue

            corrs = np.full(cfg.flow_return_n_lags, np.nan)
            for lag in range(cfg.flow_return_n_lags):
                if lag == 0:
                    corrs[lag] = _safe_corr(ofi_trimmed, spread_changes)
                elif lag < n:
                    corrs[lag] = _safe_corr(ofi_trimmed[:-lag], spread_changes[lag:])

            if label not in self._ofi_spread_corr:
                self._ofi_spread_corr[label] = []
            self._ofi_spread_corr[label].append(corrs)

    def _compute_intraday_curve(self, flow: DayFlow) -> None:
        """Compute 1-minute OFI profile across the trading day."""
        if flow.n_ofi_events == 0:
            return
        rth = rth_mask_utc(
            flow.ofi_timestamps_ns,
            utc_offset_hours=self._utc_off,
        )
        ts_rth = flow.ofi_timestamps_ns[rth]
        vals_rth = flow.ofi_values[rth]
        if len(ts_rth) == 0:
            return

        bin_minutes = self._flow_cfg.intraday_bin_minutes
        res_ns = int(bin_minutes * 60 * NS_PER_SECOND)
        resampled = resample(ts_rth, vals_rth, res_ns, agg="sum", label="intraday")

        n_bins = TRADING_MINUTES_PER_DAY // bin_minutes
        curve = np.full(n_bins, np.nan, dtype=np.float64)
        actual = resampled.values
        length = min(len(actual), n_bins)
        curve[:length] = actual[:length]
        self._intraday_ofi_bins.append(curve)

    def _compute_trade_imbalance(self, flow: DayFlow) -> None:
        """Net signed trade volume per timescale bin (directional trades only)."""
        if flow.n_trades == 0:
            return
        rth = flow.rth_mask_trades
        dir_mask = flow.directional_mask
        if len(rth) == 0 or not np.any(rth):
            return

        combined = rth & dir_mask
        if not np.any(combined):
            return

        ts_rth = flow.trade_timestamps_ns[combined]
        signed = flow.trade_sizes[combined].astype(np.float64).copy()
        signed[flow.trade_sides[combined] != SIDE_BID] *= -1.0

        for scale_s in self._flow_cfg.ofi_timescales_seconds:
            res_ns = int(scale_s * NS_PER_SECOND)
            label = seconds_to_label(scale_s)
            resampled = resample(ts_rth, signed, res_ns, agg="sum", label=label)
            filled = resampled.values[resampled.counts > 0]
            if len(filled) > 0:
                if label not in self._trade_imbalance_per_scale:
                    self._trade_imbalance_per_scale[label] = []
                self._trade_imbalance_per_scale[label].append(filled)

    def finalize(self) -> OrderFlowReport:
        return OrderFlowReport(
            symbol=self._symbol,
            n_days=self._n_days,
            ofi_distribution=self._finalize_ofi_distribution(),
            cumulative_delta=self._finalize_cumulative_delta(),
            aggressor_ratio=self._finalize_aggressor_ratio(),
            ofi_return_correlation=self._finalize_ofi_return_corr(),
            ofi_spread_correlation=self._finalize_ofi_spread_corr(),
            ofi_components=self._finalize_ofi_components(),
            ofi_autocorrelation=self._finalize_ofi_acf(),
            flow_intensity=self._finalize_flow_intensity(),
            intraday_flow_curve=self._finalize_intraday_curve(),
            trade_imbalance=self._finalize_trade_imbalance(),
            weekday_patterns=self._finalize_weekday(),
        )

    def _finalize_ofi_distribution(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label, sd in self._ofi_per_scale.items():
            ds = sd.distribution_summary()
            entry = ds.to_dict()
            if label in self._norm_ofi_per_scale:
                norm_ds = self._norm_ofi_per_scale[label].distribution_summary()
                entry["normalized"] = norm_ds.to_dict()
            result[label] = entry
        return result

    def _finalize_cumulative_delta(self) -> dict[str, Any]:
        if not self._day_records:
            return {}
        deltas = np.array([r.eod_delta for r in self._day_records])
        return {
            "mean_eod_delta": float(np.mean(deltas)),
            "std_eod_delta": float(np.std(deltas, ddof=1)) if len(deltas) > 1 else 0.0,
            "median_eod_delta": float(np.median(deltas)),
            "daily_deltas": [
                {"date": r.date, "eod_delta": r.eod_delta}
                for r in self._day_records
            ],
        }

    def _finalize_aggressor_ratio(self) -> dict[str, Any]:
        if not self._day_records:
            return {}
        fractions = []
        for r in self._day_records:
            total = r.buyer_volume + r.seller_volume
            if total > 0:
                fractions.append(r.buyer_volume / total)
        if not fractions:
            return {}
        arr = np.array(fractions)
        return {
            "mean_buyer_fraction": float(np.mean(arr)),
            "std_buyer_fraction": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "daily_fractions": fractions,
        }

    def _finalize_ofi_return_corr(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label, corr_list in self._ofi_return_corr.items():
            stacked = np.array(corr_list)
            mean_corr = np.nanmean(stacked, axis=0)
            peak_idx = int(np.argmax(np.abs(mean_corr)))
            result[label] = {
                "mean_correlation_by_lag": mean_corr.tolist(),
                "peak_lag": peak_idx,
                "peak_corr": float(mean_corr[peak_idx]),
                "n_days": len(corr_list),
            }
        return result

    def _finalize_ofi_spread_corr(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label, corr_list in self._ofi_spread_corr.items():
            stacked = np.array(corr_list)
            mean_corr = np.nanmean(stacked, axis=0)
            peak_idx = int(np.argmax(np.abs(mean_corr)))
            result[label] = {
                "mean_correlation_by_lag": mean_corr.tolist(),
                "peak_lag": peak_idx,
                "peak_corr": float(mean_corr[peak_idx]),
                "n_days": len(corr_list),
            }
        return result

    def _finalize_ofi_components(self) -> dict[str, Any]:
        if not self._ofi_component_per_day:
            return {}

        total_add = sum(r["abs_add"] for r in self._ofi_component_per_day)
        total_cancel = sum(r["abs_cancel"] for r in self._ofi_component_per_day)
        total_trade = sum(r["abs_trade"] for r in self._ofi_component_per_day)
        grand = total_add + total_cancel + total_trade

        overall: dict[str, float] = {}
        if grand > EPS:
            overall = {
                "add_fraction": total_add / grand,
                "cancel_fraction": total_cancel / grand,
                "trade_fraction": total_trade / grand,
            }

        by_regime: dict[str, dict[str, float]] = {}
        for rv in range(7):
            add_key, cancel_key, trade_key = f"abs_add_{rv}", f"abs_cancel_{rv}", f"abs_trade_{rv}"
            radd = sum(r.get(add_key, 0.0) for r in self._ofi_component_per_day)
            rcancel = sum(r.get(cancel_key, 0.0) for r in self._ofi_component_per_day)
            rtrade = sum(r.get(trade_key, 0.0) for r in self._ofi_component_per_day)
            rtotal = radd + rcancel + rtrade
            if rtotal > EPS:
                label = REGIME_LABELS.get(rv, f"regime_{rv}")
                by_regime[label] = {
                    "add_fraction": radd / rtotal,
                    "cancel_fraction": rcancel / rtotal,
                    "trade_fraction": rtrade / rtotal,
                }

        return {"overall": overall, "by_regime": by_regime}

    def _finalize_ofi_acf(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label, acf_list in self._ofi_acf_per_scale.items():
            stacked = np.array(acf_list)
            mean_acf = np.nanmean(stacked, axis=0)
            result[label] = {
                "mean_acf": mean_acf.tolist(),
                "lag1": float(mean_acf[0]) if len(mean_acf) > 0 else np.nan,
            }
        return result

    def _finalize_flow_intensity(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rv, vals in self._regime_abs_flow.items():
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            arr = np.array(vals)
            result[label] = {
                "mean_abs_flow": float(np.mean(arr)),
                "std_abs_flow": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n_days": len(arr),
            }
        return result

    def _finalize_intraday_curve(self) -> dict[str, Any]:
        if not self._intraday_ofi_bins:
            return {}
        stacked = np.array(self._intraday_ofi_bins)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            mean_curve = np.nanmean(stacked, axis=0)
            std_curve = np.nanstd(stacked, axis=0)
        return {
            "bin_minutes": self._flow_cfg.intraday_bin_minutes,
            "n_bins": len(mean_curve),
            "mean_ofi": mean_curve.tolist(),
            "std_ofi": std_curve.tolist(),
            "n_days": len(self._intraday_ofi_bins),
        }

    def _finalize_trade_imbalance(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for label, arrays in self._trade_imbalance_per_scale.items():
            combined = np.concatenate(arrays)
            ds = distribution_summary(combined)
            result[label] = ds.to_dict()
        return result

    def _finalize_weekday(self) -> dict[str, Any]:
        if not self._day_records:
            return {}
        result: dict[str, Any] = {}
        for wd in range(5):
            recs = [r for r in self._day_records if r.weekday == wd]
            if not recs:
                continue
            deltas = [r.eod_delta for r in recs]
            ofi_totals = [r.total_abs_ofi for r in recs]
            result[WEEKDAY_NAMES[wd]] = {
                "mean_eod_delta": float(np.mean(deltas)),
                "mean_abs_ofi": float(np.mean(ofi_totals)),
                "n_days": len(recs),
            }
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with NaN safety."""
    valid = np.isfinite(x) & np.isfinite(y)
    if np.sum(valid) < 3:
        return np.nan
    xv, yv = x[valid], y[valid]
    sx, sy = np.std(xv), np.std(yv)
    if sx < EPS or sy < EPS:
        return 0.0
    return float(np.corrcoef(xv, yv)[0, 1])



