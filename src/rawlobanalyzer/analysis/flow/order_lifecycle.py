"""OrderLifecycleAnalyzer: order duration, fill rate, and cancel dynamics.

Tracks individual orders across their lifecycle using ``order_id`` from MBO
events. Computes order lifetime distribution, fill rate, cancel-to-add ratio,
modify patterns, event transition matrix, duration-by-size correlation, and
regime-conditional lifecycle metrics.

This analyzer is fundamentally different from OrderFlowAnalyzer and
TradeAnalyzer: it maintains a stateful dict of active orders, bounded by
``max_active_orders`` to cap memory usage. Orders not resolved within
``order_lifetime_max_seconds`` are evicted and counted as expired.

Memory model: ~100 MB for 500K simultaneous active orders.
Eviction: two-tier strategy -- (1) time-based eviction of orders older than
``order_lifetime_max_seconds``, (2) hard-cap eviction of the oldest order
by ``add_ts`` when still at ``max_active_orders`` after tier 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple

import numpy as np

from rawlobanalyzer.analysis.base import BaseAnalyzer
from rawlobanalyzer.analysis.context import DayContext
from rawlobanalyzer.analysis.registry import register_analyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.core.constants import EPS, NS_PER_SECOND
from rawlobanalyzer.core.statistics import StreamingDistribution, WelfordAccumulator, distribution_summary
from rawlobanalyzer.core.time_utils import N_REGIMES, REGIME_LABELS, time_regime
from rawlobanalyzer.io.schema import (
    ACTION_ADD,
    ACTION_CANCEL,
    ACTION_CLEAR,
    ACTION_FILL,
    ACTION_LABELS,
    ACTION_MODIFY,
    ACTION_TRADE,
    SIDE_ASK,
    SIDE_BID,
)
from rawlobanalyzer.reports.base_report import BaseReport
from rawlobanalyzer.reports.formatters import (
    format_kv,
    format_section,
    format_subsection,
    format_table,
)


class _OrderState(NamedTuple):
    """Lightweight state for a tracked active order."""
    add_ts: int
    side: int
    price: int
    size: int
    remaining_size: int
    n_modifies: int
    n_partial_fills: int
    regime: int


# Terminal actions that end an order's lifecycle
_TERMINAL_ACTIONS = frozenset({ACTION_CANCEL, ACTION_TRADE, ACTION_FILL, ACTION_CLEAR})


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


@dataclass
class OrderLifecycleReport(BaseReport):
    """Report from the OrderLifecycleAnalyzer."""

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    symbol: str
    n_days: int
    order_lifetime: dict[str, Any]
    fill_rate: dict[str, Any]
    cancel_to_add_ratio: dict[str, Any]
    modify_patterns: dict[str, Any]
    transition_matrix: dict[str, Any]
    duration_by_size: dict[str, Any]
    regime_lifecycle: dict[str, Any]
    partial_fill_patterns: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "n_days": self.n_days,
            "order_lifetime": self.order_lifetime,
            "fill_rate": self.fill_rate,
            "cancel_to_add_ratio": self.cancel_to_add_ratio,
            "modify_patterns": self.modify_patterns,
            "transition_matrix": self.transition_matrix,
            "duration_by_size": self.duration_by_size,
            "regime_lifecycle": self.regime_lifecycle,
            "partial_fill_patterns": self.partial_fill_patterns,
            "_meta": self._meta_dict(),
        }

    def summary(self) -> str:
        lines: list[str] = []
        lines.append(format_section("ORDER LIFECYCLE ANALYSIS REPORT"))
        lines.append(format_kv({"Symbol": self.symbol, "Days analyzed": self.n_days}))

        if self.order_lifetime:
            lines.append(format_subsection("ORDER LIFETIME"))
            lines.append(format_kv({
                "Mean (s)": self.order_lifetime.get("mean_seconds", "N/A"),
                "Median (s)": self.order_lifetime.get("median_seconds", "N/A"),
                "Total resolved": self.order_lifetime.get("n_resolved", "N/A"),
                "Expired": self.order_lifetime.get("n_expired", "N/A"),
            }))

        if self.fill_rate:
            lines.append(format_subsection("FILL RATE"))
            lines.append(format_kv({
                "Overall fill rate": self.fill_rate.get("overall_fill_rate", "N/A"),
                "Bid fill rate": self.fill_rate.get("bid_fill_rate", "N/A"),
                "Ask fill rate": self.fill_rate.get("ask_fill_rate", "N/A"),
            }))

        if self.cancel_to_add_ratio:
            lines.append(format_subsection("CANCEL-TO-ADD RATIO"))
            lines.append(format_kv({
                "Mean ratio": self.cancel_to_add_ratio.get("mean_ratio", "N/A"),
            }))

        if self.modify_patterns:
            lines.append(format_subsection("MODIFY PATTERNS"))
            lines.append(format_kv({
                "Modified fraction": self.modify_patterns.get("modified_fraction", "N/A"),
                "Mean modifies/order": self.modify_patterns.get("mean_modifies_per_order", "N/A"),
            }))

        if self.transition_matrix:
            lines.append(format_subsection("EVENT TRANSITION MATRIX"))
            matrix = self.transition_matrix.get("matrix", {})
            if matrix:
                actions = sorted(matrix.keys())
                headers = ["From \\ To"] + actions
                rows = []
                for a in actions:
                    row: list[Any] = [a]
                    for b in actions:
                        row.append(matrix.get(a, {}).get(b, 0.0))
                    rows.append(row)
                lines.append(format_table(headers, rows))

        if self.partial_fill_patterns:
            lines.append(format_subsection("PARTIAL FILL PATTERNS"))
            lines.append(format_kv({
                "Partial fill fraction": self.partial_fill_patterns.get("partial_fill_fraction", "N/A"),
                "Mean fills/order": self.partial_fill_patterns.get("mean_fills_per_order", "N/A"),
            }))

        if self.regime_lifecycle:
            lines.append(format_subsection("REGIME-CONDITIONAL LIFECYCLE"))
            headers = ["Regime", "Fill rate", "Mean lifetime (s)", "Cancel ratio"]
            rows = []
            for regime, stats in sorted(self.regime_lifecycle.items()):
                if isinstance(stats, dict):
                    rows.append([
                        regime,
                        stats.get("fill_rate", np.nan),
                        stats.get("mean_lifetime_seconds", np.nan),
                        stats.get("cancel_ratio", np.nan),
                    ])
            if rows:
                lines.append(format_table(headers, rows))

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Analyzer
# ---------------------------------------------------------------------------


@register_analyzer
class OrderLifecycleAnalyzer(BaseAnalyzer[OrderLifecycleReport]):
    """Order lifecycle tracking and behavioral analysis.

    Tracks individual orders via ``order_id`` across Add/Modify/Cancel/Trade/Fill
    events. Memory-bounded by ``max_active_orders`` with LRU-style eviction.
    """

    name: ClassVar[str] = "OrderLifecycleAnalyzer"
    description: ClassVar[str] = "Order duration, fill rate, cancel dynamics, modify patterns"

    lob_columns: ClassVar[list[str] | None] = ["timestamp_ns"]
    mbo_columns: ClassVar[list[str] | None] = [
        "timestamp_ns", "order_id", "action", "side", "price", "size",
    ]
    needs_mbo: ClassVar[bool] = True
    needs_returns: ClassVar[bool] = False
    needs_flow: ClassVar[bool] = False

    def __init__(self, config: AnalysisConfig) -> None:
        super().__init__(config)
        self._symbol: str = config.symbol
        self._n_days: int = 0
        self._flow_cfg = config.thresholds.flow

        self._lifetime_dist = StreamingDistribution()
        self._bid_lifetime_acc = WelfordAccumulator()
        self._ask_lifetime_acc = WelfordAccumulator()

        self._n_resolved: int = 0
        self._n_expired: int = 0
        self._n_filled: int = 0
        self._n_cancelled: int = 0
        self._n_bid: int = 0
        self._n_ask: int = 0
        self._n_bid_filled: int = 0
        self._n_ask_filled: int = 0

        self._n_modified_orders: int = 0
        self._total_modifies: int = 0
        self._max_modifies: int = 0

        self._size_acc = WelfordAccumulator()
        self._log_dur_log_size = StreamingDistribution(reservoir_size=200_000)
        self._log_size_reservoir = StreamingDistribution(reservoir_size=200_000)

        self._add_counts: list[int] = []
        self._cancel_counts: list[int] = []

        self._transition_counts: dict[int, dict[int, int]] = {}

        self._regime_fill_counts: dict[int, int] = {}
        self._regime_cancel_counts: dict[int, int] = {}
        self._regime_total_counts: dict[int, int] = {}
        self._regime_lifetime_acc: dict[int, WelfordAccumulator] = {}

        self._fills_dist = WelfordAccumulator()
        self._n_with_partials: int = 0
        self._n_partial_then_cancel: int = 0
        self._max_fills_per_order: int = 0
        self._n_filled_orders: int = 0

    def process_day(self, ctx: DayContext) -> None:
        day = ctx.day
        self._symbol = day.symbol
        self._n_days += 1
        self._utc_off = ctx.utc_offset_hours

        mbo = day.mbo
        if mbo is None:
            return

        required = {"timestamp_ns", "order_id", "action", "side", "price", "size"}
        if not required.issubset(set(mbo.column_names)):
            return

        ts = mbo.column("timestamp_ns").to_numpy()
        order_ids = mbo.column("order_id").to_numpy()
        actions = mbo.column("action").to_numpy()
        sides = mbo.column("side").to_numpy()
        prices = mbo.column("price").to_numpy()
        sizes = mbo.column("size").to_numpy()

        n_events = len(ts)
        if n_events == 0:
            return

        regimes = time_regime(
            ts, utc_offset_hours=self._utc_off,
        )

        max_lifetime_ns = int(self._flow_cfg.order_lifetime_max_seconds * NS_PER_SECOND)
        max_active = self._flow_cfg.max_active_orders

        active: dict[int, _OrderState] = {}
        day_adds = 0
        day_cancels = 0
        prev_action: dict[int, int] = {}

        for i in range(n_events):
            oid = int(order_ids[i])
            action = int(actions[i])
            ts_i = int(ts[i])

            # Transition matrix
            if oid in prev_action:
                pa = prev_action[oid]
                if pa not in self._transition_counts:
                    self._transition_counts[pa] = {}
                if action not in self._transition_counts[pa]:
                    self._transition_counts[pa][action] = 0
                self._transition_counts[pa][action] += 1
            prev_action[oid] = action

            if action == ACTION_ADD:
                day_adds += 1
                if len(active) >= max_active:
                    self._evict_oldest(active, ts_i, max_lifetime_ns)
                active[oid] = _OrderState(
                    add_ts=ts_i,
                    side=int(sides[i]),
                    price=int(prices[i]),
                    size=int(sizes[i]),
                    remaining_size=int(sizes[i]),
                    n_modifies=0,
                    n_partial_fills=0,
                    regime=int(regimes[i]),
                )

            elif action == ACTION_MODIFY:
                if oid in active:
                    st = active[oid]
                    active[oid] = st._replace(n_modifies=st.n_modifies + 1)

            elif action in _TERMINAL_ACTIONS:
                if action == ACTION_CANCEL:
                    day_cancels += 1

                if oid in active:
                    is_fill_action = action in (ACTION_TRADE, ACTION_FILL)

                    if is_fill_action:
                        st = active[oid]
                        fill_size = int(sizes[i])
                        new_remaining = st.remaining_size - fill_size

                        if new_remaining > 0:
                            active[oid] = st._replace(
                                remaining_size=new_remaining,
                                n_partial_fills=st.n_partial_fills + 1,
                            )
                            continue
                        # Fully filled (or over-filled): resolve the order
                        st = active.pop(oid)
                        n_fill_events = st.n_partial_fills + 1
                    else:
                        st = active.pop(oid)
                        n_fill_events = st.n_partial_fills

                    lifetime_ns = ts_i - st.add_ts
                    if lifetime_ns < 0:
                        lifetime_ns = 0
                    lifetime_s = lifetime_ns / NS_PER_SECOND

                    self._lifetime_dist.add_batch(np.array([lifetime_s]))
                    self._n_resolved += 1

                    if st.side == SIDE_BID:
                        self._n_bid += 1
                        self._bid_lifetime_acc.update(lifetime_s)
                        if is_fill_action:
                            self._n_bid_filled += 1
                    elif st.side == SIDE_ASK:
                        self._n_ask += 1
                        self._ask_lifetime_acc.update(lifetime_s)
                        if is_fill_action:
                            self._n_ask_filled += 1

                    if is_fill_action:
                        self._n_filled += 1
                        self._n_filled_orders += 1
                        self._fills_dist.update(float(n_fill_events))
                        if n_fill_events > 1:
                            self._n_with_partials += 1
                        self._max_fills_per_order = max(self._max_fills_per_order, n_fill_events)
                    elif action == ACTION_CANCEL:
                        self._n_cancelled += 1
                        if n_fill_events > 0:
                            self._n_partial_then_cancel += 1
                            self._fills_dist.update(float(n_fill_events))
                            self._max_fills_per_order = max(self._max_fills_per_order, n_fill_events)

                    if st.n_modifies > 0:
                        self._n_modified_orders += 1
                    self._total_modifies += st.n_modifies
                    self._max_modifies = max(self._max_modifies, st.n_modifies)

                    self._log_dur_log_size.add_batch(np.array([np.log1p(lifetime_s)]))
                    self._log_size_reservoir.add_batch(np.array([np.log1p(float(st.size))]))

                    rv = st.regime
                    if rv not in self._regime_total_counts:
                        self._regime_total_counts[rv] = 0
                        self._regime_fill_counts[rv] = 0
                        self._regime_cancel_counts[rv] = 0
                        self._regime_lifetime_acc[rv] = WelfordAccumulator()

                    self._regime_total_counts[rv] += 1
                    self._regime_lifetime_acc[rv].update(lifetime_s)
                    if is_fill_action:
                        self._regime_fill_counts[rv] += 1
                    elif action == ACTION_CANCEL:
                        self._regime_cancel_counts[rv] += 1

                if oid in prev_action:
                    del prev_action[oid]

        # Evict remaining active orders as expired
        for oid, st in active.items():
            self._n_expired += 1

        self._add_counts.append(day_adds)
        self._cancel_counts.append(day_cancels)

    def _evict_oldest(
        self, active: dict[int, _OrderState], current_ts: int, max_ns: int,
    ) -> None:
        """Two-tier eviction to enforce the active-order hard cap.

        Tier 1: evict all orders older than ``max_ns`` (time-based cleanup).
        Tier 2: if still at capacity, evict the single oldest order by
        ``add_ts`` to make room. This guarantees ``len(active)`` never
        exceeds ``max_active_orders``.
        """
        to_evict = [
            oid for oid, st in active.items()
            if current_ts - st.add_ts > max_ns
        ]
        for oid in to_evict:
            active.pop(oid)
            self._n_expired += 1

        max_active = self._flow_cfg.max_active_orders
        if len(active) >= max_active:
            oldest_oid = min(active, key=lambda oid: active[oid].add_ts)
            active.pop(oldest_oid)
            self._n_expired += 1

    def finalize(self) -> OrderLifecycleReport:
        return OrderLifecycleReport(
            symbol=self._symbol,
            n_days=self._n_days,
            order_lifetime=self._finalize_lifetime(),
            fill_rate=self._finalize_fill_rate(),
            cancel_to_add_ratio=self._finalize_cancel_add(),
            modify_patterns=self._finalize_modify(),
            transition_matrix=self._finalize_transition(),
            duration_by_size=self._finalize_duration_size(),
            regime_lifecycle=self._finalize_regime(),
            partial_fill_patterns=self._finalize_partial_fills(),
        )

    def _finalize_lifetime(self) -> dict[str, Any]:
        if self._lifetime_dist.count == 0:
            return {}
        ds = self._lifetime_dist.distribution_summary()
        result = ds.to_dict()
        result["mean_seconds"] = ds.mean
        result["median_seconds"] = ds.percentiles.get("p50", np.nan)
        result["n_resolved"] = self._n_resolved
        result["n_expired"] = self._n_expired
        result["bid_mean_seconds"] = self._bid_lifetime_acc.mean if self._bid_lifetime_acc.count > 0 else np.nan
        result["ask_mean_seconds"] = self._ask_lifetime_acc.mean if self._ask_lifetime_acc.count > 0 else np.nan
        return result

    def _finalize_fill_rate(self) -> dict[str, Any]:
        if self._n_resolved == 0:
            return {}
        n_total = self._n_resolved
        return {
            "overall_fill_rate": self._n_filled / max(n_total, 1),
            "bid_fill_rate": self._n_bid_filled / max(self._n_bid, 1),
            "ask_fill_rate": self._n_ask_filled / max(self._n_ask, 1),
            "n_total": n_total,
            "n_filled": self._n_filled,
            "n_cancelled": self._n_cancelled,
        }

    def _finalize_cancel_add(self) -> dict[str, Any]:
        if not self._add_counts:
            return {}
        adds = np.array(self._add_counts, dtype=np.float64)
        cancels = np.array(self._cancel_counts, dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratios = np.where(adds > 0, cancels / adds, np.nan)
        valid = np.isfinite(ratios)
        return {
            "mean_ratio": float(np.nanmean(ratios)) if np.any(valid) else np.nan,
            "daily_ratios": ratios.tolist(),
            "total_adds": int(np.sum(adds)),
            "total_cancels": int(np.sum(cancels)),
        }

    def _finalize_modify(self) -> dict[str, Any]:
        if self._n_resolved == 0:
            return {}
        n = self._n_resolved
        modified_frac = self._n_modified_orders / max(n, 1)
        mean_mods = self._total_modifies / max(n, 1)
        mean_when_modified = (
            self._total_modifies / max(self._n_modified_orders, 1)
            if self._n_modified_orders > 0 else 0.0
        )
        return {
            "modified_fraction": modified_frac,
            "mean_modifies_per_order": mean_mods,
            "mean_modifies_when_modified": mean_when_modified,
            "max_modifies": self._max_modifies,
            "n_orders": n,
        }

    def _finalize_transition(self) -> dict[str, Any]:
        if not self._transition_counts:
            return {}
        all_actions = set()
        for from_a, to_dict in self._transition_counts.items():
            all_actions.add(from_a)
            all_actions.update(to_dict.keys())

        matrix: dict[str, dict[str, float]] = {}
        for from_a in sorted(all_actions):
            from_label = ACTION_LABELS.get(from_a, f"action_{from_a}")
            row_total = sum(self._transition_counts.get(from_a, {}).values())
            row: dict[str, float] = {}
            for to_a in sorted(all_actions):
                to_label = ACTION_LABELS.get(to_a, f"action_{to_a}")
                count = self._transition_counts.get(from_a, {}).get(to_a, 0)
                row[to_label] = count / max(row_total, 1)
            matrix[from_label] = row

        return {"matrix": matrix, "raw_counts": {
            ACTION_LABELS.get(a, str(a)): {
                ACTION_LABELS.get(b, str(b)): c
                for b, c in to_dict.items()
            }
            for a, to_dict in self._transition_counts.items()
        }}

    def _finalize_duration_size(self) -> dict[str, Any]:
        log_d_sample = self._log_dur_log_size.sample()
        log_s_sample = self._log_size_reservoir.sample()
        n = min(len(log_d_sample), len(log_s_sample))
        if n < 3:
            return {}

        log_d = log_d_sample[:n]
        log_s = log_s_sample[:n]
        corr = float(np.corrcoef(log_d, log_s)[0, 1]) if n > 2 else np.nan

        return {
            "log_duration_log_size_correlation": corr,
            "n_sample": n,
        }

    def _finalize_regime(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for rv in range(N_REGIMES):
            total = self._regime_total_counts.get(rv, 0)
            if total == 0:
                continue
            label = REGIME_LABELS.get(rv, f"regime_{rv}")
            fills = self._regime_fill_counts.get(rv, 0)
            cancels = self._regime_cancel_counts.get(rv, 0)
            acc = self._regime_lifetime_acc.get(rv)

            result[label] = {
                "fill_rate": fills / max(total, 1),
                "cancel_ratio": cancels / max(total, 1),
                "mean_lifetime_seconds": acc.mean if acc and acc.count > 0 else np.nan,
                "n_orders": total,
            }
        return result

    def _finalize_partial_fills(self) -> dict[str, Any]:
        n_any_partial = self._n_with_partials + self._n_partial_then_cancel
        denominator = self._n_filled_orders + self._n_partial_then_cancel
        if denominator == 0:
            return {}
        return {
            "partial_fill_fraction": n_any_partial / max(denominator, 1),
            "mean_fills_per_order": self._fills_dist.mean if self._fills_dist.count > 0 else np.nan,
            "max_fills_per_order": self._max_fills_per_order,
            "n_filled_orders": self._n_filled_orders,
            "n_with_partial_fills": self._n_with_partials,
            "n_partial_then_cancel": self._n_partial_then_cancel,
        }
