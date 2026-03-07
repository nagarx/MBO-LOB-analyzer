"""Order flow analysis domain: OFI, trade analysis, order lifecycle.

Re-exports analyzer classes for convenient imports::

    from rawlobanalyzer.analysis.flow import (
        OrderFlowAnalyzer, TradeAnalyzer, OrderLifecycleAnalyzer,
    )

Imports are deferred to avoid circular dependencies during incremental
development -- each analyzer module is imported on first access.
"""

from __future__ import annotations


def __getattr__(name: str):
    if name == "OrderFlowAnalyzer":
        from rawlobanalyzer.analysis.flow.order_flow import OrderFlowAnalyzer
        return OrderFlowAnalyzer
    if name == "TradeAnalyzer":
        from rawlobanalyzer.analysis.flow.trade import TradeAnalyzer
        return TradeAnalyzer
    if name == "OrderLifecycleAnalyzer":
        from rawlobanalyzer.analysis.flow.order_lifecycle import OrderLifecycleAnalyzer
        return OrderLifecycleAnalyzer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "OrderFlowAnalyzer",
    "OrderLifecycleAnalyzer",
    "TradeAnalyzer",
]
