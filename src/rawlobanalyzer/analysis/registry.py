"""Analyzer registry: auto-discovery and name-based resolution.

Analyzers register themselves via the ``@register_analyzer`` decorator.
The orchestrator and CLI resolve analyzer names to classes through this module.
"""

from __future__ import annotations

from typing import Type

from rawlobanalyzer.analysis.base import BaseAnalyzer

ANALYZER_REGISTRY: dict[str, Type[BaseAnalyzer]] = {}

_ALL_REGISTERED: bool = False


def register_analyzer(cls: Type[BaseAnalyzer]) -> Type[BaseAnalyzer]:
    """Decorator to register an analyzer class in the global registry.

    Usage::

        @register_analyzer
        class MyAnalyzer(BaseAnalyzer[MyReport]):
            name = "MyAnalyzer"
            ...
    """
    key = cls.name if hasattr(cls, "name") and cls.name != "BaseAnalyzer" else cls.__name__
    if key in ANALYZER_REGISTRY:
        existing = ANALYZER_REGISTRY[key]
        if existing is not cls:
            raise ValueError(
                f"Duplicate analyzer registration: {key!r} "
                f"(existing: {existing.__module__}.{existing.__qualname__}, "
                f"new: {cls.__module__}.{cls.__qualname__})"
            )
    ANALYZER_REGISTRY[key] = cls
    return cls


def _register_all() -> None:
    """Eagerly import all analyzer modules to trigger ``@register_analyzer``."""
    global _ALL_REGISTERED
    if _ALL_REGISTERED:
        return

    from rawlobanalyzer.analysis.health.data_quality import DataQualityAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.price.returns import ReturnAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.price.volatility import VolatilityAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.price.jump_risk import JumpRiskAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.price.microstructure_noise import MicrostructureNoiseAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.spread.spread import SpreadAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.spread.depth import DepthAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.spread.liquidity import LiquidityAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.flow.order_flow import OrderFlowAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.flow.trade import TradeAnalyzer  # noqa: F401
    from rawlobanalyzer.analysis.flow.order_lifecycle import OrderLifecycleAnalyzer  # noqa: F401

    _ALL_REGISTERED = True


def get_analyzer(name: str) -> Type[BaseAnalyzer]:
    """Resolve an analyzer class by name.

    Triggers eager registration on first call.

    Args:
        name: Registered analyzer name (e.g. ``"DataQualityAnalyzer"``).

    Returns:
        The analyzer class.

    Raises:
        KeyError: If no analyzer with that name is registered.
    """
    _register_all()
    if name not in ANALYZER_REGISTRY:
        available = sorted(ANALYZER_REGISTRY.keys())
        raise KeyError(
            f"Unknown analyzer: {name!r}. Available: {available}"
        )
    return ANALYZER_REGISTRY[name]


def list_analyzers() -> list[str]:
    """Return sorted list of all registered analyzer names."""
    _register_all()
    return sorted(ANALYZER_REGISTRY.keys())
