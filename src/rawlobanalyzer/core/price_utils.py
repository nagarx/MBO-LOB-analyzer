"""Price conversion and tick-level utilities.

All price data from MBO-LOB-reconstructor is in nanodollars (int64).
This module provides safe, tested conversions.
"""

from __future__ import annotations

import numpy as np

from rawlobanalyzer.core.constants import (
    BPS_FACTOR,
    EPS,
    NANODOLLARS_PER_DOLLAR,
    TICK_SIZE_USD,
)


def nanodollars_to_usd(prices: np.ndarray) -> np.ndarray:
    """Convert nanodollar int64 prices to USD float64.

    Args:
        prices: Array of int64 nanodollar prices.

    Returns:
        Array of float64 prices in USD.
    """
    return prices.astype(np.float64) / NANODOLLARS_PER_DOLLAR


def usd_to_nanodollars(prices: np.ndarray) -> np.ndarray:
    """Convert USD float64 prices to nanodollar int64.

    Args:
        prices: Array of float64 USD prices.

    Returns:
        Array of int64 nanodollar prices.
    """
    return (prices * NANODOLLARS_PER_DOLLAR).astype(np.int64)


def log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log returns from a price series.

    Formula: r_t = ln(P_t / P_{t-1})  (Campbell, Lo, MacKinlay 1997)

    Args:
        prices: Array of prices (any unit, must be > 0).

    Returns:
        Array of log returns, length ``len(prices) - 1``.
        Invalid prices (<=0, NaN) produce NaN returns.
    """
    safe = np.where(np.isfinite(prices) & (prices > 0), prices, np.nan)
    return np.diff(np.log(safe))


def spread_in_ticks(spread_usd: np.ndarray, tick_size: float = TICK_SIZE_USD) -> np.ndarray:
    """Convert spread from USD to number of ticks.

    Args:
        spread_usd: Array of spreads in USD.
        tick_size: Tick size in USD (default: $0.01 for US equities).

    Returns:
        Array of spreads expressed in ticks.
    """
    return spread_usd / tick_size


def basis_points(value: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Compute basis points: ``value / reference * 10000``.

    Division-safe: returns NaN where reference is near zero.

    Args:
        value: Numerator array.
        reference: Denominator array.

    Returns:
        Basis points as float64.
    """
    safe_ref = np.where(np.abs(reference) > EPS, reference, np.nan)
    return value / safe_ref * BPS_FACTOR
