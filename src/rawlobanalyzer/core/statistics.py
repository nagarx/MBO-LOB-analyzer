"""Streaming statistics and robust estimators.

Provides Welford's online algorithm for mean/variance, weighted quantiles,
and standard distribution summary statistics. All computations use float64.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats as sp_stats

from rawlobanalyzer.core.constants import DEFAULT_PERCENTILES, EPS


@dataclass
class WelfordAccumulator:
    """Welford's online algorithm for streaming mean and variance.

    Numerically stable single-pass computation. Follows Welford (1962)
    and Knuth TAOCP Vol 2, 3rd ed, p. 232.
    """

    count: int = 0
    mean: float = 0.0
    _m2: float = 0.0

    def update(self, value: float) -> None:
        """Incorporate a single observation."""
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self._m2 += delta * delta2

    def update_batch(self, values: np.ndarray) -> None:
        """Incorporate a batch of observations.

        Uses the parallel/batch variant of Welford's algorithm
        (Chan, Golub, LeVeque 1979). NaN and infinite values are
        excluded from the count and statistics.
        """
        if len(values) == 0:
            return

        finite_mask = np.isfinite(values)
        n_b = int(np.count_nonzero(finite_mask))
        if n_b == 0:
            return

        clean = values[finite_mask]
        mean_b = float(np.mean(clean))
        var_b = float(np.var(clean)) if n_b > 1 else 0.0

        n_a = self.count
        mean_a = self.mean
        m2_a = self._m2

        n_ab = n_a + n_b
        if n_ab == 0:
            return

        delta = mean_b - mean_a
        self.mean = (n_a * mean_a + n_b * mean_b) / n_ab
        self._m2 = m2_a + var_b * n_b + delta * delta * n_a * n_b / n_ab
        self.count = n_ab

    @property
    def variance(self) -> float:
        """Population variance."""
        if self.count < 2:
            return 0.0
        return self._m2 / self.count

    @property
    def sample_variance(self) -> float:
        """Sample variance (Bessel-corrected)."""
        if self.count < 2:
            return 0.0
        return self._m2 / (self.count - 1)

    @property
    def std(self) -> float:
        """Population standard deviation."""
        return self.variance**0.5

    @property
    def sample_std(self) -> float:
        """Sample standard deviation."""
        return self.sample_variance**0.5


@dataclass
class DistributionSummary:
    """Summary statistics for a univariate distribution."""

    count: int
    mean: float
    std: float
    skewness: float
    kurtosis: float
    min: float
    max: float
    percentiles: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "skewness": self.skewness,
            "kurtosis": self.kurtosis,
            "min": self.min,
            "max": self.max,
            "percentiles": self.percentiles,
        }


def distribution_summary(
    data: np.ndarray,
    *,
    percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    nan_policy: str = "omit",
) -> DistributionSummary:
    """Compute comprehensive distribution statistics.

    Args:
        data: 1D array of observations.
        percentiles: Percentile breakpoints to compute.
        nan_policy: ``"omit"`` to drop NaNs, ``"propagate"`` to include.

    Returns:
        ``DistributionSummary`` with all computed statistics.
    """
    if nan_policy == "omit":
        data = data[np.isfinite(data)]

    n = len(data)
    if n == 0:
        return DistributionSummary(
            count=0, mean=np.nan, std=np.nan,
            skewness=np.nan, kurtosis=np.nan,
            min=np.nan, max=np.nan,
        )

    mean_val = float(np.mean(data))
    std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
    skew_val = float(sp_stats.skew(data, nan_policy="omit")) if n > 2 else np.nan
    kurt_val = float(sp_stats.kurtosis(data, nan_policy="omit")) if n > 3 else np.nan

    pct_values = np.percentile(data, percentiles) if n > 0 else [np.nan] * len(percentiles)
    pct_dict = {f"p{p:g}": float(v) for p, v in zip(percentiles, pct_values)}

    return DistributionSummary(
        count=n,
        mean=mean_val,
        std=std_val,
        skewness=skew_val,
        kurtosis=kurt_val,
        min=float(np.min(data)),
        max=float(np.max(data)),
        percentiles=pct_dict,
    )


def var_cvar(
    returns: np.ndarray,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute Value-at-Risk and Conditional VaR (Expected Shortfall).

    VaR_alpha = quantile(returns, alpha)
    CVaR_alpha = E[r | r <= VaR_alpha]   (Acerbi & Tasche, 2002)

    Args:
        returns: 1D array of return observations.
        alpha: Tail probability (default 5%).

    Returns:
        Tuple of (VaR, CVaR) where both are negative for losses.
    """
    clean = returns[np.isfinite(returns)]
    if len(clean) < 10:
        return (np.nan, np.nan)

    var = float(np.percentile(clean, alpha * 100))
    tail = clean[clean <= var]
    cvar = float(np.mean(tail)) if len(tail) > 0 else var
    return (var, cvar)


def coefficient_of_variation(data: np.ndarray) -> float:
    """Coefficient of variation: std / |mean|.

    Returns NaN if mean is near zero.
    """
    clean = data[np.isfinite(data)]
    if len(clean) < 2:
        return np.nan
    m = float(np.mean(clean))
    if abs(m) < EPS:
        return np.nan
    return float(np.std(clean, ddof=1)) / abs(m)


class StreamingDistribution:
    """Welford accumulator + reservoir sampling for bounded-memory distribution stats.

    Maintains exact streaming mean, variance, and count via Welford's algorithm.
    Uses Algorithm R (Vitter 1985) reservoir sampling to keep a fixed-size random
    sample for quantiles, skewness, kurtosis, and other statistics that require
    the full empirical distribution.

    Memory: O(reservoir_size) regardless of total observations.
    """

    __slots__ = ("_welford", "_reservoir", "_reservoir_size", "_total_seen", "_rng")

    def __init__(self, reservoir_size: int = 100_000, seed: int = 42) -> None:
        self._welford = WelfordAccumulator()
        self._reservoir_size = reservoir_size
        self._reservoir = np.empty(reservoir_size, dtype=np.float64)
        self._total_seen: int = 0
        self._rng = np.random.default_rng(seed)

    @property
    def count(self) -> int:
        return self._welford.count

    @property
    def mean(self) -> float:
        return self._welford.mean

    @property
    def std(self) -> float:
        return self._welford.sample_std

    def add_batch(self, values: np.ndarray) -> None:
        """Incorporate a batch of observations."""
        finite = values[np.isfinite(values)]
        if len(finite) == 0:
            return

        self._welford.update_batch(finite)

        for v in finite:
            if self._total_seen < self._reservoir_size:
                self._reservoir[self._total_seen] = v
            else:
                j = int(self._rng.integers(0, self._total_seen + 1))
                if j < self._reservoir_size:
                    self._reservoir[j] = v
            self._total_seen += 1

    def sample(self) -> np.ndarray:
        """Return the reservoir sample (up to reservoir_size elements)."""
        n = min(self._total_seen, self._reservoir_size)
        return self._reservoir[:n].copy()

    def distribution_summary(
        self,
        percentiles: tuple[float, ...] = DEFAULT_PERCENTILES,
    ) -> DistributionSummary:
        """Compute approximate distribution summary from the reservoir."""
        data = self.sample()
        if len(data) == 0:
            return DistributionSummary(
                count=0, mean=np.nan, std=np.nan,
                skewness=np.nan, kurtosis=np.nan,
                min=np.nan, max=np.nan,
            )
        n = self._welford.count
        return DistributionSummary(
            count=n,
            mean=self._welford.mean,
            std=self._welford.sample_std,
            skewness=float(sp_stats.skew(data, nan_policy="omit")) if len(data) > 2 else np.nan,
            kurtosis=float(sp_stats.kurtosis(data, nan_policy="omit")) if len(data) > 3 else np.nan,
            min=float(np.min(data)),
            max=float(np.max(data)),
            percentiles={f"p{p:g}": float(v) for p, v in zip(
                percentiles, np.percentile(data, percentiles),
            )},
        )


def acf(series: np.ndarray, max_lag: int) -> np.ndarray:
    """FFT-based sample autocorrelation at lags 1 .. max_lag.

    O(N log N) regardless of ``max_lag``, using circular autocorrelation
    via the Wiener-Khinchin theorem.

    Args:
        series: 1D array of observations.
        max_lag: Maximum lag to compute.

    Returns:
        Array of ACF values at lags 1, 2, ..., max_lag.
        Empty array if the series is too short.
        Zeros if variance is near zero.
    """
    n = len(series)
    if n < max_lag + 1:
        max_lag = n - 1
    if max_lag < 1:
        return np.array([], dtype=np.float64)

    centered = series - np.mean(series)
    var = float(np.sum(centered ** 2))
    if var < EPS:
        return np.zeros(max_lag, dtype=np.float64)

    fft_size = 1
    while fft_size < 2 * n:
        fft_size *= 2
    f_centered = np.fft.rfft(centered, n=fft_size)
    acf_full = np.fft.irfft(f_centered * np.conj(f_centered), n=fft_size)[:n]

    return (acf_full[1: max_lag + 1] / var).astype(np.float64)
