# MBO-LOB-analyzer: Codebase Reference

Technical documentation of every module, data contract, and internal mechanism. This document is the single source of truth for understanding how the analysis pipeline works -- written for LLM coders who need to extend, debug, or integrate with this repo.

**Last updated**: 2026-03-05 | **Version**: 0.3.0 | **Python package**: `rawlobanalyzer` (in `src/rawlobanalyzer/`)

---

## Table of Contents

1. [Directory Layout](#1-directory-layout)
2. [Data Flow (End-to-End)](#2-data-flow-end-to-end)
3. [I/O Layer (`io/`)](#3-io-layer)
4. [Core Primitives (`core/`)](#4-core-primitives)
5. [Configuration System (`config/`)](#5-configuration-system)
6. [Analysis Framework (`analysis/`)](#6-analysis-framework)
7. [Shared Computation Engines](#7-shared-computation-engines)
8. [Health Domain (`analysis/health/`)](#8-health-domain)
9. [Price Domain (`analysis/price/`)](#9-price-domain)
10. [Spread Domain (`analysis/spread/`)](#10-spread-domain)
11. [Flow Domain (`analysis/flow/`)](#11-flow-domain)
12. [Reports (`reports/`)](#12-reports)
13. [Configuration Profiles (`configs/`)](#13-configuration-profiles)
14. [Operational Robustness](#135-operational-robustness)
15. [Test Suite (`tests/`)](#14-test-suite)
16. [Module Dependency Graph](#15-module-dependency-graph)
17. [Critical Invariants](#16-critical-invariants)

---

## 1. Directory Layout

```
MBO-LOB-analyzer/
├── configs/
│   ├── defaults.yaml              # Global default config
│   ├── profiles/                  # YAML analysis profiles
│   │   ├── flow.yaml              #   OFI + trade + lifecycle (4 analyzers)
│   │   ├── full.yaml              #   All 11 analyzers
│   │   ├── quick.yaml             #   DataQuality only (smoke test)
│   │   ├── spread.yaml            #   Spread + depth + liquidity (4 analyzers)
│   │   ├── standard.yaml          #   Balanced daily analysis (5 analyzers)
│   │   └── volatility.yaml        #   Price + vol deep-dive (5 analyzers)
│   └── sampling/                  # Curated date lists for --dates-file
│       └── nvda_stratified_50.txt #   58 dates stratified by month/volume/weekday
├── scripts/
│   ├── run_analysis.py            # CLI entry point (profiles, analyzers, dates, checkpoint)
│   └── generate_sample.py         # Stratified date sampling for --dates-file
├── src/rawlobanalyzer/
│   ├── __init__.py                # Package metadata, version
│   ├── cli.py                     # Console script entry point
│   ├── io/
│   │   ├── schema.py              # Schema constants, action/side enums, validation
│   │   ├── loader.py              # ParquetDayLoader, DayData
│   │   └── session.py             # AnalysisSession (bounded-memory iteration)
│   ├── core/
│   │   ├── constants.py           # Named constants (EPS, NS_PER_*, tick size, etc.)
│   │   ├── calendar.py            # WEEKDAY_NAMES, weekday_from_date, weekday_name
│   │   ├── time_utils.py          # RTH mask, time regime, inter-event times
│   │   ├── price_utils.py         # Nanodollar conversion, log returns, basis points
│   │   ├── resampler.py           # Vectorized multi-timescale resampling engine
│   │   ├── statistics.py          # WelfordAccumulator, StreamingDistribution, acf, VaR/CVaR
│   │   ├── intraday_accumulator.py # IntradayCurveAccumulator (streaming per-bin stats)
│   │   └── regime_accumulator.py  # RegimeStreamingAccumulator (streaming per-regime stats)
│   ├── config/
│   │   ├── analysis_config.py     # AnalysisConfig + all threshold dataclasses
│   │   ├── timescale_config.py    # TimescaleConfig, TradingHours
│   │   └── profile_loader.py      # YAML profile loading and application
│   ├── analysis/
│   │   ├── base.py                # BaseAnalyzer protocol
│   │   ├── context.py             # DayContext (shared computation cache)
│   │   ├── orchestrator.py        # BatchOrchestrator (single-pass execution)
│   │   ├── registry.py            # @register_analyzer, name resolution
│   │   ├── health/
│   │   │   └── data_quality.py    # DataQualityAnalyzer
│   │   ├── price/
│   │   │   ├── _return_engine.py  # Shared: compute_day_returns() -> DayReturns
│   │   │   ├── returns.py         # ReturnAnalyzer
│   │   │   ├── volatility.py      # VolatilityAnalyzer
│   │   │   ├── jump_risk.py       # JumpRiskAnalyzer
│   │   │   └── microstructure_noise.py  # MicrostructureNoiseAnalyzer
│   │   ├── spread/
│   │   │   ├── _spread_engine.py  # Shared: compute_day_spreads() -> DaySpreads
│   │   │   ├── spread.py          # SpreadAnalyzer
│   │   │   ├── depth.py           # DepthAnalyzer
│   │   │   └── liquidity.py       # LiquidityAnalyzer
│   │   └── flow/
│   │       ├── _flow_engine.py    # Shared: compute_day_flow() -> DayFlow
│   │       ├── order_flow.py      # OrderFlowAnalyzer
│   │       ├── trade.py           # TradeAnalyzer
│   │       └── order_lifecycle.py # OrderLifecycleAnalyzer
│   └── reports/
│       ├── base_report.py         # BaseReport abstract class
│       └── formatters.py          # Text table/section formatting
└── tests/
    ├── conftest.py                # Shared fixtures (_write_day, synthetic Parquet)
    ├── test_analysis/             # One subdir per domain + test_empty_day.py
    ├── test_config/
    ├── test_core/
    ├── test_io/
    └── test_reports/              # BaseReport, formatters tests
```

---

## 2. Data Flow (End-to-End)

```
  CLI (scripts/run_analysis.py)
       |
       |  parse args (--profile, --analyzer, --dates-file, --checkpoint-dir, --resume)
       |  load YAML profile, build AnalysisConfig (incl. dates_list, checkpoint_dir)
       v
  BatchOrchestrator(config, analyzers)
       |
       |  create AnalysisSession(data_dir, date_range, dates_list, symbol)
       |  merge lob_columns + mbo_columns across all analyzers
       |  check needs_returns / needs_spreads / needs_flow flags
       |  if resume: restore checkpoint -> completed_dates set
       v
  AnalysisSession.iter_days()  -----> yields one DayData per trading day
       |                                   |
       |                                   |  ParquetDayLoader.load_day()
       |                                   |    - column projection (only load needed columns)
       |                                   |    - schema validation (first day only)
       |                                   |    - gc.collect() after yield
       v
  FOR EACH day:
       |
       ├── if day.date in completed_dates: SKIP (checkpoint resume)
       ├── utc_offset = utc_offset_for_date(day.date)  # DST-aware: -4 or -5
       ├── day_config = config with TradingHours adjusted to utc_offset
       |
       ├── if any_need_returns:  compute_day_returns(day, day_config) -> DayReturns
       ├── if any_need_spreads:  compute_day_spreads(day, day_config) -> DaySpreads
       ├── if any_need_flow:     compute_day_flow(day, day_config)    -> DayFlow
       |
       ├── DayContext = { day, day_epoch_ns, utc_offset_hours, returns, spreads, flow }
       |
       ├── analyzer_1.process_day(ctx)
       ├── analyzer_2.process_day(ctx)
       └── ...
       |
       ├── log progress (timing, throughput, ETA, peak memory)
       ├── if checkpoint_dir: save_checkpoint(analyzers, day.date)
       ├── gc.collect()
       |
  AFTER ALL days:
       ├── analyzer_1.finalize() -> Report_1
       ├── analyzer_2.finalize() -> Report_2
       └── ...
       |
       v
  Write JSON + TXT summaries to output_dir/
```

**Key properties**:
- **Single pass**: K analyzers on N days = N data loads (not K*N)
- **Column projection**: Only columns declared by analyzers are loaded from Parquet
- **Bounded memory**: One day's data + accumulators in memory at any time; `gc.collect()` after each day. All accumulators use `StreamingDistribution` or `WelfordAccumulator` for O(reservoir_size) memory regardless of dataset size.
- **Shared computation**: Expensive intermediate results (returns, spreads, flow) computed once per day
- **DST-aware**: Per-day UTC offset computed automatically; RTH/regime/grid logic uses the correct offset for each day
- **Crash-safe**: Optional checkpoint/resume serializes all analyzer state after each day
- **Flexible date selection**: Filter by contiguous range (`--date-start`/`--date-end`), explicit date list (`--dates-file`), or both

---

## 3. I/O Layer

### `io/schema.py` -- Schema Constants and Validation

The single source of truth for the data contract with MBO-LOB-reconstructor.

**LOB Snapshot Columns** (`LOB_CORE_COLUMNS` + `LOB_DERIVED_COLUMNS`):

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp_ns` | int64 | ns since epoch (UTC) | Event timestamp |
| `sequence` | uint64 | - | Sequence number from exchange |
| `levels` | uint8 | - | Number of LOB levels available |
| `best_bid` | int64 | nanodollars | Best bid price |
| `best_ask` | int64 | nanodollars | Best ask price |
| `bid_prices` | list\<int64\>[10] | nanodollars | 10-level bid prices |
| `bid_sizes` | list\<uint32\>[10] | shares | 10-level bid sizes |
| `ask_prices` | list\<int64\>[10] | nanodollars | 10-level ask prices |
| `ask_sizes` | list\<uint32\>[10] | shares | 10-level ask sizes |
| `delta_ns` | uint64 | nanoseconds | Time since previous event |
| `triggering_action` | int8 | enum | MBO action that caused this snapshot |
| `triggering_side` | int8 | enum | MBO side that caused this snapshot |
| `mid_price` | float64 | USD | (best_bid + best_ask) / 2e9 |
| `spread` | float64 | USD | (best_ask - best_bid) / 1e9 |
| `spread_bps` | float64 | basis points | spread / mid_price * 10000 |
| `microprice` | float64 | USD | Volume-weighted mid |
| `total_bid_volume` | uint64 | shares | Sum of all bid sizes |
| `total_ask_volume` | uint64 | shares | Sum of all ask sizes |
| `depth_imbalance` | float64 | [-1, 1] | (bid_vol - ask_vol) / total |
| `book_consistency` | uint8 | enum | 0=Valid, 1=Empty, 2=Locked, 3=Crossed |

**MBO Event Columns** (`MBO_COLUMNS`):

| Column | Type | Unit | Description |
|--------|------|------|-------------|
| `timestamp_ns` | int64 | ns since epoch (UTC) | Event timestamp |
| `order_id` | uint64 | - | Unique order identifier |
| `action` | uint8 | enum byte | 65=Add, 67=Cancel, 77=Modify, 84=Trade, 70=Fill, 82=Clear |
| `side` | uint8 | enum byte | 66=Bid(B), 65=Ask(A), 78=None(N) |
| `price` | int64 | nanodollars | Order price |
| `size` | uint32 | shares | Order size |

**Action Enum** (byte values from Rust):
- `ACTION_ADD = 65` (b'A') -- New order placed
- `ACTION_CANCEL = 67` (b'C') -- Order canceled
- `ACTION_MODIFY = 77` (b'M') -- Order price/size modified
- `ACTION_TRADE = 84` (b'T') -- Order executed (trade)
- `ACTION_FILL = 70` (b'F') -- Order filled
- `ACTION_CLEAR = 82` (b'R') -- Book cleared
- `ACTION_NONE = 78` (b'N') -- No action

**Side Enum**:
- `SIDE_BID = 66` (b'B') -- Buy side
- `SIDE_ASK = 65` (b'A') -- Sell side
- `SIDE_NONE = 78` (b'N') -- No side (system/clearing trades)

**Validation functions**:
- `validate_lob_schema(arrow_schema)` -- Returns list of missing required columns
- `validate_mbo_schema(arrow_schema)` -- Returns list of missing required columns
- `validate_parquet_metadata(arrow_metadata)` -- Extracts and validates file-level metadata (symbol, schema version, source)

### `io/loader.py` -- ParquetDayLoader and DayData

**`ParquetDayLoader`**:
- Constructor takes `data_dir: Path`
- `discover_dates()` -- scans for `{date}_lob_snapshots.parquet` files, returns sorted date strings
- `load_day(date, need_lob, need_mbo, lob_columns, mbo_columns, validate)` -- loads one day's data
  - **Single file open**: Opens each Parquet file once via `pq.ParquetFile()`, reads schema/metadata from the handle, then reads data via `pf.read(columns=...)` without reopening
  - **Column projection**: if `lob_columns` / `mbo_columns` is not None, only those columns are read
  - **Schema validation**: only on first day (`validate` param) to avoid repeated overhead
  - **File naming convention**: `{date}_lob_snapshots.parquet`, `{date}_mbo_events.parquet`

**`DayData`** -- immutable per-day data container:
- `date: str` -- "YYYY-MM-DD"
- `symbol: str` -- ticker (e.g. "NVDA")
- `lob: pa.Table | None` -- raw PyArrow table (LOB)
- `mbo: pa.Table | None` -- raw PyArrow table (MBO)
- `metadata: dict` -- file-level Parquet metadata

**Lazy cached properties** (computed on first access, not at load time):
- `lob_timestamps_ns` -- `lob.column("timestamp_ns").to_numpy()` -> int64 array
- `mbo_timestamps_ns` -- `mbo.column("timestamp_ns").to_numpy()` -> int64 array
- `mid_prices` -- `lob.column("mid_price").to_numpy()` -> float64 USD
- `spreads` -- `lob.column("spread").to_numpy()` -> float64 USD
- `best_bids_usd` -- `lob.column("best_bid").to_numpy() / 1e9` -> float64 USD
- `best_asks_usd` -- `lob.column("best_ask").to_numpy() / 1e9` -> float64 USD
- `n_lob_rows`, `n_mbo_rows` -- row counts

### `io/session.py` -- AnalysisSession

Wraps `ParquetDayLoader` with bounded-memory streaming:

- Constructor: `AnalysisSession(data_dir, *, date_range, dates_list, symbol)`
  - Discovers all dates, filters by optional `date_range: (start, end)` inclusive
  - `dates_list: list[str] | None` -- explicit list of dates to include. If both `date_range` and `dates_list` are set, the intersection is used (date must pass both filters)
  - `symbol_override` replaces symbol from metadata on every yielded `DayData`
- `iter_days(need_lob, need_mbo, lob_columns, mbo_columns)` -- generator
  - Yields one `DayData` per trading day in chronological order
  - Calls `gc.collect()` after each yield to free the previous day's PyArrow tables
  - Validates schema only on the first day (i == 0)

---

## 4. Core Primitives

### `core/constants.py` -- Named Constants

Every magic number lives here. Never use raw literals in analysis code.

| Constant | Value | Source / Rationale |
|----------|-------|--------------------|
| `EPS` | 1e-15 | 10x machine epsilon for float64 |
| `NANODOLLARS_PER_DOLLAR` | 1,000,000,000 | Parquet price encoding |
| `TICK_SIZE_USD` | 0.01 | SEC Rule 612, Reg NMS |
| `TICK_SIZE_NANODOLLARS` | 10,000,000 | `TICK_SIZE_USD * 1e9` |
| `NS_PER_SECOND` | 1,000,000,000 | Time conversion |
| `NS_PER_MILLISECOND` | 1,000,000 | Time conversion |
| `NS_PER_MINUTE` | 60 * NS_PER_SECOND | Time conversion |
| `NS_PER_HOUR` | 3,600 * NS_PER_SECOND | Time conversion |
| `NS_PER_DAY` | 86,400 * NS_PER_SECOND | Time conversion |
| `DEFAULT_SIGNIFICANCE_ALPHA` | 0.05 | Default significance level for tests |
| `DEFAULT_PERCENTILES` | (1, 5, 10, 25, 50, 75, 90, 95, 99) | Standard quantile breakpoints |
| `TRADING_DAYS_PER_YEAR` | 252 | US equity annualization |
| `TRADING_HOURS_PER_DAY` | 6.5 | 9:30 AM - 4:00 PM ET |
| `TRADING_SECONDS_PER_DAY` | 23,400 | 6.5 * 3600 |
| `TRADING_MINUTES_PER_DAY` | 390 | 6.5 * 60 |
| `BPS_FACTOR` | 10,000 | Basis-point conversion (1 bps = 0.01%) |
| `ANNUALIZATION_FACTOR` | `TRADING_DAYS_PER_YEAR` | Alias for `TRADING_DAYS_PER_YEAR`; used in volatility annualization |
| `BIPOWER_MU_1` | 0.7979 | sqrt(2/pi), Barndorff-Nielsen & Shephard (2004) Eq. 4 |

### `core/time_utils.py` -- Time Utilities

All timestamps from MBO-LOB-reconstructor are **int64 nanoseconds since epoch (UTC)**.

**DST-aware timezone handling**:

- `utc_offset_for_date(date_str) -> int` -- returns `-4` (EDT) or `-5` (EST) for any `YYYY-MM-DD` date. Uses US Eastern DST rules: DST starts the 2nd Sunday in March, ends the 1st Sunday in November. LRU-cached per year for performance. The orchestrator calls this once per day and passes the result through `DayContext.utc_offset_hours` so all downstream code (RTH masks, regime classifications, grid alignment) uses the correct offset automatically.

**Functions**:

- `ns_to_seconds_since_midnight_utc(ts_ns)` -- converts to float64 seconds mod 86400
- `ns_to_hours_since_midnight_utc(ts_ns)` -- converts to float64 fractional hours [0, 24)
- `rth_mask_utc(ts_ns, *, utc_offset_hours=-5)` -- boolean mask for Regular Trading Hours (9:30-16:00 ET)
  - Accepts `utc_offset_hours` kwarg: `-5` for EST, `-4` for EDT. In orchestrated runs, the per-day offset from `DayContext` is used automatically.
  - Returns True for events with UTC hour in `[rth_open_utc, rth_close_utc)`
- `extended_hours_mask_utc(ts_ns, *, utc_offset_hours=-5)` -- boolean mask for 4:00 AM - 8:00 PM ET (handles midnight wrap)
- `time_regime(ts_ns, *, utc_offset_hours=-5)` -- classifies each timestamp into one of 7 intraday regimes (int8):

  | Value | Label | Time (ET) |
  |-------|-------|-----------|
  | 0 | pre-market | Before 9:30 |
  | 1 | open-auction | 9:30 - 9:35 |
  | 2 | morning | 9:35 - 12:00 |
  | 3 | midday | 12:00 - 14:00 |
  | 4 | afternoon | 14:00 - 15:45 |
  | 5 | close-auction | 15:45 - 16:00 |
  | 6 | after-hours | After 16:00 |

- `seconds_to_label(s)` -- converts float seconds to human-readable timescale label (e.g., `0.1` -> `"100ms"`, `5` -> `"5s"`, `300` -> `"5m"`). Used by all shared engines for consistent timescale naming.
- `rth_grid_edges_ns(day_epoch_ns, resolution_ns, utc_offset_hours=-5)` -- returns a canonical RTH bin-edge array for a given trading day and resolution. Produces a deterministic grid from market open to market close aligned to `resolution_ns`, so MBO-derived (OFI) and LOB-derived (returns, spreads) engines resampling onto the same grid get perfectly aligned bins for cross-series correlation.
- `compute_inter_event_times_ns(ts_ns)` -- `np.diff(timestamps_ns)`, returns int64 array

### `core/price_utils.py` -- Price Conversion

- `nanodollars_to_usd(prices)` -- `int64 / 1e9` -> float64 USD
- `usd_to_nanodollars(prices)` -- `float64 * 1e9` -> int64 nanodollars
- `log_returns(prices)` -- `ln(P_t / P_{t-1})` (Campbell, Lo, MacKinlay 1997). Returns NaN for non-positive prices. Length = len(prices) - 1.
- `spread_in_ticks(spread_usd, tick_size=0.01)` -- spread / tick_size
- `basis_points(value, reference)` -- `value / reference * 10000`, returns NaN where reference < EPS

### `core/resampler.py` -- Vectorized Resampling Engine

The multi-granularity engine that powers all timescale-dependent analysis. Converts event-level data into fixed-width time bins.

**`resample(timestamps_ns, values, resolution_ns, agg, label)`**:

- **Input**: sorted int64 timestamps, float64 values, bin width in nanoseconds
- **Bin assignment**: O(N) integer division -- `(ts - bin_start) // resolution_ns`
- **Returns**: `ResampledSeries(bin_edges_ns, values, counts, label)`

**Aggregation modes** (all O(N) except median):

| Mode | Output | Algorithm |
|------|--------|-----------|
| `"count"` | Events per bin | `np.bincount` |
| `"sum"` | Sum of values per bin | `np.bincount(weights=...)` |
| `"mean"` | Mean per bin (NaN for empty) | sum / count |
| `"last"` | Last value per bin (NaN for empty) | Cumulative counts + segment boundaries |
| `"first"` | First value per bin | Cumulative counts |
| `"ohlc"` | (4, n_bins) open/high/low/close | `reduceat` for high/low, segment boundaries for open/close |
| `"median"` | Median per bin | Python loop over filled bins (not events) |

**Performance**: The `_compute_segments()` helper derives segment boundaries from per-bin counts using cumulative sums, enabling O(N) `last`/`first`/`ohlc` without Python-level event loops. This was vectorized from an earlier O(N*B) implementation, yielding ~165x speedup for fine-grained resampling.

**`resample_to_grid(timestamps_ns, values, grid_edges_ns, agg, label)`**:

- **Input**: sorted int64 timestamps, float64 values, pre-computed bin edges (int64), aggregation mode, label
- **Purpose**: Resamples event data into bins defined by an externally supplied grid. Unlike `resample()` which derives bins from the data itself, `resample_to_grid()` uses edges from `rth_grid_edges_ns()` so that different data sources (e.g., MBO-derived OFI and LOB-derived returns) resampled onto the same canonical grid produce perfectly aligned bin timestamps for cross-series correlation.
- **Bin assignment**: `np.searchsorted(grid_edges_ns, timestamps_ns, side="right") - 1`
- **Returns**: `ResampledSeries(bin_edges_ns, values, counts, label)` with the same structure as `resample()`
- **Aggregation modes**: Same set as `resample()` (count, sum, mean, last, first, ohlc, median)

### `core/statistics.py` -- Statistical Primitives

**`WelfordAccumulator`** -- streaming mean/variance:
- `update(value)` -- single observation, Welford (1962)
- `update_batch(values)` -- batch variant, Chan, Golub, LeVeque (1979)
- Properties: `mean`, `variance` (population), `sample_variance` (Bessel-corrected), `std`, `sample_std`

**`DistributionSummary`** -- count, mean, std, skewness, kurtosis, min, max, percentiles dict

**`distribution_summary(data, percentiles, nan_policy)`** -- computes full summary using scipy.stats for skew/kurtosis

**`StreamingDistribution`** -- Welford accumulator + reservoir sampling for bounded-memory distribution stats:
- `add_batch(values)` -- incorporates a batch of observations (filters NaN/Inf)
- `sample()` -> reservoir sample array (up to `reservoir_size` elements, default 100K)
- `distribution_summary(percentiles)` -> `DistributionSummary` with exact streaming mean/std and approximate quantiles/skewness/kurtosis from the reservoir
- Memory: O(reservoir_size) regardless of total observations
- Uses Algorithm R (Vitter 1985) for unbiased reservoir sampling

**`acf(series, max_lag)`** -- FFT-based sample autocorrelation at lags 1..max_lag. O(N log N) via Wiener-Khinchin theorem. Centralized from duplicate implementations in `returns.py` and `order_flow.py`.

**`var_cvar(returns, alpha=0.05)`** -- Value-at-Risk and Conditional VaR (Expected Shortfall). Acerbi & Tasche (2002). Requires >= 10 observations.

**`coefficient_of_variation(data)`** -- std / |mean|, NaN if mean near zero

### `core/calendar.py` -- Trading Calendar Utilities

Single source of truth for weekday-related logic. Replaces 5 duplicate implementations across analyzers.

- `WEEKDAY_NAMES: tuple[str, ...]` -- `("Monday", "Tuesday", ..., "Friday")`
- `weekday_from_date(date_str)` -> 0-based weekday index (Monday=0) from `YYYY-MM-DD` string
- `weekday_name(date_str)` -> human-readable weekday name

### `core/intraday_accumulator.py` -- IntradayCurveAccumulator

Fixed-bin streaming accumulator for intraday curves. Used by `VolatilityAnalyzer` and `SpreadAnalyzer` to compute mean and std per time-of-day bin across multiple trading days with O(n_bins) memory.

- `add(bin_indices, values)` -- accumulate into specified bins
- `finalize(bin_width)` -> dict with `minutes`, mean, std, and `n_days` per bin

### `core/regime_accumulator.py` -- RegimeStreamingAccumulator

Per-regime streaming sum/sum-of-squares/count accumulator. Used by `VolatilityAnalyzer` and `SpreadAnalyzer` for regime-conditional statistics.

- `add(regime, values)` -- accumulate finite values for a regime
- `finalize(min_count)` -> dict of `{regime_label: {"mean", "std", "n"}}`
- `get_bucket(regime)` / `items()` -- raw access for analyzers needing custom finalize logic

---

## 5. Configuration System

### `config/timescale_config.py` -- TimescaleConfig and TradingHours

**`TimescaleConfig`** (frozen dataclass):
- `resolution_ns: int` -- bin width in nanoseconds
- `label: str` -- human-readable (e.g. "1s", "5m", "1h")
- `trading_hours_only: bool` -- if True, filter to RTH before resampling (default True)
- Factory methods: `seconds(n)`, `minutes(n)`, `hourly()`, `daily()`, `from_label("5m")`

**`TradingHours`** (frozen dataclass):
- `rth_open_utc_h`, `rth_close_utc_h` -- RTH boundaries in UTC fractional hours
- `ext_open_utc_h`, `ext_close_utc_h` -- Extended hours boundaries
- `utc_offset_hours` -- local timezone offset
- Presets: `us_equity()` (EST, -5), `us_equity_dst()` (EDT, -4)
- Resolution: `from_label("us_equity_rth")` maps to `us_equity()`

**Default timescales**: `[1s, 5s, 30s, 1m, 5m, 15m, 1h]`

### `config/analysis_config.py` -- AnalysisConfig

Top-level configuration dataclass. Every configurable threshold is here.

**`AnalysisConfig`** fields:
| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `data_dir` | Path | required | Parquet export directory |
| `symbol` | str | "UNKNOWN" | Ticker symbol |
| `date_range` | tuple[str,str]\|None | None | Inclusive YYYY-MM-DD filter |
| `dates_list` | list[str]\|None | None | Explicit date list filter (intersected with `date_range` if both set) |
| `timescales` | list[TimescaleConfig] | 7 defaults | Resampling granularities |
| `trading_hours` | TradingHours | US equity EST | Market session |
| `thresholds` | StatisticalThresholds | defaults | All statistical parameters |
| `max_rows_per_day` | int\|None | None | Subsample limit |
| `output_dir` | Path\|None | None | Report output directory |
| `checkpoint_dir` | Path\|None | None | Directory for checkpoint files (enables crash recovery) |
| `resume` | bool | False | Resume from existing checkpoint if available |
| `save_json` | bool | True | Write JSON reports |
| `save_summary` | bool | True | Write text summaries |
| `verbose` | bool | False | Progress logging |

**Threshold hierarchy** (nested dataclasses inside `StatisticalThresholds`):

```
StatisticalThresholds
├── significance_alpha: 0.05
├── correlation_weak/moderate/strong: 0.3/0.5/0.7
├── VolatilityThresholds
│   ├── signature_scales_seconds: (0.1, 0.5, 1, 2, 5, 10, 30, 60, 300)
│   ├── min_returns_per_bin: 10
│   ├── intraday_bin_minutes: 1
│   ├── max_acf_lag: 20
│   ├── jump_confidence: 0.999
│   ├── jump_threshold_sigma: 3.0
│   ├── hill_tail_fraction: 0.05
│   ├── noise_max_scale_seconds: 60
│   ├── noise_n_scales: 20
│   ├── arch_lags: (1, 5, 10, 50, 100, 500)
│   └── primary_vol_scale_seconds: 5.0
├── SpreadThresholds
│   ├── intraday_bin_minutes: 1
│   ├── max_acf_lag: 20
│   ├── wide_spread_ticks: 5.0
│   ├── narrow_spread_ticks: 1.5
│   ├── width_bucket_ticks: (1, 2, 5)
│   └── tick_size_usd: TICK_SIZE_USD (0.01)
├── DepthThresholds
│   ├── n_levels: 10
│   ├── recovery_horizons: (1, 5, 10, 50)
│   └── intraday_bin_minutes: 5
├── LiquidityThresholds
│   ├── realized_spread_horizons_seconds: (1, 5, 30)
│   ├── cost_to_trade_sizes: (100, 500, 1000, 5000, 10000)
│   └── kyle_lambda_scale_seconds: 1
└── FlowThresholds
    ├── ofi_timescales_seconds: (1, 5, 10, 30, 60, 300)
    ├── intraday_bin_minutes: 1
    ├── max_acf_lag: 20
    ├── flow_return_max_lag_seconds: 60
    ├── flow_return_n_lags: 12
    ├── large_trade_percentile: 95
    ├── trade_cluster_gap_seconds: 1
    ├── order_lifetime_max_seconds: 3600
    ├── order_lifetime_n_bins: 50
    └── max_active_orders: 500,000
```

**`__post_init__` validation**: All threshold dataclasses (`VolatilityThresholds`, `SpreadThresholds`, `DepthThresholds`, `LiquidityThresholds`, `FlowThresholds`, `StatisticalThresholds`) validate their fields on construction. Checks include positive values, range constraints (e.g., `jump_confidence` in [0, 1]), ordering constraints (e.g., `narrow_spread_ticks < wide_spread_ticks`, `correlation_weak <= moderate <= strong`), non-empty and sorted tuples, and cross-field consistency (e.g., `primary_vol_scale_seconds` must be in `signature_scales_seconds`).

### `config/profile_loader.py` -- YAML Profile Loading

- `load_profile(path)` -> `ProfileSpec(name, description, phases: list[PhaseSpec], config_overrides)`
- `PhaseSpec` contains `name`, `description`, `analyzers: list[AnalyzerSpec]`
- `apply_profile_config(base_config, profile)` -- applies `config_overrides` (timescales, trading_hours, max_rows) from the profile YAML onto a base `AnalysisConfig`

---

## 6. Analysis Framework

### `analysis/base.py` -- BaseAnalyzer Protocol

Every analyzer inherits from `BaseAnalyzer[ReportT]` and must implement:

```python
class BaseAnalyzer(ABC, Generic[ReportT]):
    # ClassVars that subclasses override:
    name: ClassVar[str]           # e.g. "OrderFlowAnalyzer"
    description: ClassVar[str]    # one-line description
    lob_columns: ClassVar[list[str] | None]  # None = all, [] = none
    mbo_columns: ClassVar[list[str] | None]  # None = not needed
    needs_mbo: ClassVar[bool] = False
    needs_returns: ClassVar[bool] = False
    needs_spreads: ClassVar[bool] = False
    needs_flow: ClassVar[bool] = False

    def process_day(self, ctx: DayContext) -> None: ...   # accumulate
    def finalize(self) -> ReportT: ...                     # produce report
    def get_extra_scales(self) -> tuple[float, ...] | None: ...  # optional
```

**Contract**:
- `process_day` is called once per trading day by the orchestrator. Must not store full `DayData` -- extract and aggregate only.
- `finalize` is called once after all days are processed. Returns a `BaseReport` subclass.
- `get_extra_scales` declares additional sampling scales in seconds beyond `config.timescales` (used by signature-plot analyzers that need sub-second sampling).

**Standalone `run()` method**: If an analyzer is run outside the orchestrator, `run(session)` loops over days itself, computing `DayReturns`/`DaySpreads`/`DayFlow` per-day as needed (no caching).

### `analysis/context.py` -- DayContext

Per-day shared computation container created by the orchestrator:

```python
@dataclass
class DayContext:
    day: DayData                          # always present
    day_epoch_ns: int = 0                 # midnight UTC of trading day (ns since epoch);
                                          # used by engines for canonical RTH grid alignment
    utc_offset_hours: int = -5            # DST-aware: -4 (EDT) or -5 (EST), computed
                                          # per-day by the orchestrator via utc_offset_for_date()
    day_returns: DayReturns | None        # present if any analyzer needs_returns
    day_spreads: DaySpreads | None        # present if any analyzer needs_spreads
    day_flow: DayFlow | None              # present if any analyzer needs_flow
```

**Lifetime**: Created at the start of each day's processing, passed to all analyzers, garbage-collected after the day is done.

**DST wiring**: The orchestrator calls `utc_offset_for_date(day.date)` for each day, passes the result to `DayContext.utc_offset_hours`, and creates a per-day `AnalysisConfig` with `TradingHours` adjusted to the correct UTC offset. All analyzers read `ctx.utc_offset_hours` for any RTH/regime/grid logic, ensuring correct DST-aware behavior across multi-month datasets.

### `analysis/orchestrator.py` -- BatchOrchestrator

Single-pass multi-analyzer execution:

1. **Column merging**: Computes the union of `lob_columns` and `mbo_columns` across all analyzers. If any analyzer requests `None` (all columns), loads all.
2. **Needs detection**: Checks `needs_mbo`, `needs_returns`, `needs_spreads`, `needs_flow` across all analyzers.
3. **Scale merging**: Collects `get_extra_scales()` from all analyzers into a single sorted tuple for the return/spread engines.
4. **Checkpoint restoration**: If `config.resume` is `True` and `config.checkpoint_dir` contains a valid checkpoint, restores analyzer state from pickle files and populates a `completed_dates` set from `manifest.txt`. Previously processed days are skipped.
5. **Day loop**: For each day from `AnalysisSession.iter_days()`:
   - Skip if date is in `completed_dates` (checkpoint resume)
   - Compute DST-aware UTC offset via `utc_offset_for_date(day.date)`
   - Create a per-day `AnalysisConfig` with adjusted `TradingHours` if offset differs from base config
   - Compute shared engines once: `compute_day_returns`, `compute_day_spreads`, `compute_day_flow`
   - Wrap in `DayContext` (with `utc_offset_hours` set)
   - Call `process_day(ctx)` on every analyzer
   - **Progress logging**: Per-day timing, throughput (rows/sec), ETA (exponential moving average, alpha=0.3), peak RSS memory (MB)
   - **Checkpoint save**: If `checkpoint_dir` is set, serialize each analyzer via `pickle` and append the date to `manifest.txt`
   - Explicit `gc.collect()` after each day
6. **Finalization**: After all days, calls `finalize()` on every analyzer, collecting reports.

**Progress log format** (one line per day):
```
[3/234] 2025-02-05  537.2s  45,000 rows/s  ETA 1.2h  peak 4,063 MB  (LOB=12,300,000 MBO=5,400,000)
```

**Checkpoint/resume**:
- **Save**: After each day, writes one pickle per analyzer (`{analyzer.name}.pkl`) + `manifest.txt` (sorted list of completed dates) to `checkpoint_dir`. Save failures are non-fatal (logged as warnings).
- **Restore**: On startup with `--resume`, reads `manifest.txt`, deserializes analyzer pickles, and skips completed dates. If any pickle is missing, starts fresh.
- **CLI**: `--checkpoint-dir <path>` enables checkpointing; `--resume` triggers restoration.

### `analysis/registry.py` -- Analyzer Registry

- `@register_analyzer` decorator: registers the class in `ANALYZER_REGISTRY` dict keyed by `cls.name`
- `_register_all()` -- eagerly imports all analyzer modules to trigger decorators (called on first `get_analyzer` or `list_analyzers`)
- `get_analyzer(name)` -> class (raises `KeyError` with available list if unknown)
- `list_analyzers()` -> sorted list of registered names

Currently registered (11): DataQualityAnalyzer, DepthAnalyzer, JumpRiskAnalyzer, LiquidityAnalyzer, MicrostructureNoiseAnalyzer, OrderFlowAnalyzer, OrderLifecycleAnalyzer, ReturnAnalyzer, SpreadAnalyzer, TradeAnalyzer, VolatilityAnalyzer

---

## 7. Shared Computation Engines

Three engines pre-compute expensive intermediate data structures that multiple analyzers consume. Each produces a frozen/immutable data object cached in `DayContext`.

### `price/_return_engine.py` -> DayReturns

**Input**: `DayData` with `mid_price` and `timestamp_ns` columns, plus optional `day_epoch_ns: int = 0`.

**Algorithm**:
1. Filter to finite, positive mid-prices
2. Compute RTH mask via `rth_mask_utc()`
3. Extract open/close prices (first/last valid RTH mid-price)
4. Compute tick-by-tick log returns: `log_returns(mids_valid)` -> `tick_returns`
5. For each timescale in `config.timescales` + `extra_scales_seconds`:
   - If `trading_hours_only`, filter to RTH
   - **Canonical RTH grid**: When `day_epoch_ns > 0` and `trading_hours_only` is True, resamples onto a canonical RTH grid via `rth_grid_edges_ns()` + `resample_to_grid()` instead of `resample()`. This ensures return bins are perfectly aligned with OFI bins from `compute_day_flow()` for cross-series correlation.
   - Otherwise, resample mid-prices at the timescale with `agg="last"` to get bin close prices
   - Compute log returns between consecutive filled-bin closes
   - Store as `ScaledReturns(label, returns, bin_timestamps_ns, n_bins_total, n_bins_filled)`

**Output**: `DayReturns(date, tick_returns, tick_timestamps_ns, scaled: dict[str, ScaledReturns], rth_mask, open_price, close_price, n_valid_prices)`

**Consumers**: ReturnAnalyzer, VolatilityAnalyzer, JumpRiskAnalyzer, MicrostructureNoiseAnalyzer, OrderFlowAnalyzer (OFI-return correlation)

### `spread/_spread_engine.py` -> DaySpreads

**Input**: `DayData` with `spread`, `best_bid`, `best_ask`, `timestamp_ns` columns.

**Algorithm**:
1. Filter to finite, positive spreads and convert to ticks via `spread_in_ticks()`
2. Compute RTH mask
3. For each timescale: resample tick-level spreads with `agg="mean"` to get `ScaledSpreads(label, spreads, bin_timestamps, n_bins_total, n_bins_filled)`

**Output**: `DaySpreads(date, tick_spreads_usd, tick_spreads_bps, tick_timestamps_ns, scaled: dict[str, ScaledSpreads], rth_mask, trade_mask, n_valid)`

**Consumers**: SpreadAnalyzer, DepthAnalyzer, LiquidityAnalyzer

### `flow/_flow_engine.py` -> DayFlow

**Input**: `DayData` with both LOB (`timestamp_ns`, `best_bid`, `best_ask`, `mid_price`, `spread`) and MBO (`timestamp_ns`, `order_id`, `action`, `side`, `price`, `size`).

**Algorithm**:

1. **Vectorized MBO-LOB alignment**: `np.searchsorted(lob_ts, mbo_ts, side="right") - 1` maps every MBO event to the LOB snapshot immediately preceding it. This gives aligned `best_bid` / `best_ask` for BBO comparison. O(N log M) where N = MBO events, M = LOB snapshots.

2. **Trade extraction (aggressor-only)**: Filter MBO events where `action == ACTION_TRADE` **and** `order_id == 0` (aggressor). Databento MBO emits two events per physical trade -- one for the aggressor (incoming/taker, `order_id=0`) and one for the passive (resting/maker, `order_id!=0`). Keeping only aggressors prevents double-counting. For each aggressor trade, record: timestamp, price (USD), size, side, pre-trade mid-price, pre-trade spread. The `side` field on the aggressor event gives the true trade direction (`SIDE_BID` = buyer-initiated, `SIDE_ASK` = seller-initiated). Compute `rth_mask_trades` and `directional_mask` (True where `side != SIDE_NONE`).

3. **OFI computation** (Cont, Kukanov & Stoikov 2014):
   - Identify events at BBO: `mbo_price == aligned_best_bid` or `mbo_price == aligned_best_ask`
   - Assign OFI sign per component (add, cancel, trade separately):
     - **+1 (buy pressure)**: Bid Add at best bid, Ask Cancel at best ask, Buyer-initiated aggressor Trade
     - **-1 (sell pressure)**: Ask Add at best ask, Bid Cancel at best bid, Seller-initiated aggressor Trade
   - Trades use aggressor-only filtering (`order_id == 0`) to prevent the passive-side event from cancelling the aggressor's OFI contribution (they have opposite signs but represent the same physical trade)
   - Total OFI = add OFI + cancel OFI + trade OFI (invariant)
   - **Component decomposition**: `ofi_add_values`, `ofi_cancel_values`, `ofi_trade_values` track the contribution of each event type separately. This allows downstream analysis of cancel-driven vs add-driven OFI (cancel OFI is a stronger short-term signal for options trading).
   - Filter to non-zero OFI events

4. **Multi-scale OFI resampling**: For each scale in `FlowThresholds.ofi_timescales_seconds` (default: 1s, 5s, 10s, 30s, 1m, 5m):
   - Filter OFI events to RTH
   - **Canonical RTH grid**: When `day_epoch_ns > 0`, resamples onto a canonical RTH grid via `rth_grid_edges_ns()` + `resample_to_grid()` instead of `resample()`. This ensures OFI bins are perfectly aligned with return bins from `compute_day_returns()` for cross-series correlation.
   - Resample with `agg="sum"` to get net OFI per time bin
   - **Normalization**: `normalized_ofi = net_ofi / std(net_ofi)`. NaN where std=0. Makes OFI comparable across days and stocks.
   - Store as `ScaledOFI(label, net_ofi, normalized_ofi, bin_timestamps_ns, counts, n_bins_total, n_bins_filled)`

**Output**: `DayFlow(date, trade_timestamps_ns, trade_prices_usd, trade_sizes, trade_sides, trade_mid_before, trade_spread_before, ofi_timestamps_ns, ofi_values, ofi_add_values, ofi_cancel_values, ofi_trade_values, scaled_ofi, rth_mask_trades, directional_mask, n_trades, n_ofi_events)`

**Critical: `directional_mask`**: Boolean array over trades. `True` = SIDE_BID or SIDE_ASK (known aggressor). `False` = SIDE_NONE (system/clearing trade, no aggressor). All directional metrics (cumulative delta, aggressor ratio, trade imbalance) must filter through this mask. Non-directional metrics (size distribution, VWAP, clustering) should include SIDE_NONE trades.

**Consumers**: OrderFlowAnalyzer, TradeAnalyzer (OrderLifecycleAnalyzer processes MBO directly)

---

## 8. Health Domain

### DataQualityAnalyzer (`analysis/health/data_quality.py`)

**Purpose**: Validates data integrity and produces an overview of the dataset before any analytical processing. This is always the first analyzer to run.

**ClassVars**:
- `needs_mbo = True` (validates MBO data too)
- `lob_columns = None` (needs all columns for comprehensive validation)
- `mbo_columns = None` (needs all columns)

**Metrics computed per day (accumulated via WelfordAccumulator)**:
- Row counts: LOB rows, MBO rows
- Timestamp range: first, last, max inter-event gap
- Median inter-event time (LOB and MBO separately)
- MBO action distribution: count per action type (Add, Cancel, Trade, etc.)
- MBO side distribution: count per side (Bid, Ask, None)
- Book consistency distribution: count per state (Valid, Empty, Locked, Crossed)
- Crossed book detection: count of events where `best_bid >= best_ask`
- Time regime distribution: fraction of events in each of the 7 intraday regimes

**Report**: `DataQualityReport` with per-day records and cross-day summary statistics.

---

## 9. Price Domain

All price-domain analyzers consume `DayReturns` from `DayContext.day_returns` (set `needs_returns = True`).

### ReturnAnalyzer (`analysis/price/returns.py`)

**Purpose**: Characterizes return distributions at every configured timescale.

**Metrics (9 sub-analyses)**:
1. **Return distribution per timescale**: mean, std, skewness, kurtosis, percentiles -- via `distribution_summary()` on `ScaledReturns.returns` accumulated across days
2. **Intraday return curve**: 390 1-minute bins, mean/std return per bin across all days
3. **Overnight return decomposition**: close-to-open vs open-to-close contributions
4. **Tail analysis**: Hill tail index (left and right tails), 5% VaR and CVaR via `var_cvar()`
5. **Return persistence**: autocorrelation at lags 1..20 (FFT-based via `np.correlate`)
6. **Absolute return clustering**: autocorrelation of |returns| (measures volatility clustering)
7. **Zero-return fraction**: per timescale (indicates potential staleness or illiquidity)
8. **Max drawdown/runup**: per-day peak-to-trough and trough-to-peak from cumulative returns
9. **Weekday patterns**: mean return and mean |return| by day-of-week

**Accumulation**: Uses `StreamingDistribution` (Welford + reservoir sampling) for bounded-memory distribution statistics at all timescales. Tail analysis, quantiles, skewness, and kurtosis are computed from the reservoir sample.

### VolatilityAnalyzer (`analysis/price/volatility.py`)

**Purpose**: Multi-scale realized volatility characterization.

**Metrics (7 sub-analyses)**:
1. **Realized volatility per timescale**: `RV = sum(r_t^2)` at each scale, annualized as `AnnVol = sqrt(RV * 252) * 100%`. Note: RV is the sum of squared returns over one day, so it already represents the day's total variance; no bins-per-day multiplier is needed. Mean/std across days.
2. **Intraday volatility curve**: 1-minute bins, mean volatility per bin (U-shaped for equities)
3. **Volatility-of-volatility**: std of daily RV values (measures regime instability)
4. **Volatility persistence**: autocorrelation of daily RV at lags 1..20
5. **Volatility clustering** (ARCH effects): Ljung-Box test on squared returns at configurable lags
6. **Weekday patterns**: mean RV by day-of-week
7. **Correlation with spread**: Pearson correlation between daily RV and daily mean spread

**Extra scales**: Declares `get_extra_scales()` returning `VolatilityThresholds.signature_scales_seconds` for the volatility signature plot.

### JumpRiskAnalyzer (`analysis/price/jump_risk.py`)

**Purpose**: Detects and characterizes price jumps using bipower variation.

**Method**: Barndorff-Nielsen & Shephard (2004) BNS test:
- `BV = (pi/2) * sum(|r_t| * |r_{t-1}|)` (bipower variation, robust to jumps)
- `J = RV - BV` (jump component, >= 0 in theory)
- `jump_fraction = J / RV`
- Test statistic: `z = (RV - BV) / sqrt(vq)` where vq uses realized quarticity
- Confidence level: configurable via `VolatilityThresholds.jump_confidence`

**Metrics (5 sub-analyses)**:
1. **BNS test summary**: mean z-statistic, fraction of days with significant jumps
2. **Jump fraction distribution**: mean, median, max across days
3. **Jump size distribution**: conditional on jump detected -- mean, std, percentiles
4. **Regime-conditional jumps**: jump rate and mean jump fraction per time regime
5. **Intraday jump curve**: 5-minute bins, fraction of jumps occurring in each bin

### MicrostructureNoiseAnalyzer (`analysis/price/microstructure_noise.py`)

**Purpose**: Estimates the noise floor of raw mid-prices and recommends optimal sampling frequency.

**Method**: Volatility signature plot approach:
- Compute RV at 20 log-spaced scales from 0.1s to 60s
- Noise variance: `(RV_fast - RV_slow) / (2 * n_fast_bins)`
- Signal-to-noise ratio: `true_var / noise_var`
- Optimal sampling: MSE-minimizing frequency from Zhang, Mykland & Ait-Sahalia (2005)

**Additional metrics**:
1. **Roll's implied spread**: `2 * sqrt(-gamma_1) * mid_level` where `gamma_1` is first-order autocovariance of tick returns. Roll (1984) Eq. 2. Negative autocovariance indicates bid-ask bounce.
2. **Roll-to-observed ratio**: Roll implied spread / actual mean spread
3. **Optimal sampling recommendation**: Rounded to nearest "nice" frequency (1s, 5s, etc.)

---

## 10. Spread Domain

All spread-domain analyzers consume `DaySpreads` from `DayContext.day_spreads` (set `needs_spreads = True`).

### SpreadAnalyzer (`analysis/spread/spread.py`)

**Purpose**: Characterizes spread distribution, regime behavior, and trade-conditional dynamics.

**Metrics (6 sub-analyses)**:
1. **Tick-level spread distribution**: mean, median, percentiles in ticks and USD
2. **Intraday spread curve**: 1-minute bins, mean spread per bin across days (U-shaped)
3. **Regime-conditional spreads**: mean/median/std spread per time regime
4. **Trade-conditional spreads**: spread at moments when MBO trades arrive vs unconditional
5. **Spread width classification**: fraction of time in 1-tick, 2-tick, 3-4 tick, 5+ tick (configurable via `SpreadThresholds.wide_spread_ticks`)
6. **Spread autocorrelation**: ACF at lags 1..20 (measures spread persistence)

### DepthAnalyzer (`analysis/spread/depth.py`)

**Purpose**: Analyzes the shape, imbalance, and resilience of the 10-level order book.

**Metrics (5 sub-analyses)**:
1. **Depth profile**: Mean bid/ask sizes at each of the 10 levels across all days
2. **Depth imbalance**: Distribution of `(total_bid - total_ask) / (total_bid + total_ask)` from LOB's `depth_imbalance` column
3. **Concentration**: Fraction of total volume at level 1 (BBO) -- higher = thinner book
4. **Post-trade depth recovery**: Depth at horizons [1, 5, 10, 50] events after a trade, measuring how fast the book replenishes
5. **Depth stability**: Coefficient of variation of total depth, measuring how erratic book size is

### LiquidityAnalyzer (`analysis/spread/liquidity.py`)

**Purpose**: Execution cost analysis combining LOB snapshots with MBO trade data.

**ClassVars**: `needs_mbo = True`, `needs_spreads = True`

**Metrics (5 sub-analyses)**:
1. **Effective spread**: `2 * |trade_price - mid_before| / mid_before * 10000` (in bps) for each MBO trade, using `np.searchsorted` to align trades with LOB snapshots. Includes distribution statistics.
2. **Volume-weighted spread**: `sum(spread * trade_size) / sum(trade_size)` -- captures the spread experienced by actual trading volume
3. **Microprice deviation**: `|microprice - mid_price| / mid_price * 10000` in bps -- measures the information content of the microprice beyond the simple midpoint
4. **Kyle's lambda**: Regression coefficient from `delta_price = lambda * signed_volume + epsilon`, estimated at 1-second scale using OLS. Kyle (1985). Higher lambda = lower liquidity / higher price impact.
5. **Amihud illiquidity**: `|return| / dollar_volume` aggregated at 5-minute scale. Amihud (2002). Higher = less liquid.

---

## 11. Flow Domain

All flow-domain analyzers (except OrderLifecycleAnalyzer) consume `DayFlow` from `DayContext.day_flow` (set `needs_flow = True`). The flow `__init__.py` uses **lazy imports** to avoid circular dependencies.

### OrderFlowAnalyzer (`analysis/flow/order_flow.py`)

**Purpose**: The core OFI analyzer -- computes the most predictive short-term signal in microstructure literature.

**ClassVars**: `needs_mbo = True`, `needs_flow = True`, `needs_returns = True`, `needs_spreads = True`

**Metrics (12 sub-analyses)**:

1. **OFI distribution per timescale**: For each of the 6 OFI scales (1s, 5s, 10s, 30s, 1m, 5m), computes mean, std, skewness, kurtosis across all RTH bins across all days. Includes normalized OFI distribution (mean ~0, std ~1) for cross-stock comparability.

2. **Cumulative delta**: End-of-day signed volume `sum(size * sign)` where sign is +1 for buyer-initiated, -1 for seller-initiated. **Excludes SIDE_NONE via `directional_mask`**. Positive = net buying, negative = net selling.

3. **Aggressor ratio**: `buyer_volume / (buyer_volume + seller_volume)` using only directional trades. Near 0.5 = balanced, >0.55 = buyer-dominated, <0.45 = seller-dominated.

4. **OFI-return cross-correlation**: At each timescale, computes Pearson correlation between **normalized** OFI bins and return bins at lags 0 to `flow_return_n_lags`. Reports the peak lag and peak correlation. **This is the key predictive signal** -- if OFI at lag=1 predicts returns at lag=0, there is a tradeable signal. Uses normalized OFI for stability across days. **Alignment**: OFI and return bins are matched by timestamp intersection (`np.intersect1d` on `bin_timestamps_ns`), not by index, ensuring correct alignment even with sparse bins. Lags that cannot be computed (insufficient data) are set to `np.nan` rather than `0.0`.

5. **OFI-spread cross-correlation**: At each timescale, computes Pearson correlation between **normalized** OFI bins and spread change (Δspread) bins at multiple lags. Reveals how order flow drives spread widening/tightening -- critical for options entry timing. **Alignment**: Same timestamp-based intersection as OFI-return correlation.

6. **OFI component breakdown**: Decomposes total |OFI| into add/cancel/trade fractions. Reports overall fractions and per-regime fractions. Cancel OFI fraction is a key signal: high cancel fraction = market maker quote-pulling (short-term directional signal for options).

7. **OFI autocorrelation**: ACF of OFI at lags 1..20. Persistent OFI (slow decay) indicates trending flow; mean-reverting OFI (fast decay) indicates noise.

8. **Flow intensity by regime**: Mean |OFI| per time regime (open-auction, morning, midday, afternoon, close-auction). Higher = more active order flow.

9. **Intraday flow curve**: 390 1-minute bins, mean OFI per bin across all days. Reveals systematic intraday flow patterns (e.g., institutional buying in the afternoon).

10. **Trade imbalance per timescale**: `(buyer_vol - seller_vol) / (buyer_vol + seller_vol)` per bin, **excluding SIDE_NONE**. Reports distribution of imbalance values.

11. **Weekday patterns**: Mean end-of-day delta and mean buyer fraction per day-of-week.

### TradeAnalyzer (`analysis/flow/trade.py`)

**Purpose**: Trade execution analysis -- size distribution, timing, impact, and execution quality.

**ClassVars**: `needs_mbo = True`, `needs_flow = True`

**Metrics (8 sub-analyses)**:

1. **Trade size distribution**: Distribution of individual trade sizes (shares), separately for RTH and all-hours. Includes mean, median, percentiles, large-trade threshold (95th percentile by default).

2. **Trade-through rate by regime**: Fraction of trades where trade_price < best_bid or trade_price > best_ask (trade at a price worse than BBO). Broken down by time regime. Higher rate at open/close indicates more aggressive execution.

3. **Trade clustering**: Inter-trade time distribution. A "cluster" is a sequence of trades with inter-trade gap < `trade_cluster_gap_seconds` (default 1s). Reports mean cluster size, max cluster size, fraction of trades in clusters.

4. **VWAP trajectory**: Cumulative VWAP throughout the day (% deviation from end-of-day VWAP at each point). Reveals whether volume is front-loaded or back-loaded.

5. **Large trade impact**: For trades above the `large_trade_percentile` threshold, measures `(trade_price - mid_before) / mid_before * 10000` (in bps). Reports mean/median impact, buyer vs seller count split (**using SIDE_ASK for seller, not `!= SIDE_BID`** to correctly exclude SIDE_NONE).

6. **Trade rate by regime**: Trades per second in each time regime. Reveals intraday execution activity patterns.

7. **Trade price level classification**: Categorizes each trade as "at_bid", "at_ask", "inside" (between bid and ask), or "outside" (beyond BBO). Uses pre-trade LOB state from `DayFlow.trade_mid_before` and `trade_spread_before`.

8. **Directional trade size distribution**: Separate size distributions for buyer-initiated and seller-initiated trades, using `directional_mask` to exclude SIDE_NONE. Reveals asymmetry in trade sizing between buyers and sellers (e.g., buyers trading in larger clips = institutional buying pressure).

### OrderLifecycleAnalyzer (`analysis/flow/order_lifecycle.py`)

**Purpose**: Tracks individual orders from birth (Add) to death (Cancel/Trade/Fill) using MBO `order_id`. Reveals market participant behavior and market maker dynamics.

**ClassVars**: `needs_mbo = True`, `needs_flow = False` (processes MBO directly, does not use DayFlow)

**Stateful tracking**: Maintains an `_active_orders: dict[int, _OrderState]` mapping `order_id` -> `(add_timestamp, add_size, remaining_size, side, n_modifies, n_partial_fills)`. Orders are evicted after `order_lifetime_max_seconds` (default 3600s) or when `max_active_orders` (default 500K) is reached. This bounds memory to approximately `500K * 56 bytes = ~28 MB`.

**Partial fill support**: When a Trade/Fill event arrives for an active order, the fill size is subtracted from `remaining_size`. If `remaining_size > 0`, the order stays active (partial fill). Only when `remaining_size <= 0` is the order fully filled and resolved. Cancel/Clear events always resolve immediately regardless of remaining size.

**Metrics (8 sub-analyses)**:

1. **Order lifetime distribution**: Histogram of time from Add to Cancel/Trade/Fill in seconds. Log-spaced bins from 1ms to `order_lifetime_max_seconds`. Reports separately for bid and ask orders. Sub-second lifetimes indicate HFT; multi-minute indicate institutional.

2. **Fill rate**: Fraction of orders that terminate via Trade/Fill vs Cancel. Reported overall and separately for bid/ask. Higher fill rate = more patient or better-calibrated orders.

3. **Cancel-to-add ratio**: `n_cancels / n_adds`. Near 1.0 = most orders are canceled (typical for HFT market making). Much less than 1.0 = many orders fill.

4. **Modify patterns**: Distribution of modify count per order. Mean, median, max modifies. Orders with many modifies may be algorithmic strategies adjusting aggressively.

5. **Event transition matrix**: 4x4 matrix of `P(next_action | current_action)` for actions {Add, Cancel, Trade, Modify}. Normalized per row. E.g., `P(Cancel | Add)` reveals what fraction of adds are immediately canceled.

6. **Duration-by-size correlation**: Pearson correlation between initial order size and order lifetime. Positive = larger orders live longer (institutional patience); negative = larger orders are picked off faster.

7. **Regime-conditional lifecycle**: Mean lifetime, fill rate, and cancel rate per time regime. Reveals how market participant behavior shifts across the trading day.

8. **Partial fill patterns**: Fraction of filled orders that received multiple fill events (`partial_fill_fraction`), mean fill events per filled order (`mean_fills_per_order`), and max fill events. High partial fill fraction indicates the stock has significant iceberg/hidden liquidity or large orders being worked algorithmically.

---

## 12. Reports

### `reports/base_report.py` -- BaseReport

Abstract base class for all analyzer reports:

- `SCHEMA_VERSION: ClassVar[str] = "1.0"` -- semantic version for report format
- `to_dict()` -> JSON-compatible dict (must include `_meta` key with schema version, timestamp)
- `summary()` -> human-readable multi-line string
- `_meta_dict()` -> `{"schema_version": ..., "created_at": ISO timestamp, "analyzer_version": ...}`
- `to_json(path, indent=2)` -> writes RFC 8259 compliant JSON to disk; all `NaN`/`Inf`/`-Inf` values are sanitized to `null` via `_sanitize_for_json()` and `allow_nan=False`

Every analyzer defines its own `*Report` dataclass inheriting from `BaseReport`.

### `reports/formatters.py` -- Text Formatting

Utility functions for consistent report rendering:

- `format_section(title)` -> `"==== TITLE ===="` header block
- `format_subsection(title)` -> `"---- TITLE ----"` header block
- `format_kv(items)` -> aligned key-value pairs
- `format_table(headers, rows)` -> right-aligned text table with separator line
- `_format_cell(value)` -> formats floats (4 decimals, scientific for tiny values), ints (comma-separated), None as "-"

---

## 13. Configuration Profiles

Profiles are YAML files in `configs/profiles/`. Each defines:

```yaml
name: <profile_name>
description: "..."
config:                    # optional overrides
  timescales: ["1s", "5s", "30s", "1m", "5m", "15m", "1h"]
  trading_hours: "us_equity_rth"
phases:
  - name: <phase_name>
    description: "..."
    analyzers:
      - AnalyzerClassName
      - name: AnalyzerClassName    # alternate form with per-analyzer config
        config: { ... }
```

**Current profiles**:

| Profile | Analyzers (count) | Description |
|---------|-------------------|-------------|
| `quick` | DataQualityAnalyzer (1) | Smoke test |
| `standard` | DataQuality, Return, Volatility, Spread, OrderFlow (5) | Balanced daily |
| `volatility` | DataQuality, Return, Volatility, JumpRisk, Noise (5) | Vol deep-dive |
| `spread` | DataQuality, Spread, Depth, Liquidity (4) | Liquidity focus |
| `flow` | DataQuality, OrderFlow, Trade, OrderLifecycle (4) | Order flow focus |
| `full` | All 11 analyzers | Complete characterization |

### Date Sampling (`scripts/generate_sample.py`)

For large datasets (e.g., 234 trading days), analyzing every day with the `full` profile is expensive. The stratified sampling script selects a representative subset.

**Strategy**:
1. Discover all dates from the export directory
2. Use MBO file size as a fast volume proxy (no data loading)
3. Force-include the top-K and bottom-K days by volume (extremes are most informative for tail analysis)
4. Stratify remaining budget by month (proportional representation)
5. Within each month, stratify by volume quartile
6. Ensure all 5 weekdays are represented
7. Deterministic: uses `numpy.random.default_rng(seed=42)`

**CLI**:
```bash
python scripts/generate_sample.py \
    --data-dir /path/to/exports/ \
    --output configs/sampling/nvda_stratified_50.txt \
    --target 50 --extreme-k 5 --seed 42
```

**Output**: Text file with one `YYYY-MM-DD` per line, suitable for `--dates-file` in `run_analysis.py`.

**Pre-generated sample**: `configs/sampling/nvda_stratified_50.txt` -- 58 dates from the full 234-day NVDA dataset (Feb 2025 - Jan 2026), covering all 12 months and all weekdays.

---

## 13.5. Operational Robustness

### Memory Safety

All 11 analyzers are designed for bounded-memory operation over arbitrarily long multi-day runs. Six analyzers that previously accumulated unbounded per-event lists were fixed:

| Analyzer | Previous Issue | Fix | Memory Bound |
|----------|---------------|-----|-------------|
| **DepthAnalyzer** | `_imbalance_all`, `_bid_l1_all`, `_ask_l1_all` -- appended full-day arrays (~10M floats/day, ~22 GB over 234 days) | Replaced with `StreamingDistribution` (reservoir 100K) | O(300K) floats |
| **LiquidityAnalyzer** | `_effective_spreads` -- one float per trade (~330M over 234 days) | Replaced with `StreamingDistribution` + per-regime `WelfordAccumulator` | O(100K + 7) |
| **OrderLifecycleAnalyzer** | 6 parallel unbounded lists + per-regime lists -- one entry per resolved order (hundreds of millions) | `StreamingDistribution` (lifetime, 2x reservoir 200K for duration-size correlation), `WelfordAccumulator` (per-side, per-regime), counters for categoricals | O(500K) active orders + O(400K) reservoir |
| **TradeAnalyzer** | `_large_impacts` -- one entry per large trade | Replaced with `StreamingDistribution` | O(100K) |
| **JumpRiskAnalyzer** | `_jump_returns` -- arrays of jump returns per jump day | Replaced with `StreamingDistribution` + `WelfordAccumulator` (pos/neg) | O(100K) |
| **SpreadAnalyzer** | `_resampled_for_acf` -- unbounded list of daily arrays | Replaced with `deque(maxlen=60)` (last 60 days) | O(23K) floats |

**Key primitives**:
- `StreamingDistribution` -- Welford accumulator + Algorithm R reservoir sampling (Vitter 1985). Provides exact streaming mean/std and approximate quantiles/skewness/kurtosis from the reservoir. Memory: O(reservoir_size) regardless of total observations.
- `WelfordAccumulator` -- O(1) streaming mean/variance (Welford 1962). Used for per-regime and per-side statistics.
- `deque(maxlen=N)` -- bounded FIFO for ordered sequences needed in their entirety (e.g., ACF input).

### DST Correctness

Datasets spanning multiple months (e.g., Feb 2025 - Jan 2026) cross US DST boundaries:
- **EST (UTC-5)**: ~Nov to ~Mar
- **EDT (UTC-4)**: ~Mar to ~Nov

The orchestrator computes the correct offset per day via `utc_offset_for_date()` and adjusts `TradingHours` accordingly. All RTH masks, regime classifications, and canonical RTH grids use the per-day DST-aware offset from `DayContext.utc_offset_hours`. Without this, RTH would be misclassified by 1 hour for ~150 of 234 days in a year-long NVDA dataset.

### Checkpoint/Resume

The orchestrator supports crash recovery for long-running overnight analyses:

```bash
python scripts/run_analysis.py \
    -d /path/to/exports/ -s NVDA --profile full \
    --checkpoint-dir /tmp/nvda_checkpoint --resume
```

After each day, all analyzer states are serialized to `checkpoint_dir`. On restart with `--resume`, completed days are skipped. This bounds data loss to a single day's work on any crash.

### Progress Logging

When `verbose=True`, the orchestrator emits per-day progress with timing, throughput, memory, and ETA:

```
[3/234] 2025-02-05  537.2s  45,000 rows/s  ETA 1.2h  peak 4,063 MB  (LOB=12,300,000 MBO=5,400,000)
```

- **ETA**: Exponential moving average (alpha=0.3) of per-day processing time, multiplied by remaining days
- **Peak memory**: RSS via `resource.getrusage` (macOS: bytes, Linux: KB)
- **Throughput**: `(LOB rows + MBO rows) / elapsed_seconds`

---

## 14. Test Suite

**380 tests** covering all modules. Tests use synthetic Parquet data generated in `tests/conftest.py`.

### `tests/conftest.py` -- Shared Test Fixtures

**`_make_day(n_lob, n_mbo, ...)`**: Generates a `DayData` with synthetic LOB and MBO Parquet tables:
- LOB: sequential timestamps, random bid/ask prices around $100, derived columns (mid_price, spread, depth_imbalance, etc.)
- MBO: sequential timestamps, mixed actions (Add/Cancel/Trade), random sides (Bid/Ask/None), prices near BBO, random sizes. Trade events automatically receive `order_id=0` (aggressor convention); non-trade events receive sequential IDs.
- Configurable: `n_lob`, `n_mbo`, `ts_start` (nanosecond start time), price level, spread

**Key design**: `ts_start` defaults to within RTH (10:00 AM ET = `1_738_594_800_000_000_000` ns) so that RTH-dependent logic works correctly in tests. Trade events in synthetic data use `order_id=0` (aggressor) to match the real Databento MBO structure. Test helpers in `test_flow_engine.py` accept an explicit `mbo_order_ids` parameter for tests that need paired aggressor+passive events.

### Test Organization

```
tests/
├── test_analysis/
│   ├── test_base.py                 # 8 tests: BaseAnalyzer protocol
│   ├── test_orchestrator.py         # 12 tests: BatchOrchestrator, DayContext caching
│   ├── test_empty_day.py            # 44 tests: zero-day finalize, empty-day process, JSON roundtrip, summary for all 11 analyzers
│   ├── test_flow/
│   │   ├── test_flow_engine.py      # 33 tests: trade extraction, OFI, SIDE_NONE, decomposition, normalization, MBO trade deduplication
│   │   ├── test_order_flow.py       # 12 tests: OFI distribution, delta, aggressor, spread corr, components
│   │   ├── test_trade.py            # 12 tests: size, clustering, VWAP, impact, directional sizes
│   │   └── test_order_lifecycle.py  # 17 tests: lifetime, fill rate, transitions, partial fills, eviction hard cap
│   ├── test_price/
│   │   ├── test_return_engine.py    # 9 tests: DayReturns, ScaledReturns
│   │   ├── test_returns.py          # 20 tests: return dist, tail, persistence
│   │   ├── test_volatility.py       # 13 tests: RV, ACF, ARCH, weekday
│   │   ├── test_jump_risk.py        # 14 tests: BNS test, jump fraction
│   │   └── test_microstructure_noise.py  # 11 tests: signature, Roll, SNR
│   └── test_spread/
│       ├── test_spread_engine.py    # 9 tests: DaySpreads, ScaledSpreads
│       ├── test_spread.py           # 6 tests: dist, regime, classification
│       ├── test_depth.py            # 5 tests: profile, imbalance, recovery
│       └── test_liquidity.py        # 4 tests: effective spread, Kyle's lambda
├── test_config/
│   └── test_config.py               # 16 tests: config, profiles, timescales
├── test_core/
│   ├── test_resampler.py            # 22 tests: all agg modes, edge cases, resample_to_grid
│   ├── test_time_utils.py           # 22 tests: RTH mask, regimes, rth_grid_edges_ns, DST boundaries
│   ├── test_price_utils.py          # 9 tests: conversions, log returns
│   └── test_statistics.py           # 15 tests: Welford (incl. NaN handling), distribution, VaR
├── test_io/
│   ├── test_loader.py               # 12 tests: loading, projection, validation
│   └── test_schema.py               # 10 tests: schema validation, enums
└── test_reports/
    ├── test_base_report.py          # 20 tests: NaN sanitization, metadata, JSON roundtrip
    └── test_formatters.py           # 25 tests: NaN/Inf/None cells, empty tables, section headers
```

---

## 15. Module Dependency Graph

```
                    io/schema.py (constants, enums)
                          |
                    io/loader.py (ParquetDayLoader, DayData)
                          |
                    io/session.py (AnalysisSession)
                          |
                    analysis/orchestrator.py
                       /     |     \
                      /      |      \
         _return_engine  _spread_engine  _flow_engine
              |               |               |
         DayReturns      DaySpreads       DayFlow
              \               |               /
               \              |              /
                  analysis/context.py (DayContext)
                          |
                  analysis/base.py (BaseAnalyzer)
                  /    |    |    \
                 /     |    |     \
         health/  price/  spread/  flow/
```

**Import rules**:
- `core/` modules have zero imports from `analysis/` or `io/` (leaf dependencies)
- `io/` modules import only from `core/`
- `analysis/` modules import from `io/` and `core/`
- Shared engines (`_return_engine`, `_spread_engine`, `_flow_engine`) import from `io/` and `core/`
- Analyzers import from their domain's shared engine + `core/` + `io/`
- `analysis/flow/__init__.py` uses lazy `__getattr__` imports to prevent circular dependencies

---

## 16. Critical Invariants

These invariants must hold for correctness. Violating any of them will produce silently wrong results.

1. **Timestamps are sorted**: All LOB and MBO Parquet files have monotonically non-decreasing `timestamp_ns`. The resampler's integer-division bin assignment and `searchsorted` alignment depend on this.

2. **Prices are in nanodollars (int64)**: LOB `best_bid`, `best_ask`, MBO `price` are nanodollars. Derived columns `mid_price`, `spread` are in USD (float64). Never mix units.

3. **SIDE_NONE exclusion**: `directional_mask` in `DayFlow` must be used for all directional calculations (cumulative delta, aggressor ratio, trade imbalance). SIDE_NONE trades carry ~34% of volume and will severely bias results if included directionally.

4. **RTH filtering**: All timescale-dependent analysis filters to RTH before resampling when `trading_hours_only = True`. Pre-market and after-hours events contaminate intraday curves and RV estimates.

5. **Division guards**: All divisions use `EPS` guard or `np.where` to handle near-zero denominators. `np.isfinite()` checks before comparisons. WelfordAccumulator returns 0 variance for count < 2.

6. **Column projection consistency**: If an analyzer declares `lob_columns = ["timestamp_ns", "mid_price"]`, those columns must exist in the Parquet file. The loader silently drops columns not present in the file (graceful degradation), but the analyzer's `cached_property` will raise `ValueError` if it tries to access an unloaded column.

7. **Order lifecycle memory bound**: `OrderLifecycleAnalyzer` tracks up to `max_active_orders` (default 500K) simultaneously. Eviction is FIFO. If the stock has more than 500K concurrent active orders, oldest orders will be silently evicted and their lifecycle metrics lost. NVDA data shows ~5.3M adds/day but median lifetime is 38ms, so simultaneous active orders are well below 500K.

8. **OFI at BBO only**: The flow engine only counts OFI for events at the best bid or best ask. Events deeper in the book do not contribute to OFI. This matches the Cont, Kukanov & Stoikov (2014) definition but means depth-level order flow is not captured.

9. **OFI component decomposition invariant**: `ofi_values == ofi_add_values + ofi_cancel_values + ofi_trade_values` element-wise. The three component arrays are computed from disjoint boolean masks (add, cancel, trade) and their sum must equal the total OFI. Verified by test `TestOFIDecomposition.test_components_sum_to_total`.

10. **Partial fill remaining_size**: In `OrderLifecycleAnalyzer`, `remaining_size` must never go below zero for a well-formed MBO stream. If fill size exceeds remaining size (data anomaly), the order is resolved as fully filled.

11. **Aggressor-only trades**: The flow engine filters MBO trade events to `order_id == 0` (aggressor side). Databento MBO emits two events per physical trade -- one for the aggressor (`order_id=0`, incoming/taker) and one for the passive (`order_id!=0`, resting/maker). Processing both double-counts every trade and zeroes out the OFI trade component (since aggressor `+1` cancels with passive `-1`). The `OrderLifecycleAnalyzer` is unaffected because `order_id=0` events never appear in its active-order tracking dict (no Add event has `order_id=0`).

12. **Canonical RTH grid alignment**: When `day_epoch_ns > 0`, both `compute_day_returns()` and `compute_day_flow()` resample RTH-only timescales onto the same canonical grid produced by `rth_grid_edges_ns()`. This guarantees that OFI bins and return bins have identical timestamps for cross-series correlation. Without this alignment, data-driven grids from MBO (OFI) and LOB (returns) sources produce different bin edges, making index-based correlation silently wrong.

13. **JSON RFC 8259 compliance**: All reports use `_sanitize_for_json()` + `allow_nan=False` to ensure output JSON contains no `NaN`/`Infinity`/`-Infinity` tokens. This is critical for downstream consumers (JavaScript, Rust serde, Go) that reject non-standard JSON.

14. **Streaming memory bounds**: All analyzers use `StreamingDistribution` (Welford + reservoir sampling) or `WelfordAccumulator` for statistics that previously accumulated unbounded lists. Memory is O(reservoir_size) per accumulator regardless of dataset size.

15. **OFI-return/spread correlation uses timestamp intersection**: `OrderFlowAnalyzer._compute_ofi_return_corr` and `_compute_ofi_spread_corr` align bins by timestamp intersection (`np.intersect1d` on `bin_timestamps_ns`), not by index. This is a safety net even with canonical grid alignment, ensuring correctness if bin timestamps differ due to missing data or configuration differences.

16. **DST-aware per-day UTC offset**: The orchestrator calls `utc_offset_for_date(day.date)` for every trading day and propagates the result through `DayContext.utc_offset_hours` and a per-day `AnalysisConfig` with adjusted `TradingHours`. All downstream code (RTH masks, regime classifications, canonical RTH grids, intraday curves) must use the offset from context, never a hardcoded value. Failure to do so misclassifies RTH by 1 hour for all EDT days (~150 of 234 in a year-long US equity dataset).

17. **Checkpoint atomicity**: Checkpoint saves write all analyzer pickles and then update `manifest.txt`. If the process crashes mid-save, the manifest may not include the partially-written day. On restore, any day not in the manifest is re-processed. This is safe because `process_day` is idempotent with respect to analyzer accumulator state (processing a day twice just adds its contribution again, which is incorrect but detectable). For this reason, checkpoint integrity should be verified by checking that all `{analyzer.name}.pkl` files exist and the manifest is consistent.
