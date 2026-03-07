# MBO-LOB-analyzer

Deep statistical analysis of raw **Market-By-Order (MBO) events** and **Limit Order Book (LOB) snapshots** for intraday trading signal discovery.

This repo is the analytical backbone of the HFT pipeline. It characterizes every stock's microstructure -- spread dynamics, volatility regime, order flow patterns, and participant behavior -- across configurable timescales (1s to daily), producing the insights that drive feature extractor configuration and model selection for profitable intraday strategies (0DTE ATM options, short-term directional).

## Objective

Analyze the **raw, unsampled** MBO + LOB data exported by [MBO-LOB-reconstructor](../MBO-LOB-reconstructor/) **before** any feature extraction or normalization. This repo answers:

1. **What is this stock's microstructure profile?** -- spread distribution, depth shape, liquidity regime, volatility signature
2. **Who is in the market?** -- order lifetimes reveal HFT vs institutional participants; fill rates and cancel dynamics expose market maker behavior
3. **Where is the directional signal?** -- OFI (Order Flow Imbalance) at multiple timescales, OFI-return cross-correlation, cumulative delta, aggressor ratio
4. **What configuration should the feature extractor use?** -- optimal sampling frequency (noise floor), jump risk thresholds, spread regime boundaries
5. **Are there exploitable patterns?** -- intraday flow curves, trade clustering, regime-conditional behavior, weekday effects

## Why MBO Data (Not Just LOB)

LOB snapshots show the *result* of order book changes. MBO events reveal the *cause*. For short-term trading, this distinction is critical:

| Capability | LOB Only | MBO + LOB |
|---|---|---|
| Spread / depth / mid-price | Yes | Yes |
| OFI (who is adding/canceling at BBO) | No | **Yes** |
| Order lifetime distribution | No | **Yes** (requires `order_id` tracking) |
| Fill rate / cancel-to-add ratio | No | **Yes** |
| Aggressor classification (buyer vs seller initiated) | No | **Yes** (MBO `side` field) |
| Iceberg order detection | No | **Planned** |
| Queue position estimation | No | **Planned** |
| Event clustering / Hawkes processes | Partial | **Planned** (full action+side+timestamp) |

## Data Contract

Consumes Parquet files exported by `MBO-LOB-reconstructor`:

| File | Contents | Key Columns |
|---|---|---|
| `{date}_lob_snapshots.parquet` | Full 10-level order book state at every MBO event | `timestamp_ns`, `best_bid`, `best_ask`, `mid_price`, `spread`, `bid_prices[10]`, `bid_sizes[10]`, `ask_prices[10]`, `ask_sizes[10]`, `depth_imbalance` |
| `{date}_mbo_events.parquet` | Raw MBO messages | `timestamp_ns`, `order_id`, `action` (Add/Cancel/Trade/Clear), `side` (Bid/Ask/None), `price`, `size` |

- **Prices**: nanodollars (int64, divide by 1e9 for USD)
- **Timestamps**: nanoseconds since epoch (UTC)
- **Sizes**: shares (uint32)

### Critical: MBO Trade Pairing (Aggressor Filtering)

Databento MBO data emits **two events per physical trade**: one for the aggressor (incoming/taker, `order_id=0`) and one for the passive (resting/maker, `order_id!=0`). The MBO-LOB-reconstructor exports both with `action='T'`. The flow engine filters to **aggressor-only** events (`order_id == 0`) to avoid double-counting trades, volumes, and OFI trade contributions. The `side` field on the aggressor event gives the true trade direction.

### Critical: SIDE_NONE Trades

~12% of MBO Trade events have `side=None` (system/clearing trades with no aggressor). These carry ~34% of volume. All directional calculations (cumulative delta, aggressor ratio, trade imbalance) **exclude** SIDE_NONE trades via the `directional_mask` in `DayFlow`. Non-directional metrics (size distribution, VWAP, clustering) include them.

## Quick Start

```bash
cd MBO-LOB-analyzer
uv pip install -e ".[dev]"

# Full analysis (all 11 analyzers)
uv run python scripts/run_analysis.py \
    --data-dir /path/to/exports/ \
    --symbol NVDA \
    --profile full

# Order flow focus (OFI, trades, lifecycle)
uv run python scripts/run_analysis.py \
    --data-dir /path/to/exports/ \
    --symbol NVDA \
    --profile flow

# Specific analyzers
uv run python scripts/run_analysis.py \
    --data-dir /path/to/exports/ \
    --symbol NVDA \
    --analyzer OrderFlowAnalyzer,TradeAnalyzer

# With date range
uv run python scripts/run_analysis.py \
    --data-dir /path/to/exports/ \
    --symbol NVDA \
    --profile full \
    --date-start 2025-02-03 --date-end 2025-02-07

# With stratified date sample
uv run python scripts/run_analysis.py \
    --data-dir /path/to/exports/ \
    --symbol NVDA \
    --profile full \
    --dates-file configs/sampling/nvda_stratified_50.txt

# Long run with checkpoint/resume (crash-safe)
nohup .venv/bin/python scripts/run_analysis.py \
    --data-dir /path/to/exports/ \
    --symbol NVDA \
    --profile full \
    --output-dir full_output \
    --checkpoint-dir full_checkpoint \
    --resume -v \
    > run.log 2>&1 &

# Discovery
uv run python scripts/run_analysis.py --list-analyzers
uv run python scripts/run_analysis.py --list-profiles
```

## Architecture

```
src/rawlobanalyzer/
├── io/              # Parquet loading, schema validation, day-by-day streaming
│   ├── loader.py        # ParquetDayLoader, DayData (column projection, lazy properties)
│   ├── session.py       # AnalysisSession (bounded-memory day iteration)
│   └── schema.py        # Schema constants, action/side enums, validation
├── core/            # Shared computation primitives
│   ├── resampler.py     # Vectorized multi-timescale resampling (sum/mean/last/first/ohlc/median)
│   ├── time_utils.py    # RTH masking, time regime classification, nanosecond conversions
│   ├── price_utils.py   # Nanodollar conversion, log returns, spread-in-ticks
│   ├── statistics.py    # WelfordAccumulator, distribution_summary, VaR/CVaR
│   └── constants.py     # NS_PER_SECOND, TRADING_MINUTES_PER_DAY, EPS
├── config/          # Configuration system
│   ├── analysis_config.py   # AnalysisConfig, StatisticalThresholds, FlowThresholds
│   ├── profile_loader.py    # YAML profile loading and application
│   └── timescale_config.py  # TimescaleConfig, TradingHours
├── analysis/        # Analyzer framework + domain implementations
│   ├── base.py          # BaseAnalyzer protocol (process_day/finalize contract)
│   ├── context.py       # DayContext (caches DayReturns, DaySpreads, DayFlow per day)
│   ├── orchestrator.py  # BatchOrchestrator (single-pass, shared computation)
│   ├── registry.py      # @register_analyzer decorator, name-based resolution
│   ├── health/          # DataQualityAnalyzer
│   ├── price/           # ReturnAnalyzer, VolatilityAnalyzer, JumpRiskAnalyzer, MicrostructureNoiseAnalyzer
│   ├── spread/          # SpreadAnalyzer, DepthAnalyzer, LiquidityAnalyzer
│   └── flow/            # OrderFlowAnalyzer, TradeAnalyzer, OrderLifecycleAnalyzer
└── reports/         # Report serialization (JSON + text summary)
```

### Key Design Patterns

1. **PyArrow-native**: Zero-copy Parquet reads, column projection, 3-5x memory savings vs Pandas
2. **Streaming accumulator**: Every analyzer implements `process_day(ctx)` / `finalize()` -- bounded memory (<4 GB) regardless of dataset size
3. **Shared engine pattern**: Expensive computations (`DayReturns`, `DaySpreads`, `DayFlow`) are computed once per day by dedicated engines and cached in `DayContext` for all analyzers
4. **Single-pass orchestration**: `BatchOrchestrator` loads each day once, fans out to all analyzers -- K analyzers on N days = N loads (not K*N)
5. **Configuration-driven**: YAML profiles, `AnalysisConfig` dataclass, no hardcoded thresholds
6. **Analyzer registry**: `@register_analyzer` decorator for auto-discovery; CLI resolves by name
7. **Checkpoint/resume**: Serializes full analyzer state after each day; crash-safe for multi-day runs
8. **DST-aware**: Dynamic UTC offset per day via `utc_offset_for_date()`; correct RTH boundaries year-round
9. **Stratified sampling**: `scripts/generate_sample.py` selects representative dates by month, volume quartile, and weekday

### Data Flow

```
Parquet files on disk
        |
        v
ParquetDayLoader (column projection, schema validation)
        |
        v
AnalysisSession.iter_days() (bounded-memory, one day at a time)
        |
        v
BatchOrchestrator (merges column requirements, computes shared engines)
        |
        +---> compute_day_returns() --> DayReturns (cached in DayContext)
        +---> compute_day_spreads() --> DaySpreads (cached in DayContext)
        +---> compute_day_flow()    --> DayFlow    (cached in DayContext)
        |
        v
DayContext { day, day_returns, day_spreads, day_flow }
        |
        +---> Analyzer1.process_day(ctx)
        +---> Analyzer2.process_day(ctx)
        +---> ...
        |
        v
Analyzer.finalize() --> Report --> JSON + TXT
```

## Implemented Analyzers (11)

### Health Domain

| Analyzer | What It Computes | Data Source |
|---|---|---|
| **DataQualityAnalyzer** | Row counts, max gap, median inter-event time, action/side distribution, book consistency, time regime distribution, crossed book detection | LOB + MBO |

### Price Domain

| Analyzer | What It Computes | Data Source |
|---|---|---|
| **ReturnAnalyzer** | Log return distribution at all timescales, intraday return curve, overnight decomposition, tail analysis (Hill index, VaR/CVaR) | LOB (mid_price) |
| **VolatilityAnalyzer** | Realized volatility, intraday vol curve, vol-of-vol, vol persistence (ACF), vol clustering (ARCH/Ljung-Box), weekday patterns | LOB (mid_price) |
| **JumpRiskAnalyzer** | Bipower variation, BNS jump test statistic, jump fraction, jump-size distribution, regime-conditional jump rates | LOB (mid_price) |
| **MicrostructureNoiseAnalyzer** | Noise variance signature plot, optimal sampling frequency, Roll's implied spread, noise-to-signal ratio, ARCH effects | LOB (mid_price) |

### Spread Domain

| Analyzer | What It Computes | Data Source |
|---|---|---|
| **SpreadAnalyzer** | Tick-level spread distribution, intraday spread curve, regime-conditional spreads, trade-conditional spreads, spread width classification | LOB (spread, best_bid, best_ask) |
| **DepthAnalyzer** | 10-level depth profile (mean sizes at each level), depth imbalance, concentration, post-trade depth recovery, depth stability | LOB (bid_prices, bid_sizes, ask_prices, ask_sizes) |
| **LiquidityAnalyzer** | Effective spread, volume-weighted spread, microprice deviation, Kyle's lambda, Amihud illiquidity | LOB + MBO |

### Flow Domain (MBO-powered)

| Analyzer | What It Computes | Data Source |
|---|---|---|
| **OrderFlowAnalyzer** | OFI at 6 timescales (1s/5s/10s/30s/1m/5m), OFI distribution (raw + normalized), **OFI component decomposition** (add/cancel/trade fractions by regime), OFI-return cross-correlation, **OFI-spread cross-correlation**, OFI autocorrelation, cumulative delta, aggressor ratio, flow intensity by regime, intraday flow curve, trade imbalance, weekday patterns | MBO + LOB |
| **TradeAnalyzer** | Trade size distribution, **directional trade size distribution** (buyer vs seller), trade-through rate (by regime), inter-trade time clustering, VWAP trajectory, large trade impact (bps), trade rate by regime, trade price level classification (at bid/ask/inside/outside) | MBO + LOB |
| **OrderLifecycleAnalyzer** | Order lifetime distribution (mean/median/percentiles, bid vs ask), fill rate (overall/bid/ask), cancel-to-add ratio, modify patterns, event transition matrix (Add->Cancel/Trade probabilities), duration-by-size correlation, regime-conditional lifecycle metrics, **partial fill patterns** (fill count per order, partial fill fraction) | MBO (order_id tracking) |

## Planned Analyzers (Next Phase)

These analyzers will extend the flow domain with deeper MBO-powered analysis:

| Priority | Analyzer | What It Will Compute | Why It Matters |
|---|---|---|---|
| P0 | **IcebergDetector** | Repeated fills at same price without displayed size decreasing; hidden liquidity estimation | Reveals undisplayed institutional activity that LOB cannot see |
| P0 | **QueuePositionEstimator** | FIFO queue depth at each price level; fill probability curves by queue position | Critical for limit order execution optimization |
| P1 | **EventClusteringAnalyzer** | Hawkes process self-excitation parameters; cascade detection (Trade -> Cancel bursts); inter-event intensity | Predicts short-term volatility spikes and market fragility |
| P1 | **InformationContentAnalyzer** | Per-event information score (Add at new level vs Cancel at BBO); surprise classification; information arrival clustering | Classifies which events actually move the market |
| P1 | **AggressorProfileAnalyzer** | Signed trade flow at finer granularity; aggressor size distribution; aggressor persistence (does a buyer keep buying?) | Deeper directional signal beyond aggregate aggressor ratio |
| P2 | **CancelDynamicsAnalyzer** | Cancel rate spike detection; cancel-before-trade patterns; quote stuffing detection; cancel speed distribution | Real-time warning signals for options traders |
| P2 | **LiquidityProviderAnalyzer** | Market maker behavior classification; quoting patterns; inventory management signals; spread-setting dynamics | Understand who provides liquidity and when they pull quotes |
| P2 | **CrossAssetFlowAnalyzer** | Multi-symbol OFI correlation; lead-lag detection between related stocks | Portfolio-level signal discovery |

## Analysis Profiles

| Profile | Analyzers | Est. Time (5 days NVDA) | Use Case |
|---|---|---|---|
| `quick` | DataQuality | ~1 min | Smoke test after export |
| `standard` | DataQuality, Return, Volatility, Spread, OrderFlow | ~3 min | Daily operational analysis |
| `volatility` | DataQuality, Return, Volatility, JumpRisk, Noise | ~3 min | Volatility deep-dive |
| `spread` | DataQuality, Spread, Depth, Liquidity | ~3 min | Liquidity analysis |
| `flow` | DataQuality, OrderFlow, Trade, OrderLifecycle | ~3 min | Order flow focus |
| `full` | All 11 analyzers | ~3 min | Complete characterization |

## Shared Computation Engines

To avoid redundant work when running multiple analyzers, expensive per-day computations are factored into shared engines:

| Engine | Output | Consumers | Cached In |
|---|---|---|---|
| `_return_engine.py` | `DayReturns` (log returns + multi-scale resampled returns) | ReturnAnalyzer, VolatilityAnalyzer, JumpRiskAnalyzer, MicrostructureNoiseAnalyzer, OrderFlowAnalyzer | `DayContext.day_returns` |
| `_spread_engine.py` | `DaySpreads` (tick spreads + multi-scale resampled spreads) | SpreadAnalyzer, DepthAnalyzer, LiquidityAnalyzer, OrderFlowAnalyzer | `DayContext.day_spreads` |
| `_flow_engine.py` | `DayFlow` (trades, OFI components, normalized OFI, directional_mask + multi-scale OFI) | OrderFlowAnalyzer, TradeAnalyzer | `DayContext.day_flow` |

## Relationship to Other Repos

```
MBO-LOB-reconstructor (Rust)          feature-extractor-MBO-LOB
         |                                      |
         | Parquet export                       | Feature engineering
         v                                      v
    MBO-LOB-analyzer  <-- validates -->  lob-dataset-analyzer
    (this repo)        feature extractor  (analyzes extracted features)
         |             correctness
         v
    Configuration recommendations for feature extractor + model selection
```

- **MBO-LOB-reconstructor** (upstream): Rust pipeline that reconstructs LOB from raw DBN MBO data and exports to Parquet
- **feature-extractor-MBO-LOB** (parallel): Applies sampling + feature engineering to the same raw data
- **lob-dataset-analyzer** (downstream): Analyzes the feature-extracted output; future cross-repo comparison planned

## Test Suite

391 tests covering all analyzers, shared engines, core utilities, config system, and I/O layer:

```bash
uv run pytest -v              # full suite
uv run pytest tests/test_analysis/test_flow/ -v   # flow domain only
```

Tests use synthetic Parquet data generated by `tests/conftest.py` with deterministic seeds.
