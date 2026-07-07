"""MBO-LOB-analyzer: Deep statistical analysis of raw MBO events and LOB snapshots.

Analyzes the full richness of Market-By-Order (MBO) data alongside reconstructed
Limit Order Book (LOB) snapshots to characterize stock microstructure, order flow
dynamics, and intraday trading patterns across configurable timescales -- all
before any feature engineering or model training.

Consumes Parquet output from MBO-LOB-reconstructor.
"""

__version__ = "0.3.0"
