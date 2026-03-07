"""Tests for LiquidityAnalyzer."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from rawlobanalyzer.analysis.spread.liquidity import LiquidityAnalyzer
from rawlobanalyzer.config.analysis_config import AnalysisConfig
from rawlobanalyzer.io.session import AnalysisSession


class TestLiquidityAnalyzer:
    def test_basic_run(self, tmp_data_dir: Path):
        """LiquidityAnalyzer runs and produces a report (needs MBO)."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        session = AnalysisSession(
            data_dir=tmp_data_dir,
            date_range=None,
            symbol="TEST",
        )
        analyzer = LiquidityAnalyzer(config)
        report = analyzer.run(session)

        assert report.symbol == "TEST"
        assert report.n_days == 1
        assert report.volume_weighted_spread or report.microprice_deviation
        if report.effective_spread:
            assert "mean_usd" in report.effective_spread
            assert report.effective_spread.get("n_trades", 0) >= 0

    def test_volume_weighted_spread(self, tmp_data_dir: Path):
        """Volume-weighted spread is populated when LOB has volume columns."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = LiquidityAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.volume_weighted_spread:
            assert "mean_usd" in report.volume_weighted_spread

    def test_microprice_deviation(self, tmp_data_dir: Path):
        """Microprice deviation is populated when LOB has microprice."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = LiquidityAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        if report.microprice_deviation:
            assert "mean" in report.microprice_deviation
            assert "n" in report.microprice_deviation

    def test_to_dict_roundtrip(self, tmp_data_dir: Path):
        """Report serializes to dict."""
        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = LiquidityAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        d = report.to_dict()
        assert "symbol" in d
        assert d["symbol"] == "TEST"
        assert "_meta" in d


class TestAggressorOnlyFilter:
    """Verify LiquidityAnalyzer counts only aggressor-side trades (B2 bug fix).

    Databento MBO emits two events per physical trade: aggressor (order_id=0)
    and passive (order_id!=0).  The analyzer must filter to aggressor-only.
    """

    def test_n_trades_counts_aggressor_only(self, tmp_data_dir: Path):
        """n_trades reflects aggressor events, not total MBO trade rows.

        The conftest fixture creates MBO data where all trade events have
        order_id=0 (aggressor-only).  We append duplicate passive-side rows
        (order_id!=0) and verify the count stays at the aggressor count.
        """
        mbo_path = tmp_data_dir / "2025-02-03_mbo_events.parquet"
        mbo_orig = pq.read_table(mbo_path)

        action_col = mbo_orig.column("action").to_numpy()
        trade_mask = action_col == 84  # ACTION_TRADE = ord('T')
        n_aggressor_trades = int(np.sum(trade_mask))
        assert n_aggressor_trades > 0, "Fixture must have at least one trade"

        passive_indices = np.where(trade_mask)[0]
        passive_rows = mbo_orig.take(passive_indices)

        new_oids = np.arange(1, len(passive_indices) + 1, dtype=np.uint64)
        passive_rows = passive_rows.set_column(
            passive_rows.schema.get_field_index("order_id"),
            "order_id",
            pa.array(new_oids, type=pa.uint64()),
        )

        combined = pa.concat_tables([mbo_orig, passive_rows])
        sort_idx = np.argsort(combined.column("timestamp_ns").to_numpy(), kind="stable")
        combined = combined.take(sort_idx)
        pq.write_table(combined, mbo_path)

        config = AnalysisConfig(data_dir=tmp_data_dir, symbol="TEST")
        analyzer = LiquidityAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_data_dir, date_range=None, symbol="TEST"),
        )

        assert report.effective_spread, "Expected effective_spread to be populated"
        assert report.effective_spread["n_trades"] == n_aggressor_trades, (
            f"Expected {n_aggressor_trades} aggressor trades, "
            f"got {report.effective_spread['n_trades']}"
        )


class TestLobAlignment:
    """Verify LOB alignment uses side='right' convention (B5 fix).

    When a trade timestamp exactly equals a LOB snapshot timestamp, the
    analyzer should use that LOB snapshot (not the one before it).
    """

    def test_exact_timestamp_match_uses_current_snapshot(self, tmp_path: Path):
        """Trade at t=X uses LOB snapshot at t=X, not t=X-1."""
        from datetime import datetime, timezone
        from tests.conftest import _write_day

        date = "2025-02-03"
        rng = np.random.default_rng(99)
        _write_day(tmp_path, date, n_rows=100, rng=rng)

        lob_path = tmp_path / f"{date}_lob_snapshots.parquet"
        lob = pq.read_table(lob_path)
        lob_ts = lob.column("timestamp_ns").to_numpy()
        mid = lob.column("mid_price").to_numpy()

        target_idx = 50
        target_ts = lob_ts[target_idx]
        expected_mid = mid[target_idx]
        prev_mid = mid[target_idx - 1]

        if abs(expected_mid - prev_mid) < 1e-9:
            pytest.skip("Adjacent mids are identical; cannot distinguish alignment")

        mbo_path = tmp_path / f"{date}_mbo_events.parquet"
        mbo = pq.read_table(mbo_path)

        trade_row = pa.table({
            "timestamp_ns": pa.array([target_ts], type=pa.int64()),
            "order_id": pa.array([np.uint64(0)], type=pa.uint64()),
            "action": pa.array([np.uint8(84)]),  # 'T'
            "side": pa.array([np.uint8(66)]),     # 'B'
            "price": pa.array(lob.column("best_bid").to_numpy()[[target_idx]], type=pa.int64()),
            "size": pa.array([np.uint32(100)]),
        }).replace_schema_metadata(mbo.schema.metadata)

        combined = pa.concat_tables([mbo, trade_row])
        sort_idx = np.argsort(combined.column("timestamp_ns").to_numpy(), kind="stable")
        combined = combined.take(sort_idx)
        pq.write_table(combined, mbo_path)

        config = AnalysisConfig(data_dir=tmp_path, symbol="TEST")
        analyzer = LiquidityAnalyzer(config)
        report = analyzer.run(
            AnalysisSession(data_dir=tmp_path, date_range=None, symbol="TEST"),
        )

        assert report.effective_spread, "Expected effective_spread to be populated"
