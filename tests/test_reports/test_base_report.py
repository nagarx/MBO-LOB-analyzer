"""Tests for BaseReport: NaN sanitization, metadata, JSON roundtrip."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import pytest

from rawlobanalyzer import __version__
from rawlobanalyzer.reports.base_report import BaseReport, _sanitize_for_json


# ------------------------------------------------------------------
# Concrete stub report for testing
# ------------------------------------------------------------------


@dataclass
class _StubReport(BaseReport):
    SCHEMA_VERSION: ClassVar[str] = "test-1.0"

    value: float = 0.0
    nested: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"value": self.value, "_meta": self._meta_dict()}
        if self.nested is not None:
            d["nested"] = self.nested
        return d

    def summary(self) -> str:
        return f"StubReport(value={self.value})"


# ------------------------------------------------------------------
# _sanitize_for_json
# ------------------------------------------------------------------


class TestSanitizeForJson:
    def test_nan_replaced_with_none(self):
        assert _sanitize_for_json(float("nan")) is None

    def test_inf_replaced_with_none(self):
        assert _sanitize_for_json(float("inf")) is None

    def test_neg_inf_replaced_with_none(self):
        assert _sanitize_for_json(float("-inf")) is None

    def test_finite_float_unchanged(self):
        assert _sanitize_for_json(3.14) == 3.14

    def test_zero_unchanged(self):
        assert _sanitize_for_json(0.0) == 0.0

    def test_int_unchanged(self):
        assert _sanitize_for_json(42) == 42

    def test_string_unchanged(self):
        assert _sanitize_for_json("hello") == "hello"

    def test_none_unchanged(self):
        assert _sanitize_for_json(None) is None

    def test_nested_dict(self):
        data = {"a": float("nan"), "b": {"c": float("inf"), "d": 1.0}}
        result = _sanitize_for_json(data)
        assert result == {"a": None, "b": {"c": None, "d": 1.0}}

    def test_nested_list(self):
        data = [float("nan"), [float("-inf"), 2.0], "ok"]
        result = _sanitize_for_json(data)
        assert result == [None, [None, 2.0], "ok"]

    def test_mixed_structure(self):
        data = {"items": [{"val": float("nan")}, {"val": 5.0}]}
        result = _sanitize_for_json(data)
        assert result["items"][0]["val"] is None
        assert result["items"][1]["val"] == 5.0


# ------------------------------------------------------------------
# Metadata
# ------------------------------------------------------------------


class TestMetaDict:
    def test_contains_schema_version(self):
        r = _StubReport()
        meta = r._meta_dict()
        assert meta["schema_version"] == "test-1.0"

    def test_contains_created_at(self):
        r = _StubReport()
        meta = r._meta_dict()
        assert "created_at" in meta
        assert "T" in meta["created_at"]

    def test_contains_analyzer_version(self):
        r = _StubReport()
        meta = r._meta_dict()
        assert meta["analyzer_version"] == __version__


# ------------------------------------------------------------------
# JSON roundtrip
# ------------------------------------------------------------------


class TestJsonRoundtrip:
    def test_basic_roundtrip(self, tmp_path: Path):
        r = _StubReport(value=42.0)
        out = tmp_path / "test.json"
        r.to_json(out)
        data = json.loads(out.read_text())
        assert data["value"] == 42.0
        assert "_meta" in data

    def test_nan_becomes_null_in_json(self, tmp_path: Path):
        r = _StubReport(value=float("nan"))
        out = tmp_path / "test_nan.json"
        r.to_json(out)
        data = json.loads(out.read_text())
        assert data["value"] is None

    def test_inf_becomes_null_in_json(self, tmp_path: Path):
        r = _StubReport(
            value=1.0,
            nested={"x": float("inf"), "y": float("-inf"), "z": 0.5},
        )
        out = tmp_path / "test_inf.json"
        r.to_json(out)
        data = json.loads(out.read_text())
        assert data["nested"]["x"] is None
        assert data["nested"]["y"] is None
        assert data["nested"]["z"] == 0.5

    def test_rfc8259_compliance(self, tmp_path: Path):
        """Verify the JSON contains no NaN/Infinity tokens."""
        r = _StubReport(
            value=float("nan"),
            nested={"a": float("inf"), "b": [float("-inf"), 1.0]},
        )
        out = tmp_path / "test_rfc.json"
        r.to_json(out)
        text = out.read_text()
        assert "NaN" not in text
        assert "Infinity" not in text
        assert "-Infinity" not in text
        json.loads(text)

    def test_creates_parent_dirs(self, tmp_path: Path):
        r = _StubReport(value=1.0)
        out = tmp_path / "sub" / "dir" / "test.json"
        r.to_json(out)
        assert out.exists()

    def test_summary_method(self):
        r = _StubReport(value=3.14)
        assert "3.14" in r.summary()
