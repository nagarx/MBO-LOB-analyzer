"""Tests for report formatting utilities: NaN, Inf, empty data, short rows."""

from __future__ import annotations

import math

import pytest

from rawlobanalyzer.reports.formatters import (
    _format_cell,
    format_kv,
    format_section,
    format_subsection,
    format_table,
)


# ------------------------------------------------------------------
# format_section / format_subsection
# ------------------------------------------------------------------


class TestSectionHeaders:
    def test_section_contains_title(self):
        result = format_section("DEPTH ANALYSIS")
        assert "DEPTH ANALYSIS" in result

    def test_section_has_separator_lines(self):
        result = format_section("TITLE", width=20)
        assert "=" * 20 in result

    def test_subsection_contains_title(self):
        result = format_subsection("Sub Title")
        assert "Sub Title" in result

    def test_subsection_has_dashes(self):
        result = format_subsection("Sub", width=30)
        assert "-" * 30 in result


# ------------------------------------------------------------------
# format_kv
# ------------------------------------------------------------------


class TestFormatKV:
    def test_empty_dict(self):
        assert format_kv({}) == ""

    def test_int_formatting(self):
        result = format_kv({"Count": 1_000_000})
        assert "1,000,000" in result

    def test_float_small_formatting(self):
        result = format_kv({"Price": 0.015})
        assert "0.015" in result

    def test_float_large_formatting(self):
        result = format_kv({"Volume": 50_000.0})
        assert "50,000.00" in result

    def test_nan_formatting(self):
        result = format_kv({"Val": float("nan")})
        assert "nan" in result.lower()

    def test_inf_formatting(self):
        result = format_kv({"Val": float("inf")})
        assert "inf" in result.lower()

    def test_string_value(self):
        result = format_kv({"Symbol": "NVDA"})
        assert "NVDA" in result


# ------------------------------------------------------------------
# format_table
# ------------------------------------------------------------------


class TestFormatTable:
    def test_basic_table(self):
        headers = ["A", "B"]
        rows = [[1, 2.0], [3, 4.0]]
        result = format_table(headers, rows)
        assert "A" in result
        assert "B" in result

    def test_empty_rows(self):
        result = format_table(["A", "B"], [])
        assert "A" in result
        assert "B" in result

    def test_short_rows_no_crash(self):
        """Rows shorter than headers should not crash."""
        headers = ["A", "B", "C"]
        rows = [[1], [2, 3]]
        result = format_table(headers, rows)
        assert "A" in result

    def test_none_cell(self):
        headers = ["X"]
        rows = [[None]]
        result = format_table(headers, rows)
        assert "-" in result

    def test_nan_cell(self):
        headers = ["X"]
        rows = [[float("nan")]]
        result = format_table(headers, rows)
        assert "nan" in result.lower()

    def test_inf_cell(self):
        headers = ["X"]
        rows = [[float("inf")]]
        result = format_table(headers, rows)
        assert "inf" in result.lower()

    def test_custom_col_widths(self):
        headers = ["Name", "Value"]
        rows = [["a", 1]]
        result = format_table(headers, rows, col_widths=[20, 10])
        assert len(result) > 0

    def test_custom_indent(self):
        headers = ["A"]
        rows = [[1]]
        result = format_table(headers, rows, indent=8)
        lines = result.split("\n")
        assert lines[0].startswith(" " * 8)


# ------------------------------------------------------------------
# _format_cell
# ------------------------------------------------------------------


class TestFormatCell:
    def test_none(self):
        assert _format_cell(None) == "-"

    def test_int(self):
        assert _format_cell(1000) == "1,000"

    def test_float_normal(self):
        result = _format_cell(3.14159)
        assert "3.14" in result

    def test_float_very_small(self):
        result = _format_cell(0.0001)
        assert "e" in result.lower()

    def test_float_zero(self):
        result = _format_cell(0.0)
        assert "0" in result

    def test_string(self):
        assert _format_cell("hello") == "hello"
