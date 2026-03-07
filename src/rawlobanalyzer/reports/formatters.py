"""Text formatting utilities for report summaries.

Provides consistent table and section formatting across all analyzers.
"""

from __future__ import annotations

from typing import Any


def format_section(title: str, width: int = 70) -> str:
    """Format a section header."""
    return f"\n{'=' * width}\n  {title}\n{'=' * width}"


def format_subsection(title: str, width: int = 60) -> str:
    """Format a subsection header."""
    return f"\n  {'-' * width}\n  {title}\n  {'-' * width}"


def format_kv(items: dict[str, Any], indent: int = 4) -> str:
    """Format key-value pairs as aligned text.

    Args:
        items: Dict of label -> value pairs.
        indent: Number of leading spaces.
    """
    if not items:
        return ""
    max_key_len = max(len(str(k)) for k in items)
    lines: list[str] = []
    prefix = " " * indent
    for k, v in items.items():
        if isinstance(v, float):
            v_str = f"{v:,.6f}" if abs(v) < 1000 else f"{v:,.2f}"
        elif isinstance(v, int):
            v_str = f"{v:,}"
        else:
            v_str = str(v)
        lines.append(f"{prefix}{str(k):<{max_key_len + 2}}{v_str}")
    return "\n".join(lines)


def format_table(
    headers: list[str],
    rows: list[list[Any]],
    *,
    col_widths: list[int] | None = None,
    indent: int = 4,
) -> str:
    """Format a simple text table.

    Args:
        headers: Column header labels.
        rows: List of rows, each a list of cell values.
        col_widths: Optional explicit column widths.
        indent: Number of leading spaces.

    Returns:
        Formatted table string.
    """
    n_cols = len(headers)
    if col_widths is None:
        col_widths = [max(len(str(h)), 10) for h in headers]
        for row in rows:
            for i, cell in enumerate(row[:n_cols]):
                cell_str = _format_cell(cell)
                col_widths[i] = max(col_widths[i], len(cell_str))

    prefix = " " * indent
    header_line = prefix + "  ".join(
        f"{str(h):>{w}}" for h, w in zip(headers, col_widths)
    )
    separator = prefix + "  ".join("-" * w for w in col_widths)

    lines = [header_line, separator]
    for row in rows:
        cells = [_format_cell(c) for c in row[:n_cols]]
        line = prefix + "  ".join(
            f"{c:>{w}}" for c, w in zip(cells, col_widths)
        )
        lines.append(line)

    return "\n".join(lines)


def _format_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if abs(value) < 0.001 and value != 0.0:
            return f"{value:.6e}"
        return f"{value:,.4f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)
