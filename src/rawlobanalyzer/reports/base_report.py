"""Base report protocol for all analyzer outputs.

Every analyzer produces a report that inherits from ``BaseReport``.
Reports provide structured serialization (``to_dict``, ``to_json``)
and human-readable summaries.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, ClassVar

from rawlobanalyzer import __version__


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf/-Inf floats with None for valid JSON.

    Python's ``json.dump`` with ``allow_nan=True`` emits non-standard
    tokens (``NaN``, ``Infinity``, ``-Infinity``) that violate RFC 8259
    and are rejected by strict parsers (JavaScript, Rust serde, Go).
    This function walks a nested dict/list structure and replaces any
    non-finite float with ``None`` (JSON ``null``).
    """
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


class BaseReport(ABC):
    """Abstract base for all analyzer reports.

    Subclasses must define:
        - ``SCHEMA_VERSION``: Semantic version for the report format.
        - ``to_dict()``: Serialize report data to a dict.
        - ``summary()``: Human-readable multi-line text summary.
    """

    SCHEMA_VERSION: ClassVar[str] = "1.0"

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """Serialize report to a JSON-compatible dictionary.

        Must include a ``_meta`` key with schema version and timestamp.
        """
        ...

    @abstractmethod
    def summary(self) -> str:
        """Human-readable multi-line text summary."""
        ...

    def _meta_dict(self) -> dict[str, Any]:
        """Standard metadata block for serialization."""
        return {
            "schema_version": self.SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "analyzer_version": __version__,
        }

    def to_json(self, path: Path | str, *, indent: int = 2) -> None:
        """Write report as formatted JSON.

        Produces valid RFC 8259 JSON. Any ``NaN`` / ``Inf`` / ``-Inf``
        values in the report dict are replaced with ``null``.

        Args:
            path: Output file path.
            indent: JSON indentation level.
        """
        data = _sanitize_for_json(self.to_dict())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=indent, default=str, allow_nan=False)
