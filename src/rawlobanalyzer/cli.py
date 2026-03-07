"""CLI entry point for the ``mbo-lob-analyze`` console script."""

from __future__ import annotations

import sys


def main() -> None:
    """Entry point for ``mbo-lob-analyze`` command."""
    from scripts.run_analysis import main as run_main
    sys.exit(run_main())
