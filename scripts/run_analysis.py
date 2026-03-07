#!/usr/bin/env python3
"""CLI entry point for MBO-LOB-analyzer.

Usage:
    # Run a profile
    python scripts/run_analysis.py -d /path/to/exports/ -s NVDA --profile standard

    # Run ALL analyzers (auto-discovers every registered analyzer)
    python scripts/run_analysis.py -d /path/to/exports/ -s NVDA --profile full

    # Run specific analyzers (comma-separated)
    python scripts/run_analysis.py -d /path/to/exports/ -s NVDA \\
        --analyzer SpreadAnalyzer,DepthAnalyzer,LiquidityAnalyzer

    # Run a single analyzer
    python scripts/run_analysis.py -d /path/to/exports/ -s NVDA -a DataQualityAnalyzer

    # With date range filter
    python scripts/run_analysis.py -d /path/to/exports/ -s NVDA \\
        --date-start 2025-02-03 --date-end 2025-02-07 --profile full

    # Discovery commands
    python scripts/run_analysis.py --list-analyzers     # show all registered analyzers
    python scripts/run_analysis.py --list-profiles       # show all available profiles
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import yaml
from rich.console import Console

console = Console(stderr=True)

_PROFILES_DIR = Path(__file__).parent.parent / "configs" / "profiles"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="MBO-LOB-analyzer: deep statistical analysis of MBO events and LOB snapshots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", "-d", type=Path, default=None,
        help="Directory containing {date}_lob_snapshots.parquet files",
    )
    parser.add_argument(
        "--symbol", "-s", type=str, default="UNKNOWN",
        help="Ticker symbol (default: UNKNOWN)",
    )
    parser.add_argument(
        "--profile", "-p", type=str, default=None,
        help="Analysis profile (quick, standard, volatility, spread, full)",
    )
    parser.add_argument(
        "--analyzer", "-a", type=str, default=None,
        help="Analyzer(s) to run, comma-separated (e.g. SpreadAnalyzer,DepthAnalyzer)",
    )
    parser.add_argument(
        "--date-start", type=str, default=None,
        help="Start date filter (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--date-end", type=str, default=None,
        help="End date filter (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--dates-file", type=Path, default=None,
        help="Text file with one date per line (YYYY-MM-DD) to restrict analysis to",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path, default=None,
        help="Output directory for reports (default: data_dir/analysis_results/)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed progress",
    )
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=None,
        help="Directory for checkpoint files (enables crash recovery)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--list-analyzers", action="store_true",
        help="List all available analyzers and exit",
    )
    parser.add_argument(
        "--list-profiles", action="store_true",
        help="List all available profiles and exit",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    from rawlobanalyzer.analysis.registry import get_analyzer, list_analyzers

    if args.list_analyzers:
        console.print("[bold]Available analyzers:[/bold]")
        for name in list_analyzers():
            cls = get_analyzer(name)
            console.print(f"  {name:35s}  {cls.description}")
        return 0

    if args.list_profiles:
        return _do_list_profiles()

    if args.data_dir is None:
        console.print("[red]Error: --data-dir is required[/red]")
        return 1

    if not args.data_dir.is_dir():
        console.print(f"[red]Error: data directory not found: {args.data_dir}[/red]")
        return 1

    if args.profile is None and args.analyzer is None:
        console.print("[red]Error: specify --profile or --analyzer[/red]")
        console.print()
        _do_list_profiles()
        console.print()
        console.print("[bold]Available analyzers:[/bold]")
        for name in list_analyzers():
            cls = get_analyzer(name)
            console.print(f"  {name:35s}  {cls.description}")
        return 1

    from rawlobanalyzer.config.analysis_config import AnalysisConfig
    from rawlobanalyzer.config.profile_loader import (
        apply_profile_config,
        load_profile,
    )

    date_range = None
    if args.date_start and args.date_end:
        date_range = (args.date_start, args.date_end)

    dates_list = None
    if args.dates_file:
        if not args.dates_file.is_file():
            console.print(f"[red]Error: dates file not found: {args.dates_file}[/red]")
            return 1
        dates_list = [
            line.strip() for line in args.dates_file.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        console.print(f"[bold]Dates file:[/bold] {args.dates_file} ({len(dates_list)} dates)")

    output_dir = args.output_dir or args.data_dir / "analysis_results"

    config = AnalysisConfig(
        data_dir=args.data_dir,
        symbol=args.symbol,
        date_range=date_range,
        dates_list=dates_list,
        output_dir=output_dir,
        checkpoint_dir=args.checkpoint_dir,
        resume=args.resume,
        verbose=args.verbose,
    )

    analyzer_names: list[str] = []

    if args.analyzer:
        analyzer_names = [n.strip() for n in args.analyzer.split(",") if n.strip()]
    elif args.profile:
        profile_path = _resolve_profile_path(args.profile)
        if profile_path is None:
            console.print(f"[red]Error: profile not found: {args.profile!r}[/red]")
            console.print("[dim]Available profiles:[/dim]")
            _do_list_profiles()
            return 1
        profile = load_profile(profile_path)
        config = apply_profile_config(config, profile)
        analyzer_names = profile.all_analyzer_names
        console.print(
            f"[bold]Profile:[/bold] {profile.name} -- {profile.description}"
        )

    console.print(f"[bold]Data directory:[/bold] {config.data_dir}")
    console.print(f"[bold]Symbol:[/bold] {config.symbol}")
    if date_range:
        console.print(f"[bold]Date range:[/bold] {date_range[0]} to {date_range[1]}")
    console.print(f"[bold]Analyzers ({len(analyzer_names)}):[/bold] {', '.join(analyzer_names)}")
    console.print()

    from rawlobanalyzer.analysis.orchestrator import BatchOrchestrator

    analyzers = []
    for name in analyzer_names:
        cls = get_analyzer(name)
        analyzers.append(cls(config))

    orchestrator = BatchOrchestrator(config, analyzers)
    t0 = time.monotonic()
    results = orchestrator.run()
    elapsed = time.monotonic() - t0

    output_dir.mkdir(parents=True, exist_ok=True)

    for i, (name, report) in enumerate(results.items()):
        console.print(f"\n{report.summary()}")

        if config.save_json:
            json_path = output_dir / f"{i + 1:02d}_{name}.json"
            report.to_json(json_path)
            console.print(f"\n  [dim]Saved: {json_path}[/dim]")

        if config.save_summary:
            txt_path = output_dir / f"{i + 1:02d}_{name}.txt"
            txt_path.write_text(report.summary())

    console.print(f"\n[bold green]Analysis complete: {elapsed:.1f}s[/bold green]")
    console.print(f"[dim]Results saved to: {output_dir}[/dim]")

    return 0


def _resolve_profile_path(name: str) -> Path | None:
    """Resolve a profile name to a YAML file path."""
    if Path(name).exists():
        return Path(name)

    candidates = [
        _PROFILES_DIR / f"{name}.yaml",
        Path("configs") / "profiles" / f"{name}.yaml",
    ]

    for c in candidates:
        if c.exists():
            return c

    return None


def _discover_profiles() -> list[tuple[str, str, int]]:
    """Discover all YAML profiles and return (name, description, analyzer_count)."""
    profiles: list[tuple[str, str, int]] = []
    for candidate_dir in [_PROFILES_DIR, Path("configs") / "profiles"]:
        if not candidate_dir.is_dir():
            continue
        for yaml_path in sorted(candidate_dir.glob("*.yaml")):
            try:
                with open(yaml_path) as f:
                    raw = yaml.safe_load(f)
                if not isinstance(raw, dict):
                    continue
                name = raw.get("name", yaml_path.stem)
                desc = raw.get("description", "")
                n_analyzers = sum(
                    len(p.get("analyzers", []))
                    for p in raw.get("phases", [])
                    if isinstance(p, dict)
                )
                profiles.append((name, desc, n_analyzers))
            except Exception:
                continue
    seen: set[str] = set()
    deduped: list[tuple[str, str, int]] = []
    for name, desc, count in profiles:
        if name not in seen:
            seen.add(name)
            deduped.append((name, desc, count))
    return deduped


def _do_list_profiles() -> int:
    """Print all available profiles."""
    profiles = _discover_profiles()
    if not profiles:
        console.print("[yellow]No profiles found in configs/profiles/[/yellow]")
        return 0

    console.print("[bold]Available profiles:[/bold]")
    for name, desc, count in profiles:
        console.print(f"  {name:15s}  ({count} analyzers)  {desc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
