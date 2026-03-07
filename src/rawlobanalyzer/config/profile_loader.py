"""YAML profile loading and validation.

Profiles define which analyzers to run, in what order, with what configuration
overrides. They are the primary user-facing interface for configuring analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from rawlobanalyzer.config.analysis_config import AnalysisConfig, StatisticalThresholds
from rawlobanalyzer.config.timescale_config import TimescaleConfig, TradingHours


@dataclass
class AnalyzerSpec:
    """Specification for a single analyzer within a profile."""

    name: str
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass
class PhaseSpec:
    """A named phase (group) of analyzers."""

    name: str
    description: str
    analyzers: list[AnalyzerSpec]


@dataclass
class ProfileSpec:
    """A complete analysis profile loaded from YAML."""

    name: str
    description: str
    phases: list[PhaseSpec]
    config_overrides: dict[str, Any] = field(default_factory=dict)

    @property
    def all_analyzer_names(self) -> list[str]:
        """Flat list of all analyzer names in execution order."""
        names: list[str] = []
        for phase in self.phases:
            for spec in phase.analyzers:
                names.append(spec.name)
        return names


class ProfileLoadError(Exception):
    """Raised when a profile YAML is invalid."""


def load_profile(path: Path) -> ProfileSpec:
    """Load and validate a YAML analysis profile.

    Args:
        path: Path to the YAML profile file.

    Returns:
        Validated ``ProfileSpec``.

    Raises:
        ProfileLoadError: If the file is missing, malformed, or invalid.
    """
    if not path.exists():
        raise ProfileLoadError(f"Profile file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ProfileLoadError(f"Profile must be a YAML mapping, got {type(raw).__name__}")

    name = raw.get("name", path.stem)
    description = raw.get("description", "")
    config_overrides = raw.get("config", {})

    phases_raw = raw.get("phases", [])
    if not isinstance(phases_raw, list):
        raise ProfileLoadError("'phases' must be a list")

    phases: list[PhaseSpec] = []
    for p in phases_raw:
        if not isinstance(p, dict):
            raise ProfileLoadError(f"Each phase must be a mapping, got {type(p).__name__}")

        phase_name = p.get("name", "unnamed")
        phase_desc = p.get("description", "")
        analyzers_raw = p.get("analyzers", [])
        if not isinstance(analyzers_raw, list):
            raise ProfileLoadError(f"Phase {phase_name!r}: 'analyzers' must be a list")

        specs: list[AnalyzerSpec] = []
        for a in analyzers_raw:
            if isinstance(a, str):
                specs.append(AnalyzerSpec(name=a))
            elif isinstance(a, dict):
                a_name = a.get("name")
                if not a_name:
                    raise ProfileLoadError(f"Analyzer spec in phase {phase_name!r} missing 'name'")
                specs.append(AnalyzerSpec(name=a_name, overrides=a.get("config", {})))
            else:
                raise ProfileLoadError(
                    f"Analyzer spec must be a string or mapping, got {type(a).__name__}"
                )

        phases.append(PhaseSpec(name=phase_name, description=phase_desc, analyzers=specs))

    return ProfileSpec(
        name=name,
        description=description,
        phases=phases,
        config_overrides=config_overrides,
    )


def apply_profile_config(
    base: AnalysisConfig,
    profile: ProfileSpec,
) -> AnalysisConfig:
    """Apply profile-level config overrides to a base config.

    Args:
        base: Base ``AnalysisConfig`` (from CLI args).
        profile: Loaded profile with optional ``config_overrides``.

    Returns:
        New ``AnalysisConfig`` with overrides applied.
    """
    overrides = profile.config_overrides
    if not overrides:
        return base

    timescales = base.timescales
    if "timescales" in overrides:
        timescales = [TimescaleConfig.from_label(l) for l in overrides["timescales"]]

    trading_hours = base.trading_hours
    if "trading_hours" in overrides:
        trading_hours = TradingHours.from_label(overrides["trading_hours"])

    max_rows = base.max_rows_per_day
    if "max_rows_per_day" in overrides:
        max_rows = overrides["max_rows_per_day"]

    return AnalysisConfig(
        data_dir=base.data_dir,
        symbol=base.symbol,
        date_range=base.date_range,
        dates_list=base.dates_list,
        timescales=timescales,
        trading_hours=trading_hours,
        thresholds=base.thresholds,
        max_rows_per_day=max_rows,
        output_dir=base.output_dir,
        checkpoint_dir=base.checkpoint_dir,
        resume=base.resume,
        save_json=base.save_json,
        save_summary=base.save_summary,
        verbose=base.verbose,
    )
