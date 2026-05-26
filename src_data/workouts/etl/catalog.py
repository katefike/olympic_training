"""Exercise catalog: aliases, reportability, and progress_metric."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LBS_PER_KG = 2.2046226218


@dataclass(frozen=True)
class ExerciseCatalogEntry:
    canonical_name: str
    aliases: tuple[str, ...]
    include_in_reports: bool
    progress_metric: str | None


@dataclass(frozen=True)
class ExerciseCatalog:
    entries: tuple[ExerciseCatalogEntry, ...]
    alias_to_canonical: dict[str, str]
    by_canonical: dict[str, ExerciseCatalogEntry]

    def canonicalize(self, raw_name: str) -> str:
        clean = raw_name.strip().lower()
        return self.alias_to_canonical.get(clean, clean)

    def entry_for(self, canonical_name: str) -> ExerciseCatalogEntry | None:
        return self.by_canonical.get(canonical_name)


def load_catalog(path: Path) -> ExerciseCatalog:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_exercises = payload.get("exercises", [])
    if not isinstance(raw_exercises, list):
        raise ValueError(f"Invalid catalog: exercises must be a list in {path}")

    entries: list[ExerciseCatalogEntry] = []
    alias_to_canonical: dict[str, str] = {}
    by_canonical: dict[str, ExerciseCatalogEntry] = {}

    for item in raw_exercises:
        if not isinstance(item, dict):
            continue
        canonical = str(item.get("canonical_name", "")).strip().lower()
        if not canonical:
            continue
        metric = item.get("progress_metric")
        if metric is not None:
            metric = str(metric).strip().lower()
            if metric not in ("duration", "reps"):
                raise ValueError(f"Invalid progress_metric for {canonical}: {metric}")
        aliases_raw = item.get("aliases", [])
        aliases: list[str] = []
        if isinstance(aliases_raw, list):
            aliases = [str(a).strip().lower() for a in aliases_raw if str(a).strip()]
        entry = ExerciseCatalogEntry(
            canonical_name=canonical,
            aliases=tuple(aliases),
            include_in_reports=bool(item.get("include_in_reports", False)),
            progress_metric=metric,
        )
        entries.append(entry)
        by_canonical[canonical] = entry
        alias_to_canonical[canonical] = canonical
        for alias in aliases:
            if alias in alias_to_canonical and alias_to_canonical[alias] != canonical:
                raise ValueError(f"Alias {alias!r} maps to both {alias_to_canonical[alias]!r} and {canonical!r}")
            alias_to_canonical[alias] = canonical

    return ExerciseCatalog(
        entries=tuple(entries),
        alias_to_canonical=alias_to_canonical,
        by_canonical=by_canonical,
    )


def normalize_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    cleaned = str(unit).strip().lower()
    supported = {"lb", "lbs", "pound", "pounds", "kg", "kgs", "kilogram", "kilograms"}
    if cleaned not in supported:
        return None
    if cleaned.startswith("kg"):
        return "kg"
    return "lbs"


def to_lbs(weight: Any, unit: str | None) -> float | None:
    if weight is None:
        return None
    try:
        numeric = float(weight)
    except (TypeError, ValueError):
        return None
    if unit == "kg":
        return round(numeric * LBS_PER_KG, 2)
    return round(numeric, 2)


def session_scores(details: dict[str, Any]) -> dict[str, float]:
    """Compute duration/reps product scores for one session-exercise JSON object."""
    unit = normalize_unit(details.get("weight_unit"))
    reps = details.get("reps_each") or details.get("reps_total")
    try:
        reps_val = int(reps) if reps is not None else 0
    except (TypeError, ValueError):
        reps_val = 0

    score_duration = 0.0
    score_reps = 0.0
    has_duration = False
    has_reps = bool(reps_val)

    for entry in details.get("sets") or []:
        if not isinstance(entry, dict):
            continue
        weight_lbs = to_lbs(entry.get("weight"), unit) or 0.0
        duration = entry.get("duration_sec")
        if duration is not None:
            has_duration = True
            try:
                score_duration += weight_lbs * int(duration)
            except (TypeError, ValueError):
                pass
        if reps_val and weight_lbs:
            score_reps += weight_lbs * reps_val

    return {
        "progress_score_duration": score_duration,
        "progress_score_reps": score_reps,
        "has_duration": float(has_duration),
        "has_reps": float(has_reps),
    }
