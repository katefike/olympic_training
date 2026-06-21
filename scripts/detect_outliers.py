#!/usr/bin/env python3
"""Flag statistical outliers for reportable exercises in training_set JSON."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
ETL_DIR = ROOT / "src_data" / "workouts" / "etl"
sys.path.insert(0, str(ETL_DIR))

from catalog import ExerciseCatalog, load_catalog, normalize_unit, to_lbs  # noqa: E402


@dataclass
class SessionRecord:
    canonical: str
    source_file: str
    heic_id: str | None
    session_index: int
    session_date: str | None
    gym: str | None
    raw_name: str
    weight_unit: str | None
    max_weight_lbs: float
    max_duration_sec: int | None
    progress_score: float
    progress_metric: str
    sets: list[dict[str, Any]]
    reps_each: int | None
    reps_total: int | None


def iqr_outliers(values: list[float]) -> tuple[float, float]:
    if len(values) < 4:
        lo, hi = min(values), max(values)
        median = statistics.median(values)
        return median * 0.5, median * 2.0
    qs = statistics.quantiles(values, n=4)
    q1, q3 = qs[0], qs[2]
    iqr = q3 - q1
    if iqr == 0:
        median = statistics.median(values)
        return median * 0.5, median * 2.0
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def collect_records(
    training_set_dir: Path,
    catalog: ExerciseCatalog,
) -> dict[str, list[SessionRecord]]:
    reportable = {
        name: entry
        for name, entry in catalog.by_canonical.items()
        if entry.include_in_reports
    }
    by_exercise: dict[str, list[SessionRecord]] = {name: [] for name in reportable}

    for file_path in sorted(training_set_dir.glob("*.json")):
        try:
            sessions = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(sessions, list):
            continue

        for session_index, session in enumerate(sessions):
            if not isinstance(session, dict):
                continue
            exercises = session.get("exercises", {})
            if not isinstance(exercises, dict):
                continue

            for raw_name, details in exercises.items():
                if not isinstance(details, dict):
                    continue
                canonical = catalog.canonicalize(raw_name)
                entry = reportable.get(canonical)
                if not entry:
                    continue

                unit = normalize_unit(details.get("weight_unit"))
                sets_raw = details.get("sets") or []
                sets_norm: list[dict[str, Any]] = []
                max_weight = 0.0
                max_duration: int | None = None
                score_duration = 0.0
                score_reps = 0.0

                reps_each = details.get("reps_each")
                reps_total = details.get("reps_total")
                try:
                    reps_val = int(reps_each or reps_total or 0)
                except (TypeError, ValueError):
                    reps_val = 0

                for set_entry in sets_raw:
                    if not isinstance(set_entry, dict):
                        continue
                    weight_lbs = to_lbs(set_entry.get("weight"), unit) or 0.0
                    duration = set_entry.get("duration_sec")
                    duration_int: int | None = None
                    if duration is not None:
                        try:
                            duration_int = int(duration)
                        except (TypeError, ValueError):
                            duration_int = None
                    sets_norm.append(
                        {
                            "weight": set_entry.get("weight"),
                            "weight_lbs": weight_lbs,
                            "duration_sec": duration_int,
                        }
                    )
                    max_weight = max(max_weight, weight_lbs)
                    if duration_int is not None:
                        max_duration = max(max_duration or 0, duration_int)
                        score_duration += weight_lbs * duration_int
                    if reps_val and weight_lbs:
                        score_reps += weight_lbs * reps_val

                metric = entry.progress_metric or "duration"
                progress_score = score_duration if metric == "duration" else score_reps

                by_exercise[canonical].append(
                    SessionRecord(
                        canonical=canonical,
                        source_file=file_path.name,
                        heic_id=session.get("heic_id"),
                        session_index=session_index,
                        session_date=session.get("date"),
                        gym=session.get("gym"),
                        raw_name=raw_name.strip().lower(),
                        weight_unit=unit,
                        max_weight_lbs=max_weight,
                        max_duration_sec=max_duration,
                        progress_score=progress_score,
                        progress_metric=metric,
                        sets=sets_norm,
                        reps_each=int(reps_each) if reps_each is not None else None,
                        reps_total=int(reps_total) if reps_total is not None else None,
                    )
                )

    return by_exercise


def format_sets(record: SessionRecord) -> str:
    parts = []
    for s in record.sets:
        w = s["weight"]
        unit = record.weight_unit or "?"
        if s["duration_sec"] is not None:
            parts.append(f"{w}{unit} x {s['duration_sec']}s")
        else:
            parts.append(f"{w}{unit}")
    reps = record.reps_each or record.reps_total
    if reps:
        return f"reps={reps}; " + ", ".join(parts)
    return ", ".join(parts)


def print_exercise_report(records: list[SessionRecord], min_sessions: int) -> None:
    if not records:
        print("  (no sessions)")
        return

    canonical = records[0].canonical
    metric = records[0].progress_metric
    weights = [r.max_weight_lbs for r in records]
    scores = [r.progress_score for r in records]
    durations = [r.max_duration_sec for r in records if r.max_duration_sec is not None]

    w_lo, w_hi = iqr_outliers(weights)
    s_lo, s_hi = iqr_outliers(scores)
    d_lo, d_hi = (iqr_outliers([float(d) for d in durations]) if durations else (0.0, 0.0))

    print(
        f"  sessions={len(records)}  metric={metric}  "
        f"weight lbs: median={statistics.median(weights):.1f} range=[{min(weights):.1f}, {max(weights):.1f}]  "
        f"IQR fence=[{w_lo:.1f}, {w_hi:.1f}]"
    )
    if metric == "duration":
        print(
            f"  volume (weight×sec): median={statistics.median(scores):.0f} "
            f"range=[{min(scores):.0f}, {max(scores):.0f}]  IQR fence=[{s_lo:.0f}, {s_hi:.0f}]"
        )
        if durations:
            print(
                f"  max set duration: median={statistics.median(durations):.0f}s "
                f"range=[{min(durations)}, {max(durations)}]  IQR fence=[{d_lo:.0f}, {d_hi:.0f}]"
            )
    else:
        print(
            f"  progress score (weight×reps): median={statistics.median(scores):.0f} "
            f"range=[{min(scores):.0f}, {max(scores):.0f}]  IQR fence=[{s_lo:.0f}, {s_hi:.0f}]"
        )

    flagged: list[tuple[str, SessionRecord, list[str]]] = []
    for record in records:
        reasons: list[str] = []
        if record.max_weight_lbs < w_lo or record.max_weight_lbs > w_hi:
            reasons.append(f"max weight {record.max_weight_lbs:.1f} lbs outside [{w_lo:.1f}, {w_hi:.1f}]")
        if record.progress_score < s_lo or record.progress_score > s_hi:
            reasons.append(f"progress score {record.progress_score:.0f} outside [{s_lo:.0f}, {s_hi:.0f}]")
        if (
            record.max_duration_sec is not None
            and metric == "duration"
            and durations
            and (record.max_duration_sec < d_lo or record.max_duration_sec > d_hi)
        ):
            reasons.append(
                f"max duration {record.max_duration_sec}s outside [{d_lo:.0f}, {d_hi:.0f}]"
            )
        if reasons:
            flagged.append(("OUTLIER", record, reasons))

    if not flagged:
        print("  No outliers detected.")
        return

    print(f"  Outliers ({len(flagged)}):")
    for _, record, reasons in sorted(flagged, key=lambda x: (-x[1].max_weight_lbs, x[1].session_date or "")):
        gym = record.gym or "(gym unknown)"
        path = f"src_data/workouts/training_set/{record.source_file}"
        print(
            f"    - {record.session_date or 'no date'} | gym={gym} | "
            f"{path} session[{record.session_index}] heic={record.heic_id} raw={record.raw_name!r}"
        )
        print(f"      {'; '.join(reasons)}")
        print(f"      sets: {format_sets(record)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect outliers in reportable exercise data.")
    parser.add_argument(
        "--training-set-dir",
        type=Path,
        default=ROOT / "src_data" / "workouts" / "training_set",
    )
    parser.add_argument(
        "--catalog-file",
        type=Path,
        default=ETL_DIR / "exercise_catalog.json",
    )
    parser.add_argument(
        "--exercise",
        action="append",
        help="Only analyze this canonical exercise (repeatable).",
    )
    args = parser.parse_args()

    catalog = load_catalog(args.catalog_file)
    by_exercise = collect_records(args.training_set_dir, catalog)

    names = sorted(by_exercise)
    if args.exercise:
        wanted = {e.strip().lower() for e in args.exercise}
        names = [n for n in names if n in wanted]
        missing = wanted - set(names)
        for name in sorted(missing):
            print(f"warning: no data for exercise {name!r}", file=sys.stderr)

    print(f"Reportable exercises with data: {len(names)}\n")
    for name in names:
        print(f"## {name}")
        print_exercise_report(by_exercise[name], min_sessions=2)
        print()


if __name__ == "__main__":
    main()
