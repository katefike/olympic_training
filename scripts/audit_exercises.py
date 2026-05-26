#!/usr/bin/env python3
"""Audit training_set exercise names against the exercise catalog."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETL_DIR = ROOT / "src_data" / "workouts" / "etl"
sys.path.insert(0, str(ETL_DIR))

from catalog import ExerciseCatalog, load_catalog, session_scores  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit exercise names in training_set JSON.")
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
    parser.add_argument("--min-sessions", type=int, default=2, help="Flag raw names below this count.")
    args = parser.parse_args()

    catalog: ExerciseCatalog | None = None
    if args.catalog_file.exists():
        catalog = load_catalog(args.catalog_file)
    else:
        print(f"warning: catalog not found at {args.catalog_file}", file=sys.stderr)

    def canonicalize(raw: str) -> str:
        if catalog:
            return catalog.canonicalize(raw)
        return raw.strip().lower()

    # canonical -> aggregated stats
    by_canonical: dict[str, dict] = defaultdict(
        lambda: {
            "sessions": 0,
            "raw_names": set(),
            "dur_sessions": 0,
            "rep_sessions": 0,
            "score_duration": 0.0,
            "score_reps": 0.0,
        }
    )
    unmapped_raw: dict[str, int] = defaultdict(int)

    for file_path in sorted(args.training_set_dir.glob("*.json")):
        try:
            sessions = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"skip invalid {file_path.name}: {exc}")
            continue
        if not isinstance(sessions, list):
            continue
        for session in sessions:
            if not isinstance(session, dict):
                continue
            exercises = session.get("exercises", {})
            if not isinstance(exercises, dict):
                continue
            for raw_name, details in exercises.items():
                if not isinstance(details, dict):
                    continue
                raw = raw_name.strip().lower()
                canon = canonicalize(raw_name)
                stats = by_canonical[canon]
                stats["sessions"] += 1
                stats["raw_names"].add(raw)
                scores = session_scores(details)
                if scores["has_duration"]:
                    stats["dur_sessions"] += 1
                if scores["has_reps"]:
                    stats["rep_sessions"] += 1
                stats["score_duration"] += scores["progress_score_duration"]
                stats["score_reps"] += scores["progress_score_reps"]

                if catalog and canon == raw and raw not in catalog.alias_to_canonical:
                    unmapped_raw[raw] += 1

    print(f"{'sessions':>8}  {'dur':>4}  {'rep':>4}  {'catalog':>8}  canonical")
    print("-" * 72)
    for canon, stats in sorted(by_canonical.items(), key=lambda x: -x[1]["sessions"]):
        in_cat = ""
        if catalog:
            entry = catalog.entry_for(canon)
            if entry:
                in_cat = f"{entry.progress_metric or '-':>4} {'Y' if entry.include_in_reports else 'n':>3}"
            else:
                in_cat = "   -   n"
        mixed = stats["dur_sessions"] and stats["rep_sessions"]
        flag = " MIXED" if mixed else ""
        raws = ",".join(sorted(stats["raw_names"])[:3])
        if len(stats["raw_names"]) > 3:
            raws += ",..."
        print(
            f"{stats['sessions']:8d}  {stats['dur_sessions']:4d}  {stats['rep_sessions']:4d}  {in_cat:>8}  {canon}{flag}"
            f"  [{raws}]"
        )

    if unmapped_raw:
        print("\nRaw names with no catalog entry (same as canonical):")
        for raw, count in sorted(unmapped_raw.items(), key=lambda x: -x[1]):
            if count < args.min_sessions:
                print(f"  {count:3d}  {raw}  (low frequency)")
            else:
                print(f"  {count:3d}  {raw}")

    mixed_canonicals = [
        c for c, s in by_canonical.items() if s["dur_sessions"] and s["rep_sessions"]
    ]
    if mixed_canonicals:
        print("\nMixed-mode canonicals (duration and rep sessions):")
        for canon in sorted(mixed_canonicals):
            s = by_canonical[canon]
            print(f"  {canon}: {s['dur_sessions']} duration, {s['rep_sessions']} rep sessions")


if __name__ == "__main__":
    main()
