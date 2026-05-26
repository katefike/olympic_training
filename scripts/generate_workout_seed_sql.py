#!/usr/bin/env python3
"""Generate docker/db/init/002_workout_seed.sql from training_set JSON (no Postgres required)."""

from __future__ import annotations

import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ETL_DIR = ROOT / "src_data" / "workouts" / "etl"
sys.path.insert(0, str(ETL_DIR))

from catalog import ExerciseCatalog, load_catalog, normalize_unit, to_lbs  # noqa: E402

OUT = ROOT / "docker" / "db" / "init" / "002_workout_seed.sql"
TRAINING_SET = ROOT / "src_data" / "workouts" / "training_set"
CATALOG_FILE = ETL_DIR / "exercise_catalog.json"


def sql_literal(value) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, date) and not isinstance(value, datetime):
        return f"'{value.isoformat()}'::date"
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


def parse_session_date(value) -> date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if len(raw) != 6 or not raw.isdigit():
        return None
    try:
        return datetime.strptime(raw, "%y%m%d").date()
    except ValueError:
        return None


def main() -> None:
    catalog = load_catalog(CATALOG_FILE)
    lines: list[str] = [
        "-- Generated workout seed data. Re-run after JSON or catalog changes:",
        "--   python scripts/generate_workout_seed_sql.py",
        "-- or (with Postgres): bash scripts/local-dev-setup.sh && bash scripts/dump-db-seed.sh",
        f"-- Generated: {datetime.now(UTC).strftime('%Y-%m-%dT%H:%M:%SZ')}",
        "",
        "SET client_encoding = 'UTF8';",
        "",
    ]

    exercise_id: dict[str, int] = {}
    alias_rows: list[tuple[str, int]] = []
    next_exercise_id = 1
    next_alias_id = 1

    for entry in catalog.entries:
        exercise_id[entry.canonical_name] = next_exercise_id
        lines.append(
            "INSERT INTO workouts.exercise "
            "(id, canonical_name, include_in_reports, progress_metric) VALUES "
            f"({next_exercise_id}, {sql_literal(entry.canonical_name)}, "
            f"{sql_literal(entry.include_in_reports)}, {sql_literal(entry.progress_metric)});"
        )
        next_exercise_id += 1
        for alias in entry.aliases:
            alias_rows.append((alias, exercise_id[entry.canonical_name]))

    def ensure_exercise(canonical: str) -> int:
        nonlocal next_exercise_id
        if canonical not in exercise_id:
            exercise_id[canonical] = next_exercise_id
            lines.append(
                "INSERT INTO workouts.exercise (id, canonical_name) VALUES "
                f"({next_exercise_id}, {sql_literal(canonical)});"
            )
            next_exercise_id += 1
        return exercise_id[canonical]

    session_id = 1
    pain_id = 1
    session_exercise_id = 1
    set_id = 1

    for file_path in sorted(TRAINING_SET.glob("*.json")):
        try:
            sessions = json.loads(file_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(sessions, list):
            continue
        for session_index, session in enumerate(sessions):
            if not isinstance(session, dict):
                continue
            session_date = parse_session_date(session.get("date"))
            lines.append(
                "INSERT INTO workouts.workout_session "
                "(id, source_file, heic_id, session_index, session_date, gym, cardio_type, "
                "cardio_duration_minutes, notes) VALUES ("
                f"{session_id}, {sql_literal(file_path.name)}, {sql_literal(session.get('heic_id'))}, "
                f"{session_index}, {sql_literal(session_date)}, {sql_literal(session.get('gym'))}, "
                f"{sql_literal(session.get('cardio_type'))}, "
                f"{sql_literal(session.get('cardio_duration_minutes'))}, "
                f"{sql_literal(session.get('notes'))});"
            )
            for pain in session.get("pain") or []:
                if not isinstance(pain, dict):
                    continue
                lines.append(
                    "INSERT INTO workouts.pain_entry "
                    f"(id, workout_session_id, pain_level, pain_location) VALUES "
                    f"({pain_id}, {session_id}, {sql_literal(pain.get('pain_level'))}, "
                    f"{sql_literal(pain.get('pain_location'))});"
                )
                pain_id += 1

            exercises = session.get("exercises", {})
            if not isinstance(exercises, dict):
                session_id += 1
                continue
            for raw_name, details in exercises.items():
                if not isinstance(details, dict):
                    continue
                canonical = catalog.canonicalize(raw_name)
                eid = ensure_exercise(canonical)
                entry = catalog.entry_for(canonical)
                if entry:
                    lines.append(
                        f"UPDATE workouts.exercise SET include_in_reports = "
                        f"{sql_literal(entry.include_in_reports)}, progress_metric = "
                        f"{sql_literal(entry.progress_metric)} WHERE id = {eid};"
                    )
                alias_rows.append((raw_name.strip().lower(), eid))
                lines.append(
                    "INSERT INTO workouts.session_exercise "
                    "(id, workout_session_id, exercise_id, raw_exercise_name, notes, reps_total, reps_each) "
                    f"VALUES ({session_exercise_id}, {session_id}, {eid}, {sql_literal(raw_name)}, "
                    f"{sql_literal(details.get('notes'))}, {sql_literal(details.get('reps_total'))}, "
                    f"{sql_literal(details.get('reps_each'))});"
                )
                unit = normalize_unit(details.get("weight_unit"))
                for set_index, entry_set in enumerate(details.get("sets") or []):
                    if not isinstance(entry_set, dict):
                        continue
                    ow = entry_set.get("weight")
                    lines.append(
                        "INSERT INTO workouts.set_entry "
                        "(id, session_exercise_id, set_index, weight_lbs, original_weight, "
                        "original_weight_unit, duration_sec) VALUES ("
                        f"{set_id}, {session_exercise_id}, {set_index}, "
                        f"{sql_literal(to_lbs(ow, unit))}, {sql_literal(ow)}, "
                        f"{sql_literal(unit)}, {sql_literal(entry_set.get('duration_sec'))});"
                    )
                    set_id += 1
                session_exercise_id += 1
            session_id += 1

    for entry in catalog.entries:
        eid = exercise_id[entry.canonical_name]
        for alias in entry.aliases:
            alias_rows.append((alias, eid))

    lines.append("")
    lines.append("-- Aliases (deduplicated)")
    seen_aliases: set[str] = set()
    for alias_name, eid in alias_rows:
        if alias_name in seen_aliases:
            continue
        seen_aliases.add(alias_name)
        lines.append(
            "INSERT INTO workouts.exercise_alias (id, alias_name, exercise_id) VALUES "
            f"({next_alias_id}, {sql_literal(alias_name)}, {eid}) "
            "ON CONFLICT (alias_name) DO NOTHING;"
        )
        next_alias_id += 1

    lines.extend(
        [
            "",
            f"SELECT setval('workouts.exercise_id_seq', (SELECT COALESCE(MAX(id), 1) FROM workouts.exercise));",
            f"SELECT setval('workouts.exercise_alias_id_seq', (SELECT COALESCE(MAX(id), 1) FROM workouts.exercise_alias));",
            f"SELECT setval('workouts.workout_session_id_seq', (SELECT COALESCE(MAX(id), 1) FROM workouts.workout_session));",
            f"SELECT setval('workouts.pain_entry_id_seq', (SELECT COALESCE(MAX(id), 1) FROM workouts.pain_entry));",
            f"SELECT setval('workouts.session_exercise_id_seq', (SELECT COALESCE(MAX(id), 1) FROM workouts.session_exercise));",
            f"SELECT setval('workouts.set_entry_id_seq', (SELECT COALESCE(MAX(id), 1) FROM workouts.set_entry));",
        ]
    )

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
