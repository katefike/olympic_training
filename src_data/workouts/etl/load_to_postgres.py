from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import psycopg

LBS_PER_KG = 2.2046226218
SUPPORTED_UNITS = {"lb", "lbs", "pound", "pounds", "kg", "kgs", "kilogram", "kilograms"}


@dataclass
class Settings:
    postgres_dsn: str
    training_set_dir: Path
    strength_standards_dir: Path | None
    aliases_file: Path


def parse_args() -> Settings:
    parser = argparse.ArgumentParser(description="Normalize workout json files and load into Postgres.")
    parser.add_argument(
        "--postgres-dsn",
        default="postgresql://postgres:postgres@localhost:5432/olympic_training",
        help="Postgres DSN",
    )
    parser.add_argument(
        "--training-set-dir",
        default=str(Path(__file__).resolve().parents[1] / "training_set"),
        help="Directory with workout session JSON files.",
    )
    parser.add_argument(
        "--strength-standards-dir",
        default=str(Path(__file__).resolve().parents[2] / "strength_standards"),
        help="Optional directory with strength standards (.csv/.json).",
    )
    parser.add_argument(
        "--aliases-file",
        default=str(Path(__file__).resolve().parent / "exercise_aliases.json"),
        help="JSON map of exercise aliases -> canonical names.",
    )
    args = parser.parse_args()
    strength_dir = Path(args.strength_standards_dir)
    return Settings(
        postgres_dsn=args.postgres_dsn,
        training_set_dir=Path(args.training_set_dir),
        strength_standards_dir=strength_dir if strength_dir.exists() and strength_dir.is_dir() else None,
        aliases_file=Path(args.aliases_file),
    )


def normalize_unit(unit: Any) -> str | None:
    if unit is None:
        return None
    cleaned = str(unit).strip().lower()
    if cleaned not in SUPPORTED_UNITS:
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


def parse_session_date(value: Any) -> datetime.date | None:
    if value is None:
        return None
    raw = str(value).strip()
    if len(raw) != 6 or not raw.isdigit():
        return None
    try:
        return datetime.strptime(raw, "%y%m%d").date()
    except ValueError:
        return None


def load_aliases(path: Path) -> dict[str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    aliases = payload.get("aliases", {})
    if not isinstance(aliases, dict):
        return {}
    return {str(k).strip().lower(): str(v).strip().lower() for k, v in aliases.items()}


def canonicalize_exercise(raw_name: str, aliases: dict[str, str]) -> str:
    clean = raw_name.strip().lower()
    return aliases.get(clean, clean)


def get_or_create_exercise(cur: psycopg.Cursor[Any], canonical_name: str) -> int:
    cur.execute(
        """
        INSERT INTO workouts.exercise (canonical_name)
        VALUES (%s)
        ON CONFLICT (canonical_name) DO UPDATE SET canonical_name = EXCLUDED.canonical_name
        RETURNING id
        """,
        (canonical_name,),
    )
    exercise_id = cur.fetchone()
    if exercise_id:
        return int(exercise_id[0])
    cur.execute("SELECT id FROM workouts.exercise WHERE canonical_name = %s", (canonical_name,))
    row = cur.fetchone()
    if not row:
        raise RuntimeError(f"Failed to resolve exercise id for {canonical_name}")
    return int(row[0])


def upsert_alias(cur: psycopg.Cursor[Any], raw_name: str, exercise_id: int) -> None:
    cur.execute(
        """
        INSERT INTO workouts.exercise_alias (alias_name, exercise_id)
        VALUES (%s, %s)
        ON CONFLICT (alias_name) DO UPDATE SET exercise_id = EXCLUDED.exercise_id
        """,
        (raw_name.strip().lower(), exercise_id),
    )


def clear_workout_tables(cur: psycopg.Cursor[Any]) -> None:
    cur.execute("TRUNCATE workouts.set_entry, workouts.session_exercise, workouts.pain_entry, workouts.workout_session RESTART IDENTITY CASCADE")


def load_workout_files(cur: psycopg.Cursor[Any], training_set_dir: Path, aliases: dict[str, str]) -> None:
    files = sorted(training_set_dir.glob("*.json"))
    for file_path in files:
        sessions = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(sessions, list):
            continue
        for session_index, session in enumerate(sessions):
            if not isinstance(session, dict):
                continue
            cur.execute(
                """
                INSERT INTO workouts.workout_session
                  (source_file, heic_id, session_index, session_date, gym, cardio_type, cardio_duration_minutes, notes)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    file_path.name,
                    session.get("heic_id"),
                    session_index,
                    parse_session_date(session.get("date")),
                    session.get("gym"),
                    session.get("cardio_type"),
                    session.get("cardio_duration_minutes"),
                    session.get("notes"),
                ),
            )
            workout_session_id = int(cur.fetchone()[0])

            pain_entries = session.get("pain", [])
            if isinstance(pain_entries, list):
                for pain in pain_entries:
                    if not isinstance(pain, dict):
                        continue
                    cur.execute(
                        """
                        INSERT INTO workouts.pain_entry (workout_session_id, pain_level, pain_location)
                        VALUES (%s, %s, %s)
                        """,
                        (workout_session_id, pain.get("pain_level"), pain.get("pain_location")),
                    )

            exercises = session.get("exercises", {})
            if not isinstance(exercises, dict):
                continue
            for raw_name, details in exercises.items():
                if not isinstance(details, dict):
                    continue
                canonical_name = canonicalize_exercise(raw_name, aliases)
                exercise_id = get_or_create_exercise(cur, canonical_name)
                upsert_alias(cur, raw_name, exercise_id)
                cur.execute(
                    """
                    INSERT INTO workouts.session_exercise
                      (workout_session_id, exercise_id, raw_exercise_name, notes, reps_total, reps_each)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    RETURNING id
                    """,
                    (
                        workout_session_id,
                        exercise_id,
                        raw_name,
                        details.get("notes"),
                        details.get("reps_total"),
                        details.get("reps_each"),
                    ),
                )
                session_exercise_id = int(cur.fetchone()[0])

                unit = normalize_unit(details.get("weight_unit"))
                sets = details.get("sets", [])
                if not isinstance(sets, list):
                    continue
                for set_index, entry in enumerate(sets):
                    if not isinstance(entry, dict):
                        continue
                    original_weight = entry.get("weight")
                    cur.execute(
                        """
                        INSERT INTO workouts.set_entry
                          (session_exercise_id, set_index, weight_lbs, original_weight, original_weight_unit, duration_sec)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        """,
                        (
                            session_exercise_id,
                            set_index,
                            to_lbs(original_weight, unit),
                            original_weight,
                            unit,
                            entry.get("duration_sec"),
                        ),
                    )


def _coerce_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _norm_key(value: str) -> str:
    return value.strip().lower().replace(" ", "_").replace("-", "_")


def _pick(record: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in record:
            return record[key]
    return None


def _load_standards_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("standards", payload) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        return []
    out: list[dict[str, Any]] = []
    for record in records:
        if isinstance(record, dict):
            out.append({str(k): v for k, v in record.items()})
    return out


def _load_standards_from_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            normalized = {_norm_key(k): v for k, v in row.items() if k is not None}
            rows.append(normalized)
    return rows


def _standard_lbs(record: dict[str, Any], tier: str) -> float | None:
    lbs = _coerce_float(_pick(record, f"{tier}_lbs", tier))
    if lbs is not None:
        return round(lbs, 2)
    kg = _coerce_float(_pick(record, f"{tier}_kg", f"{tier}_kgs"))
    if kg is not None:
        return round(kg * LBS_PER_KG, 2)
    return None


def load_strength_standards(cur: psycopg.Cursor[Any], standards_dir: Path | None, aliases: dict[str, str]) -> None:
    if standards_dir is None:
        return
    cur.execute("TRUNCATE workouts.strength_standard RESTART IDENTITY")
    files = sorted([*standards_dir.glob("*.csv"), *standards_dir.glob("*.json")])
    for standards_path in files:
        if standards_path.suffix.lower() == ".csv":
            records = _load_standards_from_csv(standards_path)
        else:
            records = _load_standards_from_json(standards_path)

        for record in records:
            exercise_name = _pick(record, "exercise", "exercise_name", "lift")
            if not exercise_name:
                continue
            canonical = canonicalize_exercise(str(exercise_name), aliases)
            if not canonical:
                continue
            exercise_id = get_or_create_exercise(cur, canonical)
            source_name = _pick(record, "source_name", "source") or standards_path.stem
            sex = _pick(record, "sex", "gender") or "female"
            age_years = _pick(record, "age_years", "age", "age_yrs") or 29
            cur.execute(
                """
                INSERT INTO workouts.strength_standard
                  (source_name, sex, age_years, exercise_id, beginner_lbs, novice_lbs, intermediate_lbs, advanced_lbs, elite_lbs)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (source_name, sex, age_years, exercise_id)
                DO UPDATE SET
                  beginner_lbs = EXCLUDED.beginner_lbs,
                  novice_lbs = EXCLUDED.novice_lbs,
                  intermediate_lbs = EXCLUDED.intermediate_lbs,
                  advanced_lbs = EXCLUDED.advanced_lbs,
                  elite_lbs = EXCLUDED.elite_lbs
                """,
                (
                    str(source_name),
                    str(sex),
                    int(age_years),
                    exercise_id,
                    _standard_lbs(record, "beginner"),
                    _standard_lbs(record, "novice"),
                    _standard_lbs(record, "intermediate"),
                    _standard_lbs(record, "advanced"),
                    _standard_lbs(record, "elite"),
                ),
            )


def main() -> None:
    settings = parse_args()
    aliases = load_aliases(settings.aliases_file)
    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.cursor() as cur:
            clear_workout_tables(cur)
            load_workout_files(cur, settings.training_set_dir, aliases)
            load_strength_standards(cur, settings.strength_standards_dir, aliases)
        conn.commit()
    print("Load complete: workouts normalized with canonical unit lbs.")


if __name__ == "__main__":
    main()
