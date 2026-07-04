from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import psycopg


@dataclass
class Settings:
    postgres_dsn: str
    exports_dir: Path


def parse_args() -> Settings:
    mfp_dir = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Load MyFitnessPal CSV exports into Postgres.")
    parser.add_argument(
        "--postgres-dsn",
        default="postgresql://postgres:postgres@localhost:5432/olympic_training",
        help="Postgres DSN",
    )
    parser.add_argument(
        "--exports-dir",
        default=str(mfp_dir / "exports"),
        help="Directory containing dated MFP export folders.",
    )
    args = parser.parse_args()
    return Settings(postgres_dsn=args.postgres_dsn, exports_dir=Path(args.exports_dir))


def _coerce_float(value: str | None) -> float | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _coerce_int(value: str | None) -> int | None:
    number = _coerce_float(value)
    if number is None:
        return None
    return int(number)


def _parse_date(value: str | None) -> date | None:
    if not value:
        return None
    raw = str(value).strip()
    try:
        return datetime.strptime(raw, "%Y-%m-%d").date()
    except ValueError:
        return None


def _latest_export_dir(exports_dir: Path) -> Path:
    if not exports_dir.is_dir():
        raise FileNotFoundError(f"Exports directory not found: {exports_dir}")
    candidates = sorted(
        (p for p in exports_dir.iterdir() if p.is_dir()),
        key=lambda p: p.name,
    )
    if not candidates:
        raise FileNotFoundError(f"No export folders under {exports_dir}")
    return candidates[-1]


def _find_summary(export_dir: Path, prefix: str) -> Path:
    matches = sorted(export_dir.glob(f"{prefix}-*.csv"))
    if not matches:
        raise FileNotFoundError(f"No {prefix}-*.csv in {export_dir}")
    return matches[-1]


def clear_tables(cur: psycopg.Cursor[Any]) -> None:
    cur.execute("TRUNCATE myfitnesspal.exercise_entry RESTART IDENTITY")
    cur.execute("TRUNCATE myfitnesspal.nutrition_meal RESTART IDENTITY")
    cur.execute("TRUNCATE myfitnesspal.measurement RESTART IDENTITY")


def load_measurements(cur: psycopg.Cursor[Any], path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            measured_on = _parse_date(row.get("Date"))
            if measured_on is None:
                continue
            values = (
                measured_on,
                _coerce_float(row.get("Weight")),
                _coerce_float(row.get("Chest")),
                _coerce_float(row.get("Chest (under armpits)")),
                _coerce_float(row.get("Hips")),
                _coerce_float(row.get("Neck")),
                _coerce_float(row.get("Suprailiac (mm)")),
                _coerce_float(row.get("Thigh (mm)")),
                _coerce_float(row.get("Total Body Fat (mm)")),
                _coerce_float(row.get("Tricep (mm)")),
                _coerce_float(row.get("Waist")),
                path.name,
            )
            if all(v is None for v in values[1:11]):
                continue
            cur.execute(
                """
                INSERT INTO myfitnesspal.measurement (
                  measured_on, weight_lbs, chest, chest_under_armpits, hips, neck,
                  suprailiac_mm, thigh_mm, total_body_fat_mm, tricep_mm, waist, source_file
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                values,
            )
            count += 1
    return count


def load_nutrition(cur: psycopg.Cursor[Any], path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logged_on = _parse_date(row.get("Date"))
            meal = (row.get("Meal") or "").strip()
            if logged_on is None or not meal:
                continue
            note = (row.get("Note") or "").strip() or None
            cur.execute(
                """
                INSERT INTO myfitnesspal.nutrition_meal (
                  logged_on, meal, calories, fat_g, saturated_fat_g, polyunsaturated_fat_g,
                  monounsaturated_fat_g, trans_fat_g, cholesterol_mg, sodium_mg, potassium_mg,
                  carbohydrates_g, fiber_g, sugar_g, protein_g, vitamin_a, vitamin_c, calcium,
                  iron, note, source_file
                ) VALUES (
                  %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                """,
                (
                    logged_on,
                    meal,
                    _coerce_float(row.get("Calories")),
                    _coerce_float(row.get("Fat (g)")),
                    _coerce_float(row.get("Saturated Fat")),
                    _coerce_float(row.get("Polyunsaturated Fat")),
                    _coerce_float(row.get("Monounsaturated Fat")),
                    _coerce_float(row.get("Trans Fat")),
                    _coerce_float(row.get("Cholesterol")),
                    _coerce_float(row.get("Sodium (mg)")),
                    _coerce_float(row.get("Potassium")),
                    _coerce_float(row.get("Carbohydrates (g)")),
                    _coerce_float(row.get("Fiber")),
                    _coerce_float(row.get("Sugar")),
                    _coerce_float(row.get("Protein (g)")),
                    _coerce_float(row.get("Vitamin A")),
                    _coerce_float(row.get("Vitamin C")),
                    _coerce_float(row.get("Calcium")),
                    _coerce_float(row.get("Iron")),
                    note,
                    path.name,
                ),
            )
            count += 1
    return count


def load_exercise(cur: psycopg.Cursor[Any], path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            logged_on = _parse_date(row.get("Date"))
            exercise_name = (row.get("Exercise") or "").strip()
            if logged_on is None or not exercise_name:
                continue
            note = (row.get("Note") or "").strip() or None
            cur.execute(
                """
                INSERT INTO myfitnesspal.exercise_entry (
                  logged_on, exercise_name, exercise_type, calories, minutes, sets,
                  reps_per_set, pounds, steps, note, source_file
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    logged_on,
                    exercise_name,
                    (row.get("Type") or "").strip() or None,
                    _coerce_float(row.get("Exercise Calories")),
                    _coerce_float(row.get("Exercise Minutes")),
                    _coerce_int(row.get("Sets")),
                    _coerce_int(row.get("Reps Per Set")),
                    _coerce_float(row.get("Pounds")),
                    _coerce_int(row.get("Steps")),
                    note,
                    path.name,
                ),
            )
            count += 1
    return count


def main() -> None:
    settings = parse_args()
    export_dir = _latest_export_dir(settings.exports_dir)
    measurement_path = _find_summary(export_dir, "Measurement-Summary")
    nutrition_path = _find_summary(export_dir, "Nutrition-Summary")
    exercise_path = _find_summary(export_dir, "Exercise-Summary")

    with psycopg.connect(settings.postgres_dsn) as conn:
        with conn.cursor() as cur:
            clear_tables(cur)
            n_meas = load_measurements(cur, measurement_path)
            n_nutr = load_nutrition(cur, nutrition_path)
            n_ex = load_exercise(cur, exercise_path)
        conn.commit()

    print(
        f"Load complete from {export_dir.name}: "
        f"{n_meas} measurements, {n_nutr} nutrition meals, {n_ex} exercise entries."
    )


if __name__ == "__main__":
    main()
