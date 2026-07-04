#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
ETL_REQUIREMENTS="$ROOT/src_data/workouts/etl/requirements.txt"
WORKOUT_LOAD_SCRIPT="$ROOT/src_data/workouts/etl/load_to_postgres.py"
MFP_LOAD_SCRIPT="$ROOT/src_data/myfitnesspal/etl/load_to_postgres.py"
MFP_EXPORTS_DIR="$ROOT/src_data/myfitnesspal/exports"

if [[ ! -f "$ETL_REQUIREMENTS" || ! -f "$WORKOUT_LOAD_SCRIPT" ]]; then
  echo "error: missing ETL files under src_data/workouts/etl/" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -r "$ETL_REQUIREMENTS"
python "$WORKOUT_LOAD_SCRIPT"

if [[ -f "$MFP_LOAD_SCRIPT" && -d "$MFP_EXPORTS_DIR" ]] && compgen -G "$MFP_EXPORTS_DIR/*/" >/dev/null; then
  python "$MFP_LOAD_SCRIPT"
else
  echo "skip: MyFitnessPal exports not found under src_data/myfitnesspal/exports/ (seed data still loads on fresh DB)"
fi
