#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/.venv"
ETL_REQUIREMENTS="$ROOT/src_data/workouts/etl/requirements.txt"
LOAD_SCRIPT="$ROOT/src_data/workouts/etl/load_to_postgres.py"

if [[ ! -f "$ETL_REQUIREMENTS" || ! -f "$LOAD_SCRIPT" ]]; then
  echo "error: missing ETL files under src_data/workouts/etl/" >&2
  exit 1
fi

if [[ ! -d "$VENV" ]]; then
  python3 -m venv "$VENV"
fi

# shellcheck source=/dev/null
source "$VENV/bin/activate"
pip install -r "$ETL_REQUIREMENTS"
python "$LOAD_SCRIPT"
