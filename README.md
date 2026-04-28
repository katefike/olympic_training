# olympic-training

Local scaffold for workout analytics with:

- Postgres for normalized workout + standards data
- Grafana for visualization
- FastAPI for app/AI integrations

Canonical weight is stored in `lbs`.

## Quick start

1. Start services:

```bash
docker compose up -d
```

2. Load normalized workout data to Postgres:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r src_data/workouts/etl/requirements.txt
python src_data/workouts/etl/load_to_postgres.py
```

3. Open:

- Grafana: `http://localhost:3000` (`admin` / `admin`)
- API docs: `http://localhost:8000/docs`

## Data model

Main tables:

- `workouts.workout_session`
- `workouts.pain_entry`
- `workouts.exercise`
- `workouts.exercise_alias`
- `workouts.session_exercise`
- `workouts.set_entry` (`weight_lbs` canonical, original weight + unit preserved)
- `workouts.strength_standard`

Views for dashboards:

- `workouts.v_daily_volume`
- `workouts.v_session_summary`

Schema diagram:

- `docs/schema.mmd`

## Standards file

Strength standards are loaded from:

- all `.csv` and `.json` files in `src_data/strength_standards/`

Example: `src_data/strength_standards/free_weights.csv`