# olympic-training

Local scaffold for workout analytics with:

- Postgres for normalized workout + standards data
- Grafana for visualization
- FastAPI for app/AI integrations

Canonical weight is stored in `lbs`.

## Quick start (fresh environment)

1. Start services (loads schema + seed data on a **new** DB volume):

```bash
docker compose up -d
```

Postgres init runs [`docker/db/init/001_schema.sql`](docker/db/init/001_schema.sql) then [`docker/db/init/002_workout_seed.sql`](docker/db/init/002_workout_seed.sql). No JSON ETL is required for the first bring-up.

2. Open:

- Grafana: `http://localhost:3000` (username:`admin` / password:`admin`)
  - **Exercise Progress** — allowlisted exercises; max weight chart, then volume (TUL × weight) for duration exercises or progress score for rep-based exercises
  - **Exercise Explorer** — all exercises + debug table
  - **Workout Overview** — daily total volume
- API docs: `http://localhost:8000/docs`
- psql: `docker exec -it olympic-training-db psql -U postgres -d olympic_training`

## Refresh data from JSON

When you change `src_data/workouts/training_set/` or [`src_data/workouts/etl/exercise_catalog.json`](src_data/workouts/etl/exercise_catalog.json):

1. Audit exercise names:

```bash
python3 scripts/audit_exercises.py
```

2. Edit the catalog (aliases, `include_in_reports`, `progress_metric`: `duration` or `reps`).

3. Load into a running Postgres:

```bash
bash scripts/local-dev-setup.sh
```

4. Regenerate the committed seed (pick one):

```bash
# With Postgres client + running DB (preferred after ETL)
bash scripts/dump-db-seed.sh

# Without Postgres (from JSON + catalog)
python3 scripts/generate_workout_seed_sql.py
```

5. Replay on a local DB volume: `docker compose down -v && docker compose up -d`, or restore manually.

## Exercise catalog

[`src_data/workouts/etl/exercise_catalog.json`](src_data/workouts/etl/exercise_catalog.json) is the source of truth for:

- **aliases** — raw JSON keys → canonical names (replaces `exercise_aliases.json`)
- **include_in_reports** — Grafana Exercise Progress allowlist
- **progress_metric** — `duration` (Σ weight×seconds, shown as volume/TUL in reports) or `reps` (Σ weight×reps) for the progress score chart

Kick directions (`side_kick_out`, `side_kick_in`, `side_kick_back`, etc.) stay **separate** canonicals; only spelling variants alias together.

## Data model

Timestamps (`created_at`) use **US Eastern** (`America/New_York`, EST/EDT). Workout days are stored as plain `DATE` values from the JSON `date` field (no timezone).

Main tables:

- `workouts.workout_session`
- `workouts.pain_entry`
- `workouts.exercise` (`include_in_reports`, `progress_metric`)
- `workouts.exercise_alias`
- `workouts.session_exercise`
- `workouts.set_entry` (`weight_lbs` canonical, original weight + unit preserved)
- `workouts.strength_standard`

Views for dashboards:

- `workouts.v_session_exercise_metrics` — per session/exercise: `max_weight_lbs`, `session_volume` (TUL × weight for duration exercises), `progress_score`, etc.
- `workouts.v_reportable_exercises` — allowlist for Grafana
- `workouts.v_daily_volume`
- `workouts.v_session_summary`

Schema diagram:

- [`docs/schema.mmd`](docs/schema.mmd)

## Standards files

Strength standards are loaded from:

- all `.csv` and `.json` files in `src_data/strength_standards/`

Example: `src_data/strength_standards/free_weights.csv`
