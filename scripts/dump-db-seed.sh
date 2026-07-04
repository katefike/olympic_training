#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

POSTGRES_USER="${POSTGRES_USER:-postgres}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-postgres}"
POSTGRES_DB="${POSTGRES_DB:-olympic_training}"
POSTGRES_HOST="${POSTGRES_HOST:-localhost}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"
DB_CONTAINER="${DB_CONTAINER:-olympic-training-db}"

run_pg_dump() {
  local schema="$1"
  if command -v pg_dump >/dev/null 2>&1; then
    export PGPASSWORD="$POSTGRES_PASSWORD"
    pg_dump \
      --host="$POSTGRES_HOST" \
      --port="$POSTGRES_PORT" \
      --username="$POSTGRES_USER" \
      --dbname="$POSTGRES_DB" \
      --schema="$schema" \
      --data-only \
      --no-owner \
      --no-privileges \
      --column-inserts
    return
  fi
  if docker ps --format '{{.Names}}' | grep -qx "$DB_CONTAINER"; then
    docker exec -e PGPASSWORD="$POSTGRES_PASSWORD" "$DB_CONTAINER" \
      pg_dump \
      --username="$POSTGRES_USER" \
      --dbname="$POSTGRES_DB" \
      --schema="$schema" \
      --data-only \
      --no-owner \
      --no-privileges \
      --column-inserts
    return
  fi
  echo "error: pg_dump not found and container ${DB_CONTAINER} is not running" >&2
  echo "  Install PostgreSQL client tools, or start: docker compose up -d db" >&2
  exit 1
}

dump_schema_seed() {
  local schema="$1"
  local out="$2"
  local header="$3"
  {
    echo "$header"
    echo "-- Generated: $(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    echo ""
    run_pg_dump "$schema"
  } >"$out"
  echo "Wrote $out ($(wc -l <"$out") lines)"
}

dump_schema_seed workouts "$ROOT/docker/db/init/002_workout_seed.sql" \
  "-- Generated workout seed data. Re-run after JSON or catalog changes:
--   bash scripts/dump-db-seed.sh  (requires running Postgres after local-dev-setup.sh)
--   python3 scripts/generate_workout_seed_sql.py  (no Postgres required)"

dump_schema_seed myfitnesspal "$ROOT/docker/db/init/003_myfitnesspal_seed.sql" \
  "-- Generated MyFitnessPal seed data. Re-run after CSV export changes:
--   bash scripts/local-dev-setup.sh && bash scripts/dump-db-seed.sh"
