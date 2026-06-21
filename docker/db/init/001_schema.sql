CREATE SCHEMA IF NOT EXISTS workouts;

CREATE TABLE IF NOT EXISTS workouts.exercise (
  id BIGSERIAL PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  movement_pattern TEXT,
  primary_muscle_group TEXT,
  include_in_reports BOOLEAN NOT NULL DEFAULT FALSE,
  progress_metric TEXT CHECK (progress_metric IN ('duration', 'reps')),
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS workouts.exercise_alias (
  id BIGSERIAL PRIMARY KEY,
  alias_name TEXT NOT NULL UNIQUE,
  exercise_id BIGINT NOT NULL REFERENCES workouts.exercise(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS workouts.workout_session (
  id BIGSERIAL PRIMARY KEY,
  source_file TEXT NOT NULL,
  heic_id TEXT,
  session_index INTEGER NOT NULL,
  session_date DATE,
  gym TEXT,
  cardio_type TEXT,
  cardio_duration_minutes NUMERIC(8,2),
  notes TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  UNIQUE (source_file, session_index)
);

CREATE TABLE IF NOT EXISTS workouts.pain_entry (
  id BIGSERIAL PRIMARY KEY,
  workout_session_id BIGINT NOT NULL REFERENCES workouts.workout_session(id) ON DELETE CASCADE,
  pain_level NUMERIC(6,2),
  pain_location TEXT
);

CREATE TABLE IF NOT EXISTS workouts.session_exercise (
  id BIGSERIAL PRIMARY KEY,
  workout_session_id BIGINT NOT NULL REFERENCES workouts.workout_session(id) ON DELETE CASCADE,
  exercise_id BIGINT NOT NULL REFERENCES workouts.exercise(id),
  raw_exercise_name TEXT NOT NULL,
  notes TEXT,
  reps_total INTEGER,
  reps_each INTEGER
);

CREATE TABLE IF NOT EXISTS workouts.set_entry (
  id BIGSERIAL PRIMARY KEY,
  session_exercise_id BIGINT NOT NULL REFERENCES workouts.session_exercise(id) ON DELETE CASCADE,
  set_index INTEGER NOT NULL,
  weight_lbs NUMERIC(10,2),
  original_weight NUMERIC(10,2),
  original_weight_unit TEXT,
  duration_sec INTEGER
);

CREATE TABLE IF NOT EXISTS workouts.strength_standard (
  id BIGSERIAL PRIMARY KEY,
  source_name TEXT NOT NULL DEFAULT 'custom',
  sex TEXT NOT NULL,
  age_years INTEGER NOT NULL,
  exercise_id BIGINT NOT NULL REFERENCES workouts.exercise(id),
  beginner_lbs NUMERIC(10,2),
  novice_lbs NUMERIC(10,2),
  intermediate_lbs NUMERIC(10,2),
  advanced_lbs NUMERIC(10,2),
  elite_lbs NUMERIC(10,2),
  UNIQUE (source_name, sex, age_years, exercise_id)
);

CREATE OR REPLACE VIEW workouts.v_daily_volume AS
SELECT
  ws.session_date,
  e.canonical_name,
  COUNT(se.id) AS set_count,
  COALESCE(SUM(se.weight_lbs), 0) AS total_weight_lbs
FROM workouts.set_entry se
JOIN workouts.session_exercise sx ON sx.id = se.session_exercise_id
JOIN workouts.exercise e ON e.id = sx.exercise_id
JOIN workouts.workout_session ws ON ws.id = sx.workout_session_id
GROUP BY ws.session_date, e.canonical_name;

CREATE OR REPLACE VIEW workouts.v_session_summary AS
SELECT
  ws.id AS workout_session_id,
  ws.session_date,
  ws.gym,
  ws.cardio_type,
  ws.cardio_duration_minutes,
  COUNT(DISTINCT sx.id) AS exercise_count,
  COUNT(se.id) AS set_count,
  COALESCE(SUM(se.weight_lbs), 0) AS total_weight_lbs
FROM workouts.workout_session ws
LEFT JOIN workouts.session_exercise sx ON sx.workout_session_id = ws.id
LEFT JOIN workouts.set_entry se ON se.session_exercise_id = sx.id
GROUP BY ws.id;

DROP VIEW IF EXISTS workouts.v_session_exercise_metrics;
CREATE VIEW workouts.v_session_exercise_metrics AS
WITH set_metrics AS (
  SELECT
    sx.id AS session_exercise_id,
    sx.workout_session_id,
    sx.exercise_id,
    sx.reps_total,
    sx.reps_each,
    COUNT(se.id) AS set_count,
    MAX(se.weight_lbs) AS max_weight_lbs,
    MAX(se.duration_sec) AS max_duration_sec,
    BOOL_OR(se.duration_sec IS NOT NULL AND se.duration_sec > 0) AS has_duration,
    SUM(COALESCE(se.weight_lbs, 0) * COALESCE(se.duration_sec, 0)) AS progress_score_duration,
    SUM(
      COALESCE(se.weight_lbs, 0)
      * COALESCE(sx.reps_each, sx.reps_total, 0)
    ) AS progress_score_reps
  FROM workouts.session_exercise sx
  JOIN workouts.set_entry se ON se.session_exercise_id = sx.id
  GROUP BY sx.id
),
classified AS (
  SELECT
    sm.*,
    CASE
      WHEN sm.has_duration THEN 'duration'
      WHEN COALESCE(sm.reps_each, sm.reps_total, 0) > 0 THEN 'reps'
      ELSE NULL
    END AS session_metric
  FROM set_metrics sm
)
SELECT
  ws.session_date,
  e.canonical_name,
  e.progress_metric AS catalog_progress_metric,
  c.session_metric,
  c.set_count,
  c.max_weight_lbs,
  c.max_duration_sec,
  c.reps_total,
  c.reps_each,
  c.progress_score_duration,
  c.progress_score_reps,
  CASE
    WHEN c.session_metric = 'duration' THEN c.progress_score_duration
    ELSE NULL
  END AS session_volume,
  CASE
    WHEN c.session_metric = 'duration' THEN c.progress_score_duration
    WHEN c.session_metric = 'reps' THEN c.progress_score_reps
    ELSE NULL
  END AS progress_score
FROM classified c
JOIN workouts.workout_session ws ON ws.id = c.workout_session_id
JOIN workouts.exercise e ON e.id = c.exercise_id
WHERE ws.session_date IS NOT NULL;

CREATE OR REPLACE VIEW workouts.v_reportable_exercises AS
SELECT canonical_name, progress_metric
FROM workouts.exercise
WHERE include_in_reports
ORDER BY canonical_name;
