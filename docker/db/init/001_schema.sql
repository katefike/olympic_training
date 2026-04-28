CREATE SCHEMA IF NOT EXISTS workouts;

CREATE TABLE IF NOT EXISTS workouts.exercise (
  id BIGSERIAL PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  movement_pattern TEXT,
  primary_muscle_group TEXT,
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
