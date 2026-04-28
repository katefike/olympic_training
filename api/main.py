from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import psycopg
from fastapi import FastAPI

app = FastAPI(title="Workout API")


def dsn_from_env() -> str:
    user = os.getenv("POSTGRES_USER", "postgres")
    pwd = os.getenv("POSTGRES_PASSWORD", "postgres")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    db = os.getenv("POSTGRES_DB", "olympic_training")
    return f"postgresql://{user}:{pwd}@{host}:{port}/{db}"


@contextmanager
def get_conn() -> Iterator[psycopg.Connection]:
    with psycopg.connect(dsn_from_env()) as conn:
        yield conn


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/sessions")
def sessions(limit: int = 50) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT workout_session_id, session_date, gym, exercise_count, set_count, total_weight_lbs
            FROM workouts.v_session_summary
            ORDER BY session_date DESC NULLS LAST, workout_session_id DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {
            "workout_session_id": r[0],
            "session_date": r[1].isoformat() if r[1] else None,
            "gym": r[2],
            "exercise_count": r[3],
            "set_count": r[4],
            "total_weight_lbs": float(r[5] or 0),
        }
        for r in rows
    ]


@app.get("/volume/daily")
def daily_volume(limit: int = 180) -> list[dict]:
    with get_conn() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT session_date, SUM(total_weight_lbs) AS total_weight_lbs
            FROM workouts.v_daily_volume
            WHERE session_date IS NOT NULL
            GROUP BY session_date
            ORDER BY session_date DESC
            LIMIT %s
            """,
            (limit,),
        )
        rows = cur.fetchall()
    return [
        {"session_date": r[0].isoformat(), "total_weight_lbs": float(r[1] or 0)}
        for r in rows
    ]
