"""SQLite helpers for suspicious-activity event records."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.alerts import Alert


SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id TEXT PRIMARY KEY,
    stream_id TEXT NOT NULL,
    track_id INTEGER,
    type TEXT NOT NULL,
    confidence REAL NOT NULL,
    timestamp REAL NOT NULL,
    snapshot_path TEXT NOT NULL,
    detail TEXT NOT NULL,
    reviewed INTEGER NOT NULL DEFAULT 0
)
"""


def initialize(db_path: str | Path) -> None:
    """Create the event database and schema if they do not exist."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as connection:
        connection.execute(SCHEMA)


def insert_event(db_path: str | Path, alert: Alert) -> None:
    """Insert one alert event into the database."""
    initialize(db_path)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO events (id, stream_id, track_id, type, confidence, timestamp, snapshot_path, detail) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (alert.id, alert.stream_id, alert.track_id, alert.type, alert.confidence, alert.timestamp,
             alert.snapshot_path, alert.detail),
        )


def list_events(db_path: str | Path, stream_id: str | None = None, event_type: str | None = None,
                limit: int = 100) -> list[dict[str, Any]]:
    """Return newest event records with optional stream/type filters."""
    if limit < 1:
        raise ValueError("limit must be positive")
    initialize(db_path)
    clauses: list[str] = []
    values: list[Any] = []
    if stream_id is not None:
        clauses.append("stream_id = ?")
        values.append(stream_id)
    if event_type is not None:
        clauses.append("type = ?")
        values.append(event_type)
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(f"SELECT * FROM events{where} ORDER BY timestamp DESC LIMIT ?",
                                  (*values, limit)).fetchall()
    return [dict(row) for row in rows]


def mark_reviewed(db_path: str | Path, event_id: str) -> bool:
    """Mark an event as reviewed and report whether it existed."""
    initialize(db_path)
    with sqlite3.connect(db_path) as connection:
        cursor = connection.execute("UPDATE events SET reviewed = 1 WHERE id = ?", (event_id,))
        return cursor.rowcount > 0
