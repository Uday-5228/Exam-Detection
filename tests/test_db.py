"""SQLite event log checks."""

from pathlib import Path

from src.alerts import Alert
from src.db import insert_event, list_events, mark_reviewed


def test_insert_filter_and_review_event(tmp_path: Path) -> None:
    """Events can be stored, filtered, and marked reviewed."""
    database = tmp_path / "events.db"
    alert = Alert("event-1", "cam1", 8, "PEEKING", 0.85, 123.0, "snap.jpg", "Head turned 40 deg")
    insert_event(database, alert)
    records = list_events(database, stream_id="cam1", event_type="PEEKING")
    assert len(records) == 1
    assert records[0]["id"] == "event-1"
    assert records[0]["reviewed"] == 0
    assert mark_reviewed(database, "event-1")
    assert list_events(database)[0]["reviewed"] == 1
    assert not mark_reviewed(database, "missing")
