"""Synthetic timeline checks for behavior rules and alert cooldown."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from src.alerts import AlertManager
from src.config import DeviceUseRuleConfig, PeekingRuleConfig, RulesConfig
from src.detector import Detection
from src.rules import AlertCandidate, RuleEngine


def _rules() -> RulesConfig:
    """Return short test thresholds for readable synthetic timelines."""
    return RulesConfig(PeekingRuleConfig(35, 3, 3, 30), DeviceUseRuleConfig(0.35, 1.5), 10)


def _person(yaw: float | None, track_id: int = 1) -> Detection:
    """Build a tracked person with keypoints encoding the requested yaw."""
    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.9
    points[5] = [40, 60, 0.9]
    points[6] = [60, 60, 0.9]
    points[0] = [50 + (yaw or 0) * 20 / 90, 40, 0.9]
    if yaw is None:
        points[0, 2] = 0.0
    return Detection(track_id, "person", 0.9, (0, 0, 100, 100), points)


def test_sustained_turn_fires_once() -> None:
    """A continuous turn reaches the configured duration and emits once."""
    engine = RuleEngine(_rules())
    results = [engine.update("cam1", [_person(50)], t) for t in (0, 1, 2, 3, 4)]
    alerts = [alert for batch in results for alert in batch]
    assert len(alerts) == 1
    assert alerts[0].type == "PEEKING"
    assert "3.0 s" in alerts[0].detail


def test_brief_glance_does_not_fire() -> None:
    """A short look toward a neighbor stays below the sustained threshold."""
    engine = RuleEngine(_rules())
    results = [engine.update("cam1", [_person(50 if t < 0.4 else 0)], t)
               for t in (0, 0.2, 0.4, 0.6, 1, 2, 3)]
    assert not [alert for batch in results for alert in batch]


def test_cooldown_suppresses_duplicate_alerts(tmp_path: Path) -> None:
    """A second same-key alert is suppressed until cooldown expires."""
    def fake_saver(_frame, path, _quality):
        """Create a placeholder snapshot for the test manager."""
        path.write_bytes(b"snapshot")

    manager = AlertManager(tmp_path, cooldown_seconds=10, snapshot_saver=fake_saver)
    candidate = lambda timestamp: AlertCandidate("cam1", 1, "PEEKING", 0.9, timestamp, "test detail")
    assert manager.create(candidate(1), np.zeros((1, 1, 3), dtype=np.uint8)) is not None
    assert manager.create(candidate(5), np.zeros((1, 1, 3), dtype=np.uint8)) is None
    assert manager.create(candidate(11), np.zeros((1, 1, 3), dtype=np.uint8)) is not None


def test_phone_overlapping_person_fires_device_use() -> None:
    """A sustained phone box inside the person box emits a DEVICE_USE candidate."""
    engine = RuleEngine(_rules())
    person = _person(0)
    phone = Detection(None, "cell phone", 0.8, (70, 70, 85, 90))
    results = [engine.update("cam1", [person, phone], t) for t in (0, 1, 1.5, 1.6)]
    alerts = [alert for batch in results for alert in batch]
    assert len(alerts) == 1
    assert alerts[0].type == "DEVICE_USE"
    assert alerts[0].track_id == 1
