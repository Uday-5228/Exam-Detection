"""Alert persistence, cooldown enforcement, and snapshot saving."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from src.rules import AlertCandidate


@dataclass(frozen=True)
class Alert:
    """Persistable suspicious-activity alert."""

    id: str
    stream_id: str
    track_id: int | None
    type: str
    confidence: float
    timestamp: float
    snapshot_path: str
    detail: str

    def to_dict(self) -> dict[str, str | int | float | None]:
        """Return JSON-compatible alert fields."""
        return {
            "id": self.id, "stream_id": self.stream_id, "track_id": self.track_id,
            "type": self.type, "confidence": self.confidence, "timestamp": self.timestamp,
            "snapshot_path": self.snapshot_path, "detail": self.detail,
        }


SnapshotSaver = Callable[[np.ndarray, Path, int], None]


def _save_jpeg(frame: np.ndarray, path: Path, quality: int) -> None:
    """Encode one frame as a JPEG snapshot using OpenCV."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to save alert snapshots") from exc
    ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError(f"Unable to encode alert snapshot: {path}")
    path.write_bytes(encoded.tobytes())


class AlertManager:
    """Apply alert cooldowns and save a snapshot for each accepted alert."""

    def __init__(self, snapshot_dir: str | Path, cooldown_seconds: float, jpeg_quality: int = 70,
                 snapshot_saver: SnapshotSaver = _save_jpeg) -> None:
        """Configure snapshot storage and per-stream/track/type cooldown."""
        self.snapshot_dir = Path(snapshot_dir)
        self.cooldown_seconds = cooldown_seconds
        self.jpeg_quality = jpeg_quality
        self.snapshot_saver = snapshot_saver
        self._last_emitted: dict[tuple[str, int | None, str], float] = {}

    def create(self, candidate: AlertCandidate, frame: np.ndarray) -> Alert | None:
        """Save and return an alert unless its cooldown is still active."""
        key = (candidate.stream_id, candidate.track_id, candidate.type)
        last_timestamp = self._last_emitted.get(key)
        if last_timestamp is not None and candidate.timestamp - last_timestamp < self.cooldown_seconds:
            return None
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        alert_id = str(uuid.uuid4())
        filename = f"{candidate.stream_id}_{candidate.type.lower()}_{alert_id}.jpg"
        snapshot_path = self.snapshot_dir / filename
        self.snapshot_saver(frame, snapshot_path, self.jpeg_quality)
        self._last_emitted[key] = candidate.timestamp
        return Alert(alert_id, candidate.stream_id, candidate.track_id, candidate.type, candidate.confidence,
                     candidate.timestamp, str(snapshot_path), candidate.detail)
