"""Thread-safe live performance and alert metrics."""

from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any


class StreamMetrics:
    """Track frame throughput, processing latency, and alert totals."""

    def __init__(self) -> None:
        """Initialize counters for one stream worker."""
        self._lock = threading.Lock()
        self._started_at: float | None = None
        self._processed_frames = 0
        self._latency_total = 0.0
        self._fps = 0.0
        self._alert_counts: Counter[str] = Counter()

    def record_frame(self, captured_at: float, annotated_at: float | None = None) -> None:
        """Record a completed annotated frame and capture-to-output latency."""
        finished_at = annotated_at if annotated_at is not None else time.time()
        with self._lock:
            if self._started_at is None:
                self._started_at = finished_at
            self._processed_frames += 1
            self._latency_total += max(0.0, finished_at - captured_at)
            elapsed = finished_at - self._started_at
            if elapsed > 0 and self._processed_frames > 1:
                self._fps = (self._processed_frames - 1) / elapsed

    def record_alert(self, alert_type: str) -> None:
        """Increment a stream's alert count for the supplied event type."""
        with self._lock:
            self._alert_counts[alert_type] += 1

    def snapshot(self, status: str = "offline") -> dict[str, Any]:
        """Return a consistent JSON-ready copy of metrics for this stream."""
        with self._lock:
            average_latency = self._latency_total / self._processed_frames if self._processed_frames else 0.0
            return {
                "status": status,
                "fps": round(self._fps, 2),
                "average_latency_ms": round(average_latency * 1000, 2),
                "processed_frames": self._processed_frames,
                "alert_counts": dict(self._alert_counts),
            }
