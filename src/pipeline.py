"""Per-stream inference workers and thread-safe frame/status access."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from src.alerts import AlertManager
from src.config import AppConfig, StreamConfig
from src.db import initialize, insert_event
from src.detector import Detection, Detector
from src.metrics import StreamMetrics
from src.rules import RuleEngine

AlertCallback = Callable[[dict[str, Any]], None]


def _resolved_source(config: AppConfig, source: str) -> str:
    """Resolve local sources relative to the active config file."""
    if source.isdigit() or source.lower().startswith(("rtsp://", "http://", "https://")):
        return source
    path = Path(source)
    return str(path if path.is_absolute() else config.config_path.parent / path)


def _annotate(frame: np.ndarray, detections: list[Detection], keypoint_confidence: float) -> np.ndarray:
    """Render neutral object boxes, IDs, pose keypoints, and yaw estimates."""
    import cv2

    from src.pose import estimate_yaw

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (55, 210, 85) if detection.cls == "person" else (45, 155, 250)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = detection.cls
        if detection.track_id is not None:
            label += f" #{detection.track_id}"
        if detection.cls == "person" and detection.keypoints is not None:
            yaw = estimate_yaw(detection.keypoints, keypoint_confidence)
            label += f" | yaw {yaw:.0f} deg" if yaw is not None else " | yaw n/a"
            for x, y, confidence in detection.keypoints:
                if confidence >= keypoint_confidence:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 180, 20), -1)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)
    return frame


class StreamWorker:
    """Capture, infer, annotate, and publish one configured video stream."""

    def __init__(self, config: AppConfig, stream: StreamConfig, on_alert: AlertCallback) -> None:
        """Create an idle worker; inference models load in its worker thread."""
        self.config = config
        self.stream = stream
        self.on_alert = on_alert
        self.metrics = StreamMetrics()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._jpeg: bytes | None = None
        self._jpeg_sequence = 0
        self._status = "starting"
        self._error: str | None = None

    def start(self) -> None:
        """Launch the inference worker as a daemon thread."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"pipeline-{self.stream.id}", daemon=True)
        self._thread.start()

    def _set_status(self, status: str, error: str | None = None) -> None:
        """Update worker status and optional diagnostic text atomically."""
        with self._lock:
            self._status = status
            self._error = error
            if status != "online":
                self._jpeg = None

    def _run(self) -> None:
        """Run one capture/inference cycle and retry failed sources."""
        while not self._stop.is_set():
            source = None
            try:
                from src.stream import VideoSource

                source = VideoSource(
                    _resolved_source(self.config, self.stream.source),
                    loop=self.stream.loop,
                    reconnect_delay=self.config.output.reconnect_seconds,
                ).start()
                self._set_status("starting")
                detector = Detector(self.config.model)
                rules = RuleEngine(self.config.rules, self.config.model.keypoint_conf)
                snapshots = Path(self.config.output.snapshot_dir)
                database = Path(self.config.output.event_db)
                if not snapshots.is_absolute():
                    snapshots = self.config.config_path.parent / snapshots
                if not database.is_absolute():
                    database = self.config.config_path.parent / database
                alerts = AlertManager(snapshots, self.config.rules.cooldown_seconds,
                                      self.config.output.jpeg_quality)
                initialize(database)
                self._set_status("online")
                last_sequence = 0
                last_frame_at = time.monotonic()
                frame_index = 0
                detections: list[Detection] = []
                while not self._stop.is_set():
                    frame, sequence, captured_at = source.read(after_sequence=last_sequence, timeout=1.0)
                    if frame is None:
                        if not source.capture_alive:
                            is_local_file = not self.stream.source.isdigit() and not self.stream.source.lower().startswith(
                                ("rtsp://", "http://", "https://")
                            )
                            if is_local_file and not self.stream.loop:
                                self._set_status("offline", "Clip ended")
                                return
                            raise RuntimeError("Video source stopped")
                        self._set_status("offline", "Waiting for video frames")
                        if time.monotonic() - last_frame_at >= self.config.output.reconnect_seconds:
                            raise RuntimeError("No frames received; reconnecting")
                        continue
                    last_frame_at = time.monotonic()
                    last_sequence = sequence
                    if frame.shape[1] > self.config.output.stream_width:
                        import cv2

                        height = round(frame.shape[0] * self.config.output.stream_width / frame.shape[1])
                        frame = cv2.resize(frame, (self.config.output.stream_width, height))
                    if frame_index % self.config.model.process_every_n_frames == 0:
                        detections = detector.detect(frame)
                    frame_index += 1
                    candidates = rules.update(self.stream.id, detections, captured_at)
                    annotated = _annotate(frame, detections, self.config.model.keypoint_conf)
                    for candidate in candidates:
                        alert = alerts.create(candidate, annotated)
                        if alert is None:
                            continue
                        insert_event(database, alert)
                        self.metrics.record_alert(alert.type)
                        self.on_alert(alert.to_dict())
                    import cv2

                    encoded, buffer = cv2.imencode(
                        ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.config.output.jpeg_quality]
                    )
                    if not encoded:
                        raise RuntimeError("Could not encode annotated frame")
                    with self._lock:
                        self._jpeg = buffer.tobytes()
                        self._jpeg_sequence += 1
                        self._status = "online"
                        self._error = None
                    self.metrics.record_frame(captured_at)
            except Exception as exc:
                self._set_status("offline", str(exc))
                if self._stop.wait(self.config.output.reconnect_seconds):
                    break
            finally:
                if source is not None:
                    source.stop()
        self._set_status("offline")

    def snapshot(self) -> dict[str, Any]:
        """Return stream metadata and live performance counters."""
        with self._lock:
            status = self._status
            error = self._error
        result = {"id": self.stream.id, "name": self.stream.name, **self.metrics.snapshot(status)}
        if error:
            result["error"] = error
        return result

    def latest_jpeg(self) -> bytes | None:
        """Return the newest encoded frame, if the stream is online."""
        with self._lock:
            return self._jpeg

    def latest_frame(self) -> tuple[int, bytes | None]:
        """Return the current encoded frame and its changing sequence number."""
        with self._lock:
            return self._jpeg_sequence, self._jpeg

    def stop(self) -> None:
        """Signal the worker to stop and wait for its thread to exit."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.config.output.reconnect_seconds + 2.0)


class PipelineManager:
    """Own all configured per-stream workers for the web application."""

    def __init__(self, config: AppConfig) -> None:
        """Create one not-yet-started worker per configured stream."""
        self.config = config
        self.workers: dict[str, StreamWorker] = {}

    def start(self, on_alert: AlertCallback) -> None:
        """Start all configured stream workers."""
        for stream in self.config.streams:
            worker = StreamWorker(self.config, stream, on_alert)
            self.workers[stream.id] = worker
            worker.start()

    def stop(self) -> None:
        """Stop all active stream workers."""
        for worker in self.workers.values():
            worker.stop()

    def get(self, stream_id: str) -> StreamWorker | None:
        """Return a worker by stream id, or None when unknown."""
        return self.workers.get(stream_id)

    def streams(self) -> list[dict[str, Any]]:
        """Return status and FPS records for all configured streams."""
        return [worker.snapshot() for worker in self.workers.values()]

    def metrics(self) -> dict[str, Any]:
        """Return per-stream metrics and aggregate session totals."""
        stream_metrics = {stream_id: worker.snapshot() for stream_id, worker in self.workers.items()}
        counts: dict[str, int] = {}
        for data in stream_metrics.values():
            for alert_type, count in data["alert_counts"].items():
                counts[alert_type] = counts.get(alert_type, 0) + count
        online = [item for item in stream_metrics.values() if item["status"] == "online"]
        return {
            "streams": stream_metrics,
            "average_latency_ms": round(
                sum(item["average_latency_ms"] for item in online) / len(online), 2
            ) if online else 0.0,
            "alert_counts": counts,
        }
