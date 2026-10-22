"""Benchmark configured clips and write an FPS/latency/alert metrics report."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.alerts import AlertManager
from src.config import AppConfig, StreamConfig, load
from src.detector import Detection, Detector
from src.pipeline import _annotate
from src.rules import RuleEngine


@dataclass(frozen=True)
class GroundTruthEvent:
    """One labeled suspicious activity interval from the ground-truth CSV."""

    clip: str
    start_s: float
    end_s: float
    event_type: str


def _local_path(config: AppConfig, source: str) -> Path | None:
    """Resolve a source path or return None for live/webcam inputs."""
    if source.isdigit() or source.lower().startswith(("rtsp://", "http://", "https://")):
        return None
    path = Path(source)
    return path if path.is_absolute() else config.config_path.parent / path


def _read_ground_truth(path: Path | None) -> list[GroundTruthEvent]:
    """Load optional clip/start/end/type annotations from CSV."""
    if path is None:
        return []
    if not path.is_file():
        raise FileNotFoundError(f"Ground-truth CSV not found: {path}")
    events: list[GroundTruthEvent] = []
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        required = {"clip", "start_s", "end_s", "type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Ground-truth CSV must contain clip,start_s,end_s,type columns")
        for row in reader:
            start_s, end_s = float(row["start_s"]), float(row["end_s"])
            if start_s < 0 or end_s < start_s:
                raise ValueError(f"Invalid ground-truth interval for {row['clip']}: {start_s}-{end_s}")
            events.append(GroundTruthEvent(Path(row["clip"]).name.casefold(), start_s, end_s,
                                           row["type"].strip().upper()))
    return events


def _no_snapshot(_frame: np.ndarray, _path: Path, _quality: int) -> None:
    """Skip snapshot writes while retaining the application's cooldown logic."""


def benchmark_clip(config: AppConfig, stream: StreamConfig) -> dict[str, Any]:
    """Measure one clip's throughput, end-to-annotation latency, and alerts."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required; install requirements.txt") from exc
    path = _local_path(config, stream.source)
    if path is None:
        raise ValueError(f"Stream {stream.id} is not a local video clip")
    if not path.is_file():
        raise FileNotFoundError(f"Clip for {stream.id} not found: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open clip: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    source_fps = fps if fps > 0 else config.demo.fallback_fps
    detector = Detector(config.model)
    rules = RuleEngine(config.rules, config.model.keypoint_conf)
    snapshots = Path(config.output.snapshot_dir)
    if not snapshots.is_absolute():
        snapshots = config.config_path.parent / snapshots
    alert_manager = AlertManager(snapshots, config.rules.cooldown_seconds, config.output.jpeg_quality,
                                snapshot_saver=_no_snapshot)
    latencies_ms: list[float] = []
    alert_counts: Counter[str] = Counter()
    alert_events: list[dict[str, Any]] = []
    detections: list[Detection] = []
    frame_count = 0
    frame_index = 0
    clip_started_at = time.time()
    started = time.perf_counter()
    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                break
            captured_at = time.time()
            if frame.shape[1] > config.output.stream_width:
                resized_height = round(frame.shape[0] * config.output.stream_width / frame.shape[1])
                frame = cv2.resize(frame, (config.output.stream_width, resized_height))
            if frame_index % config.model.process_every_n_frames == 0:
                detections = detector.detect(frame)
            video_timestamp = clip_started_at + frame_index / source_fps
            candidates = rules.update(stream.id, detections, video_timestamp)
            annotated = _annotate(frame, detections, config.model.keypoint_conf)
            finished_at = time.time()
            latencies_ms.append(max(0.0, finished_at - captured_at) * 1000)
            for candidate in candidates:
                alert = alert_manager.create(candidate, annotated)
                if alert is not None:
                    alert_counts[alert.type] += 1
                    alert_events.append({
                        "clip": path.name.casefold(),
                        "type": alert.type,
                        "time_s": max(0.0, candidate.timestamp - clip_started_at),
                    })
            frame_count += 1
            frame_index += 1
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    return {
        "stream_id": stream.id,
        "clip": path.name,
        "frames": frame_count,
        "average_fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "p95_latency_ms": float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0,
        "alert_counts": dict(alert_counts),
        "alert_events": alert_events,
    }


def _event_metrics(predictions: list[dict[str, Any]], truth: list[GroundTruthEvent]) -> tuple[float, float, int, int, int]:
    """Greedily match predictions to unique labeled intervals and score them."""
    unmatched = set(range(len(truth)))
    true_positive = 0
    for prediction in predictions:
        match = next((index for index in unmatched
                      if truth[index].clip == prediction["clip"]
                      and truth[index].event_type == prediction["type"]
                      and truth[index].start_s <= prediction["time_s"] <= truth[index].end_s), None)
        if match is not None:
            unmatched.remove(match)
            true_positive += 1
    false_positive = len(predictions) - true_positive
    false_negative = len(truth) - true_positive
    precision = true_positive / len(predictions) if predictions else 0.0
    recall = true_positive / len(truth) if truth else 0.0
    return precision, recall, true_positive, false_positive, false_negative


def _write_report(path: Path, results: list[dict[str, Any]], truth: list[GroundTruthEvent]) -> None:
    """Write aggregate and per-clip benchmark statistics as Markdown."""
    path.parent.mkdir(parents=True, exist_ok=True)
    all_predictions = [event for result in results for event in result["alert_events"]]
    all_counts: Counter[str] = Counter()
    for result in results:
        all_counts.update(result["alert_counts"])
    lines = [
        "# Smart CCTV Benchmark Report", "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}", "",
        "FPS is processed frames divided by benchmark wall time. P95 latency measures frame capture to annotation.", "",
        "## Per-clip results", "",
        "| Stream | Clip | Frames | Average FPS | P95 latency (ms) | Alerts |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        by_type = ", ".join(f"{key}: {value}" for key, value in sorted(result["alert_counts"].items())) or "None"
        lines.append(f"| {result['stream_id']} | {result['clip']} | {result['frames']} | "
                     f"{result['average_fps']:.2f} | {result['p95_latency_ms']:.2f} | {by_type} |")
    lines.extend(["", "## Alert totals", "", "| Type | Count |", "|---|---:|"])
    for event_type, count in sorted(all_counts.items()):
        lines.append(f"| {event_type} | {count} |")
    if truth:
        precision, recall, tp, fp, fn = _event_metrics(all_predictions, truth)
        lines.extend([
            "", "## Ground-truth event scoring", "",
            f"- Precision: {precision:.3f}", f"- Recall: {recall:.3f}",
            f"- Event matches: {tp}; unmatched alerts: {fp}; missed labeled events: {fn}",
            "", "A prediction matches one previously unmatched interval with the same clip and type when its timestamp falls inside that interval.",
        ])
    else:
        lines.extend(["", "## Ground-truth event scoring", "", "Not computed; pass `--ground-truth` with clip,start_s,end_s,type CSV annotations."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config: AppConfig, stream_id: str | None, ground_truth_path: Path | None,
        report_path: Path | None) -> Path:
    """Benchmark all available configured clips and save the report."""
    selected = [stream for stream in config.streams if stream_id is None or stream.id == stream_id]
    if stream_id is not None and not selected:
        raise ValueError(f"Unknown stream id: {stream_id}")
    truth = _read_ground_truth(ground_truth_path)
    results: list[dict[str, Any]] = []
    for stream in selected:
        path = _local_path(config, stream.source)
        if path is None:
            print(f"Skipping live source {stream.id}; benchmark expects recorded clips")
            continue
        if not path.is_file():
            print(f"Skipping missing clip {stream.id}: {path}")
            continue
        result = benchmark_clip(config, stream)
        results.append(result)
        print(f"{stream.id}: {result['average_fps']:.2f} FPS, p95 {result['p95_latency_ms']:.2f} ms")
    if not results:
        raise FileNotFoundError("No configured local clips were available to benchmark")
    output = report_path or Path(config.output.metrics_report)
    if not output.is_absolute():
        output = config.config_path.parent / output
    _write_report(output, results, truth)
    print(f"Metrics report: {output}")
    return output


def main() -> None:
    """Parse command-line options and run the configured clip benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument("--stream", help="Benchmark one configured stream")
    parser.add_argument("--ground-truth", type=Path, help="CSV containing clip,start_s,end_s,type columns")
    parser.add_argument("--output", type=Path, help="Override metrics report path")
    arguments = parser.parse_args()
    try:
        run(load(arguments.config), arguments.stream, arguments.ground_truth, arguments.output)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
