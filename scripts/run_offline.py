"""Annotate a configured video source and report achieved processing FPS."""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from src.config import AppConfig, StreamConfig, load
from src.detector import Detection, Detector
from src.pose import estimate_yaw
from src.stream import VideoSource


def _draw(frame: cv2.Mat, detections: list[Detection], keypoint_conf: float) -> cv2.Mat:
    """Draw object boxes, track labels, and available person keypoints."""
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        color = (40, 210, 70) if detection.cls == "person" else (40, 150, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = detection.cls
        if detection.track_id is not None:
            label += f" #{detection.track_id}"
        if detection.cls == "person" and detection.keypoints is not None:
            yaw = estimate_yaw(detection.keypoints, min_confidence=keypoint_conf)
            label += f" | yaw {yaw:.0f} deg" if yaw is not None else " | yaw n/a"
            for x, y, confidence in detection.keypoints:
                if confidence >= keypoint_conf:
                    cv2.circle(frame, (int(x), int(y)), 2, (255, 180, 20), -1)
        cv2.putText(frame, label, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    color, 2, cv2.LINE_AA)
    return frame


def run(config: AppConfig, stream: StreamConfig, source: str | None, output: str | None) -> None:
    """Process one configured source and save its annotated video."""
    actual_source = source or stream.source
    if not actual_source.lower().startswith(("rtsp://", "http://", "https://")):
        source_path = Path(actual_source)
        if not source_path.is_absolute():
            actual_source = str((config.config_path.parent / source_path).resolve())
    output_path = Path(output or config.output.annotated_video)
    if not output_path.is_absolute():
        output_path = config.config_path.parent / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    detector = Detector(config.model)
    processed = 0
    frame_index = 0
    last_sequence = 0
    last_detections: list[Detection] = []
    started = time.perf_counter()
    writer: cv2.VideoWriter | None = None
    with VideoSource(actual_source, loop=False, latest_only=False) as video:
        try:
            while True:
                frame, sequence, _captured_at = video.read(after_sequence=last_sequence, timeout=5.0)
                if frame is None:
                    break
                last_sequence = sequence
                if frame_index % config.model.process_every_n_frames == 0:
                    last_detections = detector.detect(frame)
                    processed += 1
                annotated = _draw(frame, last_detections, config.model.keypoint_conf)
                if writer is None:
                    height, width = annotated.shape[:2]
                    fps = video.fps if video.fps > 0 else 25.0
                    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                             (width, height))
                    if not writer.isOpened():
                        raise RuntimeError(f"Unable to create output video: {output_path}")
                writer.write(annotated)
                frame_index += 1
        finally:
            if writer is not None:
                writer.release()
    elapsed = time.perf_counter() - started
    fps = processed / elapsed if elapsed > 0 else 0.0
    print(f"Processed {processed} inference frames from {frame_index} captured frames in {elapsed:.2f}s ({fps:.2f} inference FPS)")
    print(f"Annotated video: {output_path}")


def main() -> None:
    """Parse command-line arguments and launch offline inference."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="YAML configuration path")
    parser.add_argument("--stream", help="Configured stream id; defaults to the first stream")
    parser.add_argument("--source", help="Override source with a file path, RTSP URL, or webcam index")
    parser.add_argument("--output", help="Override annotated video path")
    args = parser.parse_args()
    config = load(args.config)
    stream = next((item for item in config.streams if item.id == args.stream), None) if args.stream else config.streams[0]
    if stream is None:
        parser.error(f"Unknown stream id: {args.stream}")
    run(config, stream, args.source, args.output)


if __name__ == "__main__":
    main()
