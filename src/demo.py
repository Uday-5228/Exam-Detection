"""Select readable demo clips and derive stable lower-cost settings."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.config import AppConfig, StreamConfig


def _probe_score(config: AppConfig, stream: StreamConfig) -> tuple[int, float, int] | None:
    """Probe a local clip and rank it by read success, frame rate, and size."""
    source = stream.source
    if source.isdigit() or source.lower().startswith(("rtsp://", "http://", "https://")):
        return None
    path = Path(source)
    if not path.is_absolute():
        path = config.config_path.parent / path
    if not path.is_file():
        return None
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv-python is required to probe demo clips") from exc
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return None
        fps = capture.get(cv2.CAP_PROP_FPS)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        readable_frames = 0
        for _ in range(config.demo.probe_frames):
            ok, _frame = capture.read()
            if not ok:
                break
            readable_frames += 1
        if readable_frames == 0:
            return None
        fps_penalty = abs((fps if fps > 0 else config.demo.fallback_fps) - config.demo.target_fps)
        return readable_frames, fps_penalty, width * height
    finally:
        capture.release()


def select_demo_config(config: AppConfig) -> AppConfig:
    """Return a config using the best-readable local clips and demo settings."""
    allowed = set(config.demo.stream_ids)
    candidates = [stream for stream in config.streams if not allowed or stream.id in allowed]
    ranked = [(score, stream) for stream in candidates
              if (score := _probe_score(config, stream)) is not None]
    ranked.sort(key=lambda item: (-item[0][0], item[0][1], item[0][2]))
    selected = tuple(stream for _score, stream in ranked[:config.demo.max_streams])
    if not selected:
        raise FileNotFoundError(
            "No readable local demo clips were found. Add video files under data/clips and configure them in config.yaml."
        )
    model = replace(config.model, process_every_n_frames=config.demo.process_every_n_frames)
    output = replace(config.output, stream_width=config.demo.stream_width)
    return replace(config, streams=selected, model=model, output=output)
