"""Load and validate the application YAML configuration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class StreamConfig:
    """Configuration for one video source."""

    id: str
    name: str
    source: str
    loop: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Model paths and inference options."""

    detector: str
    pose: str
    device: str
    imgsz: int
    conf: float
    keypoint_conf: float
    classes: tuple[int, ...]
    process_every_n_frames: int


@dataclass(frozen=True)
class PeekingRuleConfig:
    """Thresholds reserved for the peeking rule engine."""

    yaw_threshold_deg: float
    sustained_seconds: float
    repeat_turns_count: int
    repeat_window_seconds: float
    interruption_tolerance_seconds: float = 0.5


@dataclass(frozen=True)
class DeviceUseRuleConfig:
    """Thresholds reserved for the device-use rule engine."""

    min_conf: float
    sustained_seconds: float
    person_bbox_expansion: float = 0.15


@dataclass(frozen=True)
class RulesConfig:
    """Behavior rule configuration."""

    peeking: PeekingRuleConfig
    device_use: DeviceUseRuleConfig
    cooldown_seconds: float


@dataclass(frozen=True)
class OutputConfig:
    """Output and rendering settings."""

    jpeg_quality: int
    stream_width: int
    annotated_video: str
    snapshot_dir: str
    event_db: str
    reconnect_seconds: float
    metrics_report: str


@dataclass(frozen=True)
class DemoConfig:
    """Lower-cost settings used by the stable demo mode."""

    stream_ids: tuple[str, ...]
    max_streams: int
    stream_width: int
    process_every_n_frames: int
    probe_frames: int
    target_fps: float
    fallback_fps: float
    recording_seconds: int


@dataclass(frozen=True)
class AppConfig:
    """Validated application configuration."""

    streams: tuple[StreamConfig, ...]
    model: ModelConfig
    rules: RulesConfig
    output: OutputConfig
    demo: DemoConfig
    config_path: Path


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    """Return a named mapping section or raise a clear validation error."""
    value = data.get(name)
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section '{name}' must be a mapping")
    return value


def load(path: str | Path = "config.yaml") -> AppConfig:
    """Load YAML from path and validate required fields and ranges."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")

    raw_streams = raw.get("streams")
    if not isinstance(raw_streams, list) or not raw_streams:
        raise ValueError("'streams' must be a non-empty list")
    streams: list[StreamConfig] = []
    ids: set[str] = set()
    for index, item in enumerate(raw_streams):
        if not isinstance(item, dict):
            raise ValueError(f"streams[{index}] must be a mapping")
        missing = [key for key in ("id", "name", "source") if not item.get(key)]
        if missing:
            raise ValueError(f"streams[{index}] is missing: {', '.join(missing)}")
        stream_id = str(item["id"])
        if stream_id in ids:
            raise ValueError(f"Duplicate stream id: {stream_id}")
        ids.add(stream_id)
        streams.append(StreamConfig(stream_id, str(item["name"]), str(item["source"]), bool(item.get("loop", True))))

    model = _section(raw, "model")
    try:
        model_config = ModelConfig(
            detector=str(model["detector"]), pose=str(model["pose"]), device=str(model.get("device", "auto")),
            imgsz=int(model.get("imgsz", 640)), conf=float(model.get("conf", 0.35)),
            keypoint_conf=float(model.get("keypoint_conf", 0.4)),
            classes=tuple(int(value) for value in model.get("classes", [0, 67])),
            process_every_n_frames=int(model.get("process_every_n_frames", 1)),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 'model' configuration: {exc}") from exc
    if model_config.imgsz <= 0 or model_config.process_every_n_frames < 1:
        raise ValueError("model.imgsz and model.process_every_n_frames must be positive")
    if not 0 < model_config.conf <= 1:
        raise ValueError("model.conf must be in (0, 1]")
    if not 0 <= model_config.keypoint_conf <= 1:
        raise ValueError("model.keypoint_conf must be in [0, 1]")

    rules = _section(raw, "rules")
    peeking = _section(rules, "peeking")
    device_use = _section(rules, "device_use")
    try:
        rules_config = RulesConfig(
            peeking=PeekingRuleConfig(float(peeking["yaw_threshold_deg"]), float(peeking["sustained_seconds"]),
                                      int(peeking["repeat_turns_count"]), float(peeking["repeat_window_seconds"]),
                                      float(peeking.get("interruption_tolerance_seconds", 0.5))),
            device_use=DeviceUseRuleConfig(float(device_use["min_conf"]), float(device_use["sustained_seconds"]),
                                           float(device_use.get("person_bbox_expansion", 0.15))),
            cooldown_seconds=float(rules["cooldown_seconds"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 'rules' configuration: {exc}") from exc
    if not 0 < rules_config.device_use.min_conf <= 1:
        raise ValueError("rules.device_use.min_conf must be in (0, 1]")
    if rules_config.device_use.person_bbox_expansion < 0:
        raise ValueError("rules.device_use.person_bbox_expansion cannot be negative")
    if rules_config.peeking.repeat_turns_count < 1:
        raise ValueError("rules.peeking.repeat_turns_count must be positive")
    if min(rules_config.peeking.sustained_seconds, rules_config.peeking.repeat_window_seconds,
           rules_config.device_use.sustained_seconds, rules_config.cooldown_seconds,
           rules_config.peeking.interruption_tolerance_seconds) < 0:
        raise ValueError("Rule durations and cooldown_seconds cannot be negative")

    output = _section(raw, "output")
    try:
        output_config = OutputConfig(int(output.get("jpeg_quality", 70)), int(output.get("stream_width", 640)),
                                     str(output["annotated_video"]), str(output["snapshot_dir"]),
                                     str(output["event_db"]), float(output.get("reconnect_seconds", 2.0)),
                                     str(output.get("metrics_report", "outputs/metrics_report.md")))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 'output' configuration: {exc}") from exc
    if not 0 <= output_config.jpeg_quality <= 100 or output_config.stream_width <= 0:
        raise ValueError("output.jpeg_quality must be 0-100 and output.stream_width must be positive")
    if output_config.reconnect_seconds <= 0:
        raise ValueError("output.reconnect_seconds must be positive")
    demo = raw.get("demo", {})
    if not isinstance(demo, dict):
        raise ValueError("Configuration section 'demo' must be a mapping")
    if not isinstance(demo.get("stream_ids", []), list):
        raise ValueError("demo.stream_ids must be a list")
    try:
        demo_config = DemoConfig(
            stream_ids=tuple(str(value) for value in demo.get("stream_ids", [])),
            max_streams=int(demo.get("max_streams", 4)),
            stream_width=int(demo.get("stream_width", 480)),
            process_every_n_frames=int(demo.get("process_every_n_frames", 3)),
            probe_frames=int(demo.get("probe_frames", 30)),
            target_fps=float(demo.get("target_fps", 30)),
            fallback_fps=float(demo.get("fallback_fps", 25)),
            recording_seconds=int(demo.get("recording_seconds", 120)),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 'demo' configuration: {exc}") from exc
    if min(demo_config.max_streams, demo_config.stream_width, demo_config.process_every_n_frames,
           demo_config.probe_frames, demo_config.target_fps, demo_config.fallback_fps,
           demo_config.recording_seconds) <= 0:
        raise ValueError("demo max_streams, stream_width, process_every_n_frames, and probe_frames must be positive")
    unknown_demo_streams = set(demo_config.stream_ids) - ids
    if unknown_demo_streams:
        raise ValueError(f"demo.stream_ids contains unknown ids: {', '.join(sorted(unknown_demo_streams))}")
    return AppConfig(tuple(streams), model_config, rules_config, output_config, demo_config, config_path)
