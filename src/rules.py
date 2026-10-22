"""Temporal rules for suspicious head turns and phone use."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from src.config import RulesConfig
from src.detector import Detection
from src.pose import estimate_yaw


@dataclass(frozen=True)
class AlertCandidate:
    """Alert facts produced by a rule before snapshot persistence."""

    stream_id: str
    track_id: int | None
    type: str
    confidence: float
    timestamp: float
    detail: str


@dataclass
class _TrackState:
    """Temporal state kept for one stream and track pair."""

    turn_started_at: float | None = None
    turn_low_since: float | None = None
    turn_episodes: deque[float] = field(default_factory=deque)
    turn_alerted: bool = False
    device_started_at: float | None = None
    device_alerted: bool = False
    last_seen_at: float = 0.0


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Return intersection over union for xyxy boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def _phone_overlaps_person(phone: Detection, person: Detection, expansion: float) -> bool:
    """Check box overlap or phone center inside a 15 percent expanded person box."""
    if _iou(phone.bbox, person.bbox) > 0:
        return True
    px1, py1, px2, py2 = person.bbox
    phone_x1, phone_y1, phone_x2, phone_y2 = phone.bbox
    margin_x = (px2 - px1) * expansion
    margin_y = (py2 - py1) * expansion
    center_x = (phone_x1 + phone_x2) / 2
    center_y = (phone_y1 + phone_y2) / 2
    return px1 - margin_x <= center_x <= px2 + margin_x and py1 - margin_y <= center_y <= py2 + margin_y


class RuleEngine:
    """Evaluate per-stream tracked behavior into alert candidates."""

    def __init__(self, config: RulesConfig, keypoint_confidence: float = 0.4) -> None:
        """Create rule state using validated YAML thresholds."""
        self.config = config
        self.keypoint_confidence = keypoint_confidence
        self._states: dict[tuple[str, int], _TrackState] = {}

    def update(self, stream_id: str, detections: list[Detection], timestamp: float) -> list[AlertCandidate]:
        """Update temporal state and return newly triggered candidates."""
        people = [item for item in detections if item.cls == "person" and item.track_id is not None]
        phones = [item for item in detections if item.cls == "cell phone"
                  and item.conf >= self.config.device_use.min_conf]
        present: set[int] = set()
        alerts: list[AlertCandidate] = []
        for person in people:
            track_id = int(person.track_id)
            present.add(track_id)
            state = self._states.setdefault((stream_id, track_id), _TrackState())
            state.last_seen_at = timestamp
            yaw = estimate_yaw(person.keypoints, self.keypoint_confidence)
            turn_alert = self._update_turn(stream_id, track_id, state, yaw, person.conf, timestamp)
            if turn_alert is not None:
                alerts.append(turn_alert)
            matching_phone = next((phone for phone in phones if _phone_overlaps_person(
                phone, person, self.config.device_use.person_bbox_expansion)), None)
            device_alert = self._update_device(stream_id, track_id, state, matching_phone, timestamp)
            if device_alert is not None:
                alerts.append(device_alert)

        for (state_stream, track_id), state in tuple(self._states.items()):
            if state_stream != stream_id:
                continue
            if track_id not in present:
                self._update_turn(stream_id, track_id, state, None, 0.0, timestamp)
                self._update_device(stream_id, track_id, state, None, timestamp)
            if timestamp - state.last_seen_at > self.config.peeking.repeat_window_seconds:
                del self._states[(state_stream, track_id)]
        return alerts

    def _update_turn(self, stream_id: str, track_id: int, state: _TrackState, yaw: float | None,
                     confidence: float, timestamp: float) -> AlertCandidate | None:
        """Advance one track's turn episode and evaluate both peeking rules."""
        peeking = self.config.peeking
        is_turn = yaw is not None and abs(yaw) > peeking.yaw_threshold_deg
        if is_turn:
            if state.turn_started_at is None:
                state.turn_started_at = timestamp
                state.turn_alerted = False
                state.turn_episodes.append(timestamp)
            state.turn_low_since = None
            while state.turn_episodes and timestamp - state.turn_episodes[0] > peeking.repeat_window_seconds:
                state.turn_episodes.popleft()
            elapsed = timestamp - state.turn_started_at
            repeated = len(state.turn_episodes) >= peeking.repeat_turns_count
            sustained = elapsed >= peeking.sustained_seconds
            if not state.turn_alerted and (sustained or repeated):
                state.turn_alerted = True
                if sustained:
                    detail = f"Head turned {abs(yaw):.0f} deg for {elapsed:.1f} s"
                else:
                    detail = f"Repeated head turns: {len(state.turn_episodes)} episodes within {peeking.repeat_window_seconds:.0f} s"
                return AlertCandidate(stream_id, track_id, "PEEKING", confidence, timestamp, detail)
        elif state.turn_started_at is not None:
            if state.turn_low_since is None:
                state.turn_low_since = timestamp
            elif timestamp - state.turn_low_since > peeking.interruption_tolerance_seconds:
                state.turn_started_at = None
                state.turn_low_since = None
                state.turn_alerted = False
        return None

    def _update_device(self, stream_id: str, track_id: int, state: _TrackState,
                       phone: Detection | None, timestamp: float) -> AlertCandidate | None:
        """Advance phone overlap timing and emit one sustained device-use alert."""
        if phone is None:
            state.device_started_at = None
            state.device_alerted = False
            return None
        if state.device_started_at is None:
            state.device_started_at = timestamp
            state.device_alerted = False
        elapsed = timestamp - state.device_started_at
        if not state.device_alerted and elapsed >= self.config.device_use.sustained_seconds:
            state.device_alerted = True
            detail = f"Phone detected overlapping this student for {elapsed:.1f} s"
            return AlertCandidate(stream_id, track_id, "DEVICE_USE", phone.conf, timestamp, detail)
        return None
