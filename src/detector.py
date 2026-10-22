"""YOLO object tracking and pose association."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import ModelConfig


@dataclass
class Detection:
    """Detected object with optional person tracking and pose metadata."""

    track_id: int | None
    cls: str
    conf: float
    bbox: tuple[int, int, int, int]
    keypoints: np.ndarray | None = None


def _iou(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> float:
    """Compute intersection over union for two xyxy boxes."""
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    area_b = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


class Detector:
    """Run pretrained YOLO detection/tracking and person pose inference."""

    def __init__(self, config: ModelConfig) -> None:
        """Load configured models; model initialization may download weights."""
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("ultralytics is required; install requirements.txt") from exc
        self.config = config
        self.detector = YOLO(config.detector)
        self.pose_model = YOLO(config.pose)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect and track people/phones, attaching the best matching pose."""
        tracked = self.detector.track(
            frame, persist=True, tracker="bytetrack.yaml", conf=self.config.conf,
            imgsz=self.config.imgsz, classes=list(self.config.classes), device=self._device_arg(),
            verbose=False,
        )[0]
        poses = self.pose_model.predict(
            frame, conf=self.config.conf, imgsz=self.config.imgsz, device=self._device_arg(), verbose=False
        )[0]
        pose_data: list[tuple[tuple[int, int, int, int], np.ndarray]] = []
        if poses.boxes is not None and poses.keypoints is not None:
            pose_boxes = poses.boxes.xyxy.cpu().numpy().astype(int)
            keypoint_data = poses.keypoints.data.cpu().numpy()
            pose_data = [((int(box[0]), int(box[1]), int(box[2]), int(box[3])), keypoint_data[i])
                         for i, box in enumerate(pose_boxes)]

        detections: list[Detection] = []
        if tracked.boxes is None:
            return detections
        boxes = tracked.boxes.xyxy.cpu().numpy().astype(int)
        classes = tracked.boxes.cls.cpu().numpy().astype(int)
        confidences = tracked.boxes.conf.cpu().numpy()
        ids = tracked.boxes.id
        track_ids = ids.cpu().numpy().astype(int) if ids is not None else [None] * len(boxes)
        for box, class_id, confidence, track_id in zip(boxes, classes, confidences, track_ids):
            bbox = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
            label = "person" if class_id == 0 else "cell phone" if class_id == 67 else str(class_id)
            keypoints = None
            if class_id == 0 and pose_data:
                candidate, keypoints = max(pose_data, key=lambda pose: _iou(bbox, pose[0]))
                if _iou(bbox, candidate) == 0:
                    keypoints = None
            detections.append(Detection(None if track_id is None else int(track_id), label,
                                        float(confidence), bbox, keypoints))
        return detections

    def _device_arg(self) -> str | None:
        """Map automatic device selection to Ultralytics defaults."""
        return None if self.config.device == "auto" else self.config.device
