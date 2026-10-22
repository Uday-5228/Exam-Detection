"""Lightweight head-turn estimate from YOLO pose keypoints."""

from __future__ import annotations

import numpy as np


def estimate_yaw(keypoints: np.ndarray | None, min_confidence: float = 0.4) -> float | None:
    """Estimate signed yaw in degrees from nose and shoulder positions."""
    if keypoints is None:
        return None
    points = np.asarray(keypoints)
    if points.ndim != 2 or points.shape[0] < 7 or points.shape[1] < 3:
        return None
    nose, left_shoulder, right_shoulder = points[0], points[5], points[6]
    if min(nose[2], left_shoulder[2], right_shoulder[2]) < min_confidence:
        return None
    shoulder_width = float(np.linalg.norm(left_shoulder[:2] - right_shoulder[:2]))
    if shoulder_width <= 1e-6:
        return None
    shoulder_mid_x = float((left_shoulder[0] + right_shoulder[0]) / 2)
    ratio = (float(nose[0]) - shoulder_mid_x) / shoulder_width
    return float(np.clip(ratio * 90.0, -90.0, 90.0))
