"""Synthetic keypoint checks for head-turn estimation."""

import numpy as np

from src.pose import estimate_yaw


def _keypoints(yaw: float) -> np.ndarray:
    """Create valid synthetic nose and shoulder points for a yaw value."""
    points = np.zeros((17, 3), dtype=float)
    points[:, 2] = 0.9
    points[0] = [50 + yaw * 20 / 90, 40, 0.9]
    points[5] = [40, 60, 0.9]
    points[6] = [60, 60, 0.9]
    return points


def test_estimate_yaw_maps_offset_to_degrees() -> None:
    """The configured nose offset maps to a signed angle estimate."""
    assert np.isclose(estimate_yaw(_keypoints(45)), 45.0)
    assert np.isclose(estimate_yaw(_keypoints(-30)), -30.0)


def test_estimate_yaw_ignores_low_confidence_or_missing_points() -> None:
    """Low-confidence required points and malformed inputs return None."""
    points = _keypoints(45)
    points[0, 2] = 0.2
    assert estimate_yaw(points, min_confidence=0.4) is None
    assert estimate_yaw(None) is None
    assert estimate_yaw(np.zeros((3, 2))) is None
