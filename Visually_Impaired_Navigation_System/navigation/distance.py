"""Heuristic distance helpers from a detection bounding box."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class DistanceEstimate:
    meters: float
    feet: float
    proximity_label: str
    safety_level: str


def estimate_distance_from_box(
    xyxy: np.ndarray,
    frame_height: int,
    frame_width: int,
) -> DistanceEstimate:
    """
    Approximate object distance from box size.

    This is a monocular heuristic, not true depth. It is good enough for
    "warn vs keep walking" behavior in a prototype assistant.
    """
    _x1, y1, _x2, y2 = xyxy.tolist()
    h = max(0.0, (y2 - y1) / float(frame_height))
    w = max(0.0, (xyxy[2] - xyxy[0]) / float(frame_width))
    area = h * w

    # Calibrated buckets for walking-assistant feedback.
    if h > 0.38 or area > 0.13:
        meters = 0.8
        label = "very close"
        safety = "stop"
    elif h > 0.26 or area > 0.07:
        meters = 1.5
        label = "close"
        safety = "caution"
    elif h > 0.14 or area > 0.03:
        meters = 2.8
        label = "medium distance"
        safety = "watch"
    else:
        meters = 5.0
        label = "far"
        safety = "clear"

    feet = meters * 3.28084
    return DistanceEstimate(
        meters=meters,
        feet=feet,
        proximity_label=label,
        safety_level=safety,
    )


def proximity_from_box(xyxy: np.ndarray, frame_height: int, frame_width: int) -> str:
    """Backward-compatible shorthand returning only the label."""
    return estimate_distance_from_box(xyxy, frame_height, frame_width).proximity_label
