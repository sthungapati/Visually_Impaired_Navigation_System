"""Rough proximity from bounding box size (no real depth — prototype only)."""

from __future__ import annotations

import numpy as np


def proximity_from_box(xyxy: np.ndarray, frame_height: int, frame_width: int) -> str:
    """
    Heuristic: larger normalized box height ~ closer for many street objects.
    Returns a short label for speech.
    """
    _x1, y1, _x2, y2 = xyxy.tolist()
    h = max(0.0, (y2 - y1) / float(frame_height))
    w = max(0.0, (xyxy[2] - xyxy[0]) / float(frame_width))
    area = h * w

    if h > 0.35 or area > 0.12:
        return "very close"
    if h > 0.22 or area > 0.06:
        return "close"
    if h > 0.10 or area > 0.02:
        return "medium distance"
    return "far"
