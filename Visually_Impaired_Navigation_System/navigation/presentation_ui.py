"""UI drawing helpers for the polished navigation demo."""

from __future__ import annotations

import textwrap
from typing import Sequence, Tuple

import cv2
import numpy as np

from .assistant_logic import DetectionSummary, GuidanceState


def _danger_color(level: str) -> Tuple[int, int, int]:
    if level == "CRITICAL":
        return (0, 0, 255)
    if level == "HIGH":
        return (0, 120, 255)
    if level == "MEDIUM":
        return (0, 220, 255)
    return (0, 220, 0)


def draw_center_walking_zone(frame: np.ndarray, zone: Tuple[int, int]) -> np.ndarray:
    """Draw translucent center lane overlay."""
    x1, x2 = zone
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, 0), (x2, frame.shape[0] - 1), (80, 220, 80), -1)
    out = cv2.addWeighted(overlay, 0.14, frame, 0.86, 0)
    cv2.rectangle(out, (x1, 0), (x2, frame.shape[0] - 1), (80, 220, 80), 2)
    cv2.putText(
        out,
        "CENTER WALKING ZONE",
        (x1 + 8, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (60, 255, 120),
        2,
        cv2.LINE_AA,
    )
    return out


def draw_detection_overlays(
    frame: np.ndarray,
    summaries: Sequence[DetectionSummary],
    *,
    max_labels: int = 12,
) -> np.ndarray:
    """Overlay boxes, class, confidence, region, and relative distance."""
    out = frame.copy()
    for s in summaries[:max_labels]:
        x1, y1, x2, y2 = map(int, s.det.xyxy.tolist())
        color = _danger_color("HIGH" if s.danger_score > 65 else "MEDIUM")
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = (
            f"{s.det.name} {s.det.conf:.2f} | {s.region} | "
            f"{s.distance.meters:.1f}m/{s.distance.feet:.0f}ft"
        )
        cv2.putText(
            out,
            label,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )
    return out


def _draw_wrapped_text(
    panel: np.ndarray,
    text: str,
    x: int,
    y: int,
    *,
    width_chars: int = 36,
    line_h: int = 30,
    color: Tuple[int, int, int] = (235, 235, 235),
    scale: float = 0.67,
) -> int:
    lines = textwrap.wrap(text, width=width_chars) or [text]
    for line in lines:
        cv2.putText(
            panel,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            2,
            cv2.LINE_AA,
        )
        y += line_h
    return y


def build_assistant_panel(
    frame_h: int,
    panel_w: int,
    state: GuidanceState,
) -> np.ndarray:
    """Build right-side status panel."""
    panel = np.zeros((frame_h, panel_w, 3), dtype=np.uint8)
    panel[:] = (28, 28, 36)

    cv2.putText(
        panel,
        state.assistant_name,
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 235, 120),
        2,
        cv2.LINE_AA,
    )
    cv2.line(panel, (18, 56), (panel_w - 18, 56), (70, 70, 95), 2)

    y = 96
    cv2.putText(
        panel, "Current Guidance", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 210, 255), 2
    )
    y = _draw_wrapped_text(panel, state.guidance_text, 20, y + 30)

    y += 12
    cv2.putText(
        panel, "Danger Level", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (170, 210, 255), 2
    )
    danger_color = _danger_color(state.danger_level)
    cv2.putText(
        panel,
        state.danger_level,
        (20, y + 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        danger_color,
        2,
        cv2.LINE_AA,
    )

    y += 88
    cv2.putText(
        panel,
        "Most Important Obstacle",
        (20, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (170, 210, 255),
        2,
    )
    _draw_wrapped_text(panel, state.key_obstacle, 20, y + 30, color=(240, 240, 240), line_h=28)
    return panel
