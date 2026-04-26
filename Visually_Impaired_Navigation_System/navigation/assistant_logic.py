"""Rule-based navigation logic for the presentation demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .detector import Det
from .distance import DistanceEstimate, estimate_distance_from_box

# Practical class shortlist for outdoor walking.
OUTDOOR_OBSTACLE_CLASSES: Sequence[str] = (
    "person",
    "bicycle",
    "motorcycle",
    "car",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "bench",
    "dog",
    "fire hydrant",
    "pole",
    "tree",
    "trash can",
)


@dataclass(frozen=True)
class DetectionSummary:
    det: Det
    region: str
    distance: DistanceEstimate
    danger_score: float


@dataclass(frozen=True)
class GuidanceState:
    assistant_name: str
    guidance_text: str
    danger_level: str
    key_obstacle: str
    detections: List[DetectionSummary]


def center_zone_bounds(frame_width: int, ratio: float = 0.34) -> Tuple[int, int]:
    """
    Return [x_left, x_right] bounds for the center walking zone.
    ratio=0.34 means 34% of frame width is treated as the center lane.
    """
    zone_w = int(frame_width * ratio)
    x1 = max(0, (frame_width - zone_w) // 2)
    x2 = min(frame_width - 1, x1 + zone_w)
    return x1, x2


def region_from_box(xyxy, frame_width: int, center_zone: Tuple[int, int]) -> str:
    """Map object center x-position to left/center/right region."""
    cx = float(xyxy[0] + xyxy[2]) / 2.0
    x1, x2 = center_zone
    if cx < x1:
        return "left"
    if cx > x2:
        return "right"
    return "center"


def _danger_score(det: Det, region: str, distance: DistanceEstimate) -> float:
    """
    Simple score (0-100) for obstacle importance in walking guidance.
    Higher score means more urgent.
    """
    base_by_safety = {
        "clear": 20.0,
        "watch": 45.0,
        "caution": 70.0,
        "stop": 90.0,
    }
    score = base_by_safety.get(distance.safety_level, 20.0)

    # Region priority: center obstacles are most dangerous for forward walking.
    if region == "center":
        score += 8.0
    elif region in ("left", "right"):
        score += 3.0

    # Confidence nudges urgency up/down.
    score += (det.conf - 0.5) * 12.0
    return max(0.0, min(100.0, score))


def summarize_detections(
    dets: Iterable[Det],
    frame_w: int,
    frame_h: int,
    *,
    obstacle_classes: Sequence[str] | None = None,
    center_zone_ratio: float = 0.34,
) -> Tuple[List[DetectionSummary], Tuple[int, int]]:
    """Convert raw detections into region+distance+danger summaries."""
    classes = set(c.lower() for c in obstacle_classes) if obstacle_classes else None
    zone = center_zone_bounds(frame_w, ratio=center_zone_ratio)
    out: List[DetectionSummary] = []
    for d in dets:
        if classes and d.name.lower() not in classes:
            continue
        dist = estimate_distance_from_box(d.xyxy, frame_h, frame_w)
        region = region_from_box(d.xyxy, frame_w, zone)
        score = _danger_score(d, region, dist)
        out.append(
            DetectionSummary(
                det=d,
                region=region,
                distance=dist,
                danger_score=score,
            )
        )
    out.sort(key=lambda x: x.danger_score, reverse=True)
    return out, zone


def danger_level_from_score(score: float) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 40:
        return "MEDIUM"
    return "LOW"


def guidance_from_summaries(
    summaries: Sequence[DetectionSummary],
    *,
    assistant_name: str = "NavGuide AI",
) -> GuidanceState:
    """Produce final guidance text + panel metadata."""
    if not summaries:
        return GuidanceState(
            assistant_name=assistant_name,
            guidance_text="You are clear to move forward.",
            danger_level="LOW",
            key_obstacle="None",
            detections=[],
        )

    top = summaries[0]
    top_name = top.det.name.replace("_", " ")
    score = top.danger_score
    danger = danger_level_from_score(score)
    key = f"{top_name} ({top.distance.meters:.1f} m, {top.region})"

    if top.distance.safety_level == "stop" and top.region == "center":
        text = "Stop. Object directly in front of you."
    elif top.distance.safety_level in ("stop", "caution") and top.region == "center":
        text = f"There is a {top_name} ahead. Slow down."
    elif top.distance.safety_level in ("stop", "caution") and top.region == "right":
        text = "Obstacle on your right. Move slightly left."
    elif top.distance.safety_level in ("stop", "caution") and top.region == "left":
        text = "Obstacle on your left. Move slightly right."
    elif top.distance.safety_level == "watch":
        text = f"{top_name.capitalize()} ahead at medium distance. Continue carefully."
    else:
        text = "You are clear to move forward."

    return GuidanceState(
        assistant_name=assistant_name,
        guidance_text=text,
        danger_level=danger,
        key_obstacle=key,
        detections=list(summaries),
    )
