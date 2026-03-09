from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


Color = Tuple[int, int, int]


def _draw_single_box(
    img: np.ndarray,
    det: Dict[str, Any],
    color: Color = (0, 255, 0),
    thickness: int = 2,
) -> None:
    """Draw a single bounding box and label on an image."""
    xyxy = det["bbox_xyxy"]
    x1, y1, x2, y2 = (
        int(xyxy["x1"]),
        int(xyxy["y1"]),
        int(xyxy["x2"]),
        int(xyxy["y2"]),
    )
    label = f"{det.get('class_name', '')} {det.get('confidence', 0.0):.2f}"

    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    cv2.putText(
        img,
        label,
        (x1, max(0, y1 - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )


def annotate_image_file(
    image_path: Path,
    detections: Iterable[Dict[str, Any]],
    output_path: Path,
    color: Color = (0, 255, 0),
) -> bool:
    """Annotate a single image on disk and save to output_path."""
    img = cv2.imread(str(image_path))
    if img is None:
        return False

    for det in detections:
        _draw_single_box(img, det, color=color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), img)
    return bool(ok)


def annotate_frame(
    frame: np.ndarray,
    detections: Iterable[Dict[str, Any]],
    color: Color = (0, 255, 0),
) -> np.ndarray:
    """Return an annotated copy of a video frame."""
    annotated = frame.copy()
    for det in detections:
        _draw_single_box(annotated, det, color=color)
    return annotated

