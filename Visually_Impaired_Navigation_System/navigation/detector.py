"""Load YOLOv8 and run inference on a single BGR frame."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
from ultralytics import YOLO


@dataclass
class Det:
    name: str
    conf: float
    xyxy: np.ndarray  # shape (4,) float


class DetectionEngine:
    def __init__(
        self,
        weights: str,
        conf: float = 0.25,
        device: Optional[str] = None,
    ) -> None:
        self.model = YOLO(weights)
        self.conf = conf
        self.device = device

    def predict_frame(self, frame: np.ndarray) -> List[Det]:
        """Return detections for one OpenCV BGR frame."""
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "verbose": False,
        }
        if self.device is not None:
            kwargs["device"] = self.device
        r = self.model.predict(frame, **kwargs)[0]
        out: List[Det] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        names = r.names
        for b in r.boxes:
            cid = int(b.cls[0].item())
            cf = float(b.conf[0].item())
            xyxy = b.xyxy[0].cpu().numpy()
            out.append(Det(name=str(names.get(cid, str(cid))), conf=cf, xyxy=xyxy))
        return out
