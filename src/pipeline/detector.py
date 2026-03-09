from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from ultralytics import YOLO


class YoloDetector:
    """Wrapper around Ultralytics YOLOv8 for consistent inference outputs."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        device: Optional[str] = None,
        classes: Optional[Sequence[Union[int, str]]] = None,
        verbose: bool = False,
    ) -> None:
        self.weights = weights
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.verbose = verbose

        self.model = YOLO(weights)
        self.device = self._select_device(device)

        # Resolve class filters once model is loaded.
        self.classes = self._resolve_classes(classes)

    def _select_device(self, device: Optional[str]) -> str:
        """Pick a device string for YOLO."""
        if device:
            return device
        if torch.cuda.is_available():
            return "cuda:0"
        return "cpu"

    def _resolve_classes(
        self, classes: Optional[Sequence[Union[int, str]]]
    ) -> Optional[List[int]]:
        """Map class ids or names to a list of ids understood by YOLO."""
        if not classes:
            return None

        name_to_id = {str(v): k for k, v in self.model.names.items()}
        resolved: List[int] = []
        for c in classes:
            if isinstance(c, int):
                resolved.append(c)
            else:
                key = str(c)
                if key in name_to_id:
                    resolved.append(int(name_to_id[key]))
                elif key.isdigit():
                    resolved.append(int(key))
        return resolved or None

    def _run_model(self, image: np.ndarray):
        """Run YOLO on a single image/frame."""
        return self.model(
            image,
            imgsz=self.imgsz,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=self.verbose,
        )[0]

    def infer_image(self, image_path: Union[str, Path]) -> Dict[str, Any]:
        """Run inference on a single image path."""
        from cv2 import imread

        path = Path(image_path)
        img = imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read image: {path}")
        return self.infer_array(
            img,
            source_type="image",
            source_name=path.name,
            frame_index=None,
        )

    def infer_array(
        self,
        image: np.ndarray,
        source_type: str,
        source_name: str,
        frame_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run inference on a numpy array frame and return structured detections."""
        result = self._run_model(image)
        h, w = result.orig_shape

        detections: List[Dict[str, Any]] = []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            xywh = box.xywh[0].tolist()
            class_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            class_name = str(self.model.names.get(class_id, str(class_id)))

            detections.append(
                {
                    "file_name": source_name,
                    "frame_index": frame_index,
                    "img_w": w,
                    "img_h": h,
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": {
                        "x1": float(xyxy[0]),
                        "y1": float(xyxy[1]),
                        "x2": float(xyxy[2]),
                        "y2": float(xyxy[3]),
                    },
                    "bbox_xywh": {
                        "x_center": float(xywh[0]),
                        "y_center": float(xywh[1]),
                        "w": float(xywh[2]),
                        "h": float(xywh[3]),
                    },
                }
            )

        return {
            "source_type": source_type,
            "source_name": source_name,
            "frame_index": frame_index,
            "width": w,
            "height": h,
            "detections": detections,
        }

