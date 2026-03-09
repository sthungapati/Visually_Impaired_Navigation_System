from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from pipeline.detector import YoloDetector
from pipeline.io_utils import (
    RunPaths,
    create_run_paths,
    guess_source_type,
    iter_image_files,
    resolve_path,
)
from pipeline.reporting import write_detections_csv, write_detections_json, write_summary_csv
from pipeline.viz import annotate_frame, annotate_image_file


def parse_classes(value: Optional[str]) -> Optional[Sequence[str]]:
    """Parse comma-separated class ids/names from CLI."""
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 baseline detection pipeline for urban navigation.",
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input folder of images or a single video file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output folder (default: outputs/runs/<timestamp>).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLOv8 weights file (default: yolov8n.pt).",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size (default: 640).",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (default: 0.25).",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="IoU threshold for NMS (default: 0.7).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device, e.g. cpu or cuda:0 (default: auto).",
    )
    parser.add_argument(
        "--save-annotated",
        action="store_true",
        help="Save annotated images and/or video.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        default=None,
        help="Comma-separated class ids or names to filter.",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Optional limit on number of images/frames for quick tests.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose model output.",
    )
    return parser


def run_on_images(
    detector: YoloDetector,
    input_path: Path,
    run_paths: RunPaths,
    max_items: Optional[int],
    save_annotated: bool,
) -> List[Dict[str, Any]]:
    """Process a folder or single image and return detection results."""
    all_items: List[Dict[str, Any]] = []
    image_paths = list(iter_image_files(input_path, max_items=max_items))

    if not image_paths:
        raise RuntimeError(f"No image files found under {input_path}")

    for img_path in image_paths:
        item = detector.infer_image(img_path)
        all_items.append(item)

        if save_annotated:
            out_path = run_paths.annotated_dir / img_path.name
            annotate_image_file(img_path, item.get("detections", []), out_path)

    return all_items


def run_on_video(
    detector: YoloDetector,
    video_path: Path,
    run_paths: RunPaths,
    max_items: Optional[int],
    save_annotated: bool,
) -> List[Dict[str, Any]]:
    """Process a video file frame-by-frame and return detection results."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    all_items: List[Dict[str, Any]] = []
    writer = None
    frame_idx = 0

    try:
        if save_annotated:
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            annotated_path = run_paths.annotated_dir / f"annotated_{video_path.stem}.mp4"
            run_paths.annotated_dir.mkdir(parents=True, exist_ok=True)
            writer = cv2.VideoWriter(str(annotated_path), fourcc, fps, (w, h))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            item = detector.infer_array(
                frame,
                source_type="video",
                source_name=video_path.name,
                frame_index=frame_idx,
            )
            all_items.append(item)

            if save_annotated and writer is not None:
                annotated_frame = annotate_frame(frame, item.get("detections", []))
                writer.write(annotated_frame)

            frame_idx += 1
            if max_items is not None and frame_idx >= max_items:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()

    return all_items


def compute_smoke_stats(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute simple smoke-test statistics over all detections."""
    num_items = len(items)
    class_counter: Counter = Counter()
    total_dets = 0

    for item in items:
        for det in item.get("detections", []):
            class_counter[str(det.get("class_name"))] += 1
            total_dets += 1

    top5 = class_counter.most_common(5)
    return {
        "num_items": num_items,
        "total_detections": total_dets,
        "top_classes": top5,
    }


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry point for the baseline pipeline."""
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    input_path = resolve_path(parsed.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    base_output = resolve_path(parsed.output) if parsed.output else None
    run_paths = create_run_paths(base_output=base_output)

    source_type = guess_source_type(input_path)

    classes = parse_classes(parsed.classes)

    detector = YoloDetector(
        weights=parsed.model,
        imgsz=parsed.imgsz,
        conf=parsed.conf,
        iou=parsed.iou,
        device=parsed.device,
        classes=classes,
        verbose=parsed.verbose,
    )

    if source_type == "image":
        items = run_on_images(
            detector=detector,
            input_path=input_path,
            run_paths=run_paths,
            max_items=parsed.max_items,
            save_annotated=parsed.save_annotated,
        )
    else:
        items = run_on_video(
            detector=detector,
            video_path=input_path,
            run_paths=run_paths,
            max_items=parsed.max_items,
            save_annotated=parsed.save_annotated,
        )

    # Build run metadata and write logs.
    total_detections = sum(len(i.get("detections", [])) for i in items)
    run_metadata: Dict[str, Any] = {
        "run_id": run_paths.run_id,
        "source_type": source_type,
        "input_path": str(input_path),
        "model": parsed.model,
        "imgsz": parsed.imgsz,
        "conf": parsed.conf,
        "iou": parsed.iou,
        "device": parsed.device,
        "class_filter": classes,
        "total_items": len(items),
        "total_detections": total_detections,
    }

    # Save config used for the run.
    config_path = run_paths.run_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(run_metadata, f, indent=2)

    write_detections_csv(items, run_paths.run_id, run_paths.logs_dir)
    write_detections_json(items, run_metadata, run_paths.logs_dir)
    write_summary_csv(items, run_paths.logs_dir)

    # Basic smoke-test stats (especially useful for image folders).
    stats = compute_smoke_stats(items)
    print(f"Processed items: {stats['num_items']}")
    print(f"Total detections: {stats['total_detections']}")
    if stats["top_classes"]:
        print("Top classes by count (name, count):")
        for name, count in stats["top_classes"]:
            print(f"  {name}: {count}")

    print(f"Run outputs saved to: {run_paths.run_dir}")


if __name__ == "__main__":
    main()

