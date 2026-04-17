from __future__ import annotations

import argparse
import json
from datetime import datetime
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2

from pipeline.detector import YoloDetector
from pipeline.io_utils import (
    RunPaths,
    create_run_paths,
    guess_source_type,
    iter_image_files,
    iter_video_files,
    resolve_path,
)
from pipeline.reporting import write_detections_csv, write_detections_json, write_summary_csv
from pipeline.viz import annotate_frame, annotate_image_file


REPO_TEST_CLIP_NAMES = (
    "3987696-hd_1920_1080_24fps.mp4",
    "5744823-uhd_2160_3840_24fps.mp4",
    "6082601-uhd_2160_3840_24fps.mp4",
)

NAV_CRITICAL_CLASSES = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
]


def parse_classes(value: Optional[str]) -> Optional[Sequence[str]]:
    """Parse comma-separated class ids/names from CLI."""
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    return parts or None


def repo_root() -> Path:
    """Project root (parent of ``src``)."""
    return Path(__file__).resolve().parent.parent


def default_repo_clip_paths() -> List[Path]:
    """Paths to sample MP4 clips in the repo root, if those files exist."""
    root = repo_root()
    return [root / name for name in REPO_TEST_CLIP_NAMES if (root / name).is_file()]


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="YOLOv8 baseline detection pipeline for urban navigation.",
    )
    src_group = parser.add_mutually_exclusive_group(required=True)
    src_group.add_argument(
        "--input",
        nargs="+",
        type=str,
        help=(
            "One or more paths: image folder, folder of videos, single image, or video file."
        ),
    )
    src_group.add_argument(
        "--repo-clips",
        action="store_true",
        help=(
            "Run on the three sample MP4 files in the project root "
            f"({', '.join(REPO_TEST_CLIP_NAMES)}), if present."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Base output folder (default: outputs/runs/<timestamp>).",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Optional run directory name under output base.",
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
        "--nav-critical",
        action="store_true",
        help=(
            "Use navigation-critical COCO classes: person,bicycle,car,motorcycle,"
            "bus,truck,traffic light,stop sign."
        ),
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
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print progress every N images/frames (default: 25).",
    )
    return parser


def run_on_images(
    detector: YoloDetector,
    input_path: Path,
    run_paths: RunPaths,
    max_items: Optional[int],
    save_annotated: bool,
    progress_every: int,
) -> List[Dict[str, Any]]:
    """Process a folder or single image and return detection results."""
    all_items: List[Dict[str, Any]] = []
    image_paths = list(iter_image_files(input_path, max_items=max_items))

    if not image_paths:
        raise RuntimeError(f"No image files found under {input_path}")

    total = len(image_paths)
    print(f"Starting image inference for {total} item(s)...", flush=True)

    for idx, img_path in enumerate(image_paths, start=1):
        item = detector.infer_image(img_path)
        all_items.append(item)

        if save_annotated:
            out_path = run_paths.annotated_dir / img_path.name
            annotate_image_file(img_path, item.get("detections", []), out_path)

        if idx == 1 or idx % progress_every == 0 or idx == total:
            print(f"[images] processed {idx}/{total}", flush=True)

    return all_items


def run_on_video(
    detector: YoloDetector,
    video_path: Path,
    run_paths: RunPaths,
    max_items: Optional[int],
    save_annotated: bool,
    progress_every: int,
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
            if frame_idx == 1 or frame_idx % progress_every == 0:
                print(f"[video] processed frame {frame_idx}", flush=True)
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
    unique_classes = sorted(class_counter.keys())
    return {
        "num_items": num_items,
        "total_detections": total_dets,
        "top_classes": top5,
        "unique_classes": unique_classes,
    }


def compute_stats_by_source(items: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Smoke stats grouped by ``source_name`` (e.g. per input video)."""
    by_source: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for item in items:
        key = str(item.get("source_name") or "")
        by_source[key].append(item)
    return {name: compute_smoke_stats(sub) for name, sub in sorted(by_source.items())}


def main(args: Optional[Sequence[str]] = None) -> None:
    """Entry point for the baseline pipeline."""
    parser = build_arg_parser()
    parsed = parser.parse_args(args=args)

    if parsed.repo_clips:
        input_paths = default_repo_clip_paths()
        if not input_paths:
            raise FileNotFoundError(
                "No default test clips found in repo root. Expected one or more of: "
                + ", ".join(REPO_TEST_CLIP_NAMES)
            )
    else:
        input_paths = [resolve_path(p) for p in parsed.input]
        for p in input_paths:
            if not p.exists():
                raise FileNotFoundError(f"Input path does not exist: {p}")

    base_output = resolve_path(parsed.output) if parsed.output else None
    run_paths = create_run_paths(base_output=base_output, run_name=parsed.run_name)

    classes = parse_classes(parsed.classes)
    if parsed.nav_critical:
        classes = NAV_CRITICAL_CLASSES

    run_timestamp = datetime.now().isoformat(timespec="seconds")
    detector = YoloDetector(
        weights=parsed.model,
        imgsz=parsed.imgsz,
        conf=parsed.conf,
        iou=parsed.iou,
        device=parsed.device,
        classes=classes,
        verbose=parsed.verbose,
    )

    items: List[Dict[str, Any]] = []
    source_types_seen: List[str] = []
    ran_video_input = False
    max_items = parsed.max_items
    progress_every = max(1, parsed.progress_every)

    for input_path in input_paths:
        source_type = guess_source_type(input_path)
        source_types_seen.append(source_type)
        print(f"\n=== Input: {input_path} ({source_type}) ===", flush=True)

        if source_type == "image":
            items.extend(
                run_on_images(
                    detector=detector,
                    input_path=input_path,
                    run_paths=run_paths,
                    max_items=max_items,
                    save_annotated=parsed.save_annotated,
                    progress_every=progress_every,
                )
            )
        elif source_type == "video":
            ran_video_input = True
            items.extend(
                run_on_video(
                    detector=detector,
                    video_path=input_path,
                    run_paths=run_paths,
                    max_items=max_items,
                    save_annotated=parsed.save_annotated,
                    progress_every=progress_every,
                )
            )
        elif source_type == "video_dir":
            ran_video_input = True
            video_paths = list(iter_video_files(input_path))
            if not video_paths:
                raise RuntimeError(f"No video files under {input_path}")
            for vpath in video_paths:
                print(f"\n--- Video: {vpath.name} ---", flush=True)
                items.extend(
                    run_on_video(
                        detector=detector,
                        video_path=vpath,
                        run_paths=run_paths,
                        max_items=max_items,
                        save_annotated=parsed.save_annotated,
                        progress_every=progress_every,
                    )
                )
        else:
            raise ValueError(f"Unsupported source type: {source_type}")

    # Build run metadata and write logs.
    total_detections = sum(len(i.get("detections", [])) for i in items)
    run_metadata: Dict[str, Any] = {
        "run_id": run_paths.run_id,
        "timestamp": run_timestamp,
        "run_timestamp": run_timestamp,
        "source_types": source_types_seen,
        "input_paths": [str(p) for p in input_paths],
        "output_path": str(run_paths.run_dir),
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
    print("\n=== Overall ===", flush=True)
    print(f"Processed items: {stats['num_items']}")
    print(f"Total detections: {stats['total_detections']}")
    if stats["unique_classes"]:
        print(f"Object classes identified (unique): {', '.join(stats['unique_classes'])}")
    if stats["top_classes"]:
        print("Top classes by count (name, count):")
        for name, count in stats["top_classes"]:
            print(f"  {name}: {count}")

    by_src = compute_stats_by_source(items)
    if ran_video_input or len(input_paths) > 1:
        print("\n=== By video / source file ===", flush=True)
        for src_name, s in by_src.items():
            label = src_name or "(unknown)"
            print(f"\n{label}:", flush=True)
            print(f"  Frames/items: {s['num_items']}", flush=True)
            if s["unique_classes"]:
                print(f"  Objects identified: {', '.join(s['unique_classes'])}", flush=True)
            else:
                print("  Objects identified: (none above confidence threshold)", flush=True)
            if s["top_classes"]:
                print("  Top by count:", flush=True)
                for cname, count in s["top_classes"]:
                    print(f"    {cname}: {count}", flush=True)

    print(f"\nRun outputs saved to: {run_paths.run_dir}")


if __name__ == "__main__":
    main()

