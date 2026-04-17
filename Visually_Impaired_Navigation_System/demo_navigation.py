"""
Webcam or video: detect objects, rough distance, speak summary (rate-limited).

Example:
  python demo_navigation.py --weights runs/train/nav_yolo_v1/weights/best.pt --source 0
  python demo_navigation.py --weights ../yolov8n.pt --source video.mp4
  python demo_navigation.py --source clips --headless --no-speak
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import cv2

_VIS_ROOT = Path(__file__).resolve().parent
if str(_VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIS_ROOT))

from navigation.detector import DetectionEngine
from navigation.distance import proximity_from_box
from navigation.tts import speak

REPO_TEST_CLIP_NAMES = (
    "3987696-hd_1920_1080_24fps.mp4",
    "5744823-uhd_2160_3840_24fps.mp4",
    "6082601-uhd_2160_3840_24fps.mp4",
)


def default_repo_clip_paths(repo: Path) -> List[Path]:
    return [repo / name for name in REPO_TEST_CLIP_NAMES if (repo / name).is_file()]


def run_clip_batch(
    engine: DetectionEngine,
    clip_paths: List[Path],
    *,
    headless: bool,
    speak_updates: bool,
    speak_interval: float,
    frame_stride: int,
    max_frames_per_clip: int | None,
) -> None:
    """Process each clip; print identified objects (and optional TTS)."""
    last_speak = 0.0
    for clip in clip_paths:
        print(f"\n=== {clip.name} ===", flush=True)
        cap = cv2.VideoCapture(str(clip))
        if not cap.isOpened():
            print("  (skip: could not open)", flush=True)
            continue
        class_counts: Counter[str] = Counter()
        frame_i = 0
        processed = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                if frame_i % max(1, frame_stride) != 0:
                    frame_i += 1
                    continue
                if max_frames_per_clip is not None and processed >= max_frames_per_clip:
                    break
                h, w = frame.shape[:2]
                dets = engine.predict_frame(frame)
                for d in dets:
                    class_counts[d.name] += 1
                if dets:
                    primary = max(
                        dets, key=lambda d: (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1])
                    )
                    prox = proximity_from_box(primary.xyxy, h, w)
                    phrase = f"{primary.name.replace('_', ' ')}, {prox}"
                    now = time.monotonic()
                    if speak_updates and now - last_speak >= speak_interval:
                        print(f"  {phrase}", flush=True)
                        speak(phrase)
                        last_speak = now
                    elif not speak_updates:
                        print(f"  frame {frame_i}: {phrase}", flush=True)
                    if not headless:
                        for d in dets:
                            x1, y1, x2, y2 = map(int, d.xyxy.tolist())
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(
                                frame,
                                f"{d.name} {d.conf:.2f}",
                                (x1, max(0, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.5,
                                (0, 255, 0),
                                1,
                            )
                        cv2.imshow("navigation", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            cap.release()
                            if not headless:
                                cv2.destroyAllWindows()
                            return
                processed += 1
                frame_i += 1
        finally:
            cap.release()

        if class_counts:
            unique = ", ".join(sorted(class_counts.keys()))
            print(f"  Objects identified (unique): {unique}", flush=True)
            print("  Detections by class:", flush=True)
            for name, cnt in class_counts.most_common():
                print(f"    {name}: {cnt}", flush=True)
        else:
            print("  Objects identified: (none above confidence threshold)", flush=True)

    if not headless:
        cv2.destroyAllWindows()


def main() -> None:
    vis = Path(__file__).resolve().parent
    repo = vis.parent
    parser = argparse.ArgumentParser(description="Navigation prototype: YOLO + distance + TTS")
    parser.add_argument(
        "--weights",
        type=str,
        default=str(repo / "yolov8n.pt"),
        help="Trained .pt or baseline yolov8n.pt",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help=(
            "0 for default webcam, path to a video file, or 'clips' to batch the "
            "three sample MP4 files in the project root"
        ),
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--speak-interval",
        type=float,
        default=3.0,
        help="Minimum seconds between spoken updates",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Do not show OpenCV window (speech only)",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Do not call text-to-speech (still prints to the console)",
    )
    parser.add_argument(
        "--clip-frame-stride",
        type=int,
        default=15,
        help="With --source clips, run detection every N frames (default: 15)",
    )
    parser.add_argument(
        "--clip-max-frames",
        type=int,
        default=None,
        help="With --source clips, max frames to process per clip (default: all)",
    )
    args = parser.parse_args()

    if args.source.strip().lower() == "clips":
        clip_paths = default_repo_clip_paths(repo)
        if not clip_paths:
            raise RuntimeError(
                "No sample clips in repo root. Expected: " + ", ".join(REPO_TEST_CLIP_NAMES)
            )
        engine = DetectionEngine(args.weights, conf=args.conf, device=args.device)
        run_clip_batch(
            engine,
            clip_paths,
            headless=args.headless,
            speak_updates=not args.no_speak,
            speak_interval=args.speak_interval,
            frame_stride=args.clip_frame_stride,
            max_frames_per_clip=args.clip_max_frames,
        )
        return

    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    engine = DetectionEngine(args.weights, conf=args.conf, device=args.device)
    last_speak = 0.0

    print("Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            dets = engine.predict_frame(frame)
            if dets:
                primary = max(dets, key=lambda d: (d.xyxy[2] - d.xyxy[0]) * (d.xyxy[3] - d.xyxy[1]))
                prox = proximity_from_box(primary.xyxy, h, w)
                phrase = f"{primary.name.replace('_', ' ')}, {prox}"
                now = time.monotonic()
                if now - last_speak >= args.speak_interval:
                    print(phrase)
                    speak(phrase)
                    last_speak = now
                for d in dets:
                    x1, y1, x2, y2 = map(int, d.xyxy.tolist())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"{d.name} {d.conf:.2f}",
                        (x1, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            if not args.headless:
                cv2.imshow("navigation", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
