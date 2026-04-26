"""
Webcam or video: detect the front-most object and speak updates.

Example:
  python demo_navigation.py --weights runs/train/nav_yolo_v1/weights/best.pt --source 0
  python demo_navigation.py --weights ../yolov8n.pt --source video.mp4 --front-only
  python demo_navigation.py --whatsapp-clip --headless
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List, Optional

import cv2

_VIS_ROOT = Path(__file__).resolve().parent
if str(_VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIS_ROOT))

from navigation.detector import DetectionEngine
from navigation.distance import estimate_distance_from_box
from navigation.tts import speak

REPO_TEST_CLIP_NAMES = (
    "3987696-hd_1920_1080_24fps.mp4",
    "5744823-uhd_2160_3840_24fps.mp4",
    "6082601-uhd_2160_3840_24fps.mp4",
)

WHATSAPP_CLIP_CANDIDATES = (
    "WhatsApp Video 2026-04-25 at 8.26.06 PM.mp4",
    "Whats App Video 2026-04-25 at 8.26.06 PM.mp4",
)


def default_repo_clip_paths(repo: Path) -> List[Path]:
    return [repo / name for name in REPO_TEST_CLIP_NAMES if (repo / name).is_file()]


def default_whatsapp_clip_path(repo: Path) -> Optional[Path]:
    for name in WHATSAPP_CLIP_CANDIDATES:
        p = repo / name
        if p.is_file():
            return p
    return None


def _bbox_area(xyxy) -> float:
    return max(0.0, float(xyxy[2] - xyxy[0])) * max(0.0, float(xyxy[3] - xyxy[1]))


def _center_offset_score(xyxy, frame_w: int, frame_h: int) -> float:
    cx = float(xyxy[0] + xyxy[2]) / 2.0
    cy = float(xyxy[1] + xyxy[3]) / 2.0
    dx = abs(cx - (frame_w / 2.0)) / max(1.0, frame_w / 2.0)
    dy = abs(cy - (frame_h / 2.0)) / max(1.0, frame_h / 2.0)
    return (dx * dx + dy * dy) ** 0.5


def pick_front_object(dets, frame_w: int, frame_h: int):
    """
    Select one detection as the object "right in front" of the user.
    Preference: bigger object near image center.
    """
    return max(
        dets,
        key=lambda d: (_bbox_area(d.xyxy) / max(0.20, _center_offset_score(d.xyxy, frame_w, frame_h))),
    )


def _natural_phrase_for_object(class_name: str, distance_m: float, distance_ft: float, safety: str) -> str:
    obj = class_name.replace("_", " ")
    if safety == "stop":
        return f"Stop. {obj} ahead at about {distance_m:.1f} meters, {distance_ft:.0f} feet."
    if safety == "caution":
        return f"Caution. {obj} ahead at around {distance_m:.1f} meters, {distance_ft:.0f} feet."
    if safety == "watch":
        return f"{obj} in front at about {distance_m:.1f} meters. Keep walking carefully."
    return "Path looks clear ahead. Keep walking, you are doing good."


def _progress_phrase() -> str:
    return "No nearby obstacle in front. Keep walking, you are doing good."


def run_clip_batch(
    engine: DetectionEngine,
    clip_paths: List[Path],
    *,
    headless: bool,
    speak_updates: bool,
    speak_interval: float,
    front_only: bool,
    frame_stride: int,
    max_frames_per_clip: int | None,
) -> None:
    """Process each clip; print identified objects (and optional TTS)."""
    last_speak = 0.0
    last_phrase = ""
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
                phrase = _progress_phrase()
                if dets:
                    primary = (
                        pick_front_object(dets, w, h)
                        if front_only
                        else max(dets, key=lambda d: _bbox_area(d.xyxy))
                    )
                    dist = estimate_distance_from_box(primary.xyxy, h, w)
                    phrase = _natural_phrase_for_object(
                        primary.name, dist.meters, dist.feet, dist.safety_level
                    )
                    now = time.monotonic()
                    if speak_updates and now - last_speak >= speak_interval:
                        if phrase != last_phrase or dist.safety_level in ("stop", "caution"):
                            print(f"  {phrase}", flush=True)
                            speak(phrase)
                            last_speak = now
                            last_phrase = phrase
                    elif not speak_updates:
                        print(f"  frame {frame_i}: {phrase}", flush=True)
                else:
                    now = time.monotonic()
                    if speak_updates and now - last_speak >= speak_interval:
                        if phrase != last_phrase:
                            print(f"  {phrase}", flush=True)
                            speak(phrase)
                            last_speak = now
                            last_phrase = phrase
                    elif not speak_updates:
                        print(f"  frame {frame_i}: {phrase}", flush=True)

                if dets and not headless:
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
                    if not headless:
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
        default="whatsapp",
        help=(
            "0 for default webcam, path to a video file, 'whatsapp' for repo WhatsApp "
            "clip, or 'clips' to batch sample MP4 files in the project root"
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
        "--front-only",
        action="store_true",
        default=True,
        help="Speak only the object selected as in-front (default: enabled).",
    )
    parser.add_argument(
        "--all-objects",
        dest="front_only",
        action="store_false",
        help="Disable front-only mode and use largest object instead.",
    )
    parser.add_argument(
        "--whatsapp-clip",
        action="store_true",
        help="Shortcut to use the WhatsApp clip from repo root.",
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

    source_lower = args.source.strip().lower()
    if args.whatsapp_clip:
        source_lower = "whatsapp"

    if source_lower == "clips":
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
            front_only=args.front_only,
            frame_stride=args.clip_frame_stride,
            max_frames_per_clip=args.clip_max_frames,
        )
        return

    if source_lower == "whatsapp":
        whatsapp_clip = default_whatsapp_clip_path(repo)
        if whatsapp_clip is None:
            raise RuntimeError(
                "Could not find WhatsApp clip in repo root. Looked for: "
                + ", ".join(WHATSAPP_CLIP_CANDIDATES)
            )
        src = str(whatsapp_clip)
    else:
        src = int(args.source) if args.source.isdigit() else args.source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    engine = DetectionEngine(args.weights, conf=args.conf, device=args.device)
    last_speak = 0.0
    last_phrase = ""

    print("Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            dets = engine.predict_frame(frame)
            phrase = _progress_phrase()
            if dets:
                primary = (
                    pick_front_object(dets, w, h)
                    if args.front_only
                    else max(dets, key=lambda d: _bbox_area(d.xyxy))
                )
                dist = estimate_distance_from_box(primary.xyxy, h, w)
                phrase = _natural_phrase_for_object(
                    primary.name, dist.meters, dist.feet, dist.safety_level
                )
                now = time.monotonic()
                if now - last_speak >= args.speak_interval:
                    if phrase != last_phrase or dist.safety_level in ("stop", "caution"):
                        print(phrase)
                        speak(phrase)
                        last_speak = now
                        last_phrase = phrase
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
            else:
                now = time.monotonic()
                if now - last_speak >= args.speak_interval:
                    if phrase != last_phrase:
                        print(phrase)
                        speak(phrase)
                        last_speak = now
                        last_phrase = phrase

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
