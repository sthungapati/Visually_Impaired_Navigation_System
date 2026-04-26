"""
Polished final-demo prototype for visually impaired navigation assistance.

Features:
- Runs on prerecorded video files.
- YOLO detection + annotated overlays.
- Center walking zone visualization.
- Right-side assistant panel with guidance and danger status.
- Rule-based decision logic with TTS cooldown.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from navigation.assistant_logic import (
    OUTDOOR_OBSTACLE_CLASSES,
    guidance_from_summaries,
    summarize_detections,
)
from navigation.detector import DetectionEngine
from navigation.presentation_ui import (
    build_assistant_panel,
    draw_center_walking_zone,
    draw_detection_overlays,
)
from navigation.tts import speak


WHATSAPP_CLIP_CANDIDATES = (
    "WhatsApp Video 2026-04-25 at 8.26.06 PM.mp4",
    "Whats App Video 2026-04-25 at 8.26.06 PM.mp4",
)


class SpeechController:
    """Avoid message spam: speak only on message-change or cooldown."""

    def __init__(self, cooldown_s: float) -> None:
        self.cooldown_s = cooldown_s
        self.last_text = ""
        self.last_time = 0.0

    def maybe_speak(self, text: str, *, enabled: bool) -> None:
        if not enabled or not text.strip():
            return
        now = time.monotonic()
        changed = text != self.last_text
        cooled_down = now - self.last_time >= self.cooldown_s
        if changed or cooled_down:
            speak(text)
            self.last_text = text
            self.last_time = now


def _resolve_default_video(repo_root: Path) -> Path:
    for name in WHATSAPP_CLIP_CANDIDATES:
        p = repo_root / name
        if p.is_file():
            return p
    raise FileNotFoundError(
        "No default WhatsApp clip found. Expected one of: " + ", ".join(WHATSAPP_CLIP_CANDIDATES)
    )


def _open_writer(path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(str(path), fourcc, fps, (width, height))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Polished AI navigation assistant demo")
    parser.add_argument(
        "--input-video",
        type=str,
        default=None,
        help="Path to prerecorded outdoor video (default: repo WhatsApp clip).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "yolov8n.pt"),
        help="YOLO weights file",
    )
    parser.add_argument("--conf", type=float, default=0.28, help="YOLO confidence threshold")
    parser.add_argument("--device", type=str, default=None, help="cpu, cuda:0, ...")
    parser.add_argument(
        "--output-video",
        type=str,
        default="outputs/presentation_demo/navigation_assistant_demo.mp4",
        help="Annotated presentation output video path",
    )
    parser.add_argument(
        "--assistant-name",
        type=str,
        default="PathPilot AI",
        help="Displayed assistant name in panel",
    )
    parser.add_argument(
        "--panel-width",
        type=int,
        default=430,
        help="Right-side assistant panel width in pixels",
    )
    parser.add_argument(
        "--tts-cooldown",
        type=float,
        default=4.0,
        help="Seconds before repeating same message",
    )
    parser.add_argument("--no-speak", action="store_true", help="Disable TTS")
    parser.add_argument("--display", action="store_true", help="Show live OpenCV preview window")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional quick run limit for demos",
    )
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Process every Nth frame (1 = all frames)",
    )
    parser.add_argument(
        "--center-zone-ratio",
        type=float,
        default=0.34,
        help="Fraction of frame width for walking center zone",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    in_video = Path(args.input_video) if args.input_video else _resolve_default_video(repo_root)
    if not in_video.exists():
        raise FileNotFoundError(f"Input video not found: {in_video}")

    cap = cv2.VideoCapture(str(in_video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {in_video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w = frame_w + args.panel_width
    out_h = frame_h

    out_video = Path(args.output_video)
    writer = _open_writer(out_video, fps=fps, width=out_w, height=out_h)
    detector = DetectionEngine(args.weights, conf=args.conf, device=args.device)
    speech = SpeechController(cooldown_s=args.tts_cooldown)

    processed = 0
    read_i = 0
    try:
        print(f"Input: {in_video}")
        print(f"Output: {out_video}")
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if read_i % max(1, args.frame_stride) != 0:
                read_i += 1
                continue
            if args.max_frames is not None and processed >= args.max_frames:
                break

            dets = detector.predict_frame(frame)
            summaries, zone = summarize_detections(
                dets,
                frame_w,
                frame_h,
                obstacle_classes=OUTDOOR_OBSTACLE_CLASSES,
                center_zone_ratio=args.center_zone_ratio,
            )
            state = guidance_from_summaries(summaries, assistant_name=args.assistant_name)
            annotated = draw_center_walking_zone(frame, zone)
            annotated = draw_detection_overlays(annotated, summaries)
            panel = build_assistant_panel(frame_h, args.panel_width, state)
            composed = np.hstack([annotated, panel])

            writer.write(composed)
            speech.maybe_speak(state.guidance_text, enabled=not args.no_speak)

            if args.display:
                cv2.imshow("Navigation Assistant Demo", composed)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            processed += 1
            read_i += 1
            if processed == 1 or processed % 30 == 0:
                print(f"Processed frames: {processed}", flush=True)
    finally:
        cap.release()
        writer.release()
        if args.display:
            cv2.destroyAllWindows()

    print(f"Done. Presentation demo saved to: {out_video}")


if __name__ == "__main__":
    main()
