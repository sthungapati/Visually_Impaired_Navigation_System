"""
Webcam or video: detect objects, rough distance, speak summary (rate-limited).

Example:
  python demo_navigation.py --weights runs/train/nav_yolo_v1/weights/best.pt --source 0
  python demo_navigation.py --weights ../yolov8n.pt --source video.mp4
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

_VIS_ROOT = Path(__file__).resolve().parent
if str(_VIS_ROOT) not in sys.path:
    sys.path.insert(0, str(_VIS_ROOT))

from navigation.detector import DetectionEngine
from navigation.distance import proximity_from_box
from navigation.tts import speak


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
        help="0 for default webcam, or path to video file",
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
    args = parser.parse_args()

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
