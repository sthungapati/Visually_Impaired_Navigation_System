"""
Fine-tune YOLOv8n on the Mapillary-derived subset (repo: data/mapillary_yolo_subset/).

Run from repo root or from this folder. Fixes data.yaml ``path: .`` at runtime for Ultralytics.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _vis_root() -> Path:
    return Path(__file__).resolve().parent


def _fix_data_yaml(data_yaml: Path) -> Path:
    """Ultralytics resolves path: . from CWD; force dataset root to yaml's parent."""
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid data yaml: {data_yaml}")
    cfg["path"] = str(data_yaml.parent.resolve()).replace("\\", "/")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yaml", delete=False, encoding="utf-8"
    )
    try:
        yaml.safe_dump(cfg, tmp, sort_keys=False, allow_unicode=True)
        tmp.close()
    except Exception:
        tmp.close()
        Path(tmp.name).unlink(missing_ok=True)
        raise
    return Path(tmp.name)


def main() -> None:
    repo = _repo_root()
    vis = _vis_root()
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Mapillary YOLO subset.")
    parser.add_argument(
        "--data",
        type=str,
        default=str(repo / "data" / "mapillary_yolo_subset" / "data.yaml"),
        help="Path to data.yaml",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=str(repo / "yolov8n.pt"),
        help="Starting weights (default: repo yolov8n.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs (default: 10 for fast local subset runs).",
    )
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu, 0, cuda:0, etc. (default: Ultralytics auto)",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=str(vis / "runs" / "train"),
        help="Training output project directory",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="nav_yolo_v1",
        help="Run name under project",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(
            f"data.yaml not found: {data_path}\n"
            "Run from repo root: python src/prepare_mapillary_yolo.py "
            "--output data/mapillary_yolo_subset --max-train ... --max-val ... --shuffle-seed 42"
        )
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Weights not found: {model_path}\n"
            "Place yolov8n.pt in repo root or pass --model path/to/yolov8n.pt"
        )

    fixed_yaml = _fix_data_yaml(data_path)
    project_dir = Path(args.project).resolve()
    try:
        model = YOLO(str(model_path))
        model.train(
            data=str(fixed_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project_dir),
            name=args.name,
            patience=10,
            workers=4,
        )
    finally:
        fixed_yaml.unlink(missing_ok=True)

    best = project_dir / args.name / "weights" / "best.pt"
    print(f"\nDone. Weights: {best.resolve() if best.is_file() else 'see project folder'}")


if __name__ == "__main__":
    main()
