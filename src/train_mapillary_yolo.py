"""
Fine-tune YOLOv8 on the prepared Mapillary YOLO dataset (data/mapillary_yolo/data.yaml).

Uses Ultralytics training API. Does not modify Phase 1 inference code.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO


def data_yaml_for_ultralytics(data_yaml: Path) -> Path:
    """Return a yaml path Ultralytics can load: ``path`` = absolute dataset root.

    Older ``data.yaml`` files used ``path: .``, which is resolved from CWD and breaks
    (e.g. looks for ``<repo>/images/val`` instead of ``<repo>/data/mapillary_yolo/images/val``).
    """
    data_yaml = data_yaml.resolve()
    with data_yaml.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid data yaml: {data_yaml}")

    dataset_root = data_yaml.parent.resolve()
    cfg["path"] = str(dataset_root).replace("\\", "/")

    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
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
    parser = argparse.ArgumentParser(description="Train YOLOv8 on Mapillary-derived YOLO labels.")
    parser.add_argument(
        "--data",
        type=str,
        default="data/mapillary_yolo/data.yaml",
        help="Path to data.yaml produced by prepare_mapillary_yolo.py",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="Base checkpoint (e.g. yolov8n.pt, yolov8s.pt).",
    )
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="e.g. cpu, 0, cuda:0 (default: Ultralytics auto).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default="runs/train",
        help="Ultralytics project directory.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="mapillary_nav_v1",
        help="Run name under project (e.g. mapillary_nav_v1).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Dataloader workers.",
    )
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"data.yaml not found: {data_path}")

    fixed_yaml = data_yaml_for_ultralytics(data_path)
    project_root = Path(args.project).resolve()
    try:
        model = YOLO(args.model)
        model.train(
            data=str(fixed_yaml),
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=str(project_root),
            name=args.name,
            patience=args.patience,
            workers=args.workers,
        )
    finally:
        fixed_yaml.unlink(missing_ok=True)

    save_dir = project_root / args.name
    best_pt = save_dir / "weights" / "best.pt"
    last_pt = save_dir / "weights" / "last.pt"

    print()
    print("=== Training finished ===")
    print(f"Results dir: {save_dir.resolve()}")
    print(f"best.pt:     {best_pt.resolve() if best_pt.is_file() else '(missing)'}")
    print(f"last.pt:     {last_pt.resolve() if last_pt.is_file() else '(missing)'}")
    print()
    print("Evaluate with Phase 1 inference, e.g.:")
    print(
        f'  python src/run_baseline.py --input "data/mapillary_vistas/Mapillary Vistas/validation/images" '
        f'--model "{best_pt}" --save-annotated --run-name mapillary_finetuned_eval'
    )


if __name__ == "__main__":
    main()
