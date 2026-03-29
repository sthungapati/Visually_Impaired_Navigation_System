"""
Build a YOLO-format dataset from Mapillary Vistas v2.0 polygon JSON annotations.

Reads:
  <root>/training|validation/images/*.jpg
  <root>/training|validation/v2.0/polygons/<stem>.json

Outputs:
  data/mapillary_yolo/images/{train,val}/
  data/mapillary_yolo/labels/{train,val}/*.txt
  data/mapillary_yolo/data.yaml
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as e:  # pragma: no cover
    raise SystemExit("Install PyYAML: pip install pyyaml") from e


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")


def load_class_config(path: Path) -> Tuple[Dict[int, str], Dict[str, str], Dict[str, int]]:
    """Load YAML: return id->name, mapillary_label->yolo_name, yolo_name->id."""
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    names = cfg.get("names") or {}
    id_to_name: Dict[int, str] = {int(k): v for k, v in names.items()}
    m2y = cfg.get("mapillary_to_yolo") or {}

    name_to_id: Dict[str, int] = {v: k for k, v in id_to_name.items()}
    for yolo_name in m2y.values():
        if yolo_name not in name_to_id:
            raise ValueError(f"YOLO name '{yolo_name}' not in names: {path}")

    return id_to_name, m2y, name_to_id


def polygon_to_xyxy_norm(
    polygon: List[List[float]], width: int, height: int
) -> Optional[Tuple[float, float, float, float]]:
    """Axis-aligned bbox from polygon; return normalized cx,cy,w,h or None if invalid."""
    if not polygon or len(polygon) < 2:
        return None
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    x1 = max(0.0, min(float(width), x1))
    x2 = max(0.0, min(float(width), x2))
    y1 = max(0.0, min(float(height), y1))
    y2 = max(0.0, min(float(height), y2))
    bw = x2 - x1
    bh = y2 - y1
    if bw < 1.0 or bh < 1.0:
        return None
    cx = (x1 + x2) / 2.0 / float(width)
    cy = (y1 + y2) / 2.0 / float(height)
    w = bw / float(width)
    h = bh / float(height)
    if w <= 0 or h <= 0 or cx < 0 or cy < 0 or cx > 1 or cy > 1:
        return None
    return (cx, cy, w, h)


def process_split(
    split_name: str,
    images_dir: Path,
    polygons_dir: Path,
    out_images: Path,
    out_labels: Path,
    mapillary_to_yolo: Dict[str, str],
    name_to_id: Dict[str, int],
    copy_mode: str,
    max_images: Optional[int],
    shuffle_seed: Optional[int],
) -> Tuple[int, int, Counter]:
    """Copy images and write YOLO labels. Returns (images_written, labels_lines, class_counts)."""
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    image_paths: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(sorted(images_dir.glob(f"*{ext}")))
    image_paths = sorted(set(image_paths))
    if shuffle_seed is not None:
        rng = random.Random(shuffle_seed)
        rng.shuffle(image_paths)
    if max_images is not None:
        image_paths = image_paths[: max_images]

    images_written = 0
    label_lines_total = 0
    class_counts: Counter = Counter()

    for img_path in image_paths:
        stem = img_path.stem
        js = polygons_dir / f"{stem}.json"
        if not js.is_file():
            continue

        try:
            with js.open("r", encoding="utf-8") as f:
                ann: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue

        w = int(ann.get("width") or 0)
        h = int(ann.get("height") or 0)
        if w <= 0 or h <= 0:
            continue

        lines: List[str] = []
        for obj in ann.get("objects") or []:
            label = str(obj.get("label") or "").strip()
            poly = obj.get("polygon")
            if label not in mapillary_to_yolo:
                continue
            yolo_name = mapillary_to_yolo[label]
            cid = name_to_id[yolo_name]
            xywh = polygon_to_xyxy_norm(poly, w, h)
            if xywh is None:
                continue
            cx, cy, bw, bh = xywh
            lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            class_counts[yolo_name] += 1
            label_lines_total += 1

        dest_img = out_images / img_path.name
        if copy_mode == "symlink":
            if dest_img.is_file() or dest_img.is_symlink():
                dest_img.unlink()
            try:
                dest_img.symlink_to(img_path.resolve())
            except OSError:
                shutil.copy2(img_path, dest_img)
        else:
            shutil.copy2(img_path, dest_img)

        label_path = out_labels / f"{stem}.txt"
        with label_path.open("w", encoding="utf-8") as lf:
            lf.write("\n".join(lines))
            if lines:
                lf.write("\n")

        images_written += 1

    return images_written, label_lines_total, class_counts


def write_data_yaml(
    out_root: Path,
    id_to_name: Dict[int, str],
) -> Path:
    """Write Ultralytics-style data.yaml.

    ``path`` must be the absolute dataset root. Ultralytics resolves ``path: .``
    against the *process* current working directory, not the yaml file's folder,
    which breaks training when launched from the repo root.
    """
    names_block: Dict[str, str] = {str(i): id_to_name[i] for i in sorted(id_to_name.keys())}
    root = str(out_root.resolve()).replace("\\", "/")

    payload = {
        "path": root,
        "train": "images/train",
        "val": "images/val",
        "names": names_block,
    }
    yaml_path = out_root / "data.yaml"
    with yaml_path.open("w", encoding="utf-8") as f:
        yaml.dump(payload, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return yaml_path


def write_class_map_json(
    out_path: Path,
    id_to_name: Dict[int, str],
    mapillary_to_yolo: Dict[str, str],
) -> None:
    """Save a reproducible mapping snapshot next to the dataset."""
    by_id = {str(k): v for k, v in sorted(id_to_name.items())}
    inv: Dict[str, List[str]] = {}
    for m_label, y_name in mapillary_to_yolo.items():
        inv.setdefault(y_name, []).append(m_label)

    payload = {
        "id_to_name": by_id,
        "mapillary_labels_per_yolo_class": inv,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Mapillary -> YOLO dataset for fine-tuning.")
    parser.add_argument(
        "--mapillary-root",
        type=str,
        default="data/mapillary_vistas/Mapillary Vistas",
        help="Path to extracted 'Mapillary Vistas' folder (contains training/validation).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/mapillary_nav_classes.yaml",
        help="YAML with names and mapillary_to_yolo mapping.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/mapillary_yolo",
        help="Output YOLO dataset root.",
    )
    parser.add_argument(
        "--copy-mode",
        choices=("copy", "symlink"),
        default="copy",
        help="How to place images into the YOLO folder (default: copy).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=None,
        help="Optional cap on training images (debug).",
    )
    parser.add_argument(
        "--max-val",
        type=int,
        default=None,
        help="Optional cap on validation images (debug).",
    )
    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=None,
        help=(
            "If set, shuffle image order before applying --max-train / --max-val "
            "(reproducible random subset; recommended over alphabetical first-N)."
        ),
    )
    args = parser.parse_args()

    root = Path(args.mapillary_root).resolve()
    cfg_path = Path(args.config).resolve()
    out_root = Path(args.output).resolve()

    id_to_name, mapillary_to_yolo, name_to_id = load_class_config(cfg_path)

    train_img = root / "training" / "images"
    train_poly = root / "training" / "v2.0" / "polygons"
    val_img = root / "validation" / "images"
    val_poly = root / "validation" / "v2.0" / "polygons"

    for p in (train_img, train_poly, val_img, val_poly):
        if not p.is_dir():
            raise FileNotFoundError(f"Missing directory: {p}")

    (out_root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "images" / "val").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    (out_root / "labels" / "val").mkdir(parents=True, exist_ok=True)

    print("=== Mapillary -> YOLO preparation ===")
    print(f"Mapillary root: {root}")
    print(f"Class config:   {cfg_path}")
    print(f"Output:         {out_root}")
    print(f"Classes:        {len(id_to_name)} ({', '.join(id_to_name[i] for i in sorted(id_to_name))})")
    print()

    train_seed = args.shuffle_seed
    val_seed = (args.shuffle_seed + 10_000) if args.shuffle_seed is not None else None

    n_tr, lines_tr, ctr_tr = process_split(
        "train",
        train_img,
        train_poly,
        out_root / "images" / "train",
        out_root / "labels" / "train",
        mapillary_to_yolo,
        name_to_id,
        args.copy_mode,
        args.max_train,
        train_seed,
    )
    n_va, lines_va, ctr_va = process_split(
        "val",
        val_img,
        val_poly,
        out_root / "images" / "val",
        out_root / "labels" / "val",
        mapillary_to_yolo,
        name_to_id,
        args.copy_mode,
        args.max_val,
        val_seed,
    )

    yaml_path = write_data_yaml(out_root, id_to_name)
    write_class_map_json(out_root / "class_map.json", id_to_name, mapillary_to_yolo)

    total_ctr = ctr_tr + ctr_va
    print("=== Summary ===")
    print(f"Train images written: {n_tr}")
    print(f"Val images written:   {n_va}")
    print(f"Total YOLO box lines: {lines_tr + lines_va} (train {lines_tr}, val {lines_va})")
    print()
    print("Class distribution (all splits):")
    for name in sorted(total_ctr.keys()):
        print(f"  {name:20s} {total_ctr[name]}")
    print()
    print(f"data.yaml:      {yaml_path}")
    print(f"class_map.json: {out_root / 'class_map.json'}")


if __name__ == "__main__":
    main()
