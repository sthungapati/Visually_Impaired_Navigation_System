from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


def write_detections_csv(
    items: Iterable[Mapping[str, Any]],
    run_id: str,
    logs_dir: Path,
) -> Path:
    """Write per-detection rows to detections.csv."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "detections.csv"

    fieldnames = [
        "run_id",
        "source_type",
        "source_name",
        "frame_index",
        "class_id",
        "class_name",
        "confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "x_center",
        "y_center",
        "bbox_w",
        "bbox_h",
        "img_w",
        "img_h",
    ]

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in items:
            for det in item.get("detections", []):
                xyxy = det["bbox_xyxy"]
                xywh = det["bbox_xywh"]
                writer.writerow(
                    {
                        "run_id": run_id,
                        "source_type": item.get("source_type"),
                        "source_name": item.get("source_name"),
                        "frame_index": item.get("frame_index"),
                        "class_id": det.get("class_id"),
                        "class_name": det.get("class_name"),
                        "confidence": det.get("confidence"),
                        "x1": xyxy["x1"],
                        "y1": xyxy["y1"],
                        "x2": xyxy["x2"],
                        "y2": xyxy["y2"],
                        "x_center": xywh["x_center"],
                        "y_center": xywh["y_center"],
                        "bbox_w": xywh["w"],
                        "bbox_h": xywh["h"],
                        "img_w": det.get("img_w"),
                        "img_h": det.get("img_h"),
                    }
                )

    return csv_path


def write_detections_json(
    items: Iterable[Mapping[str, Any]],
    run_metadata: Mapping[str, Any],
    logs_dir: Path,
) -> Path:
    """Write detections.json with run metadata and per-item entries."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    json_path = logs_dir / "detections.json"

    payload = {
        "run": dict(run_metadata),
        "results": list(items),
    }

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return json_path


def write_summary_csv(items: Iterable[Mapping[str, Any]], logs_dir: Path) -> Path:
    """Write per-image/frame summary and overall totals."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    csv_path = logs_dir / "summary.csv"

    # Accumulate per-item and overall stats.
    per_item_stats: List[Dict[str, Any]] = []
    overall_counts: Counter = Counter()
    overall_conf_sums = defaultdict(float)
    overall_conf_counts = defaultdict(int)

    for item in items:
        class_counts: Counter = Counter()
        class_conf_sums = defaultdict(float)
        class_conf_counts = defaultdict(int)

        for det in item.get("detections", []):
            cls = str(det.get("class_name"))
            conf = float(det.get("confidence", 0.0))
            class_counts[cls] += 1
            class_conf_sums[cls] += conf
            class_conf_counts[cls] += 1

            overall_counts[cls] += 1
            overall_conf_sums[cls] += conf
            overall_conf_counts[cls] += 1

        per_item_stats.append(
            {
                "item": item,
                "class_counts": class_counts,
                "class_conf_sums": class_conf_sums,
                "class_conf_counts": class_conf_counts,
            }
        )

    classes = sorted(overall_counts.keys())

    fieldnames = [
        "source_type",
        "source_name",
        "frame_index",
        "total_detections",
    ]
    fieldnames.extend([f"count_{c}" for c in classes])
    fieldnames.extend([f"avg_conf_{c}" for c in classes])

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Per-item rows.
        for entry in per_item_stats:
            item = entry["item"]
            class_counts: Counter = entry["class_counts"]
            class_conf_sums = entry["class_conf_sums"]
            class_conf_counts = entry["class_conf_counts"]
            total = sum(class_counts.values())

            row = {
                "source_type": item.get("source_type"),
                "source_name": item.get("source_name"),
                "frame_index": item.get("frame_index"),
                "total_detections": total,
            }

            for c in classes:
                row[f"count_{c}"] = class_counts.get(c, 0)
            for c in classes:
                cnt = class_conf_counts.get(c, 0)
                if cnt:
                    row[f"avg_conf_{c}"] = class_conf_sums[c] / cnt
                else:
                    row[f"avg_conf_{c}"] = ""

            writer.writerow(row)

        # Overall totals row.
        overall_row = {
            "source_type": "ALL",
            "source_name": "",
            "frame_index": "",
            "total_detections": sum(overall_counts.values()),
        }
        for c in classes:
            overall_row[f"count_{c}"] = overall_counts.get(c, 0)
        for c in classes:
            cnt = overall_conf_counts.get(c, 0)
            if cnt:
                overall_row[f"avg_conf_{c}"] = overall_conf_sums[c] / cnt
            else:
                overall_row[f"avg_conf_{c}"] = ""

        writer.writerow(overall_row)

    return csv_path

