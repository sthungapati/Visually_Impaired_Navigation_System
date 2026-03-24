from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Tuple


def _load_counts_and_conf(detections_csv: Path) -> Tuple[Counter, Dict[str, float], Dict[str, int]]:
    """Read detections.csv and return class counts and confidence aggregates."""
    counts: Counter = Counter()
    conf_sums: Dict[str, float] = {}
    conf_counts: Dict[str, int] = {}

    with detections_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_name = str(row.get("class_name", "")).strip()
            if not class_name:
                continue
            counts[class_name] += 1

            conf = row.get("confidence")
            try:
                conf_val = float(conf) if conf is not None and conf != "" else None
            except ValueError:
                conf_val = None
            if conf_val is not None:
                conf_sums[class_name] = conf_sums.get(class_name, 0.0) + conf_val
                conf_counts[class_name] = conf_counts.get(class_name, 0) + 1

    return counts, conf_sums, conf_counts


def _total_items_from_summary(summary_csv: Path) -> int:
    """Count per-item rows in summary.csv (excluding ALL row)."""
    total = 0
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("source_type") == "ALL":
                continue
            total += 1
    return total


def _avg_conf(class_name: str, conf_sums: Dict[str, float], conf_counts: Dict[str, int]) -> float:
    n = conf_counts.get(class_name, 0)
    if n == 0:
        return 0.0
    return conf_sums[class_name] / n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two baseline runs by class counts and confidence stats."
    )
    parser.add_argument("--run-a", required=True, help="Path to first run directory.")
    parser.add_argument("--run-b", required=True, help="Path to second run directory.")
    parser.add_argument(
        "--label-a", default="run_a", help="Display label for first run (e.g., coco_baseline)."
    )
    parser.add_argument(
        "--label-b", default="run_b", help="Display label for second run (e.g., mapillary_eval)."
    )
    parser.add_argument(
        "--top-k", type=int, default=10, help="Number of top classes to display per run."
    )
    args = parser.parse_args()

    run_a = Path(args.run_a).resolve()
    run_b = Path(args.run_b).resolve()

    a_det = run_a / "logs" / "detections.csv"
    b_det = run_b / "logs" / "detections.csv"
    a_sum = run_a / "logs" / "summary.csv"
    b_sum = run_b / "logs" / "summary.csv"

    for p in (a_det, b_det, a_sum, b_sum):
        if not p.exists():
            raise FileNotFoundError(f"Missing expected file: {p}")

    a_counts, a_conf_sums, a_conf_counts = _load_counts_and_conf(a_det)
    b_counts, b_conf_sums, b_conf_counts = _load_counts_and_conf(b_det)
    a_items = _total_items_from_summary(a_sum)
    b_items = _total_items_from_summary(b_sum)

    print("=== Basic Stats ===")
    print(f"{args.label_a}: items={a_items}, detections={sum(a_counts.values())}")
    print(f"{args.label_b}: items={b_items}, detections={sum(b_counts.values())}")
    print("")

    print(f"=== Top {args.top_k} Classes ({args.label_a}) ===")
    for c, n in a_counts.most_common(args.top_k):
        print(f"{c:20s} count={n:6d} avg_conf={_avg_conf(c, a_conf_sums, a_conf_counts):.3f}")
    print("")

    print(f"=== Top {args.top_k} Classes ({args.label_b}) ===")
    for c, n in b_counts.most_common(args.top_k):
        print(f"{c:20s} count={n:6d} avg_conf={_avg_conf(c, b_conf_sums, b_conf_counts):.3f}")
    print("")

    print("=== Shared Class Delta (run_b - run_a) ===")
    for c in sorted(set(a_counts.keys()) | set(b_counts.keys())):
        delta = b_counts.get(c, 0) - a_counts.get(c, 0)
        if delta != 0:
            print(f"{c:20s} delta={delta:+6d}")


if __name__ == "__main__":
    main()

