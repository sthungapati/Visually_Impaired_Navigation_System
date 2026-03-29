### YOLOv8 Baseline Detection Pipeline (Phase 1)

This project provides a production-style baseline inference pipeline for pretrained Ultralytics YOLOv8 models, targeting urban street-scene perception for visually impaired navigation. Phase 1 focuses on **inference, logging, and reporting only** (no training).

### Project layout

- `src/pipeline/detector.py` – wraps YOLOv8 model loading and inference, returns structured detections.
- `src/pipeline/io_utils.py` – input resolution, run directory management, helpers.
- `src/pipeline/reporting.py` – writes `detections.csv`, `detections.json`, and `summary.csv`.
- `src/pipeline/viz.py` – saves annotated images and annotated videos.
- `src/run_baseline.py` – main CLI entrypoint.
- `outputs/runs/<run_id>/` – per-run outputs:
  - `annotated/` – annotated images and/or annotated video.
  - `logs/` – `detections.csv`, `detections.json`, `summary.csv`.
  - `config.json` – configuration and metadata used for the run.
- `data/sample_images/` – placeholder for sample images (not committed).

### Installation

1. Create and activate a Python 3.9+ virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

YOLOv8 and Torch will run on CPU by default; if you have a compatible GPU and drivers, they will use CUDA when available.

### Usage

From the project root:

```bash
python src/run_baseline.py --input data/sample_images --save-annotated
```

Other examples:

- Use a different model and resolution:

```bash
python src/run_baseline.py --input data/sample_images --model yolov8s.pt --conf 0.35 --imgsz 960
```

- Run on a video file and save annotated video:

```bash
python src/run_baseline.py --input data/sample_video.mp4 --save-annotated
```

Key options:

- `--conf`, `--iou`, `--imgsz`, `--model`, `--device` – standard YOLO inference settings.
- `--classes` – comma-separated class names or ids to filter (e.g. `person,car,traffic light`).
- `--nav-critical` – shortcut for `person,bicycle,car,motorcycle,bus,truck,traffic light,stop sign`.
- `--run-name` – fixed run folder name (e.g. `mapillary_eval_all`).
- `--max-items` – quick smoke runs (limit number of images or frames).
- `--save-annotated` – write annotated images / video to the run folder.

At the end of each run, the script prints a small smoke-test summary (number of items processed, total detections, and top-5 classes) and the path to the run's output folder.

### Mapillary Vistas validation-only baseline

Use the extracted validation images path:

```bash
python src/run_baseline.py --input "data/mapillary_vistas/Mapillary Vistas/validation/images" --save-annotated --run-name mapillary_eval_all
python src/run_baseline.py --input "data/mapillary_vistas/Mapillary Vistas/validation/images" --save-annotated --nav-critical --run-name mapillary_eval_filtered
```

### Compare two runs

```bash
python src/compare_runs.py --run-a outputs/runs/mapillary_eval_all --label-a mapillary_all --run-b outputs/runs/mapillary_eval_filtered --label-b mapillary_filtered --top-k 10
```

---

### Phase 2: Fine-tune YOLOv8 on Mapillary (navigation subset)

Phase 1 inference code is unchanged. Phase 2 adds:

- `configs/mapillary_nav_classes.yaml` – navigation-focused class plan and Mapillary label → YOLO name mapping.
- `src/prepare_mapillary_yolo.py` – reads `v2.0/polygons/*.json`, converts polygons to axis-aligned boxes, writes YOLO labels.
- `src/train_mapillary_yolo.py` – thin wrapper around Ultralytics `YOLO.train()`.

**1) Prepare the YOLO dataset** (training split → `train/`, validation split → `val/`):

```bash
python src/prepare_mapillary_yolo.py --mapillary-root "data/mapillary_vistas/Mapillary Vistas" --config configs/mapillary_nav_classes.yaml --output data/mapillary_yolo
```

Quick smoke test (few images):

```bash
python src/prepare_mapillary_yolo.py --max-train 200 --max-val 50
```

**Smaller / faster local training (recommended before a full GPU run):** use a **random subset** so it is not biased toward filenames A–Z. Write to a separate folder so you keep the full YOLO export for later:

```bash
python src/prepare_mapillary_yolo.py --output data/mapillary_yolo_subset --max-train 3000 --max-val 400 --shuffle-seed 42
python src/train_mapillary_yolo.py --data data/mapillary_yolo_subset/data.yaml --model yolov8n.pt --epochs 10 --batch 8 --project runs/train --name mapillary_nav_subset
```

To free disk space, you can **delete** `data/mapillary_yolo/images/` and `labels/` (or the whole `data/mapillary_yolo/` folder) and re-run `prepare_mapillary_yolo.py` when needed. Do **not** delete `data/mapillary_vistas/` until you no longer need the original Mapillary extract for full training.

**GPU server without the full original Mapillary:** you do **not** need `data/mapillary_vistas/` on the cluster. Run `prepare_mapillary_yolo.py` once on a machine that has the extract (use `--max-train` / `--max-val` / `--shuffle-seed` to cap size). Then copy **only** the prepared folder, e.g. `data/mapillary_yolo_subset/` (contains `images/`, `labels/`, `data.yaml`, `class_map.json`), plus this repo’s `src/`, `configs/`, and `requirements.txt`. Training uses **only** that YOLO folder—not polygons, panoptic, or the zip. Increase subset sizes later if you can transfer more data.

**2) Train** (start small; adjust `--epochs`, `--batch`, `--device`):

```bash
python src/train_mapillary_yolo.py --data data/mapillary_yolo/data.yaml --model yolov8n.pt --epochs 30 --imgsz 640 --batch 8 --project runs/train --name mapillary_nav_v1
```

**3) Run inference with the fine-tuned weights** (same Phase 1 pipeline; `--model` points to `best.pt`):

```bash
python src/run_baseline.py --input "data/mapillary_vistas/Mapillary Vistas/validation/images" --model runs/train/mapillary_nav_v1/weights/best.pt --save-annotated --run-name mapillary_finetuned_eval
```

**4) Compare baseline vs fine-tuned** (same eval images, different `--model` / `--run-name`):

```bash
python src/compare_runs.py --run-a outputs/runs/mapillary_eval_all --label-a pretrained --run-b outputs/runs/mapillary_finetuned_eval --label-b finetuned --top-k 15
```

Note: COCO-pretrained class names differ from this 15-class head; for quantitative comparison on **this** label set, use validation mAP from training or a separate evaluator. Qualitative comparison of annotated outputs is still useful for navigation failures (poles, signs, trees, etc.).

