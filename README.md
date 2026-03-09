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
- `--max-items` – quick smoke runs (limit number of images or frames).
- `--save-annotated` – write annotated images / video to the run folder.

At the end of each run, the script prints a small smoke-test summary (number of items processed, total detections, and top-5 classes) and the path to the run's output folder.

