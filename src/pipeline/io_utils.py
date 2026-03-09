from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


@dataclass
class RunPaths:
    """Container for important paths of a single run."""

    run_id: str
    run_dir: Path
    annotated_dir: Path
    logs_dir: Path


def resolve_path(path_str: str) -> Path:
    """Return a resolved Path for the given string."""
    return Path(path_str).expanduser().resolve()


def is_video_file(path: Path) -> bool:
    """Return True if the path looks like a video file."""
    return path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS


def is_image_file(path: Path) -> bool:
    """Return True if the path looks like an image file."""
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def iter_image_files(input_path: Path, max_items: Optional[int] = None) -> Iterator[Path]:
    """Yield image files from a directory or a single image path."""
    count = 0

    if input_path.is_dir():
        candidates: List[Path] = []
        for ext in IMAGE_EXTENSIONS:
            candidates.extend(sorted(input_path.glob(f"*{ext}")))
        for path in sorted(set(candidates)):
            yield path
            count += 1
            if max_items is not None and count >= max_items:
                break
    elif is_image_file(input_path):
        yield input_path
    else:
        raise FileNotFoundError(f"No valid image files found at {input_path}")


def create_run_paths(
    base_output: Optional[Path] = None,
    run_name: Optional[str] = None,
) -> RunPaths:
    """Create folders for a new run and return their paths."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = run_name or timestamp

    base_root = base_output or (Path("outputs") / "runs")
    run_dir = base_root / run_id
    annotated_dir = run_dir / "annotated"
    logs_dir = run_dir / "logs"

    for d in (run_dir, annotated_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    return RunPaths(
        run_id=run_id,
        run_dir=run_dir,
        annotated_dir=annotated_dir,
        logs_dir=logs_dir,
    )


def guess_source_type(input_path: Path) -> str:
    """Return 'image' or 'video' based on the input path."""
    if input_path.is_dir():
        return "image"
    if is_video_file(input_path):
        return "video"
    if is_image_file(input_path):
        return "image"
    raise ValueError(f"Could not determine source type for {input_path}")

