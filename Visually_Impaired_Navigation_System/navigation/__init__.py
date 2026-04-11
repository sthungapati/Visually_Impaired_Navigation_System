"""Small helpers for navigation prototype: detection, distance heuristic, TTS."""

from .detector import DetectionEngine
from .distance import proximity_from_box
from .tts import speak, tts_engine

__all__ = [
    "DetectionEngine",
    "proximity_from_box",
    "tts_engine",
    "speak",
]
