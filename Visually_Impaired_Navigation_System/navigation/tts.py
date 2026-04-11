"""Text-to-speech (offline). Windows/macOS/Linux: pyttsx3."""

from __future__ import annotations

from typing import Any, Optional

_engine: Optional[Any] = None


def tts_engine():
    global _engine
    if _engine is None:
        import pyttsx3

        _engine = pyttsx3.init()
        _engine.setProperty("rate", 175)
    return _engine


def speak(text: str) -> None:
    if not text.strip():
        return
    eng = tts_engine()
    eng.say(text)
    eng.runAndWait()
