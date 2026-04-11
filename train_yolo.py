"""Forward to Visually_Impaired_Navigation_System/train_yolo.py (canonical trainer)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_script = Path(__file__).resolve().parent / "Visually_Impaired_Navigation_System" / "train_yolo.py"
raise SystemExit(subprocess.call([sys.executable, str(_script)] + sys.argv[1:]))
