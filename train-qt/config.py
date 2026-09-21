"""Paths and shared constants for train-qt."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / ".env" / "bin" / "python"
SCRIPTS = {
    "preprocess": ROOT / "train" / "preprocess.py",
    "ar": ROOT / "train" / "train_lora_ar.py",
    "nar": ROOT / "train" / "train_lora_nar.py",
    "merge": ROOT / "train" / "merge_lora.py",
}
SETTINGS = Path(__file__).resolve().parent / "settings.json"
LOGS = ROOT / "train" / "logs"
TAIL_BYTES = 200_000