"""Entry point for train-qt: small Qt launcher for the YuE2 LoRA pipeline.

Each tool (generator / preprocess / AR / NAR / merge) becomes a form; Launch
spawns the matching script under ``.env/bin/python`` detached (setsid) with a
log file, so jobs keep running after the GUI closes. The right pane tails the
selected job's log. Settings persist to settings.json next to this file.
"""
from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from window import MainWindow


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()