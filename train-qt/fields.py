"""Turn a small field spec into a Qt editor widget.

Supported kinds: ``line`` (text / dir / file picker), ``spin`` (int),
``dspin`` (float), ``combo`` and ``check``.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLineEdit, QPushButton, QSpinBox,
    QDoubleSpinBox, QWidget,
)


def build_field(kind, kw, browse_target):
    if kind == "check":
        widget = QCheckBox(kw.get("label", "flag"))
        widget.setChecked(bool(kw.get("default", False)))
        return widget
    if kind == "combo":
        widget = QComboBox()
        widget.addItems(kw.get("choices", []))
        if kw.get("default") in kw.get("choices", []):
            widget.setCurrentText(str(kw["default"]))
        return widget
    if kind == "spin":
        widget = QSpinBox()
        widget.setRange(int(kw.get("min", -(10 ** 9))), int(kw.get("max", 10 ** 9)))
        widget.setValue(int(kw.get("default", 0)))
        return widget
    if kind == "dspin":
        widget = QDoubleSpinBox()
        widget.setDecimals(6)
        widget.setRange(float(kw.get("min", -1e9)), float(kw.get("max", 1e9)))
        widget.setValue(float(kw.get("default", 0.0)))
        return widget
    widget = QLineEdit()
    widget.setText(str(kw.get("default", "")))
    widget.setPlaceholderText(kw.get("placeholder", ""))
    if kw.get("dir") or kw.get("file"):
        browse = QPushButton("...")
        browse.setMaximumWidth(36)
        wrapper = QWidget()
        row = QHBoxLayout(wrapper)
        row.setContentsMargins(0, 0, 0, 0)
        row.addWidget(widget, 1)
        row.addWidget(browse)
        browse.clicked.connect(
            lambda _=None: browse_target._browse(widget, kw.get("file", False)))
        return wrapper
    return widget