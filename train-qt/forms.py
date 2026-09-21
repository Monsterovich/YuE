"""Forms for each pipeline tool: preprocess / AR / NAR / merge / generator."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox, QFileDialog, QFormLayout, QGroupBox, QLineEdit, QPushButton,
    QSpinBox, QDoubleSpinBox, QWidget,
)

from config import PYTHON, ROOT, SCRIPTS
from fields import build_field


def _exclusive_checkbox(trigger, members, checked):
    if checked:
        for other in members:
            if other is not trigger:
                other.setChecked(False)


class ToolForm(QGroupBox):
    launchRequested = Signal(object)

    def __init__(self, title, key, script, fields, parent=None):
        super().__init__(title, parent)
        self.key = key
        self.script = script
        self.fields = {}  # name -> (widget, kind, kwargs)
        form = QFormLayout(self)
        for name, spec in fields.items():
            kind = spec[0]
            kw = spec[1] if len(spec) > 1 else {}
            widget = build_field(kind, kw, self)
            self.fields[name] = (widget, kind, kw)
            if kind == "check":
                form.addRow(widget)
            else:
                form.addRow(kw.get("label", name), widget)
        groups = {}
        for name, (widget, kind, kw) in self.fields.items():
            if kind == "check" and kw.get("group"):
                groups.setdefault(kw["group"], []).append(widget)
        for members in groups.values():
            for widget in members:
                widget.toggled.connect(
                    lambda on, w=widget, m=members: _exclusive_checkbox(w, m, on))
        launch = QPushButton("Launch")
        launch.clicked.connect(lambda: self.launchRequested.emit(self))
        form.addRow("", launch)

    def _browse(self, widget, file_mode):
        start = widget.text() or str(ROOT)
        if not os.path.isabs(start):
            start = str(ROOT / start)
        if file_mode:
            path, _ = QFileDialog.getOpenFileName(self, "Select file", start)
        else:
            path = QFileDialog.getExistingDirectory(self, "Select folder", start)
        if path:
            widget.setText(path)

    def _widget_of(self, widget):
        if isinstance(widget, QWidget) and not isinstance(
                widget, (QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox)):
            inner = widget.findChild(QLineEdit)
            if inner is not None:
                return inner
        return widget

    def _value(self, name):
        widget, kind, _kw = self.fields[name]
        widget = self._widget_of(widget)
        if kind == "check":
            return None
        if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
            return widget.value()
        if isinstance(widget, QComboBox):
            return widget.currentText()
        return widget.text()

    def collect(self):
        args = []
        group_flags = {}
        for name, (widget, kind, kw) in self.fields.items():
            if kind == "check":
                if widget.isChecked():
                    group = kw.get("group")
                    if group:
                        group_flags[group] = kw.get("flag", f"--{name}")
                    else:
                        args.append(kw.get("flag", f"--{name}"))
                continue
            value = self._value(name)
            if value == "":
                continue
            if kw.get("positional"):
                args.append(str(value))
                continue
            flag = kw.get("flag", f"--{name}")
            if kind == "combo" and not kw.get("store_value", False):
                args.append(str(value))
                continue
            args.extend([flag, str(value)])
        args.extend(group_flags.values())
        return args

    def argv(self):
        return [str(PYTHON), str(self.script)] + self.collect()

    def extra_env(self):
        return {}

    def validate(self):
        return None

    def save(self, data):
        record = {}
        for name, (widget, kind, _kw) in self.fields.items():
            if kind == "check":
                record[name] = bool(widget.isChecked())
            else:
                value = self._value(name)
                record[name] = value if isinstance(value, str) else float(value)
        data[self.key] = record

    def load(self, data):
        saved = data.get(self.key) or {}
        for name, (widget, kind, _kw) in self.fields.items():
            if name not in saved:
                continue
            value = saved[name]
            widget = self._widget_of(widget)
            if kind == "check":
                widget.setChecked(bool(value))
            elif isinstance(widget, QSpinBox):
                widget.setValue(int(value))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(value))
            elif isinstance(widget, QComboBox):
                for i in range(widget.count()):
                    if widget.itemText(i) == str(value):
                        widget.setCurrentIndex(i)
                        break
            else:
                widget.setText(str(value))


class GeneratorForm(ToolForm):
    """Wraps test.py (plan/render): same pipeline as the repo's own generator.

    All knobs are passed as ``YUE2_*`` environment variables, exactly as
    ``test.py`` reads them; the only CLI argument is the stage name.
    """

    def __init__(self, parent=None):
        style_default = ("Epic cinematic orchestral trailer music, massive "
                         "symphonic orchestra with soaring heroic strings, "
                         "thunderous taiko drums and pounding orchestral "
                         "percussion, dramatic brass fanfares, powerful "
                         "angelic and dark choir, uplifting heroic cinematic "
                         "soundscapes, massive epic climaxes, professional "
                         "studio recording")
        gen_fields = {
            "stage": ("combo", {"label": "Stage", "choices": ["render", "plan"], "default": "render"}),
            "song_name": ("line", {"label": "YUE2_SONG_NAME", "default": "new_horizon", "env": "YUE2_SONG_NAME"}),
            "style": ("line", {"label": "YUE2_STYLE (LoRA trigger)", "default": style_default, "env": "YUE2_STYLE"}),
            "lyrics_file": ("line", {"label": "YUE2_LYRICS_FILE (required path)", "default": "",
                                     "env": "YUE2_LYRICS_FILE", "file": True}),
            "melody_file": ("line", {"label": "YUE2_MELODY_FILE (empty=none)", "default": "",
                                     "env": "YUE2_MELODY_FILE", "file": True}),
            "adapter": ("line", {"label": "YUE2_ADAPTER (AR, empty=none)", "default": "train/adapter-ar", "dir": True,
                                 "env": "YUE2_ADAPTER"}),
            "lora_scale": ("dspin", {"label": "YUE2_LORA_SCALE", "default": 0.5, "min": 0.0, "max": 20.0,
                                     "env": "YUE2_LORA_SCALE"}),
            "adapter_nar": ("line", {"label": "YUE2_ADAPTER_NAR (empty=none)", "default": "",
                                     "env": "YUE2_ADAPTER_NAR", "dir": True}),
            "lora_nar_scale": ("dspin", {"label": "YUE2_LORA_NAR_SCALE", "default": 1.0, "min": 0.0, "max": 20.0,
                                         "env": "YUE2_LORA_NAR_SCALE"}),
            "guidance": ("dspin", {"label": "YUE2_GUIDANCE", "default": 1.0, "min": 0.0, "max": 20.0,
                                   "env": "YUE2_GUIDANCE"}),
            "cot_mode": ("combo", {"label": "YUE2_COT_MODE", "choices": ["full", "off", "melody"], "default": "full",
                                   "env": "YUE2_COT_MODE"}),
            "seed": ("spin", {"label": "YUE2_SEED", "default": 777, "env": "YUE2_SEED"}),
            "legacy_scale": ("check", {"label": "YUE2_LEGACY_SCALE (old buggy scaling)", "env": "YUE2_LEGACY_SCALE"}),
            "style_train": ("check", {"label": "YUE2_STYLE_TRAIN (use train-manifest style)", "env": "YUE2_STYLE_TRAIN"}),
            "train_manifest": ("line", {"label": "YUE2_TRAIN_MANIFEST (for STYLE_TRAIN)", "default": "",
                                        "env": "YUE2_TRAIN_MANIFEST", "file": True}),
            "no_lora": ("check", {"label": "YUE2_NO_LORA (run base model)", "env": "YUE2_NO_LORA"}),
        }
        super().__init__("0. Generator  (train-qt/generate.py plan|render)",
                         "generator",
                         Path(__file__).resolve().parent / "generate.py",
                         gen_fields, parent)

    def argv(self):
        stage = self._widget_of(self.fields["stage"][0]).currentText()
        return [str(PYTHON), str(self.script), stage]

    def extra_env(self):
        env = {}
        for name, (widget, kind, kw) in self.fields.items():
            env_name = kw.get("env")
            if not env_name:
                continue
            if kind == "check":
                if widget.isChecked():
                    env[env_name] = "1"
                continue
            value = self._value(name)
            if value == "":
                continue
            env[env_name] = str(value)
        return env

    def validate(self):
        lyrics = self._value("lyrics_file")
        if not lyrics:
            return "YUE2_LYRICS_FILE is empty - the generator needs a lyrics file."
        path = Path(lyrics)
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file():
            return f"YUE2_LYRICS_FILE not found: {path}"
        for field, label in (("song_name", "YUE2_SONG_NAME"),
                             ("style", "YUE2_STYLE")):
            if not self._value(field):
                return f"{label} is empty - the generator needs it."
        return None


def tool_forms():
    style_default = ("Epic cinematic orchestral trailer music, massive symphonic "
                     "orchestra, soaring heroic strings, thunderous taiko drums, "
                     "soaring choir, dramatic brass, powerful uplifting climaxes")
    preprocess = {
        "songs": ("line", {"label": "Songs folder", "default": "datasets/", "dir": True, "positional": True}),
        "out": ("line", {"label": "--out", "default": "train/dataset"}),
        "model": ("line", {"label": "--model", "default": "m-a-p/YuE2-3B"}),
        "vae": ("line", {"label": "--vae", "default": "m-a-p/YuE2-Vae"}),
        "style": ("line", {"label": "--style", "default": style_default}),
        "lyrics": ("line", {"label": "--lyrics", "default": "[No lyrics text provided]"}),
        "clip-seconds": ("dspin", {"label": "--clip-seconds (0=full)", "default": 0.0, "min": 0.0}),
        "seeds": ("line", {"label": "--seeds", "default": "1"}),
        "cfg-scale": ("dspin", {"label": "--cfg-scale", "default": 1.0, "min": 0.0, "max": 20.0}),
        "max-songs": ("spin", {"label": "--max-songs (0=all)", "default": 0, "min": 0}),
        "no-low-vram": ("check", {"label": "--no-low-vram (both paths on GPU)", "flag": "--no-low-vram"}),
        "no-chunked": ("check", {"label": "--no-chunked (disable chunked VAE)", "flag": "--no-chunked"}),
    }
    ar = {
        "dataset": ("line", {"label": "--dataset", "default": "train/dataset", "dir": True}),
        "out": ("line", {"label": "--out", "default": "train/adapter-ar"}),
        "r": ("spin", {"label": "--r", "default": 8, "min": 1}),
        "alpha": ("dspin", {"label": "--alpha", "default": 8.0, "min": 0.000001}),
        "lr": ("line", {"label": "--lr", "default": "1e-4"}),
        "wd": ("line", {"label": "--wd", "default": "1e-2"}),
        "epochs": ("spin", {"label": "--epochs", "default": 5, "min": 1}),
        "steps": ("spin", {"label": "--steps (0=epochs)", "default": 1200, "min": 0}),
        "window": ("spin", {"label": "--window", "default": 2048, "min": 64}),
        "ce-chunk": ("spin", {"label": "--ce-chunk", "default": 512, "min": 1}),
        "max-frames": ("spin", {"label": "--max-frames", "default": 7500, "min": 0}),
        "seed": ("spin", {"label": "--seed", "default": 123}),
        "checkpoint-every": ("spin", {"label": "--checkpoint-every", "default": 50, "min": 1}),
        "resume": ("line", {"label": "--resume (empty=none)", "placeholder": "train/adapter-ar", "dir": True}),
    }
    nar = {
        "dataset": ("line", {"label": "--dataset", "default": "train/dataset-nar-clips", "dir": True}),
        "out": ("line", {"label": "--out", "default": "train/adapter-nar"}),
        "r": ("spin", {"label": "--r", "default": 32, "min": 1}),
        "alpha": ("dspin", {"label": "--alpha", "default": 48.0, "min": 0.000001}),
        "lr": ("line", {"label": "--lr", "default": "1e-4"}),
        "wd": ("line", {"label": "--wd", "default": "1e-2"}),
        "epochs": ("spin", {"label": "--epochs", "default": 3, "min": 1}),
        "steps": ("spin", {"label": "--steps (0=epochs)", "default": 1200, "min": 0}),
        "max-frames": ("spin", {"label": "--max-frames", "default": 2400, "min": 0}),
        "seed": ("spin", {"label": "--seed", "default": 2026}),
        "checkpoint-every": ("spin", {"label": "--checkpoint-every", "default": 50, "min": 1}),
        "loss-variant": ("combo", {"label": "--loss-variant", "choices": ["normalized", "standard"], "default": "normalized", "flag": "--loss-variant", "store_value": True}),
        "shape-w": ("dspin", {"label": "--shape-w", "default": 0.9, "min": 0.0, "max": 1.0}),
        "disable-grad-checkpoint": ("check", {"label": "disable grad checkpoint (--no-grad-checkpoint, faster/more VRAM)", "flag": "--no-grad-checkpoint"}),
        "no-heads": ("check", {"label": "--no-heads (don't adapt vae2llm/llm2vae)", "flag": "--no-heads"}),
        "no-reuse-cache": ("check", {"label": "--no-reuse-cache (always re-prefill KV)", "flag": "--no-reuse-cache"}),
        "resume": ("line", {"label": "--resume (empty=none)", "placeholder": "train/adapter-nar", "dir": True}),
    }
    merge = {
        "model": ("line", {"label": "--model", "default": "m-a-p/YuE2-3B"}),
        "adapter": ("line", {"label": "--adapter", "default": "train/adapter-nar", "dir": True}),
        "out": ("line", {"label": "--out", "default": "train/merged/merged-nar", "dir": True}),
    }
    return {
        "preprocess": ToolForm("1. Preprocess  (preprocess.py)", "preprocess", SCRIPTS["preprocess"], preprocess),
        "ar": ToolForm("2. AR LoRA  (train_lora_ar.py)", "ar", SCRIPTS["ar"], ar),
        "nar": ToolForm("3. NAR LoRA  (train_lora_nar.py)", "nar", SCRIPTS["nar"], nar),
        "merge": ToolForm("4. Merge  (merge_lora.py)", "merge", SCRIPTS["merge"], merge),
        "generator": GeneratorForm(),
    }