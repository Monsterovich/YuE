"""Main window: left = tool forms, right = job list + live log tail."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QLabel, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPlainTextEdit, QScrollArea, QSplitter, QVBoxLayout,
    QWidget, QHBoxLayout,
)

from config import LOGS, ROOT, SETTINGS, TAIL_BYTES
from forms import tool_forms


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("train-qt — YuE2 LoRA pipeline")
        self.resize(1180, 780)
        self.forms = tool_forms()
        self.jobs = {}
        self.offsets = {}
        self._shown_pid = None
        self._pinned = None
        self._build()
        self._load_settings(auto_load=True)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(500)

    def _build(self):
        splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(splitter)

        form_box = QWidget()
        lay = QVBoxLayout(form_box)
        for form in self.forms.values():
            form.launchRequested.connect(self.launch)
            lay.addWidget(form)
        lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidget(form_box)
        scroll.setWidgetResizable(True)
        splitter.addWidget(scroll)

        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.addWidget(QLabel("Jobs:"))
        self.job_list = QListWidget()
        self.job_list.currentRowChanged.connect(self._job_selected)
        rlay.addWidget(self.job_list)
        sel = QHBoxLayout()
        sel.addWidget(QLabel("Log:"))
        self.log_selector = QComboBox()
        self.log_selector.currentIndexChanged.connect(self._switch_log)
        sel.addWidget(self.log_selector, 1)
        self.follow = QCheckBox("follow current")
        self.follow.setChecked(True)
        self.follow.toggled.connect(self._follow_toggled)
        sel.addWidget(self.follow)
        rlay.addLayout(sel)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(50_000)
        rlay.addWidget(self.console, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 4)

        bar = self.statusBar()
        self.offline_box = QCheckBox("offline (HF_HUB_OFFLINE=1)")
        self.offline_box.setChecked(True)
        bar.addPermanentWidget(self.offline_box)
        bar.showMessage(f"jobs run in: {ROOT}")

    def _job_selected(self, row):
        item = self.job_list.item(row)
        if item is None:
            return
        pid = item.data(Qt.UserRole)
        self._pinned = pid
        index = self.log_selector.findData(pid)
        if index >= 0:
            self.log_selector.blockSignals(True)
            self.log_selector.setCurrentIndex(index)
            self.log_selector.blockSignals(False)
            self._following(force=True)

    def _switch_log(self, *_):
        self._pinned = self.log_selector.currentData()
        self._following(force=True)

    def _command(self, form):
        return form.argv()

    def launch(self, form):
        problem = form.validate()
        if problem:
            QMessageBox.warning(self, "Cannot launch", problem)
            return
        argv = self._command(form)
        LOGS.mkdir(parents=True, exist_ok=True)
        tag = Path(form.script).name.rsplit(".", 1)[0]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        logpath = LOGS / f"{tag}-{stamp}.log"
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}
        env.update(form.extra_env())
        if self.offline_box.isChecked():
            env["HF_HUB_OFFLINE"] = "1"
        try:
            handle = open(logpath, "w")
            proc = subprocess.Popen(
                argv, cwd=str(ROOT), stdin=subprocess.DEVNULL,
                stdout=handle, stderr=subprocess.STDOUT,
                env=env, start_new_session=True)
        except OSError as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))
            return
        finally:
            try:
                handle.close()
            except (NameError, OSError):
                pass
        self.jobs[str(proc.pid)] = {
            "name": logpath.name, "proc": proc, "script": form.script,
            "log": logpath, "cmd": shlex.join(map(str, argv)),
        }
        self.offsets[str(proc.pid)] = 0
        self._pinned = None
        self._console(f"[train-qt] launch {form.title()} pid={proc.pid}\n"
                      f"  $ {shlex.join(map(str, argv))}\n")
        self._refresh_selector()
        if not self.follow.isChecked():
            index = self.log_selector.findData(str(proc.pid))
            if index >= 0:
                self.log_selector.setCurrentIndex(index)

    def _refresh_selector(self):
        old = self.log_selector.currentData()
        job_item = self.job_list.currentItem()
        sel_pid = job_item.data(Qt.UserRole) if job_item else None
        self.job_list.blockSignals(True)
        self.job_list.clear()
        for pid, job in self.jobs.items():
            poll = job["proc"].poll()
            state = "running" if poll is None else f"exit {poll}"
            item = QListWidgetItem(f"pid {pid}  {job['name']}  [{state}]")
            item.setData(Qt.UserRole, pid)
            self.job_list.addItem(item)
        if sel_pid is not None:
            for row in range(self.job_list.count()):
                if self.job_list.item(row).data(Qt.UserRole) == sel_pid:
                    self.job_list.setCurrentRow(row)
                    break
        self.job_list.blockSignals(False)
        self.log_selector.blockSignals(True)
        self.log_selector.clear()
        for pid, job in self.jobs.items():
            self.log_selector.addItem(f"{job['name']}  (pid {pid})", pid)
        if self._pinned in self.jobs:
            target = self._pinned
        elif self.follow.isChecked():
            target = self._newest_running() or self._shown_pid
        else:
            target = old
        index = self.log_selector.findData(target) if target is not None else -1
        if index >= 0:
            self.log_selector.setCurrentIndex(index)
        self.log_selector.blockSignals(False)

    def _follow_toggled(self, on):
        if on:
            self._pinned = None
        self._refresh_selector()

    def _newest_running(self):
        for pid in reversed(list(self.jobs)):
            if self.jobs[pid]["proc"].poll() is None:
                return pid
        return None

    def _active_pid(self):
        if self._pinned in self.jobs:
            return self._pinned
        if self.follow.isChecked():
            return self._newest_running() or self._shown_pid
        return self.log_selector.currentData()

    def _following(self, force=False):
        pid = self._active_pid()
        if pid is None or pid not in self.jobs:
            return
        if pid != self._shown_pid:
            self._shown_pid = pid
            self._console_clear()
            self.offsets[pid] = 0
            self._console(f"$ {self.jobs[pid]['cmd']}\n")
        logfile = self.jobs[pid]["log"]
        if not logfile.is_file():
            return
        size = logfile.stat().st_size
        offset = self.offsets.get(pid, 0)
        if size > TAIL_BYTES and offset < size - TAIL_BYTES:
            offset = size - TAIL_BYTES
        if not force and size <= offset:
            return
        with open(logfile, "rb") as handle:
            handle.seek(offset)
            data = handle.read(max(size - offset, 0))
        self.offsets[pid] = offset + len(data)
        text = data.decode("utf-8", errors="replace")
        if text:
            self._console(text)

    def _tick(self):
        for pid, job in list(self.jobs.items()):
            if job["proc"].poll() is not None and not job.get("reported"):
                job["reported"] = True
                if pid == self._shown_pid:
                    self._console(f"\n[train-qt] pid {pid} finished, "
                                  f"exit {job['proc'].returncode}\n")
                self._refresh_selector()
        self._following()

    def _console(self, text):
        cursor = self.console.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        for i, part in enumerate(text.replace("\r\n", "\n").split("\r")):
            if i:
                cursor.movePosition(QTextCursor.MoveOperation.StartOfBlock,
                                    QTextCursor.MoveMode.KeepAnchor)
                cursor.removeSelectedText()
            cursor.insertText(part)
        bar = self.console.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _console_clear(self):
        self.console.clear()

    def _load_settings(self, auto_load=True):
        data = {}
        if auto_load:
            try:
                data = json.loads(SETTINGS.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                data = {}
        for form in self.forms.values():
            form.load(data)
        self.offline_box.setChecked(bool(data.get("offline", True)))
        self._refresh_selector()

    def _save_settings(self):
        data = {}
        for form in self.forms.values():
            form.save(data)
        data["offline"] = self.offline_box.isChecked()
        tmp = SETTINGS.with_name(SETTINGS.name + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                       encoding="utf-8")
        os.replace(tmp, SETTINGS)

    def closeEvent(self, event):
        try:
            self._save_settings()
        except OSError:
            pass
        super().closeEvent(event)