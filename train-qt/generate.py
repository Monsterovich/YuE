"""YuE2 generator (LoRA pipeline wrapper, port of repo test.py).

Fully driven by environment variables - the GUI's Generator form fills them.
No content defaults are baked in (style, lyrics, song name, adapters, seeds).

Usage::

    .env/bin/python train-qt/generate.py plan|render

Required env:
    YUE2_SONG_NAME       artifact/output name
    YUE2_STYLE           style descriptor (LoRA trigger)
    YUE2_LYRICS_FILE     path to the lyrics text file
    YUE2_SEED            integer seed
    YUE2_LORA_SCALE      AR LoRA multiplier
    YUE2_LORA_NAR_SCALE  NAR LoRA multiplier
    YUE2_GUIDANCE        guidance override value
    YUE2_COT_MODE        "full", "melody" or "off"

Optional env:
    YUE2_ADAPTER         AR LoRA directory (empty => none)
    YUE2_ADAPTER_NAR     NAR LoRA directory (empty => none)
    YUE2_MELODY_FILE     ABC melody file (optional; render without ABC if unset)
    YUE2_LEGACY_SCALE    "1" => old buggy per-module LoRA scaling
    YUE2_NO_LORA         "1" => run the base model (no adapters at all)
    YUE2_STYLE_TRAIN     "1" => take the style from the train manifest
    YUE2_TRAIN_MANIFEST  manifest path used by YUE2_STYLE_TRAIN
    YUE2_MAX_TOKENS      semantic max tokens (default 8000)
"""
from __future__ import annotations

import gc
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from yue2 import YuE2Pipeline
from yue2.protocol import GenerationConfig, Sampling

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "train"))
from lora_pipeline import LoRAYuE2Pipeline  # noqa: E402

# ---------------------------------------------------------------------------
# 0. Configuration (everything comes from the environment)
# ---------------------------------------------------------------------------

_REQUIRED_ENV = (
    "YUE2_SONG_NAME", "YUE2_STYLE", "YUE2_LYRICS_FILE", "YUE2_SEED",
    "YUE2_LORA_SCALE", "YUE2_LORA_NAR_SCALE", "YUE2_GUIDANCE", "YUE2_COT_MODE",
)
_missing = [key for key in _REQUIRED_ENV if not os.environ.get(key)]
if _missing:
    raise SystemExit(
        f"[generate] missing required env: {', '.join(_missing)}")

SONG_NAME = os.environ["YUE2_SONG_NAME"]
STYLE = os.environ["YUE2_STYLE"]
LYRICS = Path(os.environ["YUE2_LYRICS_FILE"]).read_text(encoding="utf-8").strip()
SEED = int(os.environ["YUE2_SEED"])
GUIDANCE_OVERRIDE = float(os.environ["YUE2_GUIDANCE"])
COT_MODE = os.environ["YUE2_COT_MODE"]
MAX_TOKENS = int(os.environ.get("YUE2_MAX_TOKENS") or 8000)

# AR-LoRA (plan/melody): the main lever on style. Empty means "none".
LORA_ADAPTER = os.environ.get("YUE2_ADAPTER") or None
LORA_NAR_ADAPTER = os.environ.get("YUE2_ADAPTER_NAR") or None
LORA_SCALE = float(os.environ["YUE2_LORA_SCALE"])
LORA_NAR_SCALE = float(os.environ["YUE2_LORA_NAR_SCALE"])
LEGACY_SCALE = os.environ.get("YUE2_LEGACY_SCALE") == "1"

_TRAIN_MANIFEST = (Path(os.environ["YUE2_TRAIN_MANIFEST"])
                   if os.environ.get("YUE2_TRAIN_MANIFEST") else None)

# ---------------------------------------------------------------------------
# 1. Paths (relative to the repo root, not cwd)
# ---------------------------------------------------------------------------

OUT_DIR = BASE_DIR / "outputs" / SONG_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLAN_DIR = OUT_DIR / "plan"
PLAN_DIR.mkdir(parents=True, exist_ok=True)
# Optional; if not set, render runs without ABC.
MELODY_PATH = (Path(os.environ["YUE2_MELODY_FILE"])
               if os.environ.get("YUE2_MELODY_FILE") else None)
ABC_PATH = PLAN_DIR / "score.abc"  # result of the plan stage

# Names under which YuE2 / different pipeline versions accept guidance.
GUIDANCE_CANDIDATES = ("cfg_scale", "guidance_scale", "guidance")


def resolved_style() -> str:
    if os.environ.get("YUE2_STYLE_TRAIN"):
        if _TRAIN_MANIFEST is None:
            print("[style] YUE2_STYLE_TRAIN set but YUE2_TRAIN_MANIFEST "
                  "is empty - using YUE2_STYLE")
        elif _TRAIN_MANIFEST.exists():
            try:
                entries = json.loads(_TRAIN_MANIFEST.read_text())["entries"]
            except (OSError, ValueError, KeyError):
                entries = []
            if entries:
                return entries[0]["style"]
            print(f"[style] train manifest has no entries "
                  f"({_TRAIN_MANIFEST}) - using YUE2_STYLE")
        else:
            print(f"[style] train manifest not found "
                  f"({_TRAIN_MANIFEST}) - using YUE2_STYLE")
    return STYLE


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def section_tags(lyrics: str) -> List[str]:
    return re.findall(r"^\[([^\]]+)\]", lyrics, flags=re.MULTILINE)


def abc_section_markers(abc_text: str) -> List[str]:
    return re.findall(r"^%\s*(.+)$", abc_text, flags=re.MULTILINE)


def fix_truncated_abc(abc_text: str) -> str:
    """FIX: an ABC file that the model truncated mid-bar will fail to render."""
    lines = abc_text.rstrip().splitlines()
    if not lines:
        return abc_text

    last = lines[-1].strip()

    if not last or last.startswith("%"):
        return abc_text

    if "|" in last:
        return abc_text

    if last.startswith("V:") and not re.search(r"[A-Ga-gzZ]", last):
        print(f"[ABC] cut off at '{last}' - appending 'z16|'")
        lines.append("z16|")
        return "\n".join(lines) + "\n"

    if re.fullmatch(r'"[^"]+"', last):
        print(f"[ABC] cut off at '{last}' - appending 'z16|'")
        lines[-1] = last + "z16|"
        return "\n".join(lines) + "\n"

    print(f"[ABC] last line has no '|': '{last}' - appending 'z16|'")
    lines[-1] = last + "z16|"
    return "\n".join(lines) + "\n"


def validate_abc(abc_text: str) -> None:
    if not abc_text.strip().startswith("X:"):
        raise SystemExit("[ABC] file does not start with 'X:' - not ABC.")
    if not re.search(r"^V:\s*Vocal\b", abc_text, flags=re.MULTILINE):
        print("[WARN] ABC has no 'V: Vocal' line - the model may not sing.")


def signature_info(callable_obj) -> Tuple[set, bool]:
    """Return (explicit parameter names, has **kwargs)."""
    try:
        sig = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return set(), False

    names: set = set()
    has_var_kw = False
    for name, p in sig.parameters.items():
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_kw = True
        elif p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.add(name)
    return names, has_var_kw


def detect_guidance_kwarg(callable_obj) -> Optional[str]:
    """Find the guidance parameter of a callable, or None."""
    names, has_var_kw = signature_info(callable_obj)
    for name in GUIDANCE_CANDIDATES:
        if name in names:
            return name
    if has_var_kw:
        return GUIDANCE_CANDIDATES[0]
    return None


def free_vram(pipe=None) -> None:
    if pipe is not None:
        try:
            pipe.close()
        except Exception:
            pass
    gc.collect()
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass
        try:
            torch.cuda.synchronize()
        except Exception:
            pass


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    print("[WARN] CUDA is not available - falling back to CPU "
          "(will be very slow).")
    return "cpu"


def load_pipe() -> YuE2Pipeline:
    device = pick_device()
    print(f"[pipeline] loading YuE2-3B + LoRA "
          f"(device={device}, low_vram=True, offload_ar=True)...")
    no_lora = bool(os.environ.get("YUE2_NO_LORA"))
    lora = None if no_lora else LORA_ADAPTER
    lora_nar = None if no_lora else LORA_NAR_ADAPTER
    print(f"[pipeline] lora={lora} lora_nar={lora_nar}")
    kwargs = dict(device=device, low_vram=True, offload_ar=True, lora=lora,
                  lora_nar=lora_nar, backend="torch-eager",
                  lora_scale=LORA_SCALE, lora_nar_scale=LORA_NAR_SCALE,
                  legacy_scale=LEGACY_SCALE,
                  generation_config=GenerationConfig(
                      semantic=Sampling(max_tokens=MAX_TOKENS)))
    if device == "cpu":
        kwargs = dict(device=device, lora=lora, lora_nar=lora_nar,
                      lora_scale=LORA_SCALE, lora_nar_scale=LORA_NAR_SCALE,
                      legacy_scale=LEGACY_SCALE)
    return LoRAYuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", **kwargs)


def filter_optional(fn, optional, label):
    """Keep only the optional kwargs that the signature supports.

    If the signature cannot be inspected or has **kwargs, pass everything
    through (the old "blind" behaviour).
    """
    names, has_var_kw = signature_info(fn)
    if has_var_kw or not names:
        return optional
    kept = {}
    for name, value in optional.items():
        if name in names:
            kept[name] = value
        else:
            print(f"[{label}] '{name}' is not in the signature - skipping.")
    return kept


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------


def stage_plan() -> None:
    tags = section_tags(LYRICS)
    print(f"[lyrics] {len(tags)} sections found: {tags}")

    pipe = load_pipe()
    try:
        # plan accepts cfg_scale explicitly (see yue2/cli.py); still detect
        # the exact name, falling back to cfg_scale.
        gkw = detect_guidance_kwarg(pipe.plan)
        if gkw is None:
            gkw = "cfg_scale"
            print("[plan] guidance parameter not found in signature, "
                  "trying cfg_scale directly.")
        else:
            print(f"[plan] guidance: {gkw}={GUIDANCE_OVERRIDE}")

        # Filter only the guidance kwarg: style/lyrics are explicit
        # parameters of pipe.plan and must never be dropped.
        guid = filter_optional(pipe.plan, {gkw: GUIDANCE_OVERRIDE},
                               label="plan")

        print("[plan] building the score for lyrics and style...")
        plan = pipe.plan(style=resolved_style(), lyrics=LYRICS, **guid)
        plan.save(str(PLAN_DIR))
    finally:
        free_vram(pipe)

    if not ABC_PATH.exists():
        raise SystemExit(f"[plan] plan.save did not create {ABC_PATH}")

    abc_text = ABC_PATH.read_text(encoding="utf-8")
    markers = abc_section_markers(abc_text)
    print(f"[plan] bars in ABC: {abc_text.count('|')} (rough estimate)")
    print(f"[plan] structural markers: {markers}")


def stage_render() -> None:
    abc_text: Optional[str] = None

    if MELODY_PATH is not None and MELODY_PATH.exists():
        raw = MELODY_PATH.read_text(encoding="utf-8")
        raw = fix_truncated_abc(raw)
        validate_abc(raw)

        fixed_path = OUT_DIR / "melody_fixed.abc"
        fixed_path.write_text(raw, encoding="utf-8")
        print(f"[render] melody: {MELODY_PATH} (~{raw.count('|')} bars)")
        print(f"[render] fixed copy written to {fixed_path}")
        abc_text = raw
    else:
        source = "not set" if MELODY_PATH is None else "not found"
        print(f"[render] melody ({source}: {MELODY_PATH}) - "
              "rendering without ABC, the model invents the melody.")

    pipe = load_pipe()
    try:
        gkw = detect_guidance_kwarg(pipe.__call__)
        optional: Dict[str, Any] = {}

        if gkw is not None:
            optional[gkw] = GUIDANCE_OVERRIDE
            print(f"[render] guidance: {gkw}={GUIDANCE_OVERRIDE}")
        else:
            # Even if no explicit param and no **kwargs, try cfg_scale as is;
            # filter_optional will drop it if it is not supported.
            optional["cfg_scale"] = GUIDANCE_OVERRIDE
            print("[render] guidance not found in signature - "
                  "trying cfg_scale (filter will drop it if unsupported).")

        optional["cot"] = COT_MODE

        base_kwargs: Dict[str, Any] = dict(
            style=resolved_style(), lyrics=LYRICS, seed=SEED,
        )
        # ABC is added only when a melody is present.
        if abc_text is not None:
            base_kwargs["abc"] = abc_text

        print(f"[render] generating audio (seed={SEED}, "
              f"abc={'yes' if abc_text is not None else 'no'})...")
        print(f"[render] optional kwargs: {list(optional.keys())}")
        optional = filter_optional(pipe.__call__, optional, label="render")
        song = pipe(**base_kwargs, **optional)

        tag = str(GUIDANCE_OVERRIDE).replace(".", "_")
        suffix = "with_abc" if abc_text is not None else "no_abc"
        flac_path = OUT_DIR / f"{SONG_NAME}_seed{SEED}_g{tag}_{suffix}.flac"
        song.save(str(flac_path))
        try:
            song.save_artifacts(str(OUT_DIR))
        except AttributeError:
            print("[render] save_artifacts unavailable - skipping.")
        print(f"[done] audio written to {flac_path}")
    finally:
        free_vram(pipe)


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "render"
    if stage == "plan":
        stage_plan()
    elif stage == "render":
        stage_render()
    else:
        print(f"Unknown stage: {stage!r}. Use: plan | render")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())