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

sys.path.insert(0, str(Path(__file__).resolve().parent / "train"))
from lora_pipeline import LoRAYuE2Pipeline
# AR-LoRA (план/мелодия): основной кандидат на влияние на стиль.
LORA_ADAPTER = os.environ.get(
    "YUE2_ADAPTER", str(Path(__file__).resolve().parent / "train" / "adapter-ar-1200"))
LORA_NAR_ADAPTER = os.environ.get("YUE2_ADAPTER_NAR") or None
LORA_SCALE = float(os.environ.get("YUE2_LORA_SCALE") or 0.5)
LORA_NAR_SCALE = float(os.environ.get("YUE2_LORA_NAR_SCALE") or 1.0)
# YUE2_LEGACY_SCALE=1 enables the old (buggy) scaling behaviour where the
# multiplier was applied to every LoRA module at once instead of each
# adapter's own modules only.
LEGACY_SCALE = os.environ.get("YUE2_LEGACY_SCALE") == "1"

# ---------------------------------------------------------------------------
# 0. Пути (всё относительно __file__, а не cwd)
# ---------------------------------------------------------------------------

SONG_NAME = "new_horizon"

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "outputs" / SONG_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLAN_DIR = OUT_DIR / "plan"
PLAN_DIR.mkdir(parents=True, exist_ok=True)
MELODY_PATH = BASE_DIR / "melody.abc"   # ОПЦИОНАЛЬНО: если нет — рендер без ABC
ABC_PATH = PLAN_DIR / "score.abc"       # результат стадии plan

# ---------------------------------------------------------------------------
# 1. Стиль
# ---------------------------------------------------------------------------

STYLE = (
    # триггер LoRA — тот же дескриптор, что и в тренировочном датасете
    "Fast aggressive rock with hard electronic industrial backbone, "
    "high-speed driving rock energy in the style of Celldweller and Blue Stahli, "
    "260 BPM relentless double-time rock pulse, frantic breakbeat drums "
    "punching through a dense distorted guitar wall, "
    # ЯВНЫЙ рок-драйв
    "loud cranked distorted 8-string guitar riffs leading the mix, "
    "thick palm-muted galloping chugs, screaming lead guitar hooks, "
    "twin tracked wide stereo rhythm guitars, down-tuned drop F djent attack, "
    # электро/индастриал-основа
    "massive dark synth leads, glitchy industrial percussion, huge processed "
    "drums and thundering toms, heavy sub-bass drops, distorted wobble growling bass, "
    "epic cinematic strings and choir pads, contrast of harsh and majestic, "
    # драйв и микс
    "relentless driving rock energy, explosive drops, dark dystopian atmosphere, "
    "punchy aggressive transients, bright high frequencies, wide cinematic mix"
)

# ---------------------------------------------------------------------------
# Тренировочный стиль LoRA (для теста «совпадает ли кондиция»).
# YUE2_STYLE_TRAIN=1 включает его вместо STYLE.
# ---------------------------------------------------------------------------

_TRAIN_MANIFEST = Path(__file__).resolve().parent / "train" / "dataset-celldweller-blue-stahli-full" / "manifest.json"
TRAIN_STYLE = json.loads(_TRAIN_MANIFEST.read_text())["entries"][0]["style"] if _TRAIN_MANIFEST.exists() else STYLE


def resolved_style() -> str:
    return TRAIN_STYLE if os.environ.get("YUE2_STYLE_TRAIN") else STYLE

# ---------------------------------------------------------------------------
# 2. Лирика
# ---------------------------------------------------------------------------

LYRICS = """
[Intro]
[duet] (glitchy arpeggiator, rising synth layers, distorted 8-string guitar stab, no vocals)

[Verse 1]
[male] Machine heart, silver veins,
Running out of oxygen again,
I can hear the sirens call,
Racing down a neon hall

[Pre-Chorus]
[male] Can you feel it rising?
Rising through the wire,
One spark away from fire

[Chorus]
[duet] We are the overdrive,
Burning through the night,
Higher than the satellites,
We are the overdrive
Louder through the wires,
Never stopping fire,
We are the overdrive — tonight

[Verse 2]
[female] Eyes wide, walls of glass,
Running too fast to look back,
Every second ticking loud,
Flash of light inside the crowd

[Pre-Chorus]
[female] Can you feel it rising?
Rising like a wave,
Everything we crave

[Chorus]
[duet] We are the overdrive,
Burning through the night,
Higher than the satellites,
We are the overdrive
Louder through the wires,
Never stopping fire,
We are the overdrive — tonight

[Bridge]
[male] All the static fades to gold,
[duet] everything we leave, we hold
[female] One more heartbeat, one more chance,
[duet] burning brighter, take the dance

[Guitar Solo]
[duet] (screaming distorted 8-string guitar solo, glitchy heavy break, no vocals)

[Final Chorus]
[duet] We are the overdrive,
Burning through the night,
Higher than the satellites,
We are the overdrive
Louder through the wires,
Never stopping fire,
We are the overdrive — tonight

[Outro]
[duet] We are the overdrive...
Higher than the satellites...
"""

SEED = int(os.environ.get("YUE2_SEED") or 777)
LYRICS_OVERRIDE = os.environ.get("YUE2_LYRICS_FILE")
if LYRICS_OVERRIDE:
    LYRICS = Path(LYRICS_OVERRIDE).read_text(encoding="utf-8").strip()
GUIDANCE_OVERRIDE = 1
COT_MODE = "full"

# Имена, под которыми YuE2 / разные версии пайплайна принимают guidance.
GUIDANCE_CANDIDATES = ("cfg_scale", "guidance_scale", "guidance")

# ---------------------------------------------------------------------------
# Утилиты
# ---------------------------------------------------------------------------

def section_tags(lyrics: str) -> List[str]:
    return re.findall(r"^\[([^\]]+)\]", lyrics, flags=re.MULTILINE)


def abc_section_markers(abc_text: str) -> List[str]:
    return re.findall(r"^%\s*(.+)$", abc_text, flags=re.MULTILINE)


def fix_truncated_abc(abc_text: str) -> str:
    lines = abc_text.rstrip().splitlines()
    if not lines:
        return abc_text

    last = lines[-1].strip()

    if not last or last.startswith("%"):
        return abc_text

    if "|" in last:
        return abc_text

    if last.startswith("V:") and not re.search(r"[A-Ga-gzZ]", last):
        print(f"[ABC] обрыв на '{last}' — дописываю 'z16|'")
        lines.append("z16|")
        return "\n".join(lines) + "\n"

    if re.fullmatch(r'"[^"]+"', last):
        print(f"[ABC] обрыв на '{last}' — дописываю 'z16|'")
        lines[-1] = last + "z16|"
        return "\n".join(lines) + "\n"

    print(f"[ABC] последняя строка без '|': '{last}' — дописываю 'z16|'")
    lines[-1] = last + "z16|"
    return "\n".join(lines) + "\n"


def validate_abc(abc_text: str) -> None:
    if not abc_text.strip().startswith("X:"):
        raise SystemExit("[ABC] файл не начинается с 'X:' — это не ABC.")
    if not re.search(r"^V:\s*Vocal\b", abc_text, flags=re.MULTILINE):
        print("[WARN] в ABC нет вокальной линии 'V: Vocal' — модель может не спеть.")


def signature_info(callable_obj) -> Tuple[set, bool]:
    """
    Возвращает (множество имён явных параметров, есть ли **kwargs).
    """
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
    """
    Ищем guidance-параметр у конкретного callable.
    Возвращает имя, если оно ЯВНО в сигнатуре. Если его нет, но есть
    **kwargs — возвращаем первый кандидат (YuE2 cli-стиль прокидывает
    cfg_scale через allowed-kwargs набор).
    """
    names, has_var_kw = signature_info(callable_obj)
    for name in GUIDANCE_CANDIDATES:
        if name in names:
            return name
    if has_var_kw:
        # Явного параметра нет, но **kwargs есть — пробуем первый кандидат.
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
    print("[WARN] CUDA недоступна — переключаюсь на CPU (будет очень медленно).")
    return "cpu"


def load_pipe() -> YuE2Pipeline:
    device = pick_device()
    print(f"[pipeline] загружаю YuE2-3B + LoRA celldweller (device={device}, low_vram=True, offload_ar=True)...")
    lora = None if os.environ.get("YUE2_NO_LORA") else LORA_ADAPTER
    print(f"[pipeline] lora={lora} lora_nar={LORA_NAR_ADAPTER}")
    kwargs = dict(device=device, low_vram=True, offload_ar=True, lora=lora,
                  lora_nar=LORA_NAR_ADAPTER, backend="torch-eager",
                  lora_scale=LORA_SCALE, lora_nar_scale=LORA_NAR_SCALE,
                  legacy_scale=LEGACY_SCALE,
                  generation_config=GenerationConfig(
                      semantic=Sampling(max_tokens=8000)))
    if device == "cpu":
        kwargs = dict(device=device, lora=lora, lora_nar=LORA_NAR_ADAPTER,
                      lora_scale=LORA_SCALE, lora_nar_scale=LORA_NAR_SCALE,
                      legacy_scale=LEGACY_SCALE)
    return LoRAYuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", **kwargs)


def call_with_per_kwarg_fallback(
    fn,
    base_kwargs: Dict[str, Any],
    optional_kwargs: Dict[str, Any],
    label: str,
):
    """
    FIX: пробуем вызвать со всеми optional. Если TypeError — НЕ выкидываем
    все optional сразу, а откатываемся по одному: сначала выкидываем тот,
    на который ругнулся TypeError, остальные оставляем.

    Так cfg_scale не потеряется из-за того, что, например, `cot` не
    поддерживается сигнатурой.
    """
    current = dict(optional_kwargs)
    while True:
        try:
            return fn(**base_kwargs, **current)
        except TypeError as e:
            msg = str(e)
            dropped = None
            # Пытаемся понять, на какое имя ругается TypeError.
            for name in list(current.keys()):
                if name in msg:
                    dropped = name
                    break
            if dropped is None:
                # Не смогли определить виновника — откатываемся полностью.
                if current:
                    print(f"[{label}] TypeError без явного виновника: {msg}")
                    print(f"[{label}] откатываюсь вообще без optional-параметров.")
                    current = {}
                    continue
                raise
            print(f"[{label}] TypeError на '{dropped}': {msg}")
            print(f"[{label}] выбрасываю '{dropped}' и повторяю (остальные оставляю).")
            current.pop(dropped, None)
            if not current:
                # Осталось пусто — пробуем голый base.
                continue


# ---------------------------------------------------------------------------
# Стадии
# ---------------------------------------------------------------------------

def stage_plan() -> None:
    tags = section_tags(LYRICS)
    print(f"[lyrics] найдено {len(tags)} секций: {tags}")

    pipe = load_pipe()
    try:
        # plan у YuE2 принимает cfg_scale явно (см. yue2/cli.py), поэтому
        # пробуем найти имя параметра, а если не нашли — используем cfg_scale.
        gkw = detect_guidance_kwarg(pipe.plan)
        if gkw is None:
            gkw = "cfg_scale"
            print("[plan] guidance-параметр не найден в сигнатуре, "
                  "пробую cfg_scale напрямую.")
        else:
            print(f"[plan] guidance: {gkw}={GUIDANCE_OVERRIDE}")

        plan_kwargs: Dict[str, Any] = {
            "style": resolved_style(),
            "lyrics": LYRICS,
            gkw: GUIDANCE_OVERRIDE,
        }

        print("[plan] строю партитуру под лирику и стиль...")
        plan = pipe.plan(**plan_kwargs)
        plan.save(str(PLAN_DIR))
    finally:
        free_vram(pipe)

    if not ABC_PATH.exists():
        raise SystemExit(f"[plan] plan.save не создал {ABC_PATH}")

    abc_text = ABC_PATH.read_text(encoding="utf-8")
    markers = abc_section_markers(abc_text)
    print(f"[plan] тактов в ABC: {abc_text.count('|')} (грубая оценка)")
    print(f"[plan] структурные метки: {markers}")


def stage_render() -> None:
    # FIX: melody.abc теперь необязателен.
    abc_text: Optional[str] = None

    if MELODY_PATH.exists():
        raw = MELODY_PATH.read_text(encoding="utf-8")
        raw = fix_truncated_abc(raw)
        validate_abc(raw)

        fixed_path = OUT_DIR / "melody_fixed.abc"
        fixed_path.write_text(raw, encoding="utf-8")
        print(f"[render] мелодия: {MELODY_PATH} (~{raw.count('|')} тактов)")
        print(f"[render] дочиненная копия: {fixed_path}")
        abc_text = raw
    else:
        print(f"[render] {MELODY_PATH.name} не найден — "
              "рендерю без ABC, мелодию сочинит сама модель.")

    pipe = load_pipe()
    try:
        gkw = detect_guidance_kwarg(pipe.__call__)
        optional: Dict[str, Any] = {}

        if gkw is not None:
            optional[gkw] = GUIDANCE_OVERRIDE
            print(f"[render] guidance: {gkw}={GUIDANCE_OVERRIDE} (передаю)")
        else:
            # FIX: даже если параметра нет в сигнатуре и нет **kwargs —
            # пробуем cfg_scale как есть. call_with_per_kwarg_fallback
            # аккуратно выкинет его, если TypeError.
            optional["cfg_scale"] = GUIDANCE_OVERRIDE
            print("[render] guidance не найден в сигнатуре — "
                  "пробую cfg_scale вслепую (fallback выкинет при TypeError).")

        optional["cot"] = COT_MODE

        base_kwargs: Dict[str, Any] = dict(
            style=resolved_style(), lyrics=LYRICS, seed=SEED,
        )
        # abc добавляем ТОЛЬКО если мелодия есть.
        if abc_text is not None:
            base_kwargs["abc"] = abc_text

        print(f"[render] генерирую аудио (seed={SEED}, "
              f"abc={'yes' if abc_text is not None else 'no'})...")
        print(f"[render] optional kwargs: {list(optional.keys())}")
        song = call_with_per_kwarg_fallback(
            pipe, base_kwargs, optional, label="render",
        )

        tag = f"g{GUIDANCE_OVERRIDE}".replace(".", "_")
        suffix = "with_abc" if abc_text is not None else "no_abc"
        flac_path = OUT_DIR / f"{SONG_NAME}_seed{SEED}_{tag}_{suffix}.flac"
        song.save(str(flac_path))
        try:
            song.save_artifacts(str(OUT_DIR))
        except AttributeError:
            print("[render] save_artifacts недоступен — пропускаю.")
        print(f"[done] аудио: {flac_path}")
    finally:
        free_vram(pipe)


def main() -> int:
    stage = sys.argv[1] if len(sys.argv) > 1 else "render"
    if stage == "plan":
        stage_plan()
    elif stage == "render":
        stage_render()
    else:
        print(f"Неизвестная стадия: {stage!r}. Используйте: plan | render")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())