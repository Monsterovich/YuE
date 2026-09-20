# Training LoRA Adapters for YuE2

Fine-tuning the YuE2-3B Mixture-of-Transformers model on custom music via two
separate LoRA adapters: one for **AR** (autoregressive planning) and one for
**NAR** (acoustic flow-matching). Every experiment ran on a single ~8 GB GPU
with BF16 and aggressive CPU offloading.

---

## Architecture recap

YuE2-3B is a single transformer with two parallel paths in every MoT layer:

| Path | Role | Key modules |
|------|------|-------------|
| **AR** | Semantic planning — melody, rhythm, structure, lyrics alignment | `self_attn`, `mlp`, `embed_tokens`, `lm_head` |
| **NAR** | Acoustic flow-matching — timbre, production, mix | `nar_self_attn`, `nar_mlp`, `vae2llm`, `llm2vae`, `time_embedder` |

LoRA wraps the Q/K/V/O projection linears of `self_attn` (AR) or
`nar_self_attn` (NAR) plus the gate/up/down linears of the corresponding MLP.

---

## Pipeline overview

```
FLAC source
    │
    ▼
preprocess.py          →  tokens.npy, prefix.npy, latents.npy, manifest.json
    │
    ├── train_lora_ar.py    (teacher-forced next-token CE on prefix+codec)
    └── train_lora_nar.py   (flow-matching velocity MSE on latents)
    │
    ├── lora_pipeline.py    → inference with adapter loaded (no merge)
    └── merge_lora.py       → bake adapter into a standalone model directory
```

---

## Step 1 — Preprocessing (`preprocess.py`)

Turns a folder of FLAC tracks into paired (semantic tokens, acoustic latents):

1. **AR model** generates semantic codec tokens from a text prompt (style +
   lyrics, `cot="off"`).
2. **VAE encoder** computes acoustic latents of the matching audio slice.
3. Saves `{row}.tokens.npy`, `{row}.prefix.npy`, `{row}.latents.npy` and a
   `manifest.json` with FPS = 25 (48 000 Hz / 1 920 VAE downsample ratio).

Sidecar files per track: `*.style.txt` and `*.txt`/`*.lyrics.txt` for custom
prompts.

```bash
python train/preprocess.py /path/to/songs \
    --out train/dataset \
    --style "Industrial rock, aggressive electronic production" \
    --clip-seconds 0 \       # 0 = full track
    --seeds 1
```

Key settings:
- `clip-seconds`: 0 keeps the full track; positive values truncate early.
- VAE encoding runs chunked (30 s + 2 s overlap per side) to avoid OOM.
- Max codec length capped at 24 064 tokens to stay within `CONTEXT = 24 576`.

---

## Step 2 — AR LoRA training (`train_lora_ar.py`)

The AR stage decides melody and structure — the part most responsible for
*musical style*. Training teacher-forces next-token cross-entropy on random
windows of `prefix + codec + MUSIC_END` sequences.

```bash
python train/train_lora_ar.py \
    --dataset train/dataset \
    --out train/adapter-ar \
    --r 8 --alpha 8.0 \
    --lr 1e-4 --wd 1e-2 \
    --window 2048 \
    --steps 1200 \
    --seed 123
```

### Memory trick

AR training never touches NAR modules. All `nar_self_attn`, `nar_mlp`, etc.
stay on CPU; only the AR path + heads live on GPU. Per-layer gradient
checkpointing is applied to keep VRAM flat regardless of sequence length.

### Training loop

Each step picks a random sequence, slices a random window of `--window` tokens,
and runs one forward + backward pass through the AR layers only. Loss = chunked
cross-entropy over `--ce-chunk` positions at a time. Gradients are clipped to
1.0.

---

## Step 3 — NAR LoRA training (`train_lora_nar.py`)

The NAR stage performs acoustic flow-matching: given a noisy latent state
`x_t = t·noise + (1-t)·latents`, the model predicts the velocity
`noise − latents`. Training samples are 96 s windowed clips.

```bash
python train/train_lora_nar.py \
    --dataset train/dataset-nar-clips \
    --out train/adapter-nar \
    --r 32 --alpha 48.0 \
    --lr 1e-4 --wd 1e-2 \
    --max-frames 2400 \
    --steps 1200 \
    --seed 2026 \
    --loss-variant normalized \
    --shape-w 0.9 \
    --grad-checkpoint
```

### AR KV cache prefill

NAR training needs the full AR context but must not load AR weights on a small
GPU. Solution: **prefill every sample's AR KV cache to disk once** before
training starts.

1. Load AR weights onto GPU, run `ar_kv_cache()` → per-layer K/V tensors.
2. Save `{row}.pt` to `adapter/cache/`.
3. Move AR weights back to CPU, load NAR weights + heads onto GPU.

At each training step the cache is loaded from disk into GPU memory, so only
the NAR path and RoPE/heads are live.

### Loss variants

- `standard`: plain MSE between predicted and target velocity.
- `normalized` (default): scale-free shape MSE (normalize both sides by
  per-sample std) plus a magnitude term weighted by `--shape-w` (default 0.9).
  More stable early in training.

---

## Adapting to small GPUs

Both scripts use the same placement strategy — never host both AR and NAR on
GPU simultaneously:

| Phase | AR modules | NAR modules | Heads |
|-------|-----------|-------------|-------|
| AR training | GPU | CPU | GPU |
| NAR prefill | GPU | CPU | GPU |
| NAR training | CPU | GPU | GPU |

For a GPU with ~8 GB VRAM, AR training supports windows up to ~2 048 tokens;
NAR training supports clips up to ~2 400 latent frames (~96 s). Longer content
is windowed or skipped.

---

## Hyperparameters used

### AR runs (r = 8, α = 8.0, lr = 1e-4, wd = 1e-2)

- 1 200 steps, window ≈ 1 900 tokens → final CE ≈ 0.31
- 1 200 steps, window ≈ 2 400 tokens → final CE ≈ 1.20

### NAR runs (r = 32, α = 48.0, lr = 1e-4, wd = 1e-2)

- 1 200 steps, clips up to 2 400 latent frames → final MSE ≈ 1.34
- 1 200 steps, clips up to 2 400 latent frames → final MSE ≈ 1.38

---

## Inference (`lora_pipeline.py`)

`LoRAYuE2Pipeline` wraps `YuE2Pipeline` to load adapters at construction time
without merging into the base weights:

```python
from train.lora_pipeline import LoRAYuE2Pipeline

pipe = LoRAYuE2Pipeline.from_pretrained(
    "m-a-p/YuE2-3B",
    lora="train/adapter-ar",       # AR adapter (planning)
    lora_nar="train/adapter-nar",  # NAR adapter (timbre)
    lora_scale=2.0,                # multiply alpha/r
    lora_nar_scale=1.5,
    device="cuda",
    low_vram=True,
)
```

The adapters are attached after low-vram placement, so the base checkpoint
stays untouched. Swap adapters freely by changing the path.

Effective strength of an adapter is `alpha / r` (baked into a `lora_scale`
buffer at attach time) times the `lora_scale`/`lora_nar_scale` multiplier.
Tune the multiplier by ear: low values give a clean but faint imprint of the
training data, while values that push `alpha / r * multiplier` too high (most
often past ~1.0 on NAR) collapse into harsh noise/artifacts.

`legacy_scale=True` reproduces the **old, buggy** behaviour where each
multiplier was applied to *every* LoRA module in the model instead of only the
adapter's own modules, so the last applied multiplier silently rescaled the
other stage too. It exists only to replicate pre-fix generations; leave it off
unless you deliberately want that coupling.

---

## Merging (`merge_lora.py`)

Bakes a NAR adapter into a standalone model directory that works with vanilla
`YuE2Pipeline.from_pretrained()`:

```bash
python train/merge_lora.py \
    --model m-a-p/YuE2-3B \
    --adapter train/adapter-nar \
    --out train/merged/merged-nar
```

Output contains `model.safetensors` (merged weights), `weights_manifest.json`,
and a copy of all auxiliary files from the base checkpoint.

---

## Notes

- AR adapters affect melody, arrangement, and structure; NAR adapters affect
  timbre and production style.
- AR r = 8 is very small for a 3 B model — more rank (32–64) and more data
  is likely needed for a strong style shift.
- NAR adapters converge quickly (~1 s/step on 8 GB GPU) but the effective
  timbre change is subtle at scale ≤ 2; higher scales tend to introduce
  artifacts.
- Spectral cosine correlation between generated and source audio is a useful
  quick metric for measuring adapter effect.
