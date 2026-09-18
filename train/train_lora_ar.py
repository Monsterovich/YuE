"""Train an AR (planning) LoRA on YuE2 over the tracked codec-token sequences.

The AR stage decides melody, riffs and structure — the part of the model that
carries musical style. We wrap the AR attention/MLP linears with LoRA and
teacher-force next-token cross-entropy on random windows of the target tracks'
``prefix + codec`` sequences (the same ``ar_ids`` convention as
``train_lora_nar``). NAR modules stay on CPU so the small GPU hosts the AR path.

This produces a LoRA whose *planning* shifts toward the trained artists, unlike
the NAR-velocity adapter which only tweaks acoustics and was inaudible.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.protocol import CODEC_OFFSET, CONTEXT, MUSIC_END
from yue2.storage import resolve_model

from lora import attach_ar_lora, load_adapter, save_adapter, trainable_parameters


def _move_nar_cpu(model):
    for layer in model.model.layers:
        for module in (layer.nar_input_layernorm, layer.nar_self_attn,
                       layer.nar_pre_mlp_layernorm, layer.nar_mlp):
            module.to("cpu")


def _place_ar_cuda(model, device):
    """Move only the AR planning path (plus heads) to ``device``.

    The full AR+NAR weight set does not fit this GPU; NAR modules stay on CPU
    because AR training never touches them (mirrors low-vram placement).
    """
    for module in (model.model.embed_tokens, model.model.norm,
                   model.model.rotary_emb, model.lm_head):
        module.to(device)
    for layer in model.model.layers:
        layer.input_layernorm.to(device)
        layer.self_attn.to(device)
        layer.post_attention_layernorm.to(device)
        layer.mlp.to(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _layer_ar(x, layer, cos, sin):
    h = layer.self_attn(layer.input_layernorm(x), cos, sin)
    x = x + h
    return x + layer.mlp(layer.post_attention_layernorm(x))


def checkpointed_ar_loss(model, ids, ce_chunk=512):
    """Per-layer-checkpointed causal-LM CE over one window.

    Mirrors the AR-only forward path of ``model.forward`` without KV caches.
    """
    x = model.model.embed_tokens(ids)
    seq = ids.shape[1]
    positions = torch.arange(seq, device=x.device)[None]
    cos, sin = model.model.rotary_emb(positions)
    for layer in model.model.layers:
        x = checkpoint.checkpoint(_layer_ar, x, layer, cos, sin,
                                  use_reentrant=False)
    hidden = model.model.norm(x)[0]
    total = seq - 1
    acc = torch.zeros((), dtype=hidden.dtype, device=hidden.device)
    start = 0
    while start < total:
        end = min(start + ce_chunk, total)
        logits = model.lm_head(hidden[start:end])
        targets = ids[0, start + 1:end + 1]
        acc = acc + F.cross_entropy(logits, targets) * (end - start)
        start = end
    return acc / total


def load_ar_sequences(dataset, max_frames):
    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    data_dir = Path(dataset) / "data"
    sequences = []
    for entry in manifest["entries"]:
        row = entry["id"]
        frames = int(entry["frames"])
        if frames < 1 or frames > max_frames:
            continue
        codec = np.load(data_dir / f"{row}.tokens.npy").astype(np.int64).tolist()
        prefix = np.load(data_dir / f"{row}.prefix.npy").astype(np.int64).tolist()
        frames = min(frames, len(codec))
        codec = codec[:frames]
        ar = prefix + [int(t) + CODEC_OFFSET for t in codec] + [MUSIC_END]
        if len(ar) + frames + 2 > CONTEXT:
            continue
        sequences.append(torch.tensor(ar, dtype=torch.long))
    print(f"ar sequences: {len(sequences)}")
    return sequences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--out", type=Path, default=Path("train/adapter-ar"))
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps", type=int, default=0,
                        help="total steps (overrides epochs)")
    parser.add_argument("--window", type=int, default=2048,
                        help="teacher-forcing window length in tokens")
    parser.add_argument("--ce-chunk", type=int, default=512,
                        help="positions per lm_head/cross-entropy chunk")
    parser.add_argument("--max-frames", type=int, default=7500)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--resume", type=Path, default=None,
                        help="start from a previously trained adapter directory")
    args = parser.parse_args()

    if args.r < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("rank must be positive; alpha positive and finite")
    if args.window < 64:
        raise ValueError("window too short")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("training requires a CUDA device (BF16 backend)")

    model_dir = resolve_model(args.model)
    model = YuE2ForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.requires_grad_(False).eval()
    attach_ar_lora(model, r=args.r, alpha=args.alpha, dropout=args.dropout)
    if args.resume is not None:
        load_adapter(model, args.resume)
        print(f"resumed LoRA from {args.resume}")
    _place_ar_cuda(model, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    # HF gradient_checkpointing_enable не совместим с этой кастомной
    # архитектурой; если упрёмся в память — добавить ручной per-layer checkpoint.

    sequences = load_ar_sequences(args.dataset, args.max_frames)
    if not sequences:
        raise SystemExit("no usable sequences in dataset")

    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    total = args.steps if args.steps > 0 else len(sequences) * args.epochs
    print(f"trainable params: {sum(p.numel() for p in parameters)}")

    rng = random.Random(args.seed)
    losses, step = [], 0
    while step < total:
        seq = rng.choice(sequences)
        length = seq.numel()
        # Prefer windows late in the track and sometimes the very start.
        if length <= args.window:
            start, window = 0, length
        else:
            start = rng.randrange(0, length - args.window + 1)
            window = args.window
        ids = seq[start:start + window].to(device).unsqueeze(0)

        optimizer.zero_grad(set_to_none=True)
        loss = checkpointed_ar_loss(model, ids, ce_chunk=args.ce_chunk)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        step += 1
        losses.append(float(loss.detach().cpu()))
        if step % 10 == 0 or step == total:
            recent = losses[-10:]
            print(f"step {step}/{total} ce={sum(recent) / len(recent):.4f} "
                  f"grad={grad_norm:.4f}")
        if step % args.checkpoint_every == 0 or step == total:
            save_adapter(model, args.out, meta={
                "ar": True, "window": window, "epochs": args.epochs,
                "steps": step, "base_model": args.model, "seed": args.seed,
                "lr": args.lr, "weight_decay": args.wd,
                "last_ce": float(loss.detach().cpu())})
    print(f"saved adapter -> {args.out}/lora.safetensors")


if __name__ == "__main__":
    main()