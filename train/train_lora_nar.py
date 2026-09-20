"""Train a NAR LoRA on YuE2 over precomputed (tokens, latents) pairs.

Every Mixture-of-Transformers layer has two full paths: an AR path used for
autoregressive planning/semantics and a NAR path used for acoustic flow
matching. Only the NAR linears are adapted. The AR prefix and codec tokens for
each training sample are prefilled once into a per-layer KV cache on CPU, so a
small GPU never hosts both paths.

The flow-matching objective follows the release protocol: for a timestep
``t``, an acoustic state ``x_t = t*noise + (1-t)*latents`` and raw timestep
``logit(t)``, the model predicts the velocity ``noise - latents``.
"""
from __future__ import annotations

import argparse
import dataclasses
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

from lora import (adapter_hyperparameters, ar_kv_cache, attach_nar_lora,
                  load_adapter, nar_positional, nar_velocity, save_adapter,
                  trainable_parameters)


@dataclasses.dataclass
class Sample:
    row_id: str
    ar_ids: torch.Tensor  # [ar_length] int64 CPU
    ar_length: int
    nar_length: int
    z: torch.Tensor       # [T, 64] bf16 CPU
    frames: int
    caches: list = dataclasses.field(default_factory=list)
    cos: torch.Tensor = None
    sin: torch.Tensor = None
    pos_emb: torch.Tensor = None
    caches_path: str = ""


def _move_ar(model, device):
    for layer in model.model.layers:
        for module in (layer.input_layernorm, layer.self_attn,
                       layer.post_attention_layernorm, layer.mlp):
            module.to(device)


def _move_nar(model, device):
    for layer in model.model.layers:
        for module in (layer.nar_input_layernorm, layer.nar_self_attn,
                       layer.nar_pre_mlp_layernorm, layer.nar_mlp):
            module.to(device)


def _move_heads(model, device):
    for module in (model.model.rotary_emb, model.model.norm,
                   model.vae2llm, model.llm2vae,
                   model.time_embedder, model.latent_pos_embed):
        module.to(device)


def load_samples(dataset, max_frames):
    manifest = json.loads((Path(dataset) / "manifest.json").read_text(encoding="utf-8"))
    fps, data_dir = manifest["fps"], Path(dataset) / "data"
    samples = []
    for entry in manifest["entries"]:
        row = entry["id"]
        frames = int(entry["frames"])
        if frames < 1 or frames > max_frames:
            continue
        codec = np.load(data_dir / f"{row}.tokens.npy").astype(np.int64).tolist()
        prefix = np.load(data_dir / f"{row}.prefix.npy").astype(np.int64).tolist()
        latents = np.load(data_dir / f"{row}.latents.npy").astype(np.float32)
        if latents.shape[0] < 1 or latents.shape[1] != 64:
            continue
        frames = min(frames, len(codec), latents.shape[0])
        codec = codec[:frames]
        ar = prefix + [int(t) + CODEC_OFFSET for t in codec] + [MUSIC_END]
        if len(ar) + frames + 2 > CONTEXT:
            continue
        samples.append(Sample(
            row_id=row,
            ar_ids=torch.tensor(ar, dtype=torch.long),
            ar_length=len(ar),
            nar_length=frames + 2,
            z=torch.from_numpy(latents[:frames]).to(torch.bfloat16),
            frames=frames))
    print(f"samples: {len(samples)}")
    return samples


def flow_step(model, sample, device, base_seed, grad_checkpoint=False,
              loss_variant="normalized", shape_w=0.9):
    generator = torch.Generator(device=device).manual_seed(base_seed)
    T = sample.frames
    z = sample.z.to(device)
    noise = torch.randn(T, 64, dtype=torch.bfloat16, device=device, generator=generator)
    t = torch.rand((), generator=torch.Generator(device="cpu")
                   .manual_seed(base_seed + 1)).item()
    raw = torch.logit(torch.tensor(t, dtype=torch.float64)).clamp(-20, 20).item()
    x_t = t * noise + (1 - t) * z
    target = noise - z

    cached = torch.load(sample.caches_path, weights_only=False,
                        map_location="cpu")
    caches = [(k.to(device), v.to(device)) for k, v in cached["caches"]]
    cos, sin, pos_emb = (cached["cos"].to(device), cached["sin"].to(device),
                         cached["pos_emb"].to(device))

    if not grad_checkpoint:
        predicted = nar_velocity(model, caches, cos, sin, pos_emb, x_t, raw)
    else:
        predicted = checkpointed_velocity(model, caches, cos, sin, pos_emb, x_t, raw)
    if loss_variant == "standard":
        return F.mse_loss(predicted, target)
    eps = 1e-6
    pred_n = predicted / (predicted.std() + eps)
    target_n = target / (target.std() + eps)
    mag = F.mse_loss(predicted, target) / (target.pow(2).mean().detach() + eps)
    return shape_w * F.mse_loss(pred_n, target_n) + (1.0 - shape_w) * mag


def checkpointed_velocity(model, caches, cos, sin, pos_emb, x_t, raw):
    device, dtype = x_t.device, x_t.dtype
    T = x_t.shape[0]
    nar_length = T + 2
    x_nar = F.pad(x_t, (0, 0, 1, 1))
    shifted = model._shift_t_value(raw, device, dtype)
    x = model.vae2llm(x_nar[None])
    x = x + model.time_embedder(shifted.expand(nar_length))[None]
    x = x + pos_emb
    def run_layer(x, layer, k_cache, v_cache, cos, sin):
        from yue2 import nar as nar_mod
        q, k, v = layer.nar_self_attn.project_qkv(
            layer.nar_input_layernorm(x), cos, sin)
        k = torch.cat((k_cache, k[0]))
        v = torch.cat((v_cache, v[0]))
        h = nar_mod.attention(q[0], k, v)
        x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
        return x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))

    for layer, (k_cache, v_cache) in zip(model.model.layers, caches):
        # No detach here: with use_reentrant=False the graph stays connected
        # between layers, so every layer's LoRA receives gradients.
        x = checkpoint.checkpoint(run_layer, x, layer, k_cache, v_cache,
                                  cos, sin, use_reentrant=False)
    return model.llm2vae(model.model.norm(x))[0, 1:-1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--out", type=Path, default=Path("train/adapter"))
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=1e-2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps", type=int, default=0,
                        help="total steps (overrides epochs)")
    parser.add_argument("--max-frames", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--grad-checkpoint", dest="grad_checkpoint",
                        action="store_true", default=True)
    parser.add_argument("--no-grad-checkpoint", dest="grad_checkpoint",
                        action="store_false")
    parser.add_argument("--no-heads", dest="include_heads",
                        action="store_false", default=True)
    parser.add_argument("--loss-variant", choices=("standard", "normalized"),
                        default="normalized",
                        help="normalized: scale-free shape MSE (ignores global gain)")
    parser.add_argument("--shape-w", type=float, default=0.9,
                        help="weight of scale-free shape term in normalized loss")
    parser.add_argument("--no-reuse-cache", dest="reuse_cache",
                        action="store_false", default=True,
                        help="always re-prefill AR KV caches even if on disk")
    parser.add_argument("--resume", type=Path, default=None,
                        help="load trainable LoRA weights from an existing adapter "
                             "and continue training from there")
    args = parser.parse_args()

    if args.r < 1 or not math.isfinite(args.alpha) or args.alpha <= 0:
        raise ValueError("rank must be positive; alpha positive and finite")
    if args.steps < 0 or args.epochs < 1 or args.r < 1:
        raise ValueError("invalid training budget")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("training requires a CUDA device (BF16 backend)")

    model_dir = resolve_model(args.model)
    model = YuE2ForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.requires_grad_(False).eval()
    if not model.config.tie_word_embeddings:
        pass  # lm_head is independent; it stays frozen on CPU.
    attach_nar_lora(model, r=args.r, alpha=args.alpha, dropout=args.dropout,
                    include_heads=args.include_heads)
    if args.resume is not None:
        load_adapter(model, args.resume)
        print(f"resumed LoRA from {args.resume}")

    _move_ar(model, torch.device("cpu"))
    _move_nar(model, torch.device("cpu"))
    _move_heads(model, device)
    model.model.embed_tokens.to(torch.device("cpu"))
    model.lm_head.to(torch.device("cpu"))

    samples = load_samples(args.dataset, args.max_frames)
    if not samples:
        raise SystemExit("no usable samples in dataset")

    # Prefill every sample's AR KV cache; caches spill to disk (CPU RAM on
    # this machine can't hold all long-track KV caches at once).
    cache_dir = Path(args.out) / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    for index, sample in enumerate(samples):
        sample.caches_path = str(cache_dir / f"{sample.row_id}.pt")
        if args.reuse_cache and Path(sample.caches_path).exists() \
                and Path(sample.caches_path).stat().st_size > 0:
            print(f"reuse cache {index + 1}/{len(samples)} {sample.row_id}")
            continue
        _move_nar(model, torch.device("cpu"))
        _move_ar(model, device)
        model.model.embed_tokens.to(device)
        model.lm_head.to(device)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        caches = ar_kv_cache(model, sample.ar_ids.tolist())
        cos, sin, pos_emb = nar_positional(
            model, sample.ar_length, sample.nar_length, device)
        torch.save({
            "caches": [(k.cpu().contiguous(), v.cpu().contiguous())
                       for k, v in caches],
            "cos": cos.cpu(), "sin": sin.cpu(), "pos_emb": pos_emb.cpu(),
        }, sample.caches_path)
        sample.caches = None
        sample.cos = sample.sin = sample.pos_emb = None
        del caches, cos, sin, pos_emb
        print(f"prefill {index + 1}/{len(samples)} {sample.row_id}: {sample.frames} frames")
    _move_ar(model, torch.device("cpu"))
    model.model.embed_tokens.to(torch.device("cpu"))
    model.lm_head.to(torch.device("cpu"))
    _move_nar(model, device)
    if device.type == "cuda":
        torch.cuda.empty_cache()

    parameters = trainable_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.wd)
    total = args.steps if args.steps > 0 else len(samples) * args.epochs
    print(f"trainable params: {sum(p.numel() for p in parameters)}")

    rng = random.Random(args.seed)
    losses, step = [], 0
    while step < total:
        sample = rng.choice(samples)
        optimizer.zero_grad(set_to_none=True)
        loss = flow_step(model, sample, device, args.seed + step,
                         grad_checkpoint=args.grad_checkpoint,
                         loss_variant=args.loss_variant, shape_w=args.shape_w)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        step += 1
        losses.append(float(loss.detach().cpu()))
        if step % 10 == 0 or step == total:
            recent = losses[-10:]
            print(f"step {step}/{total} mse={sum(recent) / len(recent):.6f} "
                  f"grad={grad_norm:.4f}")
        if step % args.checkpoint_every == 0 or step == total:
            save_adapter(model, args.out, meta={
                "epochs": args.epochs, "steps": step,
                "base_model": args.model, "seed": args.seed,
                "lr": args.lr, "weight_decay": args.wd,
                "last_mse": float(loss.detach().cpu())})
    print(f"saved adapter -> {args.out}/lora.safetensors")


if __name__ == "__main__":
    main()