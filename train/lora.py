"""Minimal LoRA and trainable NAR flow-matching support for YuE2.

The acoustic (NAR) path of every Mixture-of-Transformers layer is wrapped with
low-rank adapters while the frozen AR path stays on CPU. Training reuses the
per-layer AR KV cache that ``yue2.nar.CachedNAR`` builds at inference, so a
small GPU never holds both full paths at once.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from yue2 import nar as nar_mod


class LoRALinear(nn.Linear):
    """An nn.Linear frozen base with trainable LoRA A/B factors.

    ``forward`` computes ``W x + (alpha / r) B (A x)``. The wrapped module keeps
    its standard ``weight``/``bias`` names so checkpoint loading stays trivial.
    """

    def __init__(self, linear: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        if not isinstance(r, int) or r < 1:
            raise ValueError("LoRA rank must be a positive integer")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be positive and finite")
        if not 0 <= dropout < 1:
            raise ValueError("LoRA dropout must be in [0, 1)")
        super().__init__(linear.in_features, linear.out_features,
                         bias=linear.bias is not None)
        with torch.no_grad():
            self.weight.copy_(linear.weight.detach())
            if linear.bias is not None:
                self.bias.copy_(linear.bias.detach())
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.r, self.alpha, self.dropout = r, float(alpha), float(dropout)
        work_dtype = linear.weight.dtype
        self.weight.data = self.weight.data.to(work_dtype)
        if self.bias is not None:
            self.bias.data = self.bias.data.to(work_dtype)
        self.lora_A = nn.Parameter(torch.empty(r, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.register_buffer("lora_scale", torch.tensor(self.alpha / self.r))
        self.lora_A.data = self.lora_A.data.to(work_dtype)
        self.lora_B.data = self.lora_B.data.to(work_dtype)
        self.lora_scale.data = self.lora_scale.data.to(work_dtype)
        if linear.weight.device.type != "cpu":
            self.to(device=linear.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = F.linear(x, self.weight, self.bias)
        hidden = F.linear(x, self.lora_A)
        if self.dropout and self.training:
            hidden = F.dropout(hidden, p=self.dropout, training=True)
        return result + self.lora_scale.to(result.dtype) * F.linear(hidden, self.lora_B)

    def merge(self) -> None:
        with torch.no_grad():
            self.weight.add_(
                (self.lora_scale.to(self.weight.dtype) * (self.lora_B @ self.lora_A)).to(self.weight.dtype))
            self.lora_A.zero_()
            self.lora_B.zero_()


def _wrap(parent: nn.Module, name: str, r: int, alpha: float, dropout: float,
          modules: list) -> None:
    current = getattr(parent, name)
    if not isinstance(current, nn.Linear):
        raise ValueError(f"{name} is not an nn.Linear module")
    setattr(parent, name, LoRALinear(current, r=r, alpha=alpha, dropout=dropout))
    modules.append(getattr(parent, name))


def attach_nar_lora(model, r: int = 8, alpha: float = 16.0, dropout: float = 0.0,
                    include_heads: bool = True) -> list:
    """Wrap the NAR attention/MLP linears (and optionally the latent heads).

    Returns the newly created LoRALinear modules in a stable order.
    """
    if not isinstance(include_heads, bool):
        raise TypeError("include_heads must be a bool")
    modules = []
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _wrap(layer.nar_self_attn, name, r, alpha, dropout, modules)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _wrap(layer.nar_mlp, name, r, alpha, dropout, modules)
    if include_heads:
        for name in ("vae2llm", "llm2vae"):
            _wrap(model, name, r, alpha, dropout, modules)
    return modules


def attach_ar_lora(model, r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> list:
    """Wrap the AR (planning) attention/MLP linears with LoRA.

    Targets the stage that decides melody/structure/style. Returns the new
    LoRALinear modules in a stable order.
    """
    modules = []
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            _wrap(layer.self_attn, name, r, alpha, dropout, modules)
        for name in ("gate_proj", "up_proj", "down_proj"):
            _wrap(layer.mlp, name, r, alpha, dropout, modules)
    return modules


def all_lora_modules(model):
    for module in model.modules():
        if isinstance(module, LoRALinear):
            yield module


def cast_lora_dtype(model, dtype):
    for module in all_lora_modules(model):
        module.lora_A.data = module.lora_A.data.to(dtype)
        module.lora_B.data = module.lora_B.data.to(dtype)
        module.lora_scale.data = module.lora_scale.data.to(dtype)


def trainable_parameters(model):
    return [module.lora_A for module in all_lora_modules(model)] + \
           [module.lora_B for module in all_lora_modules(model)]


def save_adapter(model, directory, meta=None):
    """Save the trainable LoRA tensors and a small config to ``directory``."""
    from safetensors.torch import save_file
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors = {name: parameter.detach().to("cpu", dtype=torch.float32).contiguous()
               for name, parameter in model.named_parameters() if parameter.requires_grad}
    if not tensors:
        raise ValueError("No trainable LoRA parameters found")
    params = list(all_lora_modules(model))
    if not params:
        raise ValueError("No LoRA modules attached")
    rank = params[0].r
    if any(module.r != rank for module in params):
        raise ValueError("All LoRA modules must share the same rank")
    alphas = {module.alpha for module in params}
    if len(alphas) != 1:
        raise ValueError("All LoRA modules must share the same alpha")
    save_file(tensors, directory / "lora.safetensors")
    config = {
        "rank": rank,
        "alpha": next(iter(alphas)),
        "dropout": params[0].dropout,
        "include_heads": any(name.split(".")[0] in {"vae2llm", "llm2vae"}
                             for name in tensors),
        "tensors": len(tensors),
    }
    if meta:
        config.update(meta)
    (directory / "adapter_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return directory / "lora.safetensors"


def adapter_hyperparameters(directory):
    directory = Path(directory)
    return json.loads((directory / "adapter_config.json").read_text(encoding="utf-8"))


def load_adapter(model, directory):
    from safetensors.torch import load_file
    directory = Path(directory)
    state = load_file(directory / "lora.safetensors")
    loaded = model.load_state_dict(state, strict=False)
    if loaded.unexpected_keys:
        raise ValueError(f"Unexpected adapter keys: {loaded.unexpected_keys}")


def merge_adapters(model):
    for module in all_lora_modules(model):
        module.merge()


@torch.no_grad()
def ar_kv_cache(model, ar_ids, attention="sdpa", query_chunk_size=None, device="cpu"):
    """Prefill the AR prefix+codec sequence and return per-layer K/V on CPU.

    Mirrors ``yue2.nar.CachedNAR._prefill`` with full visibility.
    """
    target = next(model.model.embed_tokens.parameters()).device
    ids = torch.as_tensor(ar_ids, dtype=torch.long, device=target).unsqueeze(0)
    ar_length = ids.shape[1]
    positions = torch.arange(ar_length, device=target)[None]
    cos, sin = model.model.rotary_emb(positions)
    x = model.model.embed_tokens(ids)
    caches = []
    for layer in model.model.layers:
        q, k, v = layer.self_attn.project_qkv(layer.input_layernorm(x), cos, sin)
        caches.append((k[0].detach().to(device), v[0].detach().to(device)))
        h = nar_mod.attention(q[0], k[0], v[0], causal=True, backend=attention,
                              query_chunk_size=query_chunk_size)
        x = x + layer.self_attn.o_proj(h.flatten(1)[None])
        x = x + layer.mlp(layer.post_attention_layernorm(x))
    return caches


def nar_positional(model, ar_length, nar_length, device):
    """Precompute NAR RoPE and latent position embeddings for one sample."""
    positions = torch.arange(ar_length, ar_length + nar_length, device=device)[None]
    cos, sin = model.model.rotary_emb(positions)
    local = torch.arange(nar_length, device=device).clamp(max=model.config.max_latent_frames - 1)
    pos_emb = model.latent_pos_embed(local)[None]
    return cos, sin, pos_emb


def nar_velocity(model, caches, cos, sin, pos_emb, x_t, raw_t):
    """Trainable NAR flow-matching velocity for one acoustic chunk.

    Matches ``yue2.nar.CachedNAR.velocity`` exactly but runs under Autograd for
    the wrapped LoRA parameters. ``caches`` may live on CPU; they are copied to
    ``x_t.device`` per call.
    """
    device, dtype = x_t.device, x_t.dtype
    T = x_t.shape[0]
    nar_length = T + 2
    if len(caches) != len(model.model.layers):
        raise ValueError("KV cache length must match the layer count")
    if nar_length != cos.shape[1]:
        raise ValueError("RoPE length must equal T + 2")
    x_nar = F.pad(x_t, (0, 0, 1, 1))
    shifted = model._shift_t_value(raw_t, device, dtype)
    x = model.vae2llm(x_nar[None])
    x = x + model.time_embedder(shifted.expand(nar_length))[None]
    x = x + pos_emb.to(device)
    for layer, (ar_k, ar_v) in zip(model.model.layers, caches):
        k_cache, v_cache = ar_k.to(device), ar_v.to(device)
        q, k, v = layer.nar_self_attn.project_qkv(
            layer.nar_input_layernorm(x), cos.to(device), sin.to(device))
        k, v = torch.cat((k_cache, k[0])), torch.cat((v_cache, v[0]))
        h = nar_mod.attention(q[0], k, v)
        x = x + layer.nar_self_attn.o_proj(h.flatten(1)[None])
        x = x + layer.nar_mlp(layer.nar_pre_mlp_layernorm(x))
    return model.llm2vae(model.model.norm(x))[0, 1:-1]