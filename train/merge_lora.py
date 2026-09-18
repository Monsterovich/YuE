"""Merge a saved NAR LoRA adapter into a self-contained YuE2 model directory.

The output directory keeps the same file layout as the release checkpoint
(updated ``model.safetensors``, fresh ``weights_manifest.json``, and the
auxiliary config/code files), so ``YuE2Pipeline.from_pretrained(out)`` works
with unmodified runtime code.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from safetensors import safe_open
from safetensors.torch import load_file, save_file

from yue2 import modeling_yue2 as modeling_code
from yue2.modeling_yue2 import YuE2ForCausalLM
from yue2.storage import MODEL_FILES, MODEL_LICENSES, resolve_model

from lora import (adapter_hyperparameters, attach_nar_lora, cast_lora_dtype,
                  load_adapter, merge_adapters)


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="m-a-p/YuE2-3B")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        parser.error("out directory must be empty or absent")

    hyper = adapter_hyperparameters(args.adapter)
    adapter_keys = set(load_file(args.adapter / "lora.safetensors"))
    include_heads = any(key.split(".")[0] in {"vae2llm", "llm2vae"}
                       for key in adapter_keys)
    model_dir = Path(resolve_model(args.model))
    model = YuE2ForCausalLM.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True)
    model.requires_grad_(False).eval()
    attach_nar_lora(model, r=hyper["rank"], alpha=hyper["alpha"],
                    dropout=hyper.get("dropout", 0.0),
                    include_heads=include_heads)
    load_adapter(model, args.adapter)
    cast_lora_dtype(model, torch.bfloat16)
    merge_adapters(model)

    # Write only the tensors that exist in the original checkpoint.
    source_path = next(model_dir.glob("model.safetensors"))
    with safe_open(source_path, framework="pt") as handle:
        original_keys = set(handle.keys())
    state = model.state_dict()
    merged = {name: state[name].to(torch.bfloat16).contiguous()
              for name in sorted(original_keys) if name in state}
    missing = original_keys - set(merged)
    if missing:
        raise ValueError(f"Missing base tensors after merge: {sorted(missing)[:8]}")

    out.mkdir(parents=True)
    for name in sorted(MODEL_FILES):
        if name.endswith(".safetensors") or name == "weights_manifest.json":
            continue
        source = model_dir / name
        if source.is_file():
            shutil.copyfile(source, out / name)
    (out / "modeling_yue2.py").write_bytes(Path(modeling_code.__file__).read_bytes())
    for name in sorted(MODEL_LICENSES):
        source = model_dir / "licenses" / name
        if source.is_file():
            (out / "licenses").mkdir(exist_ok=True)
            shutil.copyfile(source, out / "licenses" / name)

    save_file(merged, out / "model.safetensors")
    digest = sha256_file(out / "model.safetensors")
    (out / "weights_manifest.json").write_text(
        json.dumps({"files": {"model.safetensors": {
            "sha256": digest, "bytes": (out / "model.safetensors").stat().st_size}}},
            indent=2) + "\n", encoding="utf-8")
    print(f"merged model -> {out} ({len(merged)} tensors)")


if __name__ == "__main__":
    main()