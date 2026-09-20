"""Plug a trained LoRA adapter into the pipeline without merging weights.

Subclass of ``YuE2Pipeline``: pass ``lora="train/adapter"`` to
``from_pretrained`` and the saved NAR adapter is attached to the NAR path the
first time the flow model is loaded. The base checkpoint stays untouched, so
you can swap adapters freely:
``LoRAYuE2Pipeline.from_pretrained("m-a-p/YuE2-3B", lora="train/adapter")``.

The wrapper is applied after low-vram placement (NAR modules already on CPU,
heads on the accelerator); RoPE/LAYERS keep their original layout because AR
behaviour must remain unchanged.
"""
from __future__ import annotations

from pathlib import Path

from safetensors.torch import load_file

from yue2 import YuE2Pipeline

from lora import (LoRALinear, adapter_hyperparameters, attach_ar_lora,
                  attach_nar_lora, load_adapter)


def set_lora_scale(modules, multiplier):
    """Scale the contribution of the given LoRALinear ``modules`` by ``multiplier``.

    LoRA's effective strength is ``alpha / r`` baked into the ``lora_scale``
    buffer; re-multiplying that buffer shifts the adapter intensity without
    re-training. Pass only the modules of one adapter so that AR and NAR
    scales stay independent. ``None``/``1.0`` keeps the trained strength.
    """
    if multiplier is None:
        return
    for module in modules:
        module.lora_scale.data.mul_(module.lora_scale.new_tensor(float(multiplier)))


class LoRAYuE2Pipeline(YuE2Pipeline):
    def __init__(self, model_dir, vae_dir, *, lora=None, lora_nar=None,
                 lora_scale=None, lora_nar_scale=None, **kwargs):
        self.lora_adapter = None if lora is None else Path(lora)
        self.lora_nar_adapter = None if lora_nar is None else Path(lora_nar)
        self.lora_scale = None if lora_scale is None else float(lora_scale)
        self.lora_nar_scale = None if lora_nar_scale is None else float(lora_nar_scale)
        self._lora_attached = lora is None and lora_nar is None
        for adapter in (self.lora_adapter, self.lora_nar_adapter):
            if adapter is not None:
                adapter_hyperparameters(adapter)
        super().__init__(model_dir, vae_dir, **kwargs)

    def _load_model(self, for_nar=False):
        model = super()._load_model(for_nar)
        if not self._lora_attached:
            for adapter in (self.lora_adapter, self.lora_nar_adapter):
                if adapter is None:
                    continue
                is_ar = adapter is self.lora_adapter
                hyper = adapter_hyperparameters(adapter)
                rf = dict(r=hyper["rank"], alpha=hyper["alpha"],
                          dropout=hyper.get("dropout", 0.0))
                if hyper.get("ar"):
                    modules = attach_ar_lora(model, **rf)
                else:
                    keys = set(load_file(adapter / "lora.safetensors"))
                    include_heads = any(key.split(".")[0] in {"vae2llm", "llm2vae"}
                                       for key in keys)
                    modules = attach_nar_lora(model, include_heads=include_heads, **rf)
                load_adapter(model, adapter)
                set_lora_scale(modules, self.lora_scale if is_ar
                               else self.lora_nar_scale)
            self._lora_attached = True
        return model