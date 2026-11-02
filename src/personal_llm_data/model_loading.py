"""Shared, explicit single-GPU loading for controlled BF16/int8/NF4 runs."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig


QUANTIZATION_MODES = ("none", "int8", "nf4")


def load_causal_model(model_path: str, quantization: str = "none"):
    if quantization not in QUANTIZATION_MODES:
        raise ValueError(f"Unsupported quantization mode: {quantization}")
    kwargs = {"dtype": torch.bfloat16, "attn_implementation": "sdpa"}
    if quantization != "none":
        if not torch.cuda.is_available():
            raise RuntimeError("Controlled int8/NF4 experiments require CUDA")
        if quantization == "int8":
            quant_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)
        else:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        kwargs.update(quantization_config=quant_config, device_map={"": torch.cuda.current_device()})
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    if quantization != "none":
        flag = "is_loaded_in_8bit" if quantization == "int8" else "is_loaded_in_4bit"
        if not getattr(model, flag, False):
            raise RuntimeError(f"Requested {quantization} but the model was not quantized")
    return model
