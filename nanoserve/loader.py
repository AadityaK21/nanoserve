"""Load a HuggingFace checkpoint into our Qwen2 implementation.

transformers is used for exactly two things -- the config and the raw weight
tensors -- and then gets out of the way. Its modelling code is never on the hot
path.
"""

from __future__ import annotations

import gc

import torch

from .config import ModelConfig
from .qwen2 import Qwen2ForCausalLM

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_tokenizer(cfg: ModelConfig):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(cfg: ModelConfig):
    """Returns (model, hf_config, quant_report | None)."""
    from transformers import AutoConfig, AutoModelForCausalLM

    dtype = DTYPES[cfg.dtype]
    hf_cfg = AutoConfig.from_pretrained(cfg.model_id, trust_remote_code=cfg.trust_remote_code)
    if hf_cfg.model_type != "qwen2":
        raise ValueError(
            f"nanoserve implements the Qwen2 architecture; got model_type="
            f"{hf_cfg.model_type!r}. Llama-family support is a small change to "
            f"qwen2.py (drop the q/k/v biases)."
        )

    # Materialise on CPU first: on 8 GB you do not want two copies of the
    # weights resident on the GPU while the state dict is being transplanted.
    hf_model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id, torch_dtype=dtype, trust_remote_code=cfg.trust_remote_code
    )
    state = hf_model.state_dict()

    model = Qwen2ForCausalLM(hf_cfg, max_position=cfg.max_model_len)
    model = model.to(dtype)
    model.load_hf_state_dict(state)

    del hf_model, state
    gc.collect()

    # Recorded before quantisation: afterwards the weights live in buffers, not
    # parameters, and a parameter count would silently report ~0.
    model.num_params = sum(p.numel() for p in model.parameters())

    quant_report = None
    if cfg.quantization:
        from .quant import quantize_model

        quant_report = quantize_model(
            model, cfg.quantization, cfg.quant_group_size, cfg.quant_skip
        )

    model = model.to(cfg.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()

    return model, hf_cfg, quant_report


def weight_bytes(model) -> int:
    from .quant import _weight_bytes

    return _weight_bytes(model)
