"""Model loading plus the memory arithmetic that drives cache sizing."""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ModelConfig

DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def load_model(cfg: ModelConfig):
    """Load weights and tokenizer.

    Left padding matters: for a decoder-only model, right padding would put pad
    tokens after the real prompt, so the last position of the sequence is a pad
    and the logits we sample from are garbage.
    """
    tok = AutoTokenizer.from_pretrained(
        cfg.model_id, trust_remote_code=cfg.trust_remote_code
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=DTYPES[cfg.dtype],
        trust_remote_code=cfg.trust_remote_code,
    )
    model = model.to(cfg.device).eval()
    return model, tok


def kv_bytes_per_token(model, dtype: torch.dtype) -> int:
    """Bytes of KV cache consumed by one token of one sequence.

    2 (K and V) x layers x kv_heads x head_dim x itemsize.

    This is the single most important number in the whole project. It sets how
    many concurrent sequences fit in VRAM, which sets the batch size the
    scheduler can actually reach, which sets throughput.
    """
    c = model.config
    n_layers = c.num_hidden_layers
    n_kv_heads = getattr(c, "num_key_value_heads", c.num_attention_heads)
    head_dim = getattr(c, "head_dim", c.hidden_size // c.num_attention_heads)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    return 2 * n_layers * n_kv_heads * head_dim * itemsize


def describe(model, dtype: torch.dtype) -> dict:
    params = sum(p.numel() for p in model.parameters())
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    per_tok = kv_bytes_per_token(model, dtype)

    info = {
        "params": params,
        "weight_mib": weight_bytes / 2**20,
        "kv_bytes_per_token": per_tok,
        "kv_mib_per_1k_tokens": per_tok * 1024 / 2**20,
    }

    if torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory
        free = total - weight_bytes
        info["gpu_total_mib"] = total / 2**20
        # Rough headroom estimate; Phase 2 replaces this with a real profiler.
        info["kv_tokens_in_80pct_free"] = int(free * 0.8 / per_tok)

    return info


def build_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Position ids from a left-padded attention mask.

    Left padding breaks the default assumption that position == index. A row
    padded with 3 tokens must start counting at 0 on its 4th slot, not its 1st,
    or RoPE rotates every query by the wrong angle and the outputs quietly
    degrade without ever erroring.
    """
    pos = attention_mask.long().cumsum(-1) - 1
    return pos.clamp(min=0)
