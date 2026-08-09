"""Shared fixtures.

Everything here is randomly initialised and tiny. No checkpoint is downloaded,
so the whole torch test suite runs on CPU in seconds.

The torch import is guarded rather than assumed, so that the pure-Python half
of the suite (block manager, scheduler -- where most of the interesting bugs
live) stays runnable in an environment with no torch at all. A broken CUDA
install raises ValueError rather than ImportError, which importorskip does not
catch, hence the broad except.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

try:
    import torch
except Exception as exc:  # pragma: no cover - environment dependent
    torch = None
    TORCH_ERROR = str(exc)
else:
    TORCH_ERROR = None

TORCH_TESTS = [
    "test_paged_attention.py",
    "test_model.py",
    "test_quant.py",
    "test_sampler.py",
    "test_integration.py",
]
collect_ignore = list(TORCH_TESTS) if torch is None else []


@dataclass
class TinyConfig:
    """Same field names transformers' Qwen2Config uses, so our model code does
    not need to know the difference."""

    vocab_size: int = 128
    hidden_size: int = 64
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2      # GQA: 2 query heads share each KV head
    intermediate_size: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10000.0
    max_position_embeddings: int = 512
    tie_word_embeddings: bool = False
    model_type: str = "qwen2"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


@pytest.fixture
def tiny_config():
    return TinyConfig()


@pytest.fixture
def tiny_model(tiny_config):
    from nanoserve.qwen2 import Qwen2ForCausalLM

    torch.manual_seed(0)
    m = Qwen2ForCausalLM(tiny_config, max_position=512).to(torch.float32).eval()
    for p in m.parameters():
        p.requires_grad_(False)
        # Default init leaves biases at zero and weights near-orthogonal, which
        # can hide indexing bugs by making outputs insensitive to position.
        p.normal_(0, 0.05)
    return m


@pytest.fixture
def make_cache():
    from nanoserve.kv_cache import PagedKVCache

    def _make(cfg, num_blocks=64, block_size=8, dtype=None, device="cpu"):
        head_dim = getattr(cfg, "head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
        return PagedKVCache(
            num_blocks=num_blocks,
            block_size=block_size,
            num_layers=cfg.num_hidden_layers,
            num_kv_heads=getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
            head_dim=head_dim,
            dtype=dtype or torch.float32,
            device=device,
        )

    return _make


needs_cuda = pytest.mark.skipif(
    torch is None or not torch.cuda.is_available(), reason="requires a CUDA device"
)
