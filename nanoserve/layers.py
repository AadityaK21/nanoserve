"""Model building blocks.

Written out rather than imported from transformers for two reasons. Attention
has to talk to the paged cache, which no stock module knows about. And Phase 4
swaps every Linear for a quantised one, which is a one-line change here and a
monkey-patching exercise otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Qwen2 uses RMSNorm, not LayerNorm: no mean subtraction, no bias.

    The cast to fp32 for the reduction is not optional. In fp16 the sum of
    squares over 896 channels overflows for perfectly ordinary activations, and
    you get inf -> nan several layers later with nothing pointing at the cause.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.variance_epsilon)
        return (self.weight.to(torch.float32) * x).to(dtype)


class RotaryEmbedding(nn.Module):
    """RoPE, NeoX/HF convention (rotate_half over the two contiguous halves).

    Indexing by an explicit position tensor rather than by slot in the batch is
    what makes flat batching work: token t of sequence i gets its own absolute
    position, so sequences at completely different lengths pack into one tensor
    with no padding and no per-row bookkeeping.
    """

    def __init__(self, head_dim: int, max_position: int, base: float = 10000.0, device: str = "cpu") -> None:
        super().__init__()
        self.head_dim = head_dim
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        t = torch.arange(max_position, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)                 # [max_pos, head_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)          # [max_pos, head_dim]
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        h = x.shape[-1] // 2
        return torch.cat((-x[..., h:], x[..., :h]), dim=-1)

    def forward(self, q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor):
        """q: [T, Hq, D], k: [T, Hkv, D], positions: [T]."""
        cos = self.cos_cached[positions].unsqueeze(1).to(q.dtype)   # [T, 1, D]
        sin = self.sin_cached[positions].unsqueeze(1).to(q.dtype)
        q = q * cos + self._rotate_half(q) * sin
        k = k * cos + self._rotate_half(k) * sin
        return q, k


class SwiGLU(nn.Module):
    """Qwen2 MLP: down(silu(gate(x)) * up(x))."""

    def __init__(self, hidden_size: int, intermediate_size: int, linear_cls=nn.Linear) -> None:
        super().__init__()
        self.gate_proj = linear_cls(hidden_size, intermediate_size, bias=False)
        self.up_proj = linear_cls(hidden_size, intermediate_size, bias=False)
        self.down_proj = linear_cls(intermediate_size, hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
