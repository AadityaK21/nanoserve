"""Qwen2 forward pass, rewritten against the paged KV cache.

Module names mirror HuggingFace exactly (model.layers.N.self_attn.q_proj, ...)
so a stock checkpoint loads with load_state_dict and no key remapping.

Two things differ from the HF implementation, and both matter for serving:

  Flat batching. Activations are [num_tokens, hidden], not
  [batch, seq, hidden]. There is no padding, so a batch mixing a 900-token
  prefill chunk with 40 one-token decodes costs 940 token-slots, not
  41 x 900. Padding waste is the second-largest source of waste in a naive
  server after KV fragmentation.

  Logits are computed lazily. lm_head is [hidden, 151936]; materialising logits
  for all 2048 tokens of a prefill step would allocate 600 MiB in fp16 to then
  use 1 row of it. compute_logits() takes only the token indices the sampler
  actually needs.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import AttentionMetadata
from .layers import RMSNorm, RotaryEmbedding, SwiGLU


class Qwen2Attention(nn.Module):
    def __init__(self, cfg, layer_idx: int, rotary: RotaryEmbedding, linear_cls=nn.Linear) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = cfg.hidden_size
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        self.scale = self.head_dim ** -0.5
        self.rotary = rotary

        # Qwen2 puts biases on q/k/v but not on o.
        self.q_proj = linear_cls(self.hidden_size, self.num_heads * self.head_dim, bias=True)
        self.k_proj = linear_cls(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.v_proj = linear_cls(self.hidden_size, self.num_kv_heads * self.head_dim, bias=True)
        self.o_proj = linear_cls(self.num_heads * self.head_dim, self.hidden_size, bias=False)

    def forward(self, hidden, cos, sin, kv_cache, md: AttentionMetadata, attn_fn):
        T = hidden.shape[0]
        q = self.q_proj(hidden).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden).view(T, self.num_kv_heads, self.head_dim)

        # cos/sin were looked up once for the whole forward pass, not here.
        q, k = self.rotary.apply(q, k, cos, sin)

        # Write before reading: a prefill chunk must attend to the keys it just
        # produced, and a decode step must attend to its own new key.
        kv_cache.write(self.layer_idx, k, v, md.slot_mapping)

        out = attn_fn(
            q,
            kv_cache.k_cache[self.layer_idx],
            kv_cache.v_cache[self.layer_idx],
            md,
            self.scale,
        )
        return self.o_proj(out.reshape(T, self.num_heads * self.head_dim))


class Qwen2DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx: int, rotary: RotaryEmbedding, linear_cls=nn.Linear) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(cfg, layer_idx, rotary, linear_cls)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.intermediate_size, linear_cls)
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, hidden, cos, sin, kv_cache, md, attn_fn):
        residual = hidden
        hidden = self.input_layernorm(hidden)
        hidden = self.self_attn(hidden, cos, sin, kv_cache, md, attn_fn)
        hidden = residual + hidden

        residual = hidden
        hidden = self.post_attention_layernorm(hidden)
        hidden = self.mlp(hidden)
        return residual + hidden


class Qwen2Model(nn.Module):
    def __init__(self, cfg, linear_cls=nn.Linear, max_position: int = 4096) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        # Sized to max_model_len, not to the checkpoint's max_position_embeddings.
        # Qwen2.5 advertises 32768, and precomputing cos/sin for all of it costs
        # 16 MiB of VRAM that would otherwise be KV cache we can actually use.
        rotary = RotaryEmbedding(head_dim, max_position=max_position,
                                 base=getattr(cfg, "rope_theta", 10000.0))
        self.rotary = rotary
        self.layers = nn.ModuleList(
            [Qwen2DecoderLayer(cfg, i, rotary, linear_cls) for i in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)

    def forward(self, input_ids, positions, kv_cache, md, attn_fn):
        hidden = self.embed_tokens(input_ids)
        # One RoPE table lookup for the whole model. The positions are the same
        # for every layer, so doing this inside the loop would issue 48 gather
        # kernels per step to produce 48 identical tensors.
        cos, sin = self.rotary.cos_sin(positions, hidden.dtype)
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, kv_cache, md, attn_fn)
        return self.norm(hidden)


class Qwen2ForCausalLM(nn.Module):
    def __init__(self, cfg, linear_cls=nn.Linear, max_position: int = 4096) -> None:
        super().__init__()
        self.config = cfg
        self.model = Qwen2Model(cfg, linear_cls, max_position)
        # lm_head stays fp16 even under quantisation: it is one matrix, and
        # quantising the output projection costs noticeably more perplexity per
        # byte saved than quantising the 24 decoder layers.
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        self.tie_word_embeddings = getattr(cfg, "tie_word_embeddings", False)

    def forward(self, input_ids, positions, kv_cache, md, attn_fn):
        return self.model(input_ids, positions, kv_cache, md, attn_fn)

    def compute_logits(self, hidden: torch.Tensor, indices: torch.Tensor | None = None):
        if indices is not None:
            hidden = hidden.index_select(0, indices)
        return self.lm_head(hidden)

    # ---- weight loading -------------------------------------------------
    def load_hf_state_dict(self, state_dict: dict) -> None:
        """Load a stock HF checkpoint.

        Tied embeddings are the one wrinkle: Qwen2.5-0.5B sets
        tie_word_embeddings=True and ships no lm_head.weight, so it has to be
        aliased to the embedding matrix. Missing that gives you a randomly
        initialised output head and fluent-looking garbage.
        """
        sd = dict(state_dict)
        if self.tie_word_embeddings:
            sd.pop("lm_head.weight", None)
        missing, unexpected = self.load_state_dict(sd, strict=False)

        if self.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
            missing = [m for m in missing if m != "lm_head.weight"]

        # rotary buffers are recomputed, never loaded
        missing = [m for m in missing if "rotary" not in m and "cos_cached" not in m and "sin_cached" not in m]
        unexpected = [u for u in unexpected if "rotary_emb" not in u]
        if missing:
            raise RuntimeError(f"missing weights: {missing[:8]}{'...' if len(missing) > 8 else ''}")
        if unexpected:
            raise RuntimeError(f"unexpected weights: {unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")

    # ---- shape helpers used by the cache profiler ------------------------
    @property
    def num_layers(self) -> int:
        return self.config.num_hidden_layers

    @property
    def num_kv_heads(self) -> int:
        return getattr(self.config, "num_key_value_heads", self.config.num_attention_heads)

    @property
    def head_dim(self) -> int:
        return getattr(self.config, "head_dim", None) or (
            self.config.hidden_size // self.config.num_attention_heads
        )
