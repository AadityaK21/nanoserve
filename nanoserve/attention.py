"""Paged attention.

One function handles prefill, decode and chunked prefill, because in a paged
engine they are the same operation with different query lengths. A sequence
arrives with seq_len tokens of context, of which the last query_len are new.
Query j attends to keys [0, seq_len - query_len + j]. Set query_len == seq_len
and it is a prefill; set query_len == 1 and it is a decode; anything in between
is a prefill chunk. No separate code paths, so no chance of the two drifting.

Two backends:

  torch   -- gathers each sequence's KV out of the paged cache and calls SDPA.
             Correct everywhere, including native Windows and CPU. The gather
             is real memory traffic: for decode it moves the entire context
             again every step, which is exactly the cost a fused kernel exists
             to remove. This is the reference the Triton kernel is tested
             against.
  triton  -- fuses gather, softmax and the V accumulation into one pass over
             the block table, so context KV is read once, from the cache, in
             the layout it already lives in. See triton_attention.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class AttentionMetadata:
    """Everything the attention op needs that is not a weight or an activation.

    Built once per step by the model runner and shared by every layer.
    """

    slot_mapping: torch.Tensor      # [num_tokens] int64, where to write new KV
    query_start_loc: torch.Tensor   # [num_seqs + 1] int32, prefix sums of query_lens
    seq_lens: torch.Tensor          # [num_seqs] int32, context length incl. this step
    block_tables: torch.Tensor      # [num_seqs, max_blocks] int32
    max_query_len: int
    max_seq_len: int
    num_seqs: int
    num_tokens: int
    is_decode_only: bool            # every query_len == 1

    # Kept on CPU for the python-side loop in the torch backend. Calling
    # .tolist() on the GPU copies instead would force a device sync once per
    # layer per step -- 24 stalls per iteration on a 24-layer model, on the
    # critical path of a decode step that only takes a few milliseconds.
    query_lens_cpu: list[int] = None          # type: ignore[assignment]
    seq_lens_cpu: list[int] = None            # type: ignore[assignment]
    query_start_loc_cpu: list[int] = None     # type: ignore[assignment]


def _detect_gqa_support() -> bool:
    """Does this torch's SDPA broadcast KV heads for us?

    torch >= 2.5 takes enable_gqa and handles the 1-to-many head mapping inside
    the kernel. Without it we have to materialise an expanded copy of K and V,
    which for Qwen2.5-0.5B means writing out 7x more KV than we read. Detect by
    calling it rather than by version number, because the flag has moved
    between releases.
    """
    try:
        q = torch.zeros(1, 2, 1, 8)
        kv = torch.zeros(1, 1, 1, 8)
        F.scaled_dot_product_attention(q, kv, kv, enable_gqa=True)
        return True
    except Exception:
        return False


_HAS_GQA = _detect_gqa_support()


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """[*, H_kv, L, D] -> [*, H_kv * n_rep, L, D] for grouped-query attention.

    Qwen2.5-0.5B has 14 query heads and 2 KV heads, so each KV head is shared
    by 7 query heads. That 7x reduction in KV heads is why the cache is small
    enough to serve dozens of concurrent sequences on 8 GB -- and why
    materialising the expansion, when SDPA could do it internally, is such a
    waste of bandwidth.
    """
    if n_rep == 1 or _HAS_GQA:
        return x
    return x.repeat_interleave(n_rep, dim=-3)


def _sdpa(q, k, v, attn_mask, scale, n_rep):
    kwargs = {"attn_mask": attn_mask, "scale": scale}
    if _HAS_GQA and n_rep > 1:
        kwargs["enable_gqa"] = True
    return F.scaled_dot_product_attention(q, k, v, **kwargs)


@torch.inference_mode()
def paged_attention_torch(
    query: torch.Tensor,        # [num_tokens, num_heads, head_dim]
    k_cache: torch.Tensor,      # [num_blocks, block_size, num_kv_heads, head_dim]
    v_cache: torch.Tensor,
    md: AttentionMetadata,
    scale: float,
) -> torch.Tensor:
    """Reference paged attention. Returns [num_tokens, num_heads, head_dim]."""
    num_tokens, num_heads, head_dim = query.shape
    num_blocks, block_size, num_kv_heads, _ = k_cache.shape
    n_rep = num_heads // num_kv_heads

    kf = k_cache.view(num_blocks * block_size, num_kv_heads, head_dim)
    vf = v_cache.view(num_blocks * block_size, num_kv_heads, head_dim)

    if md.is_decode_only:
        return _decode_batched(query, kf, vf, md, scale, n_rep, block_size)

    out = torch.empty_like(query)
    qsl = md.query_start_loc_cpu or md.query_start_loc.tolist()
    for i in range(md.num_seqs):
        s, e = qsl[i], qsl[i + 1]
        ql = e - s
        if ql == 0:
            continue
        sl = md.seq_lens_cpu[i]
        n_blk = (sl + block_size - 1) // block_size
        blocks = md.block_tables[i, :n_blk].to(torch.long)

        # Gather this sequence's whole context out of the paged cache.
        idx = (blocks[:, None] * block_size + torch.arange(block_size, device=query.device)[None, :])
        idx = idx.reshape(-1)[:sl]
        k = kf.index_select(0, idx)          # [sl, H_kv, D]
        v = vf.index_select(0, idx)

        q_i = query[s:e].transpose(0, 1)                       # [H, ql, D]
        k_i = _repeat_kv(k.transpose(0, 1), n_rep)             # [H, sl, D]
        v_i = _repeat_kv(v.transpose(0, 1), n_rep)

        # Query j sits at absolute position sl - ql + j and may attend to every
        # key at or before it. Getting this offset wrong is the classic chunked
        # prefill bug: it silently lets a chunk see its own future.
        q_pos = torch.arange(sl - ql, sl, device=query.device)
        k_pos = torch.arange(sl, device=query.device)
        mask = k_pos[None, :] <= q_pos[:, None]                # [ql, sl] True = attend

        o = _sdpa(
            q_i.unsqueeze(0), k_i.unsqueeze(0), v_i.unsqueeze(0),
            mask[None, None, :, :], scale, n_rep,
        )
        out[s:e] = o.squeeze(0).transpose(0, 1)
    return out


def _decode_batched(query, kf, vf, md, scale, n_rep, block_size):
    """Fast path when every sequence contributes exactly one query token.

    One padded gather and one SDPA call for the whole batch instead of a Python
    loop over sequences. Still moves the full context per step -- that is
    inherent to doing this in torch -- but it is the honest torch baseline the
    Triton kernel has to beat.
    """
    device = query.device
    S = md.num_seqs
    num_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads = kf.shape[1]

    max_blk = (md.max_seq_len + block_size - 1) // block_size
    assert md.block_tables.shape[1] >= max_blk, (
        f"block_tables has {md.block_tables.shape[1]} columns but max_seq_len="
        f"{md.max_seq_len} needs {max_blk}"
    )
    bt = md.block_tables[:, :max_blk].to(torch.long)                     # [S, max_blk]
    idx = bt[:, :, None] * block_size + torch.arange(block_size, device=device)[None, None, :]
    idx = idx.reshape(S, max_blk * block_size)                            # [S, ctx]
    ctx = idx.shape[1]

    k = kf.index_select(0, idx.reshape(-1)).view(S, ctx, num_kv_heads, head_dim)
    v = vf.index_select(0, idx.reshape(-1)).view(S, ctx, num_kv_heads, head_dim)

    k = _repeat_kv(k.permute(0, 2, 1, 3), n_rep)                          # [S, H, ctx, D]
    v = _repeat_kv(v.permute(0, 2, 1, 3), n_rep)
    q = query.view(S, 1, num_heads, head_dim).permute(0, 2, 1, 3)         # [S, H, 1, D]

    # Slots past seq_len hold stale KV from whichever sequence owned the block
    # before. Masking them is not an optimisation, it is correctness.
    seq_lens = md.seq_lens.to(device=device, dtype=torch.long)
    valid = torch.arange(ctx, device=device)[None, :] < seq_lens[:, None]  # [S, ctx]

    o = _sdpa(q, k, v, valid[:, None, None, :], scale, n_rep)
    return o.permute(0, 2, 1, 3).reshape(S * 1, num_heads, head_dim)


def get_attention_backend(name: str = "auto"):
    """Resolve the backend name to a callable with paged_attention_torch's signature."""
    if name in ("auto", "triton"):
        try:
            from .triton_attention import paged_attention_triton, triton_available

            if triton_available():
                return paged_attention_triton, "triton"
        except Exception:
            if name == "triton":
                raise
    if name == "triton":
        raise RuntimeError("triton backend requested but Triton is unavailable")
    return paged_attention_torch, "torch"
