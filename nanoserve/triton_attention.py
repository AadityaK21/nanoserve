"""Fused paged attention in Triton.

What the torch backend does per decode step, per layer: gather every
sequence's entire KV context out of the paged cache into a fresh contiguous
tensor, then run SDPA on it. Decode is memory-bandwidth bound, so that gather
is not overhead around the work -- it *is* most of the work, and it doubles the
KV traffic (once to materialise the gather, once for SDPA to read it).

This kernel removes it. One program per (sequence, query head) walks the
sequence's block table, and for each block loads K straight from the cache,
accumulates the softmax online (Flash-Attention style running max and sum), and
folds V in immediately. Context KV is read exactly once, from where it already
lives, and nothing of size O(context) is ever written to HBM.

Correctness is not asserted, it is tested: tests/test_paged_attention.py checks
this kernel against paged_attention_torch on random block tables and ragged
sequence lengths. If Triton is unavailable the engine silently uses torch.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - depends on the host
    _HAS_TRITON = False


def triton_available() -> bool:
    """Triton needs a real CUDA device; the import alone is not enough."""
    return _HAS_TRITON and torch.cuda.is_available()


if _HAS_TRITON:

    @triton.jit
    def _paged_attn_decode_kernel(
        Out,            # [S, H, D]
        Q,              # [S, H, D]
        K_cache,        # [NB, BS, H_kv, D]
        V_cache,        # [NB, BS, H_kv, D]
        BlockTables,    # [S, MAXBLK] int32
        SeqLens,        # [S] int32
        scale,
        stride_oz, stride_oh,
        stride_qz, stride_qh,
        stride_kb, stride_ks, stride_kh,
        stride_vb, stride_vs, stride_vh,
        stride_bt,
        KV_GROUP: tl.constexpr,     # query heads per kv head
        BLOCK_SIZE: tl.constexpr,   # tokens per cache block
        HEAD_DIM: tl.constexpr,
    ):
        s = tl.program_id(0)
        h = tl.program_id(1)
        kv_h = h // KV_GROUP

        seq_len = tl.load(SeqLens + s)

        offs_d = tl.arange(0, HEAD_DIM)
        offs_t = tl.arange(0, BLOCK_SIZE)

        q = tl.load(Q + s * stride_qz + h * stride_qh + offs_d).to(tl.float32)
        q = q * scale

        # Online softmax state, as rank-1 tensors rather than Python scalars.
        # A loop-carried value has to keep the same type across iterations, and
        # a bare `m_i = float("-inf")` would enter the loop as a Python float
        # and leave it as a Triton scalar, which fails to compile.
        #
        # m starts at -inf so the first block's rescale factor exp(m_i - m_new)
        # is exp(-inf) = 0, which correctly discards the zeroed accumulator.
        m_i = tl.zeros([1], dtype=tl.float32) - float("inf")
        l_i = tl.zeros([1], dtype=tl.float32)
        acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

        num_blocks = tl.cdiv(seq_len, BLOCK_SIZE)
        for b in range(0, num_blocks):
            phys = tl.load(BlockTables + s * stride_bt + b).to(tl.int64)
            valid = (b * BLOCK_SIZE + offs_t) < seq_len

            k_off = (
                phys * stride_kb
                + offs_t[:, None] * stride_ks
                + kv_h * stride_kh
                + offs_d[None, :]
            )
            k = tl.load(K_cache + k_off, mask=valid[:, None], other=0.0).to(tl.float32)

            qk = tl.sum(q[None, :] * k, axis=1)                 # [BLOCK_SIZE]
            qk = tl.where(valid, qk, float("-inf"))

            m_new = tl.maximum(m_i, tl.max(qk, axis=0))
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new)

            v_off = (
                phys * stride_vb
                + offs_t[:, None] * stride_vs
                + kv_h * stride_vh
                + offs_d[None, :]
            )
            v = tl.load(V_cache + v_off, mask=valid[:, None], other=0.0).to(tl.float32)

            l_i = l_i * alpha + tl.sum(p, axis=0)
            acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
            m_i = m_new

        out = acc / l_i
        tl.store(Out + s * stride_oz + h * stride_oh + offs_d, out.to(Out.dtype.element_ty))


@torch.inference_mode()
def paged_attention_triton(query, k_cache, v_cache, md, scale):
    """Same signature as paged_attention_torch.

    Only the decode path is fused. Prefill is compute-bound (a real GEMM per
    sequence) rather than gather-bound, so torch + SDPA is already close to
    optimal there and a hand-written prefill kernel would be effort spent in
    the wrong place.
    """
    from .attention import paged_attention_torch

    if not md.is_decode_only or not triton_available():
        return paged_attention_torch(query, k_cache, v_cache, md, scale)

    S = md.num_seqs
    num_heads, head_dim = query.shape[1], query.shape[2]
    num_kv_heads = k_cache.shape[2]
    block_size = k_cache.shape[1]

    # tl.arange needs power-of-two extents. Rather than pad and mask, fall back
    # to the reference backend -- a silently wrong kernel is far worse than a
    # slower correct one.
    if head_dim & (head_dim - 1) or block_size & (block_size - 1):
        return paged_attention_torch(query, k_cache, v_cache, md, scale)

    q = query.view(S, num_heads, head_dim).contiguous()
    out = torch.empty_like(q)
    bt = md.block_tables.contiguous().to(torch.int32)
    seq_lens = md.seq_lens.contiguous().to(torch.int32)

    _paged_attn_decode_kernel[(S, num_heads)](
        out, q, k_cache, v_cache, bt, seq_lens,
        scale,
        out.stride(0), out.stride(1),
        q.stride(0), q.stride(1),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2),
        bt.stride(0),
        KV_GROUP=num_heads // num_kv_heads,
        BLOCK_SIZE=block_size,
        HEAD_DIM=head_dim,
        num_warps=4,
        num_stages=2,
    )
    return out.view(query.shape)
