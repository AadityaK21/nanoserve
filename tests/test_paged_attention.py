"""Paged attention against a dense reference, and Triton against torch.

The dense reference is deliberately written the slow, obvious way -- one head
at a time, explicit mask, explicit softmax. It has no block tables and no
gathers, so it cannot share a bug with the thing it is testing.

Block tables are randomly permuted rather than sequential on purpose. If the
implementation ever accidentally assumes a sequence's blocks are contiguous,
sequential tables would hide it and permuted ones expose it immediately.
"""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from nanoserve.attention import AttentionMetadata, paged_attention_torch  # noqa: E402
from nanoserve.kv_cache import PagedKVCache  # noqa: E402


def dense_reference(q, k, v, q_start, n_rep):
    """q: [ql, Hq, D]; k, v: [sl, Hkv, D]. Query j sits at position q_start + j."""
    ql, Hq, D = q.shape
    sl = k.shape[0]
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros_like(q)
    for h in range(Hq):
        kh, vh = k[:, h // n_rep], v[:, h // n_rep]
        s = (q[:, h] @ kh.T) * scale
        pos = torch.arange(sl)[None, :] <= (q_start + torch.arange(ql))[:, None]
        s = s.masked_fill(~pos, float("-inf"))
        out[:, h] = s.softmax(-1) @ vh
    return out


def build(seqs, num_blocks, block_size, Hkv, D, device="cpu", dtype=torch.float32, shuffle=True):
    """Lay random sequences into a paged cache at scattered physical blocks."""
    g = torch.Generator().manual_seed(1234)
    cache = PagedKVCache(num_blocks, block_size, 1, Hkv, D, dtype, device)
    kf, vf = cache.flat(0)

    perm = torch.randperm(num_blocks, generator=g).tolist() if shuffle else list(range(num_blocks))
    cursor = 0
    tables, truth = [], []
    for sl in seqs:
        nb = (sl + block_size - 1) // block_size
        table = perm[cursor:cursor + nb]
        cursor += nb
        K = torch.randn(sl, Hkv, D, generator=g, dtype=dtype)
        V = torch.randn(sl, Hkv, D, generator=g, dtype=dtype)
        slots = torch.tensor(
            [table[p // block_size] * block_size + p % block_size for p in range(sl)]
        )
        kf.index_copy_(0, slots, K)
        vf.index_copy_(0, slots, V)
        tables.append(table)
        truth.append((K, V))
    return cache, tables, truth


def make_md(seqs, query_lens, tables, block_size, device="cpu"):
    qsl = [0]
    for q in query_lens:
        qsl.append(qsl[-1] + q)
    max_seq = max(seqs)
    max_blk = (max_seq + block_size - 1) // block_size
    bt = torch.zeros(len(seqs), max_blk, dtype=torch.int32)
    for i, t in enumerate(tables):
        # A sequence can own more blocks than this step's context needs -- the
        # chunked-prefill case allocates for the full prompt but only attends to
        # the prefix computed so far. Truncate rather than overflow the row, the
        # same way ModelRunner.build_inputs does.
        n = min(len(t), max_blk)
        bt[i, :n] = torch.tensor(t[:n], dtype=torch.int32)
    return AttentionMetadata(
        slot_mapping=torch.zeros(sum(query_lens), dtype=torch.long, device=device),
        query_start_loc=torch.tensor(qsl, dtype=torch.int32, device=device),
        seq_lens=torch.tensor(seqs, dtype=torch.int32, device=device),
        block_tables=bt.to(device),
        max_query_len=max(query_lens),
        max_seq_len=max_seq,
        num_seqs=len(seqs),
        num_tokens=sum(query_lens),
        is_decode_only=all(q == 1 for q in query_lens),
        query_lens_cpu=list(query_lens),
        seq_lens_cpu=list(seqs),
    )


# ---- prefill -------------------------------------------------------------
@pytest.mark.parametrize("block_size", [4, 8, 16])
@pytest.mark.parametrize("seqs", [[7], [16], [5, 13, 1], [1, 1, 1, 1]])
def test_prefill_matches_dense(block_size, seqs):
    Hq, Hkv, D = 4, 2, 8
    cache, tables, truth = build(seqs, 64, block_size, Hkv, D)
    md = make_md(seqs, seqs, tables, block_size)

    g = torch.Generator().manual_seed(7)
    q = torch.randn(sum(seqs), Hq, D, generator=g)
    got = paged_attention_torch(q, cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D))

    off = 0
    for i, sl in enumerate(seqs):
        K, V = truth[i]
        ref = dense_reference(q[off:off + sl], K, V, 0, Hq // Hkv)
        torch.testing.assert_close(got[off:off + sl], ref, atol=2e-5, rtol=2e-5)
        off += sl


# ---- decode --------------------------------------------------------------
@pytest.mark.parametrize("block_size", [4, 16])
@pytest.mark.parametrize("seqs", [[1, 9, 33], [16, 16], [64, 3, 17, 40]])
def test_decode_matches_dense(block_size, seqs):
    Hq, Hkv, D = 6, 2, 8
    cache, tables, truth = build(seqs, 128, block_size, Hkv, D)
    md = make_md(seqs, [1] * len(seqs), tables, block_size)
    assert md.is_decode_only

    g = torch.Generator().manual_seed(11)
    q = torch.randn(len(seqs), Hq, D, generator=g)
    got = paged_attention_torch(q, cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D))

    for i, sl in enumerate(seqs):
        K, V = truth[i]
        ref = dense_reference(q[i:i + 1], K, V, sl - 1, Hq // Hkv)
        torch.testing.assert_close(got[i:i + 1], ref, atol=2e-5, rtol=2e-5)


def test_decode_ignores_stale_kv_in_padded_slots():
    """Blocks past a sequence's length hold another sequence's old KV.

    The batched decode path gathers a rectangular [S, max_ctx] window, so short
    sequences see slots that belong to someone else. If the validity mask is
    wrong the error is small and plausible-looking, which is the worst kind.
    """
    Hq, Hkv, D, block_size = 4, 2, 8, 8
    seqs = [3, 40]
    cache, tables, truth = build(seqs, 64, block_size, Hkv, D)
    # Poison every slot the short sequence does not own.
    kf, vf = cache.flat(0)
    owned = {tables[0][p // block_size] * block_size + p % block_size for p in range(seqs[0])}
    for b in tables[0]:
        for o in range(block_size):
            if b * block_size + o not in owned:
                kf[b * block_size + o] = 1e3
                vf[b * block_size + o] = 1e3

    md = make_md(seqs, [1, 1], tables, block_size)
    g = torch.Generator().manual_seed(3)
    q = torch.randn(2, Hq, D, generator=g)
    got = paged_attention_torch(q, cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D))

    K, V = truth[0]
    ref = dense_reference(q[0:1], K, V, seqs[0] - 1, Hq // Hkv)
    torch.testing.assert_close(got[0:1], ref, atol=2e-5, rtol=2e-5)


# ---- chunked prefill ------------------------------------------------------
def test_chunked_prefill_matches_whole_prefill():
    """Query offset inside a chunk must be absolute, not chunk-relative."""
    Hq, Hkv, D, block_size = 4, 2, 8, 8
    sl = 20
    cache, tables, truth = build([sl], 32, block_size, Hkv, D)
    K, V = truth[0]
    g = torch.Generator().manual_seed(5)
    q = torch.randn(sl, Hq, D, generator=g)

    whole = paged_attention_torch(
        q, cache.k_cache[0], cache.v_cache[0],
        make_md([sl], [sl], tables, block_size), 1.0 / math.sqrt(D),
    )

    pieces = []
    for start, end in ((0, 7), (7, 13), (13, 20)):
        md = make_md([end], [end - start], tables, block_size)
        pieces.append(
            paged_attention_torch(
                q[start:end], cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D)
            )
        )
    torch.testing.assert_close(torch.cat(pieces), whole, atol=2e-5, rtol=2e-5)


def test_mixed_batch_of_prefill_and_decode():
    Hq, Hkv, D, block_size = 4, 2, 8, 8
    seqs = [12, 30, 5]
    query_lens = [12, 1, 5]        # fresh prefill, decode, fresh prefill
    cache, tables, truth = build(seqs, 64, block_size, Hkv, D)
    md = make_md(seqs, query_lens, tables, block_size)
    assert not md.is_decode_only

    g = torch.Generator().manual_seed(17)
    q = torch.randn(sum(query_lens), Hq, D, generator=g)
    got = paged_attention_torch(q, cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D))

    off = 0
    for i, (sl, ql) in enumerate(zip(seqs, query_lens)):
        K, V = truth[i]
        ref = dense_reference(q[off:off + ql], K, V, sl - ql, Hq // Hkv)
        torch.testing.assert_close(got[off:off + ql], ref, atol=2e-5, rtol=2e-5)
        off += ql


# ---- triton --------------------------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("block_size", [16, 32])
@pytest.mark.parametrize("seqs", [[1, 9, 33], [128, 7], [64] * 8])
def test_triton_matches_torch(block_size, seqs):
    from nanoserve.triton_attention import paged_attention_triton, triton_available

    if not triton_available():
        pytest.skip("triton not installed")

    Hq, Hkv, D = 6, 2, 64
    cache, tables, _ = build(seqs, 512, block_size, Hkv, D, device="cuda", dtype=torch.float16)
    md = make_md(seqs, [1] * len(seqs), tables, block_size, device="cuda")

    q = torch.randn(len(seqs), Hq, D, device="cuda", dtype=torch.float16)
    ref = paged_attention_torch(q, cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D))
    got = paged_attention_triton(q, cache.k_cache[0], cache.v_cache[0], md, 1.0 / math.sqrt(D))
    torch.testing.assert_close(got, ref, atol=3e-3, rtol=3e-3)
