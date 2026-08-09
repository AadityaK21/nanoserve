"""Paging arithmetic. No torch, no GPU -- runs anywhere.

These are the bugs that are murder to find later: an off-by-one in slot_mapping
does not raise, it silently reads another sequence's KV and the model produces
fluent nonsense that looks like a bad checkpoint.
"""

import pytest

from nanoserve.block_manager import BlockAllocator, BlockSpaceManager
from nanoserve.config import SamplingParams
from nanoserve.sequence import Sequence


def mkseq(seq_id=0, prompt_len=10, max_new=5):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling=SamplingParams(max_new_tokens=max_new),
    )


# ---- allocator ----------------------------------------------------------
def test_allocator_roundtrip():
    a = BlockAllocator(8)
    assert a.num_free == 8
    b = a.allocate(3)
    assert len(b) == 3 and a.num_free == 5 and a.num_used == 3
    a.free(b)
    assert a.num_free == 8 and a.num_used == 0


def test_allocator_oom():
    a = BlockAllocator(2)
    with pytest.raises(MemoryError):
        a.allocate(3)
    assert a.num_free == 2      # failed allocation must not consume blocks


def test_allocator_double_free():
    a = BlockAllocator(4)
    b = a.allocate(2)
    a.free(b)
    with pytest.raises(ValueError):
        a.free(b)


def test_no_block_handed_out_twice():
    a = BlockAllocator(16)
    x, y = a.allocate(5), a.allocate(5)
    assert not set(x) & set(y)
    a.free(x)
    z = a.allocate(5)
    assert not set(z) & set(y)


# ---- block space manager -------------------------------------------------
def test_blocks_for_rounds_up():
    m = BlockSpaceManager(block_size=16, num_gpu_blocks=100)
    assert m.blocks_for(0) == 0
    assert m.blocks_for(1) == 1
    assert m.blocks_for(16) == 1
    assert m.blocks_for(17) == 2


def test_allocate_is_incremental():
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=100)
    s = mkseq(prompt_len=10)
    assert m.allocate(s, 10)
    assert len(s.block_table) == 3           # ceil(10/4)
    assert m.allocate(s, 12)
    assert len(s.block_table) == 3           # still fits, no new block
    assert m.allocate(s, 13)
    assert len(s.block_table) == 4


def test_allocate_reports_oom_without_partial_state():
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=2)
    s = mkseq(prompt_len=20)
    assert not m.allocate(s, 20)             # needs 5 blocks, only 2 exist
    assert s.block_table == []               # and grabbed none of them
    assert m.num_free_blocks == 2


def test_watermark_reserves_headroom():
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=100, watermark=0.1)
    assert m.watermark_blocks == 10
    s = mkseq(prompt_len=4 * 95)
    assert not m.can_allocate(s, 4 * 95)                       # 95 > 100 - 10
    assert m.can_allocate(s, 4 * 95, respect_watermark=False)


def test_free_returns_everything():
    m = BlockSpaceManager(block_size=8, num_gpu_blocks=32)
    seqs = [mkseq(i, prompt_len=20) for i in range(4)]
    for s in seqs:
        assert m.allocate(s, 20)
    assert m.num_free_blocks == 32 - 4 * 3
    for s in seqs:
        m.free(s)
    assert m.num_free_blocks == 32
    assert all(s.block_table == [] for s in seqs)


# ---- slot mapping: the part that must be exactly right -------------------
def test_slot_mapping_matches_the_formula():
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
    s = mkseq(prompt_len=10)
    m.allocate(s, 10)
    bt = s.block_table
    assert m.slot_mapping(s, 0, 10) == [
        bt[p // 4] * 4 + p % 4 for p in range(10)
    ]


def test_slot_mapping_is_contiguous_within_a_block():
    m = BlockSpaceManager(block_size=8, num_gpu_blocks=16)
    s = mkseq(prompt_len=8)
    m.allocate(s, 8)
    slots = m.slot_mapping(s, 0, 8)
    assert slots == list(range(slots[0], slots[0] + 8))


def test_slot_mapping_of_a_decode_step():
    """One token appended at position n reuses the tail of the current block."""
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
    s = mkseq(prompt_len=6)
    m.allocate(s, 6)
    prefill = m.slot_mapping(s, 0, 6)
    m.allocate(s, 7)
    assert m.slot_mapping(s, 6, 7)[0] == prefill[4] + 2       # block 1, offset 2


def test_slot_mapping_beyond_allocation_raises():
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=16)
    s = mkseq(prompt_len=4)
    m.allocate(s, 4)
    with pytest.raises(IndexError):
        m.slot_mapping(s, 0, 5)


def test_no_slot_collision_across_sequences():
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=64)
    seqs = [mkseq(i, prompt_len=13) for i in range(5)]
    seen = set()
    for s in seqs:
        m.allocate(s, 13)
        slots = m.slot_mapping(s, 0, 13)
        assert not seen & set(slots)
        seen.update(slots)


def test_reused_blocks_do_not_alias_live_ones():
    """A freed block coming back must not still be addressed by the old seq."""
    m = BlockSpaceManager(block_size=4, num_gpu_blocks=4)
    a, b = mkseq(0, prompt_len=8), mkseq(1, prompt_len=8)
    m.allocate(a, 8)
    m.free(a)
    assert m.allocate(b, 8)
    assert a.block_table == []
    assert len(b.block_table) == 2


# ---- fragmentation -------------------------------------------------------
def test_fragmentation_is_bounded_by_block_size():
    m = BlockSpaceManager(block_size=16, num_gpu_blocks=256)
    seqs = [mkseq(i, prompt_len=17 + i) for i in range(10)]
    for s in seqs:
        m.allocate(s, s.num_tokens)
    waste = m.fragmentation_waste(seqs)
    assert 0 <= waste <= 15 * len(seqs)
