"""Physical block allocation and logical-to-physical address translation.

This is the paging layer, and it is the whole reason a serving engine can hold
more concurrent sequences than a naive one.

A naive engine reserves a contiguous KV buffer of max_model_len per sequence.
A request that asks for 4096 tokens but stops at 200 has 95% of its reservation
sitting idle, and no other request can use it. That is internal fragmentation,
and at serving scale it is where most of your VRAM goes.

Paging fixes it the same way an OS does. KV memory is carved into fixed-size
blocks. A sequence holds a block_table -- a list of physical block ids -- and
grows it one block at a time as it generates. Waste per sequence is bounded by
block_size - 1 tokens instead of max_model_len - actual_len.

No torch here on purpose: this is pure index arithmetic, so it is testable
without a GPU and the bugs it would otherwise hide (off-by-one in slot mapping,
double-free, leaked blocks) surface in a unit test instead of as silently wrong
logits.
"""

from __future__ import annotations

from collections import deque


class BlockAllocator:
    """Free list over [0, num_blocks)."""

    def __init__(self, num_blocks: int) -> None:
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        self.num_blocks = num_blocks
        self._free: deque[int] = deque(range(num_blocks))
        self._allocated: set[int] = set()

    @property
    def num_free(self) -> int:
        return len(self._free)

    @property
    def num_used(self) -> int:
        return len(self._allocated)

    def allocate(self, n: int = 1) -> list[int]:
        if n > len(self._free):
            raise MemoryError(f"requested {n} blocks, {len(self._free)} free")
        out = [self._free.popleft() for _ in range(n)]
        self._allocated.update(out)
        return out

    def free(self, blocks: list[int]) -> None:
        for b in blocks:
            if b not in self._allocated:
                raise ValueError(f"double free of block {b}")
            self._allocated.discard(b)
            self._free.append(b)

    def reset(self) -> None:
        self._free = deque(range(self.num_blocks))
        self._allocated.clear()


class BlockSpaceManager:
    """Maps sequences onto physical blocks.

    The watermark keeps a small slice of blocks unallocated. Without it the
    scheduler happily admits a new request using the last free block, and then
    every running sequence deadlocks on its next decode step because there is
    nothing left to append to. Reserving a margin means admission fails before
    the running set starves.
    """

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        watermark: float = 0.01,
    ) -> None:
        self.block_size = block_size
        self.allocator = BlockAllocator(num_gpu_blocks)
        self.watermark_blocks = int(watermark * num_gpu_blocks)

    # ---- capacity queries ---------------------------------------------
    @property
    def num_free_blocks(self) -> int:
        return self.allocator.num_free

    @property
    def num_total_blocks(self) -> int:
        return self.allocator.num_blocks

    def blocks_for(self, num_tokens: int) -> int:
        return (num_tokens + self.block_size - 1) // self.block_size

    def blocks_needed(self, seq, num_tokens: int) -> int:
        """Extra blocks required to hold num_tokens for this sequence."""
        return max(0, self.blocks_for(num_tokens) - len(seq.block_table))

    def can_allocate(self, seq, num_tokens: int, respect_watermark: bool = True) -> bool:
        need = self.blocks_needed(seq, num_tokens)
        budget = self.allocator.num_free
        if respect_watermark:
            budget -= self.watermark_blocks
        return need <= budget

    # ---- mutation ------------------------------------------------------
    def allocate(self, seq, num_tokens: int) -> bool:
        """Grow seq.block_table so it can address num_tokens. False if OOM."""
        need = self.blocks_needed(seq, num_tokens)
        if need == 0:
            return True
        if need > self.allocator.num_free:
            return False
        seq.block_table.extend(self.allocator.allocate(need))
        return True

    def free(self, seq) -> None:
        if seq.block_table:
            self.allocator.free(seq.block_table)
            seq.block_table = []

    # ---- address translation -------------------------------------------
    def slot_mapping(self, seq, start: int, end: int) -> list[int]:
        """Flat cache slots for logical token positions [start, end).

        A slot is the index into a cache tensor viewed as
        [num_blocks * block_size, num_kv_heads, head_dim], which is what lets
        the write be a single scatter instead of a per-block loop:

            slot = block_table[pos // block_size] * block_size
                   + pos % block_size
        """
        bs = self.block_size
        table = seq.block_table
        slots = []
        for pos in range(start, end):
            block_idx = pos // bs
            if block_idx >= len(table):
                raise IndexError(
                    f"seq {seq.seq_id}: position {pos} needs block {block_idx} "
                    f"but block_table has {len(table)} entries"
                )
            slots.append(table[block_idx] * bs + pos % bs)
        return slots

    # ---- reporting ------------------------------------------------------
    def utilization(self) -> float:
        return self.allocator.num_used / self.allocator.num_blocks

    def fragmentation_waste(self, seqs) -> int:
        """Tokens of KV space allocated but unused across the given sequences.

        Bounded by (block_size - 1) per sequence. Worth reporting because it is
        the cost side of paging, and it is small.
        """
        return sum(
            len(s.block_table) * self.block_size - s.num_tokens
            for s in seqs
            if s.block_table
        )
