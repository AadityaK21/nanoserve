"""Physical KV cache tensors and the profiler that decides how big they get.

Layout, per layer, K and V held separately:

    [num_blocks, block_size, num_kv_heads, head_dim]

Viewed as [num_blocks * block_size, num_kv_heads, head_dim] this makes a write
a single index_copy_ against a precomputed slot mapping, and a read a single
gather against a block table. vLLM reshapes K into a vectorised layout to help
its kernel coalesce; the simpler layout costs a little kernel bandwidth and
saves a lot of debugging, and the Triton kernel here is written against it
directly.
"""

from __future__ import annotations

import torch


def kv_bytes_per_token(num_layers: int, num_kv_heads: int, head_dim: int, dtype: torch.dtype) -> int:
    """2 (K and V) x layers x kv_heads x head_dim x itemsize.

    The single most important number in the project. It sets how many tokens of
    KV fit in VRAM, which sets max concurrency, which sets throughput.
    """
    itemsize = torch.tensor([], dtype=dtype).element_size()
    return 2 * num_layers * num_kv_heads * head_dim * itemsize


class PagedKVCache:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
        device: str,
    ) -> None:
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.dtype = dtype
        self.device = device

        shape = (num_blocks, block_size, num_kv_heads, head_dim)
        self.k_cache = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]
        self.v_cache = [
            torch.zeros(shape, dtype=dtype, device=device) for _ in range(num_layers)
        ]

    @property
    def num_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k_cache + self.v_cache)

    @property
    def capacity_tokens(self) -> int:
        return self.num_blocks * self.block_size

    def flat(self, layer: int):
        """Views for scatter/gather: [num_blocks * block_size, H, D]."""
        n = self.num_blocks * self.block_size
        return (
            self.k_cache[layer].view(n, self.num_kv_heads, self.head_dim),
            self.v_cache[layer].view(n, self.num_kv_heads, self.head_dim),
        )

    def write(self, layer: int, key: torch.Tensor, value: torch.Tensor, slot_mapping: torch.Tensor) -> None:
        """Scatter this step's K/V into their slots.

        key/value: [num_tokens, num_kv_heads, head_dim]
        slot_mapping: [num_tokens] int64
        """
        kf, vf = self.flat(layer)
        kf.index_copy_(0, slot_mapping, key.to(kf.dtype))
        vf.index_copy_(0, slot_mapping, value.to(vf.dtype))

    def zero_(self) -> None:
        for t in self.k_cache + self.v_cache:
            t.zero_()


def profile_num_blocks(
    model,
    engine_cfg,
    kv_bytes_tok: int,
    block_size: int,
    warmup_fn=None,
) -> int:
    """Decide num_gpu_blocks empirically instead of by arithmetic.

    The arithmetic answer (total - weights) is always wrong, because it ignores
    activations, the CUDA context, cuBLAS workspaces and allocator
    fragmentation. Running one real forward pass at the largest batch the
    scheduler can produce, then measuring what is actually left, is the only
    number that survives contact with the GPU.
    """
    if not torch.cuda.is_available() or not str(engine_cfg.model.device).startswith("cuda"):
        # CPU / test path: pick something small and deterministic.
        return 1024

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    if warmup_fn is not None:
        warmup_fn()
        torch.cuda.synchronize()
        # The profiling run's activations are freed by Python but still held by
        # the caching allocator, so mem_get_info would report them as used and
        # we would size the cache far too small. Hand them back to the driver
        # before measuring.
        torch.cuda.empty_cache()

    free, total = torch.cuda.mem_get_info()
    util = engine_cfg.cache.gpu_memory_utilization

    # Leave (1 - util) of *total* as headroom for allocator churn and the
    # transient buffers a big prefill step needs.
    budget = free - int((1.0 - util) * total)
    if budget <= 0:
        raise RuntimeError(
            f"no KV budget: {free / 2**20:.0f} MiB free of {total / 2**20:.0f} MiB "
            f"at gpu_memory_utilization={util}. Lower it, or use a smaller model."
        )

    bytes_per_block = kv_bytes_tok * block_size
    n = int(budget // bytes_per_block)
    if n < 16:
        raise RuntimeError(f"only {n} KV blocks fit; raise gpu_memory_utilization")
    return n
