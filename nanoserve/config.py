"""Configuration objects for the engine.

Every knob that changes a benchmark number lives here, so a run is fully
described by the config it was given.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    dtype: str = "float16"
    device: str = "cuda"
    trust_remote_code: bool = False
    max_model_len: int = 4096

    # Phase 4. None = fp16 weights.
    quantization: str | None = None      # None | "int8" | "int4"
    quant_group_size: int = 128          # only used by int4
    quant_skip: tuple[str, ...] = ("lm_head",)


@dataclass
class SamplingParams:
    """Per-request sampling.

    Benchmark defaults are deliberately deterministic. Greedy decoding plus
    ignore_eos means every request emits exactly max_new_tokens. Without that,
    output length varies run to run and throughput becomes unreproducible --
    you end up measuring the model's verbosity instead of the server.
    """

    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    ignore_eos: bool = True
    seed: int | None = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0.0


# Kept for backwards compatibility with the Phase 0 driver.
SamplingConfig = SamplingParams


@dataclass
class CacheConfig:
    """Paged KV cache sizing.

    block_size is the tokens-per-block granularity. Small blocks waste less on
    internal fragmentation (a sequence wastes at most block_size-1 slots) but
    make block tables longer and the attention kernel's inner loop shorter.
    16 is the usual sweet spot; the sweep in Phase 5 measures it.
    """

    block_size: int = 16
    gpu_memory_utilization: float = 0.85
    num_gpu_blocks: int | None = None    # set by the profiler unless forced
    swap_space_gib: float = 0.0          # CPU offload is out of scope


@dataclass
class SchedulerConfig:
    """Continuous batching knobs.

    max_num_batched_tokens is the per-iteration token budget shared by prefill
    and decode. It is the single lever that trades TTFT against TPOT: a large
    budget lets a long prompt prefill in one step (good TTFT for that request,
    bad TPOT for everyone already decoding), a small budget chops the prompt
    into chunks that interleave with decodes.
    """

    max_num_seqs: int = 256
    max_num_batched_tokens: int = 2048
    enable_chunked_prefill: bool = True
    preemption_mode: str = "recompute"   # "recompute" | "none"
    watermark: float = 0.01              # fraction of blocks kept in reserve


@dataclass
class EngineConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    attention_backend: str = "auto"      # "auto" | "torch" | "triton"
    seed: int = 0
