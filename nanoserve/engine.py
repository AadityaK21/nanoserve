"""The serving engine: put the pieces together and run steps.

Lifecycle of a request:

    add_request -> scheduler.waiting -> (admitted) running -> chunked prefill
    -> decode ... -> finished -> RequestOutput

The engine itself is thin. That is the point: all the interesting behaviour
lives in the scheduler (when to run what) and the block manager (where the KV
goes), and both are testable without a GPU.
"""

from __future__ import annotations

import time

import torch

from .block_manager import BlockSpaceManager
from .config import EngineConfig, SamplingParams
from .kv_cache import PagedKVCache, kv_bytes_per_token, profile_num_blocks
from .loader import DTYPES, load_model, load_tokenizer, weight_bytes
from .model_runner import ModelRunner
from .scheduler import Scheduler
from .sequence import RequestOutput, SeqStatus, Sequence


class LLMEngine:
    def __init__(self, cfg: EngineConfig) -> None:
        self.cfg = cfg
        self.dtype = DTYPES[cfg.model.dtype]
        self.device = cfg.model.device

        t0 = time.perf_counter()
        self.tokenizer = load_tokenizer(cfg.model)
        self.model, self.hf_config, self.quant_report = load_model(cfg.model)
        self.load_time = time.perf_counter() - t0

        self.eos_token_id = self.tokenizer.eos_token_id
        self.weight_bytes = weight_bytes(self.model)
        self.kv_bytes_per_token = kv_bytes_per_token(
            self.model.num_layers, self.model.num_kv_heads, self.model.head_dim, self.dtype
        )

        # Runner is built before the cache so profile_run can measure peak
        # activations with a throwaway cache, then the real cache is sized from
        # whatever memory is left.
        self.runner = ModelRunner(
            self.model, None, None, self.device,
            backend=cfg.attention_backend,
            seed=cfg.seed,
            block_size=cfg.cache.block_size,
            dtype=self.dtype,
        )

        num_blocks = cfg.cache.num_gpu_blocks
        if num_blocks is None:
            num_blocks = profile_num_blocks(
                self.model,
                cfg,
                self.kv_bytes_per_token,
                cfg.cache.block_size,
                warmup_fn=lambda: self.runner.profile_run(
                    cfg.scheduler.max_num_batched_tokens
                ),
            )
        self.num_gpu_blocks = num_blocks

        self.kv_cache = PagedKVCache(
            num_blocks=num_blocks,
            block_size=cfg.cache.block_size,
            num_layers=self.model.num_layers,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self.block_manager = BlockSpaceManager(
            cfg.cache.block_size, num_blocks, cfg.scheduler.watermark
        )
        self.runner.kv_cache = self.kv_cache
        self.runner.block_manager = self.block_manager

        self.scheduler = Scheduler(cfg.scheduler, self.block_manager)
        self._next_id = 0
        self._outputs: list[RequestOutput] = []

    # ---- introspection ---------------------------------------------------
    def describe(self) -> dict:
        info = {
            "model": self.cfg.model.model_id,
            "dtype": self.cfg.model.dtype,
            "quantization": self.cfg.model.quantization,
            "attention_backend": self.runner.backend_name,
            "params": getattr(self.model, "num_params", None)
            or sum(p.numel() for p in self.model.parameters()),
            "weight_mib": self.weight_bytes / 2**20,
            "kv_bytes_per_token": self.kv_bytes_per_token,
            "kv_kib_per_token": self.kv_bytes_per_token / 1024,
            "num_gpu_blocks": self.num_gpu_blocks,
            "block_size": self.cfg.cache.block_size,
            "kv_cache_mib": self.kv_cache.num_bytes / 2**20,
            "kv_capacity_tokens": self.kv_cache.capacity_tokens,
            "max_concurrent_512tok_seqs": self.kv_cache.capacity_tokens // 512,
            "load_time_s": self.load_time,
        }
        if self.quant_report:
            info["quant"] = self.quant_report
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            free, total = torch.cuda.mem_get_info()
            info["gpu_total_mib"] = total / 2**20
            info["gpu_free_mib"] = free / 2**20
        return info

    # ---- request API -----------------------------------------------------
    def add_request(
        self,
        prompt: str | None = None,
        sampling: SamplingParams | None = None,
        prompt_token_ids: list[int] | None = None,
        request_id: int | None = None,
        arrival: float | None = None,
    ) -> int:
        if prompt_token_ids is None:
            if prompt is None:
                raise ValueError("need prompt or prompt_token_ids")
            prompt_token_ids = self.tokenizer(prompt).input_ids
        rid = request_id if request_id is not None else self._next_id
        self._next_id = max(self._next_id, rid) + 1

        seq = Sequence(
            seq_id=rid,
            prompt_token_ids=list(prompt_token_ids),
            sampling=sampling or SamplingParams(),
            arrival=arrival if arrival is not None else time.perf_counter(),
        )
        max_len = self.cfg.model.max_model_len
        total = seq.prompt_len + seq.sampling.max_new_tokens
        if total > max_len:
            raise ValueError(f"request {rid} needs {total} tokens > max_model_len {max_len}")
        self.scheduler.add(seq)
        return rid

    @property
    def has_unfinished_requests(self) -> bool:
        return self.scheduler.has_work

    # ---- the step --------------------------------------------------------
    def step(self) -> list[RequestOutput]:
        sched = self.scheduler.schedule()
        if not sched:
            return []

        results = self.runner.execute(sched.scheduled)
        now = time.perf_counter()

        for seq, token_id in results:
            seq.append_token(token_id)
            m = seq.metrics
            if m.first_token == 0.0:
                m.first_token = now
            m.output_tokens = seq.num_output_tokens
            m.end = now
            seq.check_stop(self.eos_token_id)

        done = self.scheduler.free_finished()
        outs = [self._finalize(s) for s in done]
        self._outputs.extend(outs)
        return outs

    def _finalize(self, seq: Sequence) -> RequestOutput:
        reason = {
            SeqStatus.FINISHED_LENGTH: "length",
            SeqStatus.FINISHED_EOS: "stop",
            SeqStatus.FINISHED_ABORTED: "abort",
        }.get(seq.status, "unknown")
        return RequestOutput(
            request_id=seq.seq_id,
            prompt_token_ids=seq.prompt_token_ids,
            output_token_ids=seq.output_token_ids,
            finish_reason=reason,
            metrics=seq.metrics,
            num_preemptions=seq.num_preemptions,
        )

    # ---- convenience drivers ---------------------------------------------
    def run_all(self, detokenize: bool = False) -> list[RequestOutput]:
        """Offline: everything is already queued, drain it."""
        outs: list[RequestOutput] = []
        while self.has_unfinished_requests:
            outs.extend(self.step())
        if detokenize:
            for o in outs:
                o.text = self.tokenizer.decode(o.output_token_ids, skip_special_tokens=True)
        return outs

    def generate(self, prompts: list[str], sampling: SamplingParams | None = None) -> list[RequestOutput]:
        for p in prompts:
            self.add_request(p, sampling)
        outs = self.run_all(detokenize=True)
        return sorted(outs, key=lambda o: o.request_id)

    def reset(self) -> None:
        """Clear state between sweep points without reloading the weights."""
        for s in list(self.scheduler.running) + list(self.scheduler.waiting):
            self.block_manager.free(s)
        self.scheduler = Scheduler(self.cfg.scheduler, self.block_manager)
        self.block_manager.allocator.reset()
        self._outputs = []
        if torch.cuda.is_available() and str(self.device).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
