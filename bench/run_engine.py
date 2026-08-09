"""Benchmark driver for the continuous-batching engine.

    # offline throughput: everything queued at t=0
    python -m bench.run_engine --n 64 --workload skewed

    # online latency: Poisson arrivals at 8 req/s
    python -m bench.run_engine --n 128 --workload skewed --request-rate 8

    # ablations
    python -m bench.run_engine --n 64 --no-chunked-prefill
    python -m bench.run_engine --n 64 --quantization int4
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nanoserve.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SamplingParams,
    SchedulerConfig,
)
from nanoserve.engine import LLMEngine
from nanoserve.metrics import print_summary, summarize
from nanoserve.workload import skewed_workload, uniform_workload

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_engine(args) -> LLMEngine:
    cfg = EngineConfig(
        model=ModelConfig(
            model_id=args.model,
            dtype=args.dtype,
            device=args.device,
            max_model_len=args.max_model_len,
            quantization=args.quantization,
            quant_group_size=args.group_size,
        ),
        cache=CacheConfig(
            block_size=args.block_size,
            gpu_memory_utilization=args.gpu_util,
            num_gpu_blocks=args.num_blocks,
        ),
        scheduler=SchedulerConfig(
            max_num_seqs=args.max_num_seqs,
            max_num_batched_tokens=args.max_batched_tokens,
            enable_chunked_prefill=not args.no_chunked_prefill,
            preemption_mode="none" if args.no_preemption else "recompute",
        ),
        attention_backend=args.backend,
        seed=args.seed,
    )
    return LLMEngine(cfg)


def run(engine: LLMEngine, reqs, request_rate: float | None):
    """Drive the engine. Offline if request_rate is None, else timed arrivals."""
    engine.reset()
    sampling_for = {
        r.request_id: SamplingParams(max_new_tokens=r.max_new_tokens, ignore_eos=True)
        for r in reqs
    }

    t0 = time.perf_counter()
    if request_rate is None:
        for r in reqs:
            engine.add_request(
                prompt=r.prompt,
                sampling=sampling_for[r.request_id],
                request_id=r.request_id,
                arrival=t0,
            )
        outs = []
        while engine.has_unfinished_requests:
            outs.extend(engine.step())
    else:
        # Tokenize up front so the arrival clock measures the server, not the
        # tokenizer.
        pre = [(r, engine.tokenizer(r.prompt).input_ids) for r in reqs]
        pending = list(pre)
        outs = []
        while pending or engine.has_unfinished_requests:
            now = time.perf_counter()
            while pending and pending[0][0].arrival_offset <= now - t0:
                r, ids = pending.pop(0)
                engine.add_request(
                    prompt_token_ids=ids,
                    sampling=sampling_for[r.request_id],
                    request_id=r.request_id,
                    arrival=t0 + r.arrival_offset,
                )
            if engine.has_unfinished_requests:
                outs.extend(engine.step())
            elif pending:
                time.sleep(min(0.002, max(0.0, pending[0][0].arrival_offset - (time.perf_counter() - t0))))

    wall = time.perf_counter() - t0
    return outs, wall


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ModelConfig.model_id)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backend", default="auto", choices=["auto", "torch", "triton"])

    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--workload", default="skewed", choices=["uniform", "skewed"])
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--request-rate", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--num-blocks", type=int, default=None)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--no-chunked-prefill", action="store_true")
    ap.add_argument("--no-preemption", action="store_true")

    ap.add_argument("--quantization", default=None, choices=[None, "int8", "int4"])
    ap.add_argument("--group-size", type=int, default=128)

    ap.add_argument("--tag", default="engine")
    args = ap.parse_args()

    engine = build_engine(args)
    info = engine.describe()
    print(json.dumps(info, indent=2, default=str))

    tok = engine.tokenizer
    if args.workload == "uniform":
        reqs = uniform_workload(args.n, tok, args.prompt_len, args.output_len)
    else:
        reqs = skewed_workload(
            args.n, tok,
            prompt_mean=args.prompt_len,
            output_mean=args.output_len,
            request_rate=args.request_rate,
            seed=args.seed,
        )

    # Warmup: first CUDA calls pay for autotuning, lazy init and Triton JIT.
    warm = reqs[: min(4, len(reqs))]
    run(engine, warm, None)

    outs, wall = run(engine, reqs, args.request_rate)
    summary = summarize([o.metrics for o in outs], wall)
    summary["preemptions"] = sum(o.num_preemptions for o in outs)
    summary["mean_batch_size"] = engine.scheduler.stats.mean_batch_size
    summary["scheduler_steps"] = engine.scheduler.stats.total_steps
    summary["mixed_steps"] = engine.scheduler.stats.mixed_steps
    summary["prefill_tokens"] = engine.scheduler.stats.total_prefill_tokens
    summary["decode_tokens"] = engine.scheduler.stats.total_decode_tokens
    if str(args.device).startswith("cuda"):
        summary["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20

    label = f"continuous n={args.n} rate={args.request_rate or 'offline'}"
    print_summary(label, summary)
    print(f"  mean batch size     {summary['mean_batch_size']:.1f}")
    print(f"  preemptions         {summary['preemptions']}")
    print(f"  mixed steps         {summary['mixed_steps']} / {summary['scheduler_steps']}")

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.tag}_{args.workload}_{args.request_rate or 'offline'}.json"
    path.write_text(json.dumps({"config": vars(args), "engine": info, "summary": summary}, indent=2, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
