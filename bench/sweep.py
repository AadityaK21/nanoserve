"""Phase 5: the full sweep that produces the numbers in the report.

    python -m bench.sweep --quick        # ~10 min, sanity check
    python -m bench.sweep                # the real run

Every experiment answers one question, and the question is in the name:

  static_vs_continuous   does iteration-level scheduling actually beat static
                         batching, and by how much, under skewed lengths?
  latency_vs_load        where does the p99 knee sit as arrival rate rises?
  chunked_prefill        does chunking prompts cut tail TPOT, and what does it
                         cost in TTFT?
  block_size             what does paging granularity cost or buy?
  quantization           how much KV capacity does int4 free, and does the
                         extra concurrency turn into throughput?
  backend                torch gather vs fused Triton kernel, decode only.

Results land in results/sweep.json for bench/plot.py.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import torch

from nanoserve.baseline import run_static_batched
from nanoserve.config import (
    CacheConfig,
    EngineConfig,
    ModelConfig,
    SchedulerConfig,
)
from nanoserve.engine import LLMEngine
from nanoserve.metrics import print_summary, summarize
from nanoserve.workload import skewed_workload, uniform_workload

RESULTS = Path(__file__).resolve().parent.parent / "results"


def make_engine(device, dtype, quantization=None, block_size=16, chunked=True,
                max_batched=2048, backend="auto", num_blocks=None, gpu_util=0.85):
    cfg = EngineConfig(
        model=ModelConfig(dtype=dtype, device=device, quantization=quantization),
        cache=CacheConfig(block_size=block_size, gpu_memory_utilization=gpu_util,
                          num_gpu_blocks=num_blocks),
        scheduler=SchedulerConfig(max_num_batched_tokens=max_batched,
                                  enable_chunked_prefill=chunked),
        attention_backend=backend,
    )
    return LLMEngine(cfg)


def drive(engine, reqs, request_rate=None):
    from bench.run_engine import run

    outs, wall = run(engine, reqs, request_rate)
    s = summarize([o.metrics for o in outs], wall)
    s["preemptions"] = sum(o.num_preemptions for o in outs)
    s["mean_batch_size"] = engine.scheduler.stats.mean_batch_size
    s["scheduler_steps"] = engine.scheduler.stats.total_steps
    s["mixed_steps"] = engine.scheduler.stats.mixed_steps
    if torch.cuda.is_available():
        s["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
        torch.cuda.reset_peak_memory_stats()
    return s


def guard(name, fn, results):
    """Run one experiment; a failure should not lose the ones that passed."""
    print(f"\n{'=' * 62}\n{name}\n{'=' * 62}")
    try:
        results[name] = fn()
    except Exception as exc:  # pragma: no cover
        print(f"!! {name} failed: {exc}")
        traceback.print_exc()
        results[name] = {"error": str(exc)}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "sweep.json").write_text(json.dumps(results, indent=2, default=str))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip", default="", help="comma-separated experiment names")
    args = ap.parse_args()

    n = 32 if args.quick else 128
    rates = [2, 4, 8] if args.quick else [1, 2, 4, 8, 16, 32]
    batch_sizes = [1, 4, 16] if args.quick else [1, 2, 4, 8, 16, 32, 64]
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}

    # Merge into an existing sweep.json rather than clobbering it, so --skip
    # keeps previous results. Re-running a 30-minute static baseline to fix one
    # failed experiment is not a reasonable workflow.
    results: dict = {}
    prior = RESULTS / "sweep.json"
    if prior.exists():
        try:
            results = json.loads(prior.read_text())
            print(f"-- merging into existing {prior} "
                  f"({', '.join(k for k in results if k != 'meta')})")
        except Exception:
            results = {}
    results["meta"] = {"device": args.device, "dtype": args.dtype, "n": n,
                       "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if torch.cuda.is_available():
        results["meta"]["gpu"] = torch.cuda.get_device_name(0)

    engine = make_engine(args.device, args.dtype)
    results["meta"]["engine"] = engine.describe()
    print(json.dumps(results["meta"]["engine"], indent=2, default=str))
    tok = engine.tokenizer

    skewed = skewed_workload(n, tok, prompt_mean=256, output_mean=128, seed=0)
    uniform = uniform_workload(n, tok, 256, 128)

    # ---- 1. static vs continuous ------------------------------------------
    # The static baseline runs through the Phase 0 harness on the stock HF
    # model, so it is a genuinely independent reference rather than our engine
    # with a flag flipped. Loading a second copy of the weights is why it frees
    # the model before returning -- on 8 GB there is no room for two.
    def exp_static_baseline():
        from nanoserve.model import load_model as load_hf
        from nanoserve.config import ModelConfig as MC

        hf_model, hf_tok = load_hf(MC(dtype=args.dtype, device=args.device))
        out = {}
        for name, reqs in (("skewed", skewed), ("uniform", uniform)):
            out[name] = {}
            for bs in batch_sizes:
                t0 = time.perf_counter()
                recs = run_static_batched(hf_model, hf_tok, reqs, bs, device=args.device)
                s = summarize(recs, time.perf_counter() - t0)
                if torch.cuda.is_available():
                    s["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
                    torch.cuda.reset_peak_memory_stats()
                print_summary(f"static bs={bs} / {name}", s)
                out[name][f"bs{bs}"] = s
        del hf_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return out

    def exp_continuous():
        out = {}
        for name, reqs in (("skewed", skewed), ("uniform", uniform)):
            s = drive(engine, reqs)
            print_summary(f"continuous / {name}", s)
            out[name] = s
        return out

    def exp_latency_vs_load():
        out = {}
        for rate in rates:
            reqs = skewed_workload(n, tok, prompt_mean=256, output_mean=128,
                                   request_rate=rate, seed=0)
            s = drive(engine, reqs, request_rate=rate)
            print_summary(f"rate={rate} req/s", s)
            out[f"rate{rate}"] = s
        return out

    def exp_chunked_prefill():
        out = {}
        long_reqs = skewed_workload(n, tok, prompt_mean=768, prompt_sigma=0.8,
                                    output_mean=128, request_rate=8, seed=1)
        for chunked in (True, False):
            for budget in ([512, 2048] if not args.quick else [512, 2048]):
                key = f"{'chunked' if chunked else 'nochunk'}_budget{budget}"
                e = make_engine(args.device, args.dtype, chunked=chunked, max_batched=budget)
                try:
                    s = drive(e, long_reqs, request_rate=8)
                    print_summary(key, s)
                    out[key] = s
                except ValueError as ve:
                    # Not a bug -- it is the ablation's result. Without chunked
                    # prefill, any prompt longer than the whole token budget can
                    # never be scheduled, so the engine rejects it loudly. That
                    # *is* the argument for chunking; record it as such.
                    print(f"  {key}: rejected as expected -- {ve}")
                    out[key] = {"rejected": str(ve),
                                "meaning": "prompts longer than the budget are "
                                           "unschedulable without chunked prefill"}
                finally:
                    e.shutdown()
                    del e
        return out

    def exp_block_size():
        out = {}
        for bs_blk in ([8, 32] if args.quick else [8, 16, 32, 64]):
            e = make_engine(args.device, args.dtype, block_size=bs_blk)
            try:
                s = drive(e, skewed)
                s["num_gpu_blocks"] = e.num_gpu_blocks
                s["kv_capacity_tokens"] = e.kv_cache.capacity_tokens
                s["fragmentation_waste_bound"] = bs_blk - 1
                print_summary(f"block_size={bs_blk}", s)
                out[f"block{bs_blk}"] = s
            finally:
                e.shutdown()
                del e
        return out

    def exp_quantization():
        out = {}
        for q in (None, "int8", "int4"):
            e = make_engine(args.device, args.dtype, quantization=q)
            try:
                info = e.describe()
                s = drive(e, skewed)
                s["engine"] = info
                print_summary(f"quant={q}", s)
                print(f"  weights {info['weight_mib']:.0f} MiB   "
                      f"KV blocks {info['num_gpu_blocks']}   "
                      f"KV capacity {info['kv_capacity_tokens']:,} tok")
                out[str(q)] = s
            finally:
                e.shutdown()
                del e
        return out

    def exp_backend():
        out = {}
        for backend in ("torch", "triton"):
            try:
                e = make_engine(args.device, args.dtype, backend=backend)
            except RuntimeError as exc:
                out[backend] = {"error": str(exc)}
                continue
            try:
                s = drive(e, uniform)
                s["backend_used"] = e.runner.backend_name
                print_summary(f"backend={backend}", s)
                out[backend] = s
            finally:
                e.shutdown()
                del e
        return out

    # Ordering is a memory constraint, not a preference. The shared engine holds
    # ~4.8 GiB of KV cache, so everything that reuses it has to run before it is
    # torn down, and everything that builds its own engine has to run after.
    shared = {
        "continuous": exp_continuous,
        "latency_vs_load": exp_latency_vs_load,
    }
    standalone = {
        "static_baseline": exp_static_baseline,
        "chunked_prefill": exp_chunked_prefill,
        "block_size": exp_block_size,
        "quantization": exp_quantization,
        "backend": exp_backend,
    }

    for name, fn in shared.items():
        if name in skip:
            print(f"-- skipping {name}")
            continue
        guard(name, fn, results)

    # Rebind rather than `del`: the closures above capture this name, and
    # deleting it turns them into a NameError waiting to happen.
    engine.shutdown()
    engine = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"\n-- released shared engine: {free / 2**20:.0f} MiB free of "
              f"{total / 2**20:.0f} MiB")

    for name, fn in standalone.items():
        if name in skip:
            print(f"-- skipping {name}")
            continue
        guard(name, fn, results)

    print(f"\nwrote {RESULTS / 'sweep.json'}")


if __name__ == "__main__":
    main()
