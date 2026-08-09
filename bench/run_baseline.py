"""Phase 0 benchmark driver.

    python -m bench.run_baseline --mode sequential --n 16
    python -m bench.run_baseline --mode static --batch-sizes 1,2,4,8,16,32
    python -m bench.run_baseline --mode static --batch-sizes 8 --workload skewed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nanoserve.baseline import run_sequential, run_static_batched
from nanoserve.config import ModelConfig
from nanoserve.metrics import print_summary, summarize
from nanoserve.model import DTYPES, describe, load_model
from nanoserve.workload import skewed_workload, uniform_workload

RESULTS = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=ModelConfig.model_id)
    ap.add_argument("--dtype", default="float16", choices=list(DTYPES))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--mode", default="static", choices=["sequential", "static"])
    ap.add_argument("--workload", default="uniform", choices=["uniform", "skewed"])
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--batch-sizes", default="1,2,4,8,16,32")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--tag", default="baseline")
    args = ap.parse_args()

    cfg = ModelConfig(model_id=args.model, dtype=args.dtype, device=args.device)
    model, tok = load_model(cfg)

    info = describe(model, DTYPES[args.dtype])
    print(f"model               {args.model}")
    print(f"params              {info['params'] / 1e6:.1f} M")
    print(f"weights             {info['weight_mib']:.0f} MiB")
    print(f"KV per token        {info['kv_bytes_per_token'] / 1024:.1f} KiB")
    print(f"KV per 1k tokens    {info['kv_mib_per_1k_tokens']:.1f} MiB")
    if "kv_tokens_in_80pct_free" in info:
        print(f"KV tokens that fit  ~{info['kv_tokens_in_80pct_free']:,}")

    if args.workload == "uniform":
        reqs = uniform_workload(args.n, tok, args.prompt_len, args.output_len)
    else:
        reqs = skewed_workload(
            args.n, tok, prompt_mean=args.prompt_len, output_mean=args.output_len
        )

    # Warmup: the first CUDA calls pay for kernel autotuning and lazy init.
    # Timing those would make batch size 1 look far worse than it is.
    if args.warmup:
        run_static_batched(model, tok, reqs[: args.warmup], 1, device=args.device)

    all_results = {"config": vars(args), "model_info": info, "runs": {}}

    if args.mode == "sequential":
        t0 = time.perf_counter()
        recs = run_sequential(model, tok, reqs, device=args.device)
        s = summarize(recs, time.perf_counter() - t0)
        print_summary("sequential (batch=1)", s)
        all_results["runs"]["sequential"] = s
    else:
        for bs in [int(x) for x in args.batch_sizes.split(",")]:
            t0 = time.perf_counter()
            recs = run_static_batched(model, tok, reqs, bs, device=args.device)
            s = summarize(recs, time.perf_counter() - t0)
            print_summary(f"static batch={bs}", s)
            all_results["runs"][f"bs{bs}"] = s
            if args.device.startswith("cuda"):
                s["peak_gpu_mib"] = torch.cuda.max_memory_allocated() / 2**20
                torch.cuda.reset_peak_memory_stats()

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{args.tag}_{args.mode}_{args.workload}.json"
    path.write_text(json.dumps(all_results, indent=2))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
