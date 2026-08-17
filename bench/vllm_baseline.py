"""vLLM on the identical workload, for an honest external reference point.

    python -m bench.vllm_baseline            # writes results/vllm_baseline.json
    python -m bench.vllm_baseline --compare  # prints the table vs results/sweep.json

Why this exists: "1.65x over my own static baseline" is a claim about my
baseline, not about the state of the art. The first question anyone serious
asks is how it compares to vLLM, and not answering looks like avoidance.

Expect to lose, and by a lot. vLLM has FlashAttention/FlashInfer kernels, CUDA
graph capture for decode, a compiled C++ scheduler, and years of tuning. This
project has a Python scheduler and one hand-written Triton kernel, on a GPU
where §2 of the report shows decode is ~80% launch overhead -- which is exactly
what CUDA graphs exist to remove and exactly what nanoserve does not have.

A measured gap plus a correct explanation of where it comes from is a far
better answer than no number. It also makes the gap actionable: if the ratio is
close to the launch-overhead fraction, CUDA graphs are the whole story, and
that is a testable prediction rather than an excuse.

INSTALL IN A SEPARATE VIRTUALENV. `pip install vllm` pins its own torch build
and will likely replace the one nanoserve is working against:

    python3 -m venv ~/vllm-venv
    source ~/vllm-venv/bin/activate
    pip install vllm numpy

Running under WSL2 needs three workarounds, all environmental rather than
anything to do with this script:

    # 1. WSL2 does not expose Unified Virtual Addressing, which vLLM's V2 model
    #    runner allocates unconditionally.  (vllm-project/vllm#47387)
    export VLLM_WSL2_ENABLE_PIN_MEMORY=1

    # 2. FlashInfer JIT-compiles its sampling kernels and needs nvcc. WSL gives
    #    you the CUDA driver, not the toolkit, so use the PyTorch sampler.
    export VLLM_USE_FLASHINFER_SAMPLER=0

    # 3. WSL inherits the Windows PATH, which can put an unexecutable Windows
    #    `nvcc` in front of everything; torch inductor shells out to it and
    #    dies with PermissionError. Trim PATH to the Linux side.
    PATH=~/vllm-venv/bin:/usr/local/bin:/usr/bin:/bin \
        python -m bench.vllm_baseline

Add --enforce-eager if compilation still refuses; see the flag's help.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results"


def build_workload(n, model_id, prompt_mean, output_mean, seed):
    """The same requests nanoserve was measured on.

    Uses nanoserve.workload with the same seed, so the prompts and the output
    lengths are identical token-for-token. Comparing against a differently
    generated workload would measure the workload, not the engine.
    """
    from transformers import AutoTokenizer

    from nanoserve.workload import skewed_workload

    tok = AutoTokenizer.from_pretrained(model_id)
    return skewed_workload(
        n, tok, prompt_mean=prompt_mean, output_mean=output_mean, seed=seed
    ), tok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--prompt-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    # Match nanoserve's default exactly. vLLM defaults to 0.90, which would
    # hand it a bigger KV cache and quietly make the comparison unfair.
    ap.add_argument("--gpu-util", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        help="disable torch.compile and CUDA graphs in vLLM. Needed under WSL2, "
             "where inductor's nvcc probe fails; also gives the like-for-like "
             "comparison, since nanoserve has neither.",
    )
    ap.add_argument("--compare", action="store_true", help="only print the table")
    args = ap.parse_args()

    if args.compare:
        return compare()

    from vllm import LLM, SamplingParams

    reqs, _ = build_workload(args.n, args.model, args.prompt_len, args.output_len, args.seed)

    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )

    prompts = [r.prompt for r in reqs]
    params = [
        SamplingParams(temperature=0.0, max_tokens=r.max_new_tokens, ignore_eos=True)
        for r in reqs
    ]

    # Warmup so the comparison is steady-state, matching how nanoserve is timed.
    llm.generate(prompts[:2], params[:2], use_tqdm=False)

    t0 = time.perf_counter()
    outs = llm.generate(prompts, params, use_tqdm=False)
    wall = time.perf_counter() - t0

    out_tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tokens = sum(len(o.prompt_token_ids) for o in outs)

    result = {
        "engine": "vllm",
        "model": args.model,
        "n": args.n,
        "workload": "skewed",
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_util,
        # Records what vLLM was actually allowed to use. Without this the number
        # is uninterpretable: eager vLLM and compiled vLLM are different systems.
        "enforce_eager": args.enforce_eager,
        "cuda_graphs": not args.enforce_eager,
        "wall_time_s": wall,
        "prompt_tokens": in_tokens,
        "output_tokens": out_tokens,
        "output_tok_per_s": out_tokens / wall,
        "total_tok_per_s": (in_tokens + out_tokens) / wall,
    }

    try:
        import vllm

        result["vllm_version"] = vllm.__version__
    except Exception:
        pass

    print(json.dumps(result, indent=2))
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "vllm_baseline.json").write_text(json.dumps(result, indent=2))
    print(f"\nwrote {RESULTS / 'vllm_baseline.json'}")
    compare()


def compare() -> None:
    """Side by side with whatever the sweep measured for nanoserve."""
    v = RESULTS / "vllm_baseline.json"
    s = RESULTS / "sweep.json"
    if not v.exists():
        raise SystemExit("no results/vllm_baseline.json -- run without --compare first")
    vd = json.loads(v.read_text())

    label = "vLLM (eager)" if vd.get("enforce_eager") else "vLLM (compiled+graphs)"
    rows = [(label, vd["output_tok_per_s"])]
    if s.exists():
        sw = json.loads(s.read_text())
        cont = sw.get("continuous", {}).get("skewed")
        if isinstance(cont, dict) and "output_tok_per_s" in cont:
            rows.append(("nanoserve (continuous)", cont["output_tok_per_s"]))
        static = sw.get("static_baseline", {}).get("skewed", {})
        best = None
        for k, val in (static or {}).items():
            if isinstance(val, dict) and "output_tok_per_s" in val:
                if best is None or val["output_tok_per_s"] > best[1]:
                    best = (f"static batching ({k})", val["output_tok_per_s"])
        if best:
            rows.append(best)

    print()
    print("=" * 62)
    print(f"skewed workload, n={vd['n']}, same prompts and output lengths")
    print("=" * 62)
    ref = rows[0][1]
    for name, tps in rows:
        print(f"  {name:<26} {tps:8.1f} tok/s   {tps / ref:5.2f}x vLLM")

    print()
    if vd.get("enforce_eager"):
        print("  vLLM ran EAGER here: no torch.compile, no CUDA graphs. That")
        print("  removes its single biggest advantage, so this is not 'vs vLLM")
        print("  at its best' -- it is a like-for-like comparison of engine")
        print("  design, since nanoserve has neither either. It still keeps")
        print("  FlashAttention, a compiled scheduler and prefix caching.")
        print("  Read the gap as: what is left after compilation is excluded.")
    else:
        print("  vLLM ran with torch.compile and CUDA graph capture. On a GPU")
        print("  where decode is ~80% launch overhead (report section 2), graph")
        print("  capture alone should account for most of any gap -- a testable")
        print("  prediction for the CUDA-graphs work in section 9.")


if __name__ == "__main__":
    main()
