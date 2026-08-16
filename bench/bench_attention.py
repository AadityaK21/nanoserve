"""Paged attention microbenchmark: torch gather vs fused Triton kernel.

    python -m bench.bench_attention

Isolates the attention op from the rest of the engine, because a whole-engine
comparison buries the kernel under ~1300 unrelated launches per step and you
cannot tell what the kernel itself did.

Sweep both context length and batch size, because the kernel wins for two
different reasons and they pull in opposite directions.

  Small context or small batch -- the win is kernel count. The torch path
  issues roughly ten ops (arange, block-table arithmetic, two index_selects, a
  permute, a head expansion, SDPA) where the kernel issues one. At batch 1 the
  torch path costs ~0.5 ms regardless of context, which is the signature of
  being launch-bound rather than data-bound.

  Large context and batch -- the win is memory traffic. The gather materialises
  an O(context) temporary, so KV crosses the bus twice: once to write the
  gathered copy, once for SDPA to read it. The kernel walks the block table and
  accumulates an online softmax in registers, touching KV exactly once.

The consequence, which is worth predicting before you look: speedup *falls*
with context at low batch (the kernel's real work grows while torch's fixed
overhead does not) and *stabilises* at high batch (both scale with context, so
the ratio settles at the traffic saving). A single headline speedup number
would hide all of that.

Correctness is checked at every point before timing it. A fast wrong kernel is
worth nothing.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from nanoserve.attention import AttentionMetadata, paged_attention_torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def warm_gpu(device, seconds: float = 6.0) -> None:
    """Laptop GPUs idle at a few hundred MHz. Measuring before they boost gives
    numbers that are wrong by up to 12x -- see the methodology note in the
    report."""
    if not torch.cuda.is_available():
        return
    a = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    b = torch.randn(4096, 4096, device=device, dtype=torch.float16)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < seconds:
        for _ in range(20):
            a @ b
        _sync()
    del a, b
    torch.cuda.empty_cache()


def timed(fn, iters: int = 50, warmup: int = 10) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / 1000.0 / iters


def build_case(batch, ctx, block_size, num_heads, num_kv_heads, head_dim, device, dtype):
    """One decode step: `batch` sequences each holding `ctx` tokens of context.

    Block tables are randomly permuted, not sequential. If the kernel ever
    assumed a sequence's blocks were contiguous, sequential tables would hide
    it and permuted ones expose it on the first run.
    """
    blocks_per_seq = (ctx + block_size - 1) // block_size
    num_blocks = batch * blocks_per_seq

    g = torch.Generator(device="cpu").manual_seed(0)
    k_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                          generator=g, dtype=torch.float32).to(device=device, dtype=dtype)
    v_cache = torch.randn(num_blocks, block_size, num_kv_heads, head_dim,
                          generator=g, dtype=torch.float32).to(device=device, dtype=dtype)

    perm = torch.randperm(num_blocks, generator=g).tolist()
    bt = torch.tensor(
        [perm[i * blocks_per_seq:(i + 1) * blocks_per_seq] for i in range(batch)],
        dtype=torch.int32, device=device,
    )

    md = AttentionMetadata(
        slot_mapping=torch.zeros(batch, dtype=torch.long, device=device),
        query_start_loc=torch.arange(batch + 1, dtype=torch.int32, device=device),
        seq_lens=torch.full((batch,), ctx, dtype=torch.int32, device=device),
        block_tables=bt,
        max_query_len=1,
        max_seq_len=ctx,
        num_seqs=batch,
        num_tokens=batch,
        is_decode_only=True,
        query_lens_cpu=[1] * batch,
        seq_lens_cpu=[ctx] * batch,
        query_start_loc_cpu=list(range(batch + 1)),
    )
    q = torch.randn(batch, num_heads, head_dim, generator=g,
                    dtype=torch.float32).to(device=device, dtype=dtype)
    return q, k_cache, v_cache, md


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", default="128,512,1024,2048,4096")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--block-size", type=int, default=16)
    ap.add_argument("--heads", type=int, default=14)        # Qwen2.5-0.5B
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--dtype", default="float16")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("needs a CUDA device")

    from nanoserve.triton_attention import paged_attention_triton, triton_available

    if not triton_available():
        raise SystemExit(
            "Triton is not available. On native Windows this is expected -- "
            "run this under WSL2. See SETUP.md."
        )

    device = "cuda"
    dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    scale = 1.0 / math.sqrt(args.head_dim)
    contexts = [int(x) for x in args.contexts.split(",")]
    batches = [int(x) for x in args.batches.split(",")]

    print(f"gpu                 {torch.cuda.get_device_name(0)}")
    print(f"config              {args.heads} q-heads / {args.kv_heads} kv-heads / "
          f"head_dim {args.head_dim} / block {args.block_size} / {args.dtype}")
    print("warming up          6 s of sustained GEMMs")
    warm_gpu(device)

    print()
    print(f"{'batch':>6} {'ctx':>6} {'torch ms':>10} {'triton ms':>10} "
          f"{'speedup':>8} {'max err':>10}")
    print("-" * 56)

    def run_case(batch, ctx):
        """Kept in its own scope so the cache tensors are freed on return --
        a 64-sequence 4096-token case allocates a few hundred MiB."""
        q, kc, vc, md = build_case(batch, ctx, args.block_size, args.heads,
                                   args.kv_heads, args.head_dim, device, dtype)

        ref = paged_attention_torch(q, kc, vc, md, scale)
        got = paged_attention_triton(q, kc, vc, md, scale)
        err = (got.float() - ref.float()).abs().max().item()
        if err > 3e-2:
            return {"batch": batch, "ctx": ctx, "max_abs_err": err, "status": "MISMATCH"}

        t_torch = timed(lambda: paged_attention_torch(q, kc, vc, md, scale))
        t_triton = timed(lambda: paged_attention_triton(q, kc, vc, md, scale))

        # KV bytes any correct kernel must read: 2 (K and V) x ctx x kv_heads x dim.
        itemsize = torch.tensor([], dtype=dtype).element_size()
        kv_bytes = 2 * batch * ctx * args.kv_heads * args.head_dim * itemsize
        return {
            "batch": batch, "ctx": ctx,
            "torch_ms": t_torch * 1000, "triton_ms": t_triton * 1000,
            "speedup": t_torch / t_triton,
            "max_abs_err": err,
            "triton_gbs": kv_bytes / t_triton / 1e9,
            "torch_gbs": kv_bytes / t_torch / 1e9,
        }

    rows = []
    for batch in batches:
        for ctx in contexts:
            row = run_case(batch, ctx)
            rows.append(row)
            if row.get("status") == "MISMATCH":
                print(f"{batch:>6} {ctx:>6}   MISMATCH max abs err "
                      f"{row['max_abs_err']:.4f} -- kernel is wrong, stop and fix")
            else:
                print(f"{batch:>6} {ctx:>6} {row['torch_ms']:>10.3f} "
                      f"{row['triton_ms']:>10.3f} {row['speedup']:>7.2f}x "
                      f"{row['max_abs_err']:>10.2e}")
            torch.cuda.empty_cache()

    ok = [r for r in rows if "speedup" in r]
    if ok:
        print()
        print("=" * 56)
        best = max(ok, key=lambda r: r["speedup"])
        print(f"  best speedup      {best['speedup']:.2f}x (batch {best['batch']}, ctx {best['ctx']})")
        print(f"  worst speedup     {min(ok, key=lambda r: r['speedup'])['speedup']:.2f}x")

        for b in (min(batches), max(batches)):
            trend = sorted([r for r in ok if r["batch"] == b], key=lambda r: r["ctx"])
            if len(trend) > 1:
                print(f"  batch {b:<3} ctx {trend[0]['ctx']}->{trend[-1]['ctx']}: "
                      f"{trend[0]['speedup']:.2f}x -> {trend[-1]['speedup']:.2f}x   "
                      f"(torch {trend[0]['torch_ms']:.2f}->{trend[-1]['torch_ms']:.2f} ms, "
                      f"triton {trend[0]['triton_ms']:.2f}->{trend[-1]['triton_ms']:.2f} ms)")

        # How close is the kernel to the hardware? Compare KV bytes it must read
        # against achieved d2d bandwidth. Anything far below means the kernel is
        # occupancy- or latency-bound, not bandwidth-bound -- which is a lead,
        # not a failure.
        big = max(ok, key=lambda r: r["batch"] * r["ctx"])
        print(f"  largest case      batch {big['batch']}, ctx {big['ctx']}: "
              f"{big['triton_gbs']:.0f} GB/s of KV read (triton) vs "
              f"{big['torch_gbs']:.0f} GB/s (torch)")
        print("  compare that to the ~220 GB/s this GPU sustains on a plain copy:")
        print("  a large gap means one program per (sequence, head) with a serial")
        print("  loop over blocks is not enough parallelism to saturate the bus.")
        print("  Splitting the context across programs (flash-decoding) is the fix.")

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "attention_backend.json"
    out.write_text(json.dumps({
        "gpu": torch.cuda.get_device_name(0),
        "config": vars(args),
        "rows": rows,
    }, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
