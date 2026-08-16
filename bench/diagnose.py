"""Where does a decode step actually go?

    python -m bench.diagnose

A decode step has a hard floor: it must read every weight once. Divide weight
bytes by the GPU's achieved memory bandwidth and you get the fastest a step can
possibly be. Measure the real step next to that floor and you learn which of
two completely different problems you have.

  near the floor   memory-bound. Normal. Gains come from reading fewer bytes
                   (quantisation) or serving more sequences per read (batching).

  far above it     the GPU is idle, waiting on the CPU. Gains come from issuing
                   fewer, larger kernels -- or from not re-issuing them at all
                   (CUDA graphs).

The tie-breaker is GPU-busy time, measured with the profiler: sum the duration
of every kernel in a step and compare it to the wall time of that step. That
ratio is the honest utilisation number, and it does not care about anyone's
theory.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def gpu_clocks():
    """(SM MHz, watts) from nvidia-smi, or (None, None) if unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.sm,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        ).stdout.strip().splitlines()[0]
        sm, pw = (x.strip() for x in out.split(","))
        return float(sm), float(pw)
    except Exception:
        return None, None


def warm_gpu(device, seconds: float = 8.0) -> None:
    """Drag the GPU out of its idle power state before measuring anything.

    A laptop GPU sits at ~4 W and a few hundred MHz until something makes it
    work. Small decode kernels are not enough to trigger a boost, so the first
    few configurations in a sweep get measured on a sleepy card and the last
    ones on a hot one. That produces the classic nonsense result where a larger
    batch appears *faster* than a smaller one.

    Sustained large GEMMs push the clocks to their steady state so every
    configuration is measured under the same conditions. This costs ten seconds
    and is the difference between numbers you can defend and numbers you can't.
    """
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


def timed(fn, iters: int, warmup: int = 5) -> float:
    """Seconds per call, CUDA-event timed, warmup excluded."""
    for _ in range(warmup):
        fn()
    _sync()
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / 1000.0 / iters
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    return (time.perf_counter() - t0) / iters


def profile_step(step, reps: int = 3):
    """(device ops per step, GPU-busy seconds, wall seconds) -- all per step.

    Wall time is measured *inside* the profiled run, not taken from the
    unprofiled timing above. Profiling inflates kernel durations, so dividing
    profiled busy time by unprofiled wall time can hand you a utilisation over
    100% -- which is how I know it is the wrong comparison rather than a
    surprising result.

    Counting entries that consumed device time is a proxy for kernel launches.
    """
    try:
        from torch.profiler import ProfilerActivity, profile

        _sync()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            t0 = time.perf_counter()
            for _ in range(reps):
                step()
            _sync()
            wall = time.perf_counter() - t0

        # Count only real kernel events. key_averages() attributes device time
        # to BOTH the aten:: op and the kernel it launched, so summing
        # everything with device time counts every kernel twice -- which is how
        # you get a 166% utilisation and know the metric is wrong.
        try:
            from torch.autograd import DeviceType

            cuda_kind = DeviceType.CUDA
        except Exception:
            cuda_kind = None

        total_us, count = 0.0, 0
        for e in prof.key_averages():
            if cuda_kind is not None and getattr(e, "device_type", None) != cuda_kind:
                continue
            dt = getattr(e, "self_device_time_total", None)
            if dt is None:
                dt = getattr(e, "self_cuda_time_total", 0.0)
            if dt and dt > 0:
                total_us += dt
                count += e.count
        return count / reps, total_us / reps / 1e6, wall / reps
    except Exception:
        return None, None, None


# ---- hardware limits ------------------------------------------------------
def measure_bandwidth(device) -> float:
    n = 256 * 1024 * 1024 // 2          # 256 MiB of fp16
    a = torch.empty(n, dtype=torch.float16, device=device)
    b = torch.empty_like(a)
    per_call = timed(lambda: b.copy_(a), iters=30)
    return 2 * a.numel() * 2 / per_call / 1e9      # read + write


def measure_gemm_tflops(device) -> float:
    n = 4096
    a = torch.randn(n, n, device=device, dtype=torch.float16)
    b = torch.randn(n, n, device=device, dtype=torch.float16)
    per_call = timed(lambda: a @ b, iters=30)
    return 2 * n**3 / per_call / 1e12


def measure_launch_overhead(device) -> float:
    """Seconds per trivial kernel, GPU otherwise idle.

    Note this is latency, not pure submission cost: a laptop GPU sitting in a
    low power state will not clock up for 64-element kernels, so this number
    includes whatever the driver and the clock ramp cost too. That is the right
    number for our purposes, because it is what a small model actually pays.
    """
    x = torch.zeros(64, device=device)

    def burst():
        for _ in range(100):
            x.add_(1.0)

    return timed(burst, iters=20) / 100


# ---- decode steps ---------------------------------------------------------
def make_hf_step(model, device, batch: int, ctx: int = 256):
    from nanoserve.model import build_position_ids

    ids = torch.randint(0, 1000, (batch, ctx), device=device)
    attn = torch.ones_like(ids)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=attn,
                    position_ids=build_position_ids(attn), use_cache=True)
        state = {"attn": attn, "past": out.past_key_values,
                 "tok": out.logits[:, -1:].argmax(-1)}

    @torch.inference_mode()
    def step():
        a = torch.cat([state["attn"], torch.ones_like(state["tok"])], dim=-1)
        pos = a.sum(-1, keepdim=True) - 1
        o = model(input_ids=state["tok"], attention_mask=a, position_ids=pos,
                  past_key_values=state["past"], use_cache=True)
        state["attn"] = a
        state["past"] = o.past_key_values
        state["tok"] = o.logits[:, -1:].argmax(-1)

    return step


def make_engine_step(engine, batch: int, ctx: int = 256):
    """Drain prefill, then hand back a callable that is pure steady-state decode."""
    from nanoserve.config import SamplingParams

    engine.reset()
    prompt = list(range(ctx))
    budget = engine.cfg.model.max_model_len - ctx - 64
    for i in range(batch):
        engine.add_request(
            prompt_token_ids=prompt,
            sampling=SamplingParams(max_new_tokens=budget, ignore_eos=True),
            request_id=i,
        )
    for _ in range(64):
        engine.step()
        if engine.scheduler.running and not any(s.is_prefilling for s in engine.scheduler.running):
            break
    return engine.step


def probe(label, batch, step, floor, rows, report_rows, device="cuda"):
    """Re-warm, time, profile, time again.

    The re-warm matters more than it looks. Loading a checkpoint is seconds of
    disk and CPU work with the GPU idle, so the card drops back to its low
    power state and whichever configuration runs first after a load gets
    measured on a cold GPU. That is a 12x clock difference here, which is
    larger than any effect we are trying to measure.

    The repeat is not paranoia either. If two timings taken seconds apart
    disagree, the machine was changing underneath the measurement and any
    conclusion from a single number is an artefact.
    """
    warm_gpu(device, seconds=2.0)

    t1 = timed(step, iters=40, warmup=20)
    ops, busy, wall_prof = profile_step(step)
    t2 = timed(step, iters=40, warmup=5)

    t = min(t1, t2)
    spread = abs(t1 - t2) / t
    util = (busy / wall_prof * 100) if busy and wall_prof else float("nan")

    rows.append((label, batch, t))
    report_rows.append({
        "impl": label, "batch": batch, "ms": t * 1000,
        "ms_run1": t1 * 1000, "ms_run2": t2 * 1000, "spread_pct": spread * 100,
        "device_ops": ops, "gpu_busy_ms": busy * 1000 if busy else None,
        "profiled_wall_ms": wall_prof * 1000 if wall_prof else None,
        "gpu_util_pct": util,
    })

    ops_s = f"{ops:6.0f}" if ops else "     ?"
    busy_s = f"{busy * 1000:6.2f}" if busy else "     ?"
    flag = ""
    if spread > 0.15:
        flag = f"  <- UNSTABLE, runs differed {spread * 100:.0f}%"
    elif util == util and util > 100:
        flag = "  <- busy > wall, measurement inconsistent"
    print(f"  {label:<13} batch {batch:>3}   {t * 1000:7.2f} ms   {batch / t:7.1f} tok/s   "
          f"{ops_s} ops   {busy_s} ms busy   {min(util, 999):5.1f}% util{flag}")
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batches", default="1,8,32")
    ap.add_argument("--skip-engine", action="store_true")
    args = ap.parse_args()

    dev = args.device
    batches = [int(b) for b in args.batches.split(",")]
    report: dict = {"device": dev, "torch": torch.__version__}

    print("=" * 78)
    print("hardware")
    print("=" * 78)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info()
        report.update(gpu=props.name, gpu_total_mib=total / 2**20, gpu_free_mib=free / 2**20)
        print(f"  gpu                 {props.name}")
        print(f"  torch / cuda        {torch.__version__} / {torch.version.cuda}")
        print(f"  memory              {free / 2**20:.0f} MiB free of {total / 2**20:.0f} MiB")
        if (total - free) / 2**20 > 400:
            print(f"  !! {(total - free) / 2**20:.0f} MiB held by other apps -- that is KV cache you cannot use")
    else:
        print("  no CUDA device -- the numbers below are meaningless for this project")

    sm_idle, w_idle = gpu_clocks()
    print(f"  clocks, idle        {sm_idle:.0f} MHz @ {w_idle:.0f} W" if sm_idle
          else "  clocks, idle        (nvidia-smi unavailable)")
    print("  warming up          8 s of sustained GEMMs, so every measurement below")
    print("                      sees the same clocks")
    warm_gpu(dev)
    sm_hot, w_hot = gpu_clocks()
    if sm_hot:
        print(f"  clocks, warm        {sm_hot:.0f} MHz @ {w_hot:.0f} W")
        report.update(sm_mhz_idle=sm_idle, sm_mhz_warm=sm_hot, watts_warm=w_hot)
        if sm_idle and sm_hot > sm_idle * 1.5:
            print(f"  !! clocks rose {sm_hot / sm_idle:.1f}x from idle. Anything measured on a")
            print("     cold GPU would have been wrong by roughly that factor.")

    bw = measure_bandwidth(dev)
    tflops = measure_gemm_tflops(dev)
    launch = measure_launch_overhead(dev)
    report.update(bandwidth_gbs=bw, gemm_tflops=tflops, launch_overhead_us=launch * 1e6)
    print(f"  memory bandwidth    {bw:.0f} GB/s   (achieved, d2d copy)")
    print(f"  fp16 GEMM           {tflops:.1f} TFLOP/s")
    print(f"  small-kernel cost   {launch * 1e6:.1f} us each")

    print()
    print("=" * 78)
    print("decode step")
    print("=" * 78)

    from nanoserve.config import ModelConfig
    from nanoserve.model import load_model as load_hf

    hf_model, _ = load_hf(ModelConfig(model_id=args.model, dtype=args.dtype, device=dev))
    weight_bytes = sum(p.numel() * p.element_size() for p in hf_model.parameters())
    floor = weight_bytes / (bw * 1e9)
    report.update(weight_mib=weight_bytes / 2**20, bandwidth_floor_ms=floor * 1000)

    print(f"  weights             {weight_bytes / 2**20:.0f} MiB")
    print(f"  bandwidth floor     {floor * 1000:.2f} ms/step  -- nothing can beat this")
    print()

    rows, report_rows = [], []
    for b in batches:
        probe("transformers", b, make_hf_step(hf_model, dev, b), floor, rows, report_rows, dev)

    del hf_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not args.skip_engine:
        print()
        try:
            from nanoserve.config import CacheConfig, EngineConfig, SchedulerConfig
            from nanoserve.engine import LLMEngine

            engine = LLMEngine(EngineConfig(
                model=ModelConfig(model_id=args.model, dtype=args.dtype, device=dev),
                # Fixed modest cache: this is a latency probe, not a capacity test.
                cache=CacheConfig(block_size=16, num_gpu_blocks=4096),
                scheduler=SchedulerConfig(max_num_batched_tokens=8192),
            ))
            print(f"  (attention backend: {engine.runner.backend_name})")
            for b in batches:
                probe("nanoserve", b, make_engine_step(engine, b), floor, rows, report_rows, dev)
        except Exception as exc:
            import traceback

            print(f"  !! engine probe failed: {exc}")
            traceback.print_exc()

    report["rows"] = report_rows

    print()
    print("=" * 78)
    print("verdict")
    print("=" * 78)

    unstable = [r for r in report_rows if r["spread_pct"] > 15]
    if unstable:
        print("  !! Some rows were unstable between repeats. Treat them as indicative")
        print("     only, and re-run with everything else on the machine closed:")
        for r in unstable:
            print(f"       {r['impl']} batch {r['batch']}: "
                  f"{r['ms_run1']:.1f} vs {r['ms_run2']:.1f} ms")
        print()
    hf1 = next((r for r in report_rows if r["impl"] == "transformers" and r["batch"] == 1), None)
    if hf1:
        ratio = hf1["ms"] / 1000 / floor
        util = hf1["gpu_util_pct"]
        report["batch1_over_floor"] = ratio
        if ratio < 2:
            print(f"  Memory-bound. Batch-1 step is {ratio:.1f}x the floor -- about as good as")
            print("  it gets. Throughput now comes from batching and quantisation.")
        else:
            print(f"  Overhead-bound. Batch-1 step is {ratio:.0f}x the bandwidth floor.")
            if util == util:      # not NaN
                print(f"  The kernels in that step add up to {hf1['gpu_busy_ms']:.1f} ms of "
                      f"{hf1['ms']:.1f} ms wall, so the GPU")
                print(f"  is genuinely idle {100 - util:.0f}% of the time. With "
                      f"{hf1['device_ops']:.0f} device ops per step at")
                print(f"  ~{launch * 1e6:.0f} us of launch latency each, that idle time is launch overhead.")
            print()
            print("  Consequences worth writing down, because they shape the whole report:")
            print("   - fewer, larger kernels beat clever math here. That is exactly what")
            print("     paged attention plus flat batching buys, independent of memory.")
            print("   - continuous batching wins bigger than the literature suggests,")
            print("     because a fixed per-step cost is amortised over more sequences.")
            print("   - CUDA graphs are the highest-value next step: decode shapes are")
            print("     static, so the launches can be replayed instead of re-issued.")
            print()
            print("  Before blaming the OS: measure it. The small-kernel cost above varies")
            print("  by ~35% between runs on one machine, which is wider than the gap")
            print("  between Windows and Linux on this hardware. Kernel *count* is the")
            print("  variable that actually moves the step time; the driver model is not.")

    ns1 = next((r for r in report_rows if r["impl"] == "nanoserve" and r["batch"] == 1), None)
    if hf1 and ns1:
        print()
        print(f"  nanoserve vs transformers at batch 1: {hf1['ms'] / ns1['ms']:.2f}x "
              f"({hf1['ms']:.1f} -> {ns1['ms']:.1f} ms/step)")
        if ns1["device_ops"] and hf1["device_ops"]:
            print(f"  device ops per step: {hf1['device_ops']:.0f} -> {ns1['device_ops']:.0f}")

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "diagnose.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {RESULTS / 'diagnose.json'}")


if __name__ == "__main__":
    main()
