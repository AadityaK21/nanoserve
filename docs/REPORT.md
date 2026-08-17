# nanoserve — results

**Hardware:** RTX 4060 Laptop 8 GB (Ada) · Windows 11, WDDM driver model ·
PyTorch 2.11.0+cu128 · CUDA 12.8
**Model:** Qwen2.5-0.5B-Instruct, fp16 · 494M params · 24 layers · 14 query
heads / 2 KV heads (GQA 7:1) · head_dim 64
**Workloads:** `--quick` sweep, n=32 requests. *Skewed* = lognormal prompt and
output lengths (realistic traffic). *Uniform* = identical 256-in / 128-out
(the optimistic case). Greedy + `ignore_eos`, so output length is exactly
controlled. All numbers from `results/sweep.json` and `results/diagnose.json`.

Run-to-run noise on this machine is ~10%, so only same-run comparisons and
effects well above that are claimed.

---

## 1. Where the memory goes

| | MiB |
|---|---|
| Weights (fp16) | 943 |
| KV cache (profiled, `gpu_memory_utilization=0.85`) | 4,834 |
| **KV cost per token** | **12 KiB** (2 × 24 layers × 2 KV heads × 64 × 2 B) |
| **KV capacity** | **412,480 tokens** = 25,780 blocks × 16 |

GQA is doing enormous work here: with 14 KV heads instead of 2, per-token cost
would be 84 KiB and capacity ~59k tokens. The 7:1 head sharing is why a laptop
GPU can hold ~800 concurrent 512-token sequences at all.

A naive server reserving contiguous `max_model_len = 4096` per request fits
**100** concurrent requests in the same memory; paging bounds waste to at most
15 tokens per sequence instead of `4096 − actual`, which is where the 8×
concurrency headroom comes from.

## 2. Roofline: this GPU is overhead-bound, not memory-bound

`python -m bench.diagnose` → `results/diagnose.json`

Most inference writing assumes decode is memory-bandwidth bound. On this
machine it is not, and every result below has to be read in that light.

Measured on both OSes on the same laptop: Windows 11 (WDDM, torch 2.11+cu128)
and WSL2 Ubuntu (torch 2.13+cu130).

| | Windows | WSL2 |
|---|---|---|
| Achieved memory bandwidth | 209–222 GB/s | 218–226 GB/s |
| fp16 GEMM | 29.6–31.6 TFLOP/s | 30.1–30.7 TFLOP/s |
| Small-kernel cost | 7.8–15 µs | 6.8–9.2 µs |
| **Bandwidth floor per decode step** | ~4.5 ms | ~4.5 ms |
| Batch-1 step, transformers | 23.7 ms | 16.6 ms |
| GPU-busy within that step | 6.6 ms | 6.0 ms |
| **GPU utilisation during decode** | **18%** | **22%** |

The GPU is idle ~80% of every decode step on **both** operating systems,
waiting on the CPU to issue ~1,000–1,350 kernels.

**Correcting an earlier hypothesis.** I attributed much of this to Windows'
WDDM driver model routing launches through the OS scheduler, and predicted a
large improvement on Linux. The measurement does not support that:

- Small-kernel cost measured **6.8 µs and 9.2 µs on two consecutive runs of the
  same Linux setup** — a 35% spread that overlaps the Windows range. The OS
  difference is not resolvable above that noise.
- The two OSes also ran different PyTorch builds (2.11+cu128 vs 2.13+cu130),
  which changed device-ops-per-step from 1,350 to 1,014 for *identical*
  transformers code. That is a library difference, not an OS one, and it
  confounds any Windows-vs-Linux claim.
- GPU utilisation is 18% vs 22%. Overhead-bound on both.

The honest conclusion is narrower and more useful: **kernel count is the
variable that moves decode time on this hardware; the driver model is not.**
Every optimisation that paid (§2 optimisation table, §8's fused kernel) reduced
op count. Nothing that changed the OS did.

The signature is visible everywhere: step time is ~flat from batch 1 to batch
32 (28 → 34 ms), so throughput scales almost linearly with batch size —
**35.5 → 941.8 tok/s, a 26× gain for 20% more step time**. Batching is close to
free until the fixed per-step cost is amortised away.

### Measurement bugs found on the way (kept, deliberately)

The first four attempts at this measurement produced wrong numbers, each
plausible-looking:

- **Clock ramp.** The GPU idles at 210 MHz / 3 W and boosts to 2595 MHz /
  101 W — a 12.4× swing. Whichever config ran first was measured cold, which
  once produced "batch 32 is faster than batch 1" and a fake 3.4× win over
  transformers. Fix: 8 s of sustained GEMMs before measuring, re-warmed before
  every row (model loading lets clocks fall back).
- **Double-counted kernels.** `key_averages()` attributes device time to both
  the `aten::` op and its kernel; summing both reported 166% GPU utilisation.
- **Profiled busy ÷ unprofiled wall.** Utilisation must use busy and wall time
  from the same (profiled) run; profiling inflates kernel durations.
- **Instability.** Every config is now timed twice and flagged if the runs
  disagree by >15%.

### Optimisations this motivated (batch-1 step, same run pair: 33.7 → 28.2 ms, −16%)

| Change | Why it mattered here |
|---|---|
| RoPE cos/sin hoisted out of the layer loop | positions are identical across all 24 layers; the per-layer lookup built 48 identical tensors per step |
| `F.rms_norm` (fused) | replaced an 8-kernel hand-written norm at 48 sites/step |
| `enable_gqa=True` in SDPA | stopped materialising a 7× expanded copy of K and V |
| One pinned H2D copy for all per-step metadata | was six separate `torch.tensor(list, device=cuda)` transfers per step |
| Block-wise slot mapping | 128 loop iterations per 2048-token chunk instead of 2048 |

After these, nanoserve issues fewer device ops per step than eager
transformers (1,257 vs 1,350) and overtakes it at batch 32 (941.8 vs 925.0
tok/s). At batch 1 it remains 0.84× — the torch paged-attention backend
re-gathers the whole KV context every step, which is the cost the Triton
kernel removes (§8).

## 3. Continuous batching vs static batching

The headline experiment. Same 32 requests, same model, same GPU.

**Skewed lengths (realistic):**

| | output tok/s | p99 TTFT | p99 E2E | wall time |
|---|---|---|---|---|
| static bs=1 | 10.0 | 653 s | 675 s | 678 s |
| static bs=4 | 22.9 | 248 s | 285 s | 297 s |
| static bs=16 (best) | 61.0 | 60.8 s | 110.8 s | 112 s |
| **continuous** | **100.4** | **5.2 s** | **65.4 s** | **68 s** |

**1.65× the throughput of the best static configuration, with 12× better p99
TTFT.** TTFT is the number that collapses: static batching makes request 32
wait for every earlier batch to fully finish, so tail TTFT is queueing time.
Continuous batching admits a request into the next iteration.

**Uniform lengths (the honest control):** continuous 223.5 vs static bs=16
152.3 tok/s — **1.47×**. The gap narrows exactly as theory predicts: with
identical lengths every sequence in a static batch finishes together, so there
is little straggler waste to recover. The win on skewed traffic is the real
one, and skewed is what production traffic looks like.

Static TPOT is ~100 ms at *every* batch size (§2's flat-step-time signature),
so static throughput is `batch_size / 100 ms` — it scales only by raising batch
size, and its latency cost scales with it.

## 3b. Against vLLM

`python -m bench.vllm_baseline` (WSL2, separate virtualenv) →
`results/vllm_baseline.json`

Same model, same GPU, same prompts and output lengths (identical seed through
`nanoserve.workload`), same `gpu_memory_utilization=0.85` — vLLM defaults to
0.90, which would hand it a larger KV cache and quietly rig the comparison.

vLLM 0.27.1, `enforce_eager=False` — full torch.compile plus CUDA graph capture
(`FULL_AND_PIECEWISE`, 51 piecewise + 35 full graphs), FlashAttention 2,
prefix caching on. Its strongest configuration, not a handicapped one.

| | output tok/s | vs vLLM |
|---|---|---|
| vLLM (compiled + CUDA graphs) | 2103.5 | 1.00× |
| **nanoserve (continuous, Triton)** | **663.0** | **0.32×** |
| static batching (best, bs=16) | 236.9 | 0.11× |

A comparison against my own static baseline is a claim about my baseline, not
about the state of the art, so this row belongs here even though it is
unflattering. **nanoserve is 2.8× the static baseline and 0.32× vLLM.**

**An independent check that fell out of this.** vLLM's memory profiler
allocated a KV cache of **402,016 tokens** at `gpu_memory_utilization=0.85`;
nanoserve's profiler independently computed **412,480** on the same GPU with
the same setting — a 2.6% difference. Two implementations that share no code
agreeing on the memory arithmetic is decent evidence that §1 is right.

**Where the 3.2× goes.** §2 makes this predictable rather than mysterious:

- **CUDA graphs.** Decode here is ~80% launch overhead. vLLM captures 86 graphs
  at startup and replays them; nanoserve re-issues ~680 kernels every step
  through the Python interpreter. This should be the single largest term, and
  §9 lists it first for exactly this reason.
- **Compiled scheduler and model.** vLLM's `torch.compile` pass took 10.9 s at
  startup and fuses across the whole graph. nanoserve's scheduler is
  interpreted Python running between every step.
- **FlashAttention 2** vs one hand-written Triton kernel that §8 shows runs at
  ~6% of achievable bandwidth.
- **Prefix caching**, which nanoserve does not implement (§9 item 4). The
  workload shares filler text across prompts, so vLLM gets some of this for
  free — a genuine confound in vLLM's favour that a fairer run would disable.

The honest read: the architecture is right and the mechanisms are the same
ones vLLM uses; what is missing is the compilation and kernel engineering layer
underneath. That is a much better position than being slow for reasons nobody
can name.

## 4. Latency vs offered load (skewed, Poisson arrivals)

| rate (req/s) | output tok/s | p99 TTFT | p99 TPOT | p99 E2E |
|---|---|---|---|---|
| 2 | 84.5 | 2.51 s | 370 ms | 69.6 s |
| 4 | 92.5 | 2.41 s | 279 ms | 67.0 s |
| 8 | 119.5 | 2.93 s | 192 ms | 52.8 s |

On a bandwidth-bound system, more load means worse tail latency. Here **p99
TPOT improves with load** (370 → 192 ms) because a fuller batch amortises the
fixed per-step overhead — an overhead-bound system runs *more efficiently*
under pressure. The saturation knee is not reached by 8 req/s at n=32; the
full (non-quick) sweep extends to 32 req/s to find it.

## 5. Chunked prefill

Long-prompt workload (lognormal, mean 768 tokens), 8 req/s:

| config | outcome |
|---|---|
| chunked, budget 512 | works: p99 TTFT 42.3 s, p99 TPOT 890 ms under heavy prefill load |
| no chunking, budget 512 | **rejected: a 1,012-token prompt exceeds the whole per-step budget and can never be scheduled** |

The ablation produced a stronger result than a slowdown: without chunking,
long prompts are not slower — they are *unschedulable* at this budget. Chunking
is what makes a small token budget (which protects decode TPOT) compatible
with long prompts at all. The budget-2048 comparison quantifying the TTFT/TPOT
trade is in the full sweep.

## 6. Paging granularity

| block_size | output tok/s (skewed) |
|---|---|
| 8 | 102.2 |
| 16 | 102.5 |
| 32 | 101.6 |

No measurable effect — differences are inside run noise. Expected on this
hardware: the gather cost is launch-dominated, not layout-dominated, and at
n=32 capacity is nowhere near binding, so the fragmentation differences
(bounded at 7 vs 31 tokens/seq) never matter. 16 is kept as the default.

## 7. Quantisation: capacity up, throughput down — and why

| | weights | KV capacity (tok) | output tok/s | p99 TPOT |
|---|---|---|---|---|
| fp16 | 943 MiB | 412,320 | 102.5 | 184 ms |
| INT8 per-channel | 603 MiB | 439,792 | 79.6 (0.78×) | 209 ms |
| INT4 group-128 | 445 MiB | 458,048 | 49.2 (0.48×) | 299 ms |

The memory result is exactly as designed: INT4 frees ~500 MiB, worth ~46k
tokens of extra KV capacity.

The throughput result is negative, and the mechanism is worth stating
precisely. This implementation dequantises to fp16 before every matmul —
weight-only quantisation without a fused dequant-GEMM kernel *adds* kernels
and per-step latency. That is a pure cost unless the workload is
capacity-limited, and at n=32 the fp16 cache already holds every request, so
the extra capacity buys nothing. The freed memory would start paying at the
concurrency where fp16 runs out of blocks (~800 × 512-token sequences) and
INT4 admits more; this workload never gets there.

Honest summary: **quantisation here is a capacity feature, demonstrated; the
speed feature requires a fused dequant-GEMM kernel, which is future work.**
Claiming otherwise would not survive the first follow-up question.

## 8. Attention backend: fused Triton kernel vs torch gather

`python -m bench.bench_attention` (WSL2) → `results/attention_backend.json`,
`fig7_attention_kernel.png`

Measured on the same 4060, decode-only, Qwen2.5-0.5B shape (14 query heads /
2 KV heads / head_dim 64, block_size 16, fp16). Correctness checked against the
torch backend at every point before timing: **max absolute deviation ≤ 1.2e-4**
across all cases, on randomly permuted block tables.

**Speedup (torch ÷ triton):**

| batch | ctx 128 | 512 | 1024 | 2048 | 4096 |
|---|---|---|---|---|---|
| 1 | **12.48×** | 12.15× | 8.10× | 4.80× | 2.06× |
| 8 | 12.41× | 4.23× | 4.27× | 3.86× | 3.44× |
| 32 | 4.17× | 3.75× | 3.85× | 3.71× | **4.09×** |

**The shape of this table is the interesting part, and it contradicted my
prediction.** I expected speedup to *grow* with context, since the gather the
kernel eliminates is O(context). It shrinks — 12.5× → 2.1× at batch 1.

The raw timings explain it. At batch 1 the torch path costs 0.512, 0.460,
0.437, 0.525, 0.443 ms for contexts 128→4096: **flat across a 32× increase in
data.** It is not gather-bound there, it is launch-bound — ten small ops
(block-table arithmetic, two `index_select`s, permute, head expansion, SDPA)
whose cost is dominated by launch latency, not bytes. Triton meanwhile goes
0.041 → 0.215 ms, growing with context because it is actually doing the work.
The ratio collapses as real work catches up with fixed overhead.

At batch 32 the picture inverts: torch goes 0.689 → 19.556 ms (28× for 32×
context), so the gather now dominates, both implementations scale together, and
the ratio settles at the genuine traffic saving of ~4×.

So the kernel wins for two different reasons in two different regimes:

- **small batch/context → fewer launches** (1 kernel vs ~10 ops)
- **large batch/context → less memory traffic** (KV read once instead of
  written-then-read)

**Where the kernel still falls short.** At batch 32 / ctx 4096 it reads 67 MiB
of KV in 4.78 ms — about **14 GB/s**, against the ~220 GB/s this GPU sustains
on a plain device-to-device copy. So the kernel is at roughly 6% of achievable
bandwidth and is *not* bandwidth-bound; it is parallelism-bound. The launch
grid is one program per (sequence, query head) = 32 × 14 = 448 programs, each
serially walking 256 blocks. That is too few programs and too long a serial
chain to hide memory latency. The standard fix is flash-decoding: split the
context across programs, compute partial softmax statistics, and combine in a
second pass. That would be the next kernel-level change, and it is worth more
than another 4× on paper.

**Effect on the whole engine** (WSL2, `bench.diagnose`, torch backend replaced
by Triton):

| | transformers | nanoserve + Triton | |
|---|---|---|---|
| batch 1 | 60.2 tok/s | 58.3 tok/s | 0.97× |
| batch 8 | 377.6 tok/s | **465.7 tok/s** | **1.23×** |
| batch 32 | 1441.3 tok/s | **1682.3 tok/s** | **1.17×** |
| device ops / step | 1,014 | **682** | −33% |
| GPU utilisation @ b32 | 23.2% | **35.9%** | |

This is the engine's first outright win over eager transformers, and the
mechanism is visible in the op count: the fused kernel replaced ~10 ops per
layer with one, cutting the whole step from 1,014 to 682 device ops. Utilisation
at batch 32 rose from 23% to 36%, which is the same statement in different
units — less time spent issuing, more spent computing.

Note the shape: parity at batch 1, widening to 1.23× at batch 8. A serving
engine has nothing to offer a single request; its advantage is per-step fixed
cost amortised across a full batch, and that is exactly what the curve shows.

Still overhead-bound at 36% utilisation, which caps what any further kernel
work can buy. CUDA graphs (§9) attack the remaining 64%.

## 9. What I would do next, in order of expected value

1. **CUDA graphs for decode.** ~1,300 launches/step at ~8 µs on a machine
   that is 82% launch-idle. Decode shapes are static; capture once, replay.
   This is the biggest lever on this hardware by a wide margin — bigger than
   anything in §7 or §8, because it attacks the 82% rather than the 18%.
2. **Flash-decoding split in the Triton kernel.** §8 shows it runs at ~6% of
   achievable bandwidth because 448 programs each walk 256 blocks serially.
   Splitting context across programs with a second combine pass is the
   textbook fix.
3. **Fused dequant-GEMM** — turns §7's capacity-only result into a speed
   result.
4. **Prefix caching** — block tables already support it; needs content hashing
   and refcounts.

## 10. Reproducing

```bash
pip install -r requirements.txt
pytest                       # 92 tests, no GPU needed
python -m bench.diagnose     # roofline + step anatomy
python -m bench.sweep        # full sweep (--quick for ~15 min)
python -m bench.plot
```

Raw JSON for every number above is in `results/`.
