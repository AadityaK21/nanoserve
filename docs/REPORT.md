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

| | measured |
|---|---|
| Achieved memory bandwidth | 209–222 GB/s |
| fp16 GEMM | ~30 TFLOP/s |
| **Bandwidth floor per decode step** | **~4.6 ms** (943 MiB / bandwidth) |
| Actual batch-1 step | 23.7 ms (transformers) / 28.2 ms (nanoserve) |
| GPU-busy time within that step | 6.6 / 7.2 ms |
| **GPU utilisation during decode** | **~18%** |

The GPU is idle ~82% of every decode step, waiting on the CPU to issue ~1,300
kernels at ~8 µs each. Windows' WDDM driver model routes every launch through
the OS scheduler, which inflates that latency relative to Linux.

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

## 8. Attention backend

Both rows ran the torch backend: Triton is unavailable on native Windows. The
torch backend's known cost — re-gathering the full KV context every decode
step — is the residual 0.6 ms/step of extra GPU work vs transformers at batch
1 (§2). The Triton kernel (written, correctness-tested against the torch
backend on random block tables) removes that gather by walking the block table
with an online softmax. Benchmarking it is a WSL2 task.

## 9. What I would do next, in order of expected value

1. **CUDA graphs for decode.** ~1,300 launches/step at ~8 µs on a machine
   that is 82% launch-idle. Decode shapes are static; capture once, replay.
   This is the biggest lever on this hardware by a wide margin.
2. **WSL2 pass.** Quantifies the WDDM tax and unlocks the Triton benchmark.
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
