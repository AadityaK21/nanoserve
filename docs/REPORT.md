# nanoserve — results

> Fill this in from `results/sweep.json` and the figures in `results/`.
> Every `TODO` is a number the benchmark produces. Keep the framing; replace
> the numbers. Do not round in your favour — the point of the report is that
> you can defend it.

**Hardware:** RTX 4060 8 GB (Ada, TODO GB/s memory bandwidth) · TODO driver ·
PyTorch TODO · CUDA TODO
**Model:** Qwen2.5-0.5B-Instruct, fp16 · 24 layers · 14 query heads · 2 KV heads
· head_dim 64
**KV cost:** `2 × 24 × 2 × 64 × 2 B` = **12 KiB per token** (TODO confirm from
`engine.describe()`)

---

## 1. Where the memory goes

| | MiB |
|---|---|
| Weights (fp16) | TODO |
| Activations, worst-case step | TODO |
| KV cache after profiling | TODO |
| **KV capacity** | **TODO tokens** (TODO blocks × 16) |

At 12 KiB/token, 8 GB of VRAM is roughly TODO tokens of KV before you account
for weights. That number, not FLOPs, is what caps concurrency on this GPU.

## 2. Static batching: where it stops scaling

`results/fig1_throughput.png`

| Batch size | Output tok/s (uniform) | Output tok/s (skewed) | p99 E2E (ms) |
|---|---|---|---|
| 1 | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO |
| 64 | TODO | TODO | TODO |

Throughput flattens around batch size **TODO**. Past that point decode is
memory-bandwidth bound: each step reads the full weight matrix regardless of
batch size, so adding rows costs nothing until the KV traffic itself saturates
the bus.

Note the gap between the uniform and skewed columns. With uniform lengths every
sequence in a static batch finishes at the same step and almost nothing is
wasted. With lognormal lengths — which is what real traffic looks like — the
batch runs until its longest member finishes, and the average row spends TODO%
of its life computing tokens nobody asked for.

## 3. Continuous batching

`results/fig1_throughput.png`

| | Static (best batch size) | Continuous | Δ |
|---|---|---|---|
| Output tok/s, skewed | TODO | TODO | **TODO×** |
| Output tok/s, uniform | TODO | TODO | TODO× |
| p99 TTFT (ms) | TODO | TODO | TODO |
| p99 E2E (ms) | TODO | TODO | TODO |
| Mean batch size | (fixed) | TODO | |

The win is concentrated in the skewed workload, and that is the honest framing:
continuous batching does not make the GPU faster, it stops the batch from going
stale. Under uniform lengths there is almost nothing to recover, and the
numbers should show that.

## 4. Tail latency vs offered load

`results/fig2_latency_vs_load.png`

| Rate (req/s) | p50 TTFT | p99 TTFT | p50 TPOT | p99 TPOT | p99 E2E | Preemptions |
|---|---|---|---|---|---|---|
| 1 | TODO | TODO | TODO | TODO | TODO | TODO |
| 4 | TODO | TODO | TODO | TODO | TODO | TODO |
| 8 | TODO | TODO | TODO | TODO | TODO | TODO |
| 16 | TODO | TODO | TODO | TODO | TODO | TODO |
| 32 | TODO | TODO | TODO | TODO | TODO | TODO |

The knee sits at roughly **TODO req/s**. Below it, TTFT is prefill time. Above
it, the queue never drains and TTFT becomes queueing delay, which is why it
grows without bound while TPOT stays roughly flat — the GPU is still decoding
at the same rate, there is just a longer line for it.

## 5. Chunked prefill

`results/fig3_chunked_prefill.png`

| Config | p99 TTFT | p99 TPOT | Output tok/s |
|---|---|---|---|
| No chunking, budget 2048 | TODO | TODO | TODO |
| Chunked, budget 2048 | TODO | TODO | TODO |
| Chunked, budget 512 | TODO | TODO | TODO |

The trade is explicit: a smaller chunk budget raises TTFT (a long prompt takes
more steps to finish prefilling) and lowers p99 TPOT (decodes stop being
blocked behind whole prefills). At budget 512 the p99 TPOT improves by TODO%
for TODO% worse median TTFT. Which side of that trade you want depends on
whether the product is a chat UI (protect TPOT) or a batch job (protect
throughput).

## 6. Paging granularity

`results/fig4_block_size.png`

| block_size | Blocks | KV capacity (tok) | Output tok/s | Waste bound |
|---|---|---|---|---|
| 8 | TODO | TODO | TODO | 7 tok/seq |
| 16 | TODO | TODO | TODO | 15 tok/seq |
| 32 | TODO | TODO | TODO | 31 tok/seq |
| 64 | TODO | TODO | TODO | 63 tok/seq |

Small blocks waste less memory per sequence but make block tables longer and
the kernel's inner loop shorter. TODO was best here. For comparison, a naive
non-paged server reserving `max_model_len = 4096` per sequence fits only
TODO concurrent sequences in the same VRAM — that is the number paging exists
to fix.

## 7. Quantisation

`results/fig5_quantization.png`

| | Weights (MiB) | KV blocks | KV capacity (tok) | Output tok/s | Mean batch |
|---|---|---|---|---|---|
| fp16 | TODO | TODO | TODO | TODO | TODO |
| INT8 per-channel | TODO | TODO | TODO | TODO | TODO |
| INT4 group-128 | TODO | TODO | TODO | TODO | TODO |

INT4 frees **TODO MiB**, which at 12 KiB/token is **TODO more tokens** of KV
capacity and raises the achievable batch size from TODO to TODO.

Be precise about the mechanism: weights are dequantised to fp16 before the
matmul, so this does not reduce FLOPs. The throughput change comes entirely
from the extra concurrency the freed memory buys. On a workload that was not
memory-limited, INT4 would be slightly *slower* because of the dequant
overhead — check whether that shows up at low batch sizes.

Quality check: TODO (cosine similarity of logits vs fp16, or perplexity on a
fixed passage).

## 8. Attention backend

`results/fig6_backend.png`

| Backend | Output tok/s | p50 TPOT (ms) | p99 TPOT (ms) |
|---|---|---|---|
| torch (gather + SDPA) | TODO | TODO | TODO |
| Triton (fused) | TODO | TODO | TODO |

The torch backend materialises each sequence's whole context into a fresh
tensor every decode step, so it reads the KV cache twice per step and writes it
once. The Triton kernel walks the block table and accumulates an online softmax
in registers, so context KV is read exactly once. Measured speedup on decode:
**TODO×**, growing with context length because the gather cost is O(context)
while the kernel's advantage is a constant factor of the same term.

Correctness: `pytest tests/test_paged_attention.py -k triton` — max abs
deviation from the torch backend TODO.

## 9. What I would do next

- **Fused dequant-GEMM.** The obvious missing piece. Weight-only INT4 currently
  buys memory but not compute; a Triton kernel that dequantises inside the GEMM
  tiles would buy both.
- **Prefix caching.** Shared system prompts are recomputed per request today.
  The block indirection already exists; this is a content hash over blocks plus
  a refcount.
- **CUDA graphs for decode.** At batch 64 the per-step Python and launch
  overhead is TODO ms of a TODO ms step. Decode shapes are static, which is
  exactly the case graphs are for.
- **Speculative decoding.** With a 0.5B target the draft model would have to be
  tiny, so this matters more at 7B+.

## 10. Reproducing

```bash
pip install -r requirements.txt
pytest -q
python -m bench.run_baseline --mode static --batch-sizes 1,2,4,8,16,32 --n 32
python -m bench.sweep
python -m bench.plot
```

Raw JSON for every number above is in `results/`.
