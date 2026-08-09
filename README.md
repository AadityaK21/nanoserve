# nanoserve

An LLM inference server built from scratch: paged KV cache, continuous
batching, a fused Triton attention kernel, weight-only INT4/INT8 quantisation,
and a benchmark harness that reports throughput and tail latency.

No vLLM, no TGI, no TensorRT-LLM. `transformers` is used for exactly two
things — the tokenizer and the raw weight tensors. The model forward pass,
the scheduler, the cache manager, the attention kernels and the quantiser are
all in this repo.

## Why this exists

A single request through a 0.5B model is trivially fast. Serving is where the
systems work is: 200 requests arrive with wildly different prompt and output
lengths, VRAM is finite, and you are judged on p99 rather than the mean.

Three things dominate, and each has a phase:

1. **Memory is the constraint, not compute.** A naive server reserves
   `max_model_len` of contiguous KV per request and wastes most of it. Paging
   bounds the waste to `block_size - 1` tokens per sequence.
2. **The batch must never go stale.** Static batching runs until the longest
   sequence finishes, so a batch of 32 spends most of its life as a batch of 1.
   Continuous batching re-decides every iteration.
3. **Smaller weights buy concurrency.** Weight-only INT4 hands ~700 MiB back to
   the KV cache, which is tens of thousands of extra tokens of context.

## Architecture

```
   add_request
        |
        v
   Scheduler ......... iteration-level admission, token budget,
        |              chunked prefill, preemption-by-recompute
        v
   ModelRunner ....... flattens the batch (no padding), builds slot mappings
        |
        v
   Qwen2ForCausalLM .. our forward pass
        |
        v
   PagedAttention .... torch gather backend  |  fused Triton kernel
        |
        v
   PagedKVCache ...... [num_blocks, block_size, kv_heads, head_dim] per layer
        ^
        |
   BlockSpaceManager . free list + per-sequence block tables
```

| File | Responsibility |
|---|---|
| `nanoserve/block_manager.py` | Block allocator, block tables, slot mapping. Pure Python. |
| `nanoserve/kv_cache.py` | Physical cache tensors, memory profiler that sizes them. |
| `nanoserve/attention.py` | Unified prefill/decode/chunked paged attention, torch backend. |
| `nanoserve/triton_attention.py` | Fused decode kernel: online softmax over the block table. |
| `nanoserve/qwen2.py` | Qwen2 forward pass against the paged cache, flat batching. |
| `nanoserve/scheduler.py` | Continuous batching. Pure Python. |
| `nanoserve/model_runner.py` | Sequences → tensors → sampled tokens. |
| `nanoserve/quant.py` | INT8 per-channel and group-wise INT4 weight-only quantisation. |
| `nanoserve/engine.py` | Wires it together, exposes `add_request` / `step`. |

`block_manager.py` and `scheduler.py` deliberately import no torch. The bugs
that hurt most in a serving engine are bookkeeping bugs, and those are far
cheaper to catch in a CPU unit test than in a 40-minute benchmark.

## Phases

| Phase | What gets built | What it proves |
|---|---|---|
| 0 | Baseline harness: sequential + static batching, metrics, workload generator | Measure before optimising |
| 1 | Own the model forward pass and decode loop; sequence state machine; batched sampler | You own the generation loop |
| 2 | Paged KV cache: allocator, block tables, paged attention, then a Triton kernel | Memory is the real constraint |
| 3 | Continuous batching: token budget, chunked prefill, preemption | The headline feature |
| 4 | Weight-only INT8 and group-wise INT4 quantisation | Smaller weights become more KV cache |
| 5 | Full sweep, plots, writeup | Numbers you can defend |

## Quick start

```bash
pip install -r requirements.txt

# the tests need no GPU and no model download
pytest -q

# Phase 0 reference numbers
python -m bench.run_baseline --mode static --batch-sizes 1,2,4,8,16,32 --n 32

# the engine
python -m bench.run_engine --n 64 --workload skewed
python -m bench.run_engine --n 128 --workload skewed --request-rate 8
python -m bench.run_engine --n 64 --quantization int4

# everything, then the figures
python -m bench.sweep
python -m bench.plot
```

Results land in `results/` as JSON and PNG. Write conclusions into
`docs/REPORT.md`.

## Metrics

- **TTFT** — time to first token, dominated by prefill and queueing.
- **TPOT** — time per output token after the first, dominated by decode.
- **E2E** — arrival to last token. p50 / p90 / p99.
- **Output throughput** — generated tokens/sec across the whole system.

Mean latency hides everything interesting. Report percentiles.

Benchmarks run greedy with `ignore_eos=True` so every request emits exactly
`max_new_tokens`. Otherwise output length varies run to run and you end up
measuring the model's verbosity instead of the server.

## What this does not do

Stated plainly, because the gaps are more interesting than the features:

- **No fused dequant-GEMM.** Quantised weights are expanded to fp16 before the
  matmul, so INT4 cuts resident memory but not FLOPs. The throughput win comes
  from the extra KV cache, not from faster arithmetic.
- **No prefix caching.** Shared system prompts are recomputed per request. The
  block table indirection is already in place, so this is mostly a hashing
  layer over block contents.
- **No CPU/disk swap.** Preemption recomputes rather than swapping out.
- **No tensor or pipeline parallelism.** Single GPU.
- **Qwen2 architecture only.** Llama-family support is a small edit to
  `qwen2.py` (drop the q/k/v biases).

## Environment

Native Windows works for everything except the Triton kernel, which needs
WSL2 + CUDA. The engine detects Triton and falls back to the torch backend
automatically, so nothing breaks without it. See `SETUP.md`.
