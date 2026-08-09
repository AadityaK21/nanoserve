"""Synthetic request workloads.

Two knobs decide whether continuous batching looks impressive or pointless:

  * output length variance -- with uniform lengths, static batching loses almost
    nothing, because every sequence in the batch finishes at the same step. Real
    traffic is heavily skewed, and that is where a static batch wastes most of
    its compute waiting on one long straggler.
  * arrival rate -- offline mode (all requests present at t=0) measures peak
    throughput. Poisson arrivals measure what latency users actually see.

Phase 0 uses the uniform/offline case so the baseline is honest. Phase 3
switches these on to show the win.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Request:
    request_id: int
    prompt: str
    max_new_tokens: int
    arrival_offset: float = 0.0   # seconds after benchmark start


_FILLER = (
    "The system processes incoming requests and returns generated tokens. "
    "Throughput and latency are both measured. "
)


def make_prompt(target_tokens: int, tokenizer) -> str:
    """Build a prompt of roughly target_tokens by repeating filler text."""
    text = _FILLER
    while len(tokenizer(text).input_ids) < target_tokens:
        text += _FILLER
    ids = tokenizer(text).input_ids[:target_tokens]
    return tokenizer.decode(ids)


def uniform_workload(
    n: int,
    tokenizer,
    prompt_len: int = 256,
    output_len: int = 128,
) -> list[Request]:
    """All requests identical, all present at t=0. The optimistic case."""
    prompt = make_prompt(prompt_len, tokenizer)
    return [Request(i, prompt, output_len) for i in range(n)]


def skewed_workload(
    n: int,
    tokenizer,
    prompt_mean: int = 256,
    prompt_sigma: float = 0.6,
    output_mean: int = 128,
    output_sigma: float = 0.8,
    request_rate: float | None = None,
    seed: int = 0,
) -> list[Request]:
    """Lognormal lengths, optional Poisson arrivals. The realistic case."""
    rng = np.random.default_rng(seed)

    p_lens = rng.lognormal(np.log(prompt_mean), prompt_sigma, n).astype(int)
    o_lens = rng.lognormal(np.log(output_mean), output_sigma, n).astype(int)
    p_lens = np.clip(p_lens, 16, 2048)
    o_lens = np.clip(o_lens, 8, 1024)

    if request_rate is None:
        offsets = np.zeros(n)
    else:
        # Poisson process: exponential gaps with mean 1/rate.
        gaps = rng.exponential(1.0 / request_rate, n)
        offsets = np.cumsum(gaps)

    # Cache prompts by length so we tokenize each distinct length once.
    cache: dict[int, str] = {}
    reqs = []
    for i in range(n):
        pl = int(p_lens[i])
        if pl not in cache:
            cache[pl] = make_prompt(pl, tokenizer)
        reqs.append(Request(i, cache[pl], int(o_lens[i]), float(offsets[i])))
    return reqs
