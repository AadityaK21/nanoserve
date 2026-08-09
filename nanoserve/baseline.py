"""Phase 0 baselines.

Two reference points, both using our own decode loop rather than
`model.generate`. We need the loop anyway for Phase 1, and it gives exact TTFT
instead of guessing where prefill ended.

  sequential  -- one request at a time. Latency floor, throughput floor.
  static      -- fixed batch, run to the longest sequence in the batch, then
                 start the next batch. Throughput ceiling for naive batching,
                 and the thing continuous batching has to beat.
"""

from __future__ import annotations

import time

import torch

from .metrics import RequestMetrics
from .model import build_position_ids
from .workload import Request


def _sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


@torch.inference_mode()
def run_static_batch(
    model,
    tok,
    reqs: list[Request],
    device: str = "cuda",
    t_zero: float | None = None,
) -> list[RequestMetrics]:
    """Prefill the whole batch, then decode in lockstep.

    The batch runs until the *longest* request finishes. Every sequence that
    finished earlier keeps occupying a row of the batch and burning FLOPs on
    tokens nobody asked for. That waste is the whole reason continuous batching
    exists, so we measure it here rather than hiding it.
    """
    if t_zero is None:
        t_zero = time.perf_counter()

    prompts = [r.prompt for r in reqs]
    enc = tok(prompts, return_tensors="pt", padding=True)
    input_ids = enc.input_ids.to(device)
    attn = enc.attention_mask.to(device)

    real_lens = attn.sum(-1).tolist()
    recs = [
        RequestMetrics(
            request_id=r.request_id,
            prompt_tokens=int(real_lens[i]),
            arrival=t_zero,
            start=time.perf_counter(),
        )
        for i, r in enumerate(reqs)
    ]

    budgets = torch.tensor([r.max_new_tokens for r in reqs], device=device)
    max_steps = int(budgets.max().item())

    # ---- prefill ----
    _sync(device)
    out = model(
        input_ids=input_ids,
        attention_mask=attn,
        position_ids=build_position_ids(attn),
        use_cache=True,
    )
    past = out.past_key_values
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    _sync(device)
    t_first = time.perf_counter()

    produced = torch.ones(len(reqs), device=device, dtype=torch.long)
    for i, rec in enumerate(recs):
        rec.first_token = t_first
        rec.output_tokens = 1
        # A sequence asking for exactly 1 token is already done.
        if budgets[i].item() <= 1:
            rec.end = t_first

    # ---- decode ----
    for _ in range(max_steps - 1):
        attn = torch.cat([attn, torch.ones_like(next_tok)], dim=-1)
        pos = attn.sum(-1, keepdim=True) - 1
        out = model(
            input_ids=next_tok,
            attention_mask=attn,
            position_ids=pos,
            past_key_values=past,
            use_cache=True,
        )
        past = out.past_key_values
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)

        _sync(device)
        now = time.perf_counter()

        still_running = produced < budgets
        produced = produced + still_running.long()
        for i, rec in enumerate(recs):
            if still_running[i]:
                rec.output_tokens += 1
                if rec.output_tokens >= reqs[i].max_new_tokens:
                    rec.end = now

    now = time.perf_counter()
    for rec in recs:
        if rec.end == 0.0:
            rec.end = now
    return recs


@torch.inference_mode()
def run_sequential(
    model, tok, reqs: list[Request], device: str = "cuda"
) -> list[RequestMetrics]:
    """One request at a time. Batch size 1, no queueing."""
    t_zero = time.perf_counter()
    recs = []
    for r in reqs:
        recs.extend(run_static_batch(model, tok, [r], device=device, t_zero=t_zero))
    return recs


@torch.inference_mode()
def run_static_batched(
    model, tok, reqs: list[Request], batch_size: int, device: str = "cuda"
) -> list[RequestMetrics]:
    """Chunk the request list into fixed batches and run them back to back.

    Note the queueing artefact: a request in the last batch waits for every
    earlier batch to complete, so its e2e latency includes all of that queue
    time. That is real and it is exactly what shows up in p99.
    """
    t_zero = time.perf_counter()
    recs = []
    for i in range(0, len(reqs), batch_size):
        chunk = reqs[i : i + batch_size]
        recs.extend(run_static_batch(model, tok, chunk, device=device, t_zero=t_zero))
    return recs
