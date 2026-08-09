"""Batched sampling.

Every sequence in a continuous batch can carry different sampling parameters,
so the sampler is vectorised over per-request temperature / top-k / top-p
rather than looping. A Python loop over 64 sequences here would add 64 tiny
CUDA launches per step, which on a decode step that takes ~15 ms is real.

Greedy is special-cased because the benchmarks run greedy: argmax skips the
sort that top-p needs, and the sort over a 151936-wide vocabulary is not free.
"""

from __future__ import annotations

import torch


class Sampler:
    def __init__(self, device: str, seed: int = 0) -> None:
        self.device = device
        self.generator = torch.Generator(device=device)
        self.generator.manual_seed(seed)

    @torch.inference_mode()
    def sample(self, logits: torch.Tensor, params_list) -> list[int]:
        """logits: [num_seqs, vocab]. Returns one token id per sequence."""
        if all(p.greedy for p in params_list):
            return logits.argmax(dim=-1).tolist()

        logits = logits.float()

        temps = torch.tensor(
            [p.temperature if not p.greedy else 1.0 for p in params_list],
            device=logits.device,
        ).unsqueeze(1)
        logits = logits / temps.clamp(min=1e-5)

        top_ks = [p.top_k for p in params_list]
        if any(k > 0 for k in top_ks):
            logits = self._mask_top_k(logits, top_ks)

        top_ps = [p.top_p for p in params_list]
        if any(p < 1.0 for p in top_ps):
            logits = self._mask_top_p(logits, top_ps)

        probs = torch.softmax(logits, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1, generator=self.generator).squeeze(1)

        # Requests that asked for greedy still get argmax, even in a mixed batch.
        greedy_rows = [i for i, p in enumerate(params_list) if p.greedy]
        if greedy_rows:
            idx = torch.tensor(greedy_rows, device=logits.device)
            sampled[idx] = logits.index_select(0, idx).argmax(dim=-1)
        return sampled.tolist()

    @staticmethod
    def _mask_top_k(logits: torch.Tensor, top_ks: list[int]) -> torch.Tensor:
        vocab = logits.shape[-1]
        k_t = torch.tensor(
            [k if k > 0 else vocab for k in top_ks], device=logits.device
        ).clamp(max=vocab)
        max_k = int(k_t.max().item())
        vals, _ = logits.topk(max_k, dim=-1)
        # kth largest value per row, using each row's own k
        kth = vals.gather(1, (k_t - 1).unsqueeze(1))
        return logits.masked_fill(logits < kth, float("-inf"))

    @staticmethod
    def _mask_top_p(logits: torch.Tensor, top_ps: list[int]) -> torch.Tensor:
        p_t = torch.tensor(top_ps, device=logits.device, dtype=logits.dtype).unsqueeze(1)
        sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
        cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        # Drop the tail beyond p, but always keep the top-1 token so a row can
        # never end up with every logit masked to -inf.
        drop = cum - sorted_logits.softmax(dim=-1) > p_t
        drop[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(drop, float("-inf"))
        return sorted_logits.scatter(1, sorted_idx, sorted_logits)
