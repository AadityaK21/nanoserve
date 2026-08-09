"""Turns a scheduler decision into tensors, runs the model, returns tokens.

This is the only place that knows both about Sequence objects and about
tensors. Keeping the boundary here means the scheduler stays pure Python (and
unit-testable without a GPU) and the model stays ignorant of scheduling.
"""

from __future__ import annotations

import torch

from .attention import AttentionMetadata, get_attention_backend
from .sampler import Sampler


class ModelRunner:
    def __init__(
        self,
        model,
        kv_cache,
        block_manager,
        device: str,
        backend: str = "auto",
        seed: int = 0,
        block_size: int = 16,
        dtype=None,
    ) -> None:
        self.model = model
        # kv_cache is None during memory profiling: the runner has to be able to
        # run a forward pass before the real cache exists, because the size of
        # that cache is what the profiling run is measuring.
        self.kv_cache = kv_cache
        self.block_manager = block_manager
        self.device = device
        self.attn_fn, self.backend_name = get_attention_backend(backend)
        self.sampler = Sampler(device, seed)
        self.block_size = kv_cache.block_size if kv_cache is not None else block_size
        self.dtype = dtype or (kv_cache.dtype if kv_cache is not None else torch.float16)

    # ---- metadata construction -----------------------------------------
    def build_inputs(self, scheduled):
        """scheduled: list of (Sequence, num_tokens_to_compute).

        Returns (input_ids, positions, metadata, sample_indices, sample_seqs).

        sample_indices are positions in the flat token array whose logits the
        sampler needs. A sequence mid-prefill contributes no sample index: its
        chunk produces KV, not a token. Only the step that consumes a
        sequence's final uncomputed token yields a new token for it.
        """
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        query_start_loc = [0]
        seq_lens: list[int] = []
        query_lens: list[int] = []
        sample_indices: list[int] = []
        sample_seqs = []

        for seq, q in scheduled:
            start = seq.num_computed_tokens
            end = start + q
            input_ids.extend(seq.slice_token_ids(start, end))
            positions.extend(range(start, end))
            slot_mapping.extend(self.block_manager.slot_mapping(seq, start, end))

            query_start_loc.append(query_start_loc[-1] + q)
            seq_lens.append(end)
            query_lens.append(q)

            if end == seq.num_tokens:
                sample_indices.append(query_start_loc[-1] - 1)
                sample_seqs.append(seq)

        max_seq_len = max(seq_lens)
        max_blocks = (max_seq_len + self.block_size - 1) // self.block_size
        bt = torch.zeros((len(scheduled), max_blocks), dtype=torch.int32)
        for i, (seq, _) in enumerate(scheduled):
            n = min(len(seq.block_table), max_blocks)
            bt[i, :n] = torch.tensor(seq.block_table[:n], dtype=torch.int32)

        dev = self.device
        md = AttentionMetadata(
            slot_mapping=torch.tensor(slot_mapping, dtype=torch.long, device=dev),
            query_start_loc=torch.tensor(query_start_loc, dtype=torch.int32, device=dev),
            seq_lens=torch.tensor(seq_lens, dtype=torch.int32, device=dev),
            block_tables=bt.to(dev),
            max_query_len=max(query_lens),
            max_seq_len=max_seq_len,
            num_seqs=len(scheduled),
            num_tokens=len(input_ids),
            is_decode_only=all(q == 1 for q in query_lens),
            query_lens_cpu=query_lens,
            seq_lens_cpu=seq_lens,
            query_start_loc_cpu=query_start_loc,
        )

        return (
            torch.tensor(input_ids, dtype=torch.long, device=dev),
            torch.tensor(positions, dtype=torch.long, device=dev),
            md,
            torch.tensor(sample_indices, dtype=torch.long, device=dev) if sample_indices else None,
            sample_seqs,
        )

    # ---- execution -------------------------------------------------------
    @torch.inference_mode()
    def execute(self, scheduled):
        """Run one engine step. Returns list of (Sequence, token_id)."""
        input_ids, positions, md, sample_idx, sample_seqs = self.build_inputs(scheduled)

        hidden = self.model(input_ids, positions, self.kv_cache, md, self.attn_fn)

        # Advance the computed watermark only after the forward succeeded, so a
        # failed step leaves the sequences replayable rather than half-committed.
        for seq, q in scheduled:
            seq.advance_computed(q)

        if sample_idx is None:
            return []

        logits = self.model.compute_logits(hidden, sample_idx)
        token_ids = self.sampler.sample(logits, [s.sampling for s in sample_seqs])
        return list(zip(sample_seqs, token_ids))

    @torch.inference_mode()
    def profile_run(self, num_tokens: int, num_seqs: int = 1):
        """Force peak-activation allocation without touching the real cache.

        Called before the KV cache exists, to find out how much memory the
        largest possible step needs. Uses a throwaway one-block cache so the
        measurement reflects activations only.
        """
        from .kv_cache import PagedKVCache

        dummy = PagedKVCache(
            num_blocks=max(1, (num_tokens + self.block_size - 1) // self.block_size),
            block_size=self.block_size,
            num_layers=self.model.num_layers,
            num_kv_heads=self.model.num_kv_heads,
            head_dim=self.model.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        n_blocks = dummy.num_blocks
        md = AttentionMetadata(
            slot_mapping=torch.arange(num_tokens, dtype=torch.long, device=self.device),
            query_start_loc=torch.tensor([0, num_tokens], dtype=torch.int32, device=self.device),
            seq_lens=torch.tensor([num_tokens], dtype=torch.int32, device=self.device),
            block_tables=torch.arange(n_blocks, dtype=torch.int32, device=self.device).view(1, -1),
            max_query_len=num_tokens,
            max_seq_len=num_tokens,
            num_seqs=1,
            num_tokens=num_tokens,
            is_decode_only=False,
            query_lens_cpu=[num_tokens],
            seq_lens_cpu=[num_tokens],
            query_start_loc_cpu=[0, num_tokens],
        )
        ids = torch.zeros(num_tokens, dtype=torch.long, device=self.device)
        pos = torch.arange(num_tokens, dtype=torch.long, device=self.device)
        hidden = self.model(ids, pos, dummy, md, self.attn_fn)
        self.model.compute_logits(hidden, torch.tensor([num_tokens - 1], device=self.device))
        del dummy, hidden
