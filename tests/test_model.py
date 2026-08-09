"""End-to-end model correctness on a tiny randomly-initialised Qwen2.

Three properties, in increasing order of how much they would hurt to get wrong:

  1. Incremental decode == one-shot prefill. If the KV cache, the slot mapping
     or the position ids are wrong, these diverge. This is the test that catches
     almost everything.
  2. Batching invariance. A request's output must not depend on who it happens
     to share a step with. Violating this means cross-sequence contamination,
     which is the single worst bug class in a serving engine because it is
     invisible in single-request testing.
  3. Parity with transformers' own Qwen2 implementation, on identical weights.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanoserve.block_manager import BlockSpaceManager  # noqa: E402
from nanoserve.config import SamplingParams  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.sequence import Sequence  # noqa: E402


def harness(model, cfg, make_cache, block_size=8, num_blocks=128):
    cache = make_cache(cfg, num_blocks=num_blocks, block_size=block_size)
    bm = BlockSpaceManager(block_size, num_blocks, watermark=0.0)
    runner = ModelRunner(model, cache, bm, "cpu", backend="torch")
    return runner, bm


def mkseq(seq_id, tokens, max_new=4):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(tokens),
        sampling=SamplingParams(max_new_tokens=max_new, ignore_eos=True),
    )


def logits_after(runner, bm, seq, plan):
    """Feed `seq` through the runner in the given chunk sizes; return logits."""
    hidden = None
    for q in plan:
        assert bm.allocate(seq, seq.num_computed_tokens + q)
        ids, pos, md, sample_idx, _ = runner.build_inputs([(seq, q)])
        hidden = runner.model(ids, pos, runner.kv_cache, md, runner.attn_fn)
        seq.advance_computed(q)
    return runner.model.compute_logits(hidden, torch.tensor([hidden.shape[0] - 1]))


# ---- 1. incremental == one-shot ------------------------------------------
@pytest.mark.parametrize("block_size", [4, 8, 16])
def test_token_by_token_matches_one_shot_prefill(tiny_model, tiny_config, make_cache, block_size):
    tokens = torch.randint(0, tiny_config.vocab_size, (23,)).tolist()

    r1, bm1 = harness(tiny_model, tiny_config, make_cache, block_size)
    whole = logits_after(r1, bm1, mkseq(0, tokens), [len(tokens)])

    r2, bm2 = harness(tiny_model, tiny_config, make_cache, block_size)
    one_at_a_time = logits_after(r2, bm2, mkseq(0, tokens), [1] * len(tokens))

    torch.testing.assert_close(one_at_a_time, whole, atol=2e-4, rtol=2e-4)


def test_chunked_prefill_matches_one_shot(tiny_model, tiny_config, make_cache):
    tokens = torch.randint(0, tiny_config.vocab_size, (30,)).tolist()

    r1, bm1 = harness(tiny_model, tiny_config, make_cache)
    whole = logits_after(r1, bm1, mkseq(0, tokens), [30])

    r2, bm2 = harness(tiny_model, tiny_config, make_cache)
    chunked = logits_after(r2, bm2, mkseq(0, tokens), [7, 11, 12])

    torch.testing.assert_close(chunked, whole, atol=2e-4, rtol=2e-4)


# ---- 2. batching invariance ------------------------------------------------
def test_output_is_independent_of_batchmates(tiny_model, tiny_config, make_cache):
    a = torch.randint(0, tiny_config.vocab_size, (12,)).tolist()
    b = torch.randint(0, tiny_config.vocab_size, (25,)).tolist()

    r_alone, bm_alone = harness(tiny_model, tiny_config, make_cache)
    alone = logits_after(r_alone, bm_alone, mkseq(0, a), [len(a)])

    r, bm = harness(tiny_model, tiny_config, make_cache)
    sa, sb = mkseq(0, a), mkseq(1, b)
    for s, q in ((sa, len(a)), (sb, len(b))):
        assert bm.allocate(s, q)
    ids, pos, md, sample_idx, sample_seqs = r.build_inputs([(sa, len(a)), (sb, len(b))])
    hidden = r.model(ids, pos, r.kv_cache, md, r.attn_fn)
    logits = r.model.compute_logits(hidden, sample_idx)

    together = logits[sample_seqs.index(sa)].unsqueeze(0)
    torch.testing.assert_close(together, alone, atol=2e-4, rtol=2e-4)


def test_decode_is_independent_of_batchmates(tiny_model, tiny_config, make_cache):
    """The mixed-batch version: one sequence decoding beside another prefilling."""
    a = torch.randint(0, tiny_config.vocab_size, (9,)).tolist()
    b = torch.randint(0, tiny_config.vocab_size, (17,)).tolist()

    def decode_step_of_a(with_neighbour):
        r, bm = harness(tiny_model, tiny_config, make_cache)
        sa = mkseq(0, a)
        assert bm.allocate(sa, len(a))
        ids, pos, md, si, _ = r.build_inputs([(sa, len(a))])
        h = r.model(ids, pos, r.kv_cache, md, r.attn_fn)
        sa.advance_computed(len(a))
        sa.append_token(int(r.model.compute_logits(h, si).argmax(-1).item()))

        plan = [(sa, 1)]
        if with_neighbour:
            sb = mkseq(1, b)
            assert bm.allocate(sb, len(b))
            plan.append((sb, len(b)))
        for s, q in plan:
            assert bm.allocate(s, s.num_computed_tokens + q)
        ids, pos, md, si, seqs = r.build_inputs(plan)
        h = r.model(ids, pos, r.kv_cache, md, r.attn_fn)
        return r.model.compute_logits(h, si)[seqs.index(sa)]

    torch.testing.assert_close(
        decode_step_of_a(True), decode_step_of_a(False), atol=2e-4, rtol=2e-4
    )


def test_block_size_does_not_change_results(tiny_model, tiny_config, make_cache):
    tokens = torch.randint(0, tiny_config.vocab_size, (37,)).tolist()
    outs = []
    for bsz in (4, 8, 16, 32):
        r, bm = harness(tiny_model, tiny_config, make_cache, block_size=bsz)
        outs.append(logits_after(r, bm, mkseq(0, tokens), [10, 1, 1, 25]))
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol=2e-4, rtol=2e-4)


# ---- 3. parity with transformers -------------------------------------------
def test_matches_transformers_qwen2(tiny_config, make_cache):
    """Same weights, same input, same logits as the reference implementation."""
    pytest.importorskip("transformers")
    from transformers import Qwen2Config, Qwen2ForCausalLM as HFQwen2

    from nanoserve.qwen2 import Qwen2ForCausalLM

    hf_cfg = Qwen2Config(
        vocab_size=tiny_config.vocab_size,
        hidden_size=tiny_config.hidden_size,
        num_hidden_layers=tiny_config.num_hidden_layers,
        num_attention_heads=tiny_config.num_attention_heads,
        num_key_value_heads=tiny_config.num_key_value_heads,
        intermediate_size=tiny_config.intermediate_size,
        rms_norm_eps=tiny_config.rms_norm_eps,
        rope_theta=tiny_config.rope_theta,
        max_position_embeddings=tiny_config.max_position_embeddings,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    hf = HFQwen2(hf_cfg).to(torch.float32).eval()

    ours = Qwen2ForCausalLM(hf_cfg, max_position=tiny_config.max_position_embeddings)
    ours = ours.to(torch.float32).eval()
    ours.load_hf_state_dict(hf.state_dict())

    tokens = torch.randint(0, hf_cfg.vocab_size, (19,))
    with torch.inference_mode():
        ref = hf(input_ids=tokens.unsqueeze(0)).logits[0]

        r, bm = harness(ours, hf_cfg, make_cache, block_size=8)
        seq = mkseq(0, tokens.tolist())
        assert bm.allocate(seq, len(tokens))
        ids, pos, md, _, _ = r.build_inputs([(seq, len(tokens))])
        hidden = r.model(ids, pos, r.kv_cache, md, r.attn_fn)
        got = r.model.compute_logits(hidden)

    torch.testing.assert_close(got, ref, atol=1e-3, rtol=1e-3)
