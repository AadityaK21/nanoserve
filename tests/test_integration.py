"""Scheduler + runner + model driven together, on a tiny random Qwen2.

The property under test is the one that actually matters for a server:

    what a request generates must not depend on the scheduling decisions the
    engine happened to make around it.

So the same requests are run under a generous configuration and under a
deliberately hostile one -- tiny token budget, barely enough KV blocks to force
repeated preemption, chunked prefill on -- and the token streams must be
identical. If paging, chunking or preemption-by-recompute is subtly wrong, this
is where it shows up, because a recomputed sequence takes a completely
different path through the cache to reach the same answer.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanoserve.block_manager import BlockSpaceManager  # noqa: E402
from nanoserve.config import SamplingParams, SchedulerConfig  # noqa: E402
from nanoserve.model_runner import ModelRunner  # noqa: E402
from nanoserve.scheduler import Scheduler  # noqa: E402
from nanoserve.sequence import Sequence  # noqa: E402


class MiniEngine:
    """LLMEngine minus the HuggingFace loading, so it runs offline."""

    def __init__(self, model, cfg, make_cache, num_blocks=256, block_size=8, **sched_kwargs):
        self.cache = make_cache(cfg, num_blocks=num_blocks, block_size=block_size)
        self.bm = BlockSpaceManager(block_size, num_blocks, watermark=0.0)
        self.runner = ModelRunner(model, self.cache, self.bm, "cpu", backend="torch")
        self.scheduler = Scheduler(SchedulerConfig(**sched_kwargs), self.bm)

    def add(self, seq_id, tokens, max_new):
        self.scheduler.add(
            Sequence(seq_id, list(tokens),
                     SamplingParams(max_new_tokens=max_new, ignore_eos=True))
        )

    def run(self, max_steps=100_000):
        done = {}
        for _ in range(max_steps):
            out = self.scheduler.schedule()
            if not out:
                break
            for seq, token in self.runner.execute(out.scheduled):
                seq.append_token(token)
                seq.check_stop(None)
            for seq in self.scheduler.free_finished():
                done[seq.seq_id] = seq
        assert not self.scheduler.has_work, "engine did not drain"
        return done


@pytest.fixture
def prompts(tiny_config):
    torch.manual_seed(42)
    return {
        i: torch.randint(0, tiny_config.vocab_size, (n,)).tolist()
        for i, n in enumerate([5, 19, 33, 8, 27])
    }


def run_config(tiny_model, tiny_config, make_cache, prompts, max_new, **kw):
    e = MiniEngine(tiny_model, tiny_config, make_cache, **kw)
    for i, p in prompts.items():
        e.add(i, p, max_new)
    return e, e.run()


def test_all_requests_finish_with_the_right_length(tiny_model, tiny_config, make_cache, prompts):
    _, done = run_config(tiny_model, tiny_config, make_cache, prompts, max_new=6)
    assert set(done) == set(prompts)
    assert all(s.num_output_tokens == 6 for s in done.values())


def reference(tiny_model, tiny_config, make_cache, prompts, max_new=8):
    """Roomy config: one prompt per step, no chunking, no memory pressure."""
    return run_config(
        tiny_model, tiny_config, make_cache, prompts, max_new=max_new,
        num_blocks=512, block_size=16,
        max_num_batched_tokens=4096, enable_chunked_prefill=False,
    )


def test_chunked_prefill_does_not_change_output(tiny_model, tiny_config, make_cache, prompts):
    easy_engine, easy = reference(tiny_model, tiny_config, make_cache, prompts)
    hard_engine, hard = run_config(
        tiny_model, tiny_config, make_cache, prompts, max_new=8,
        num_blocks=512, block_size=8,
        max_num_batched_tokens=6,             # every prompt gets chopped
        enable_chunked_prefill=True,
    )
    assert hard_engine.scheduler.stats.total_steps > easy_engine.scheduler.stats.total_steps, (
        "the chunked config did not actually take more steps, so nothing was chunked"
    )
    for rid in prompts:
        assert hard[rid].output_token_ids == easy[rid].output_token_ids, f"request {rid} diverged"


def test_output_survives_preemption(tiny_model, tiny_config, make_cache, prompts):
    """Preemption-by-recompute must be invisible in the output.

    The block budget is chosen so preemption is forced rather than hoped for.
    Prompts are 5/19/33/8/27 tokens; at block_size 8 their prefills need
    1+3+5+1+4 = 14 blocks, so with exactly 14 the whole batch is admitted and
    then cannot grow by a single token. The first decode step has to evict
    someone.
    """
    _, easy = reference(tiny_model, tiny_config, make_cache, prompts)
    hard_engine, hard = run_config(
        tiny_model, tiny_config, make_cache, prompts, max_new=8,
        num_blocks=14, block_size=8,
        max_num_batched_tokens=4096, enable_chunked_prefill=False,
    )
    assert hard_engine.scheduler.stats.total_preemptions > 0, (
        "no preemption occurred, so this test proves nothing"
    )
    for rid in prompts:
        assert hard[rid].output_token_ids == easy[rid].output_token_ids, f"request {rid} diverged"


def test_running_one_at_a_time_gives_the_same_answer(tiny_model, tiny_config, make_cache, prompts):
    """Batching invariance at engine level, not just at one forward pass."""
    _, batched = run_config(tiny_model, tiny_config, make_cache, prompts, max_new=5)
    for rid, p in prompts.items():
        _, alone = run_config(tiny_model, tiny_config, make_cache, {rid: p}, max_new=5)
        assert alone[rid].output_token_ids == batched[rid].output_token_ids


def test_no_blocks_leak_across_a_full_run(tiny_model, tiny_config, make_cache, prompts):
    e = MiniEngine(tiny_model, tiny_config, make_cache, num_blocks=64, block_size=8,
                   max_num_batched_tokens=32)
    total = e.bm.num_free_blocks
    for i, p in prompts.items():
        e.add(i, p, 5)
    e.run()
    assert e.bm.num_free_blocks == total


def test_requests_added_mid_flight_are_served(tiny_model, tiny_config, make_cache, prompts):
    """Continuous batching's actual selling point: joining an in-progress batch."""
    e = MiniEngine(tiny_model, tiny_config, make_cache, max_num_batched_tokens=64)
    e.add(0, prompts[1], 30)
    done = {}
    for step in range(400):
        out = e.scheduler.schedule()
        if not out:
            break
        for seq, token in e.runner.execute(out.scheduled):
            seq.append_token(token)
            seq.check_stop(None)
        for seq in e.scheduler.free_finished():
            done[seq.seq_id] = seq
        if step == 5:
            e.add(1, prompts[2], 4)           # arrives while 0 is decoding
    assert set(done) == {0, 1}
    assert done[1].num_output_tokens == 4
    assert e.scheduler.stats.mixed_steps > 0, "prefill never shared a step with decode"
