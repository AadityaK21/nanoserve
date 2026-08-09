"""Continuous batching logic. Pure python -- no torch, no GPU.

The scheduler is where a serving engine either works or livelocks, and every
failure mode here is a bookkeeping bug rather than a numerical one. That makes
it exactly the thing worth testing on the CPU, in milliseconds, instead of
discovering it 40 minutes into a benchmark.
"""

import pytest

from nanoserve.block_manager import BlockSpaceManager
from nanoserve.config import SamplingParams, SchedulerConfig
from nanoserve.scheduler import Scheduler
from nanoserve.sequence import SeqStatus, Sequence


def mkseq(seq_id, prompt_len, max_new=8):
    return Sequence(
        seq_id=seq_id,
        prompt_token_ids=list(range(prompt_len)),
        sampling=SamplingParams(max_new_tokens=max_new, ignore_eos=True),
    )


def mksched(num_blocks=64, block_size=16, max_batched=1024, chunked=True,
            max_seqs=256, preemption="recompute", watermark=0.0):
    bm = BlockSpaceManager(block_size, num_blocks, watermark)
    cfg = SchedulerConfig(
        max_num_seqs=max_seqs,
        max_num_batched_tokens=max_batched,
        enable_chunked_prefill=chunked,
        preemption_mode=preemption,
        watermark=watermark,
    )
    return Scheduler(cfg, bm), bm


def fake_step(sched, out, tokens_per_seq=1):
    """Stand in for ModelRunner.execute: advance and append like the real one."""
    for seq, q in out.scheduled:
        seq.advance_computed(q)
    for seq, q in out.scheduled:
        if seq.num_computed_tokens == seq.num_tokens:
            for _ in range(tokens_per_seq):
                seq.append_token(0)
            seq.metrics.output_tokens = seq.num_output_tokens
            seq.check_stop(None)
    return sched.free_finished()


# ---- admission -----------------------------------------------------------
def test_empty_schedule():
    s, _ = mksched()
    assert not s.schedule()


def test_single_prefill_then_decode():
    s, _ = mksched()
    s.add(mkseq(0, prompt_len=10, max_new=3))

    out = s.schedule()
    assert [(q) for _, q in out.scheduled] == [10]
    assert out.num_prefill_seqs == 1 and out.num_decode_seqs == 0
    fake_step(s, out)

    out = s.schedule()
    assert [(q) for _, q in out.scheduled] == [1]
    assert out.num_decode_seqs == 1


def test_token_budget_is_respected():
    s, _ = mksched(max_batched=100)
    for i in range(5):
        s.add(mkseq(i, prompt_len=40))
    out = s.schedule()
    assert out.num_batched_tokens <= 100


def test_max_num_seqs_caps_concurrency():
    s, _ = mksched(max_batched=10_000, max_seqs=3)
    for i in range(10):
        s.add(mkseq(i, prompt_len=8))
    out = s.schedule()
    assert len(out.scheduled) == 3


def test_runs_to_completion_and_frees_everything():
    s, bm = mksched(num_blocks=128, block_size=16)
    total = bm.num_free_blocks
    for i in range(6):
        s.add(mkseq(i, prompt_len=20, max_new=5))
    finished = []
    for _ in range(400):
        out = s.schedule()
        if not out:
            break
        finished.extend(fake_step(s, out))
    assert len(finished) == 6
    assert all(f.num_output_tokens == 5 for f in finished)
    assert bm.num_free_blocks == total          # no leaked blocks


# ---- chunked prefill ------------------------------------------------------
def test_chunked_prefill_splits_a_long_prompt():
    s, _ = mksched(max_batched=64, chunked=True)
    s.add(mkseq(0, prompt_len=200, max_new=2))
    chunks = []
    for _ in range(10):
        out = s.schedule()
        if not out:
            break
        chunks.append(out.scheduled[0][1])
        fake_step(s, out)
        if s.running and not s.running[0].is_prefilling:
            break
    assert len(chunks) > 1
    assert sum(chunks) == 200
    assert max(chunks) <= 64


def test_chunked_prefill_mixes_with_decode():
    """The point of chunking: a big prompt must not stall running decodes."""
    s, _ = mksched(max_batched=64, chunked=True)
    s.add(mkseq(0, prompt_len=8, max_new=50))
    fake_step(s, s.schedule())                  # get seq 0 decoding

    s.add(mkseq(1, prompt_len=500, max_new=2))
    out = s.schedule()
    assert out.num_decode_seqs == 1 and out.num_prefill_seqs == 1
    assert out.num_batched_tokens <= 64


def test_without_chunking_a_prompt_is_all_or_nothing():
    s, _ = mksched(max_batched=64, chunked=False)
    s.add(mkseq(0, prompt_len=40))
    s.add(mkseq(1, prompt_len=40))
    out = s.schedule()
    assert len(out.scheduled) == 1 and out.scheduled[0][1] == 40


def test_without_chunking_an_oversized_prompt_is_rejected_loudly():
    s, _ = mksched(max_batched=32, chunked=False)
    s.add(mkseq(0, prompt_len=100))
    with pytest.raises(ValueError):
        s.schedule()


# ---- memory pressure and preemption ---------------------------------------
def test_admission_stops_when_blocks_run_out():
    s, bm = mksched(num_blocks=4, block_size=16, max_batched=10_000)
    for i in range(10):
        s.add(mkseq(i, prompt_len=32))          # 2 blocks each
    out = s.schedule()
    assert len(out.scheduled) == 2
    assert bm.num_free_blocks == 0


def test_preemption_frees_memory_and_requeues():
    s, bm = mksched(num_blocks=4, block_size=16, max_batched=10_000)
    for i in range(2):
        s.add(mkseq(i, prompt_len=32, max_new=200))
    fake_step(s, s.schedule())                  # both prefilled, 4/4 blocks used
    assert bm.num_free_blocks == 0

    # Decode until someone needs a fifth block; a preemption has to happen.
    preempted_any = False
    for _ in range(60):
        out = s.schedule()
        if out.preempted:
            preempted_any = True
            break
        if not out:
            break
        fake_step(s, out)
    assert preempted_any
    assert s.stats.total_preemptions >= 1


def test_preempted_sequence_keeps_its_generated_tokens():
    s, _ = mksched(num_blocks=4, block_size=16, max_batched=10_000)
    seq = mkseq(0, prompt_len=32, max_new=200)
    s.add(seq)
    s.add(mkseq(1, prompt_len=32, max_new=200))
    for _ in range(60):
        out = s.schedule()
        if not out:
            break
        fake_step(s, out)
        victim = next((x for x in out.preempted), None)
        if victim is not None:
            assert victim.num_computed_tokens == 0
            assert victim.block_table == []
            assert victim.status is SeqStatus.PREEMPTED
            # recompute preemption re-prefills prompt + what it already made
            assert victim.num_tokens == victim.prompt_len + victim.num_output_tokens
            return
    pytest.skip("no preemption triggered in this configuration")


def test_preemption_goes_to_the_front_of_the_queue():
    s, _ = mksched(num_blocks=4, block_size=16, max_batched=10_000)
    for i in range(2):
        s.add(mkseq(i, prompt_len=32, max_new=200))
    fake_step(s, s.schedule())
    s.add(mkseq(99, prompt_len=16, max_new=4))   # newcomer, queued behind
    for _ in range(60):
        out = s.schedule()
        if out.preempted:
            assert s.waiting[0] is out.preempted[0]
            return
        if not out:
            break
        fake_step(s, out)
    pytest.skip("no preemption triggered")


def test_no_preemption_mode_fails_loudly_instead_of_hanging():
    """The ablation: without preemption, KV exhaustion has no recovery path.

    A server that silently stops making progress here is far worse than one
    that raises, so assert it raises.
    """
    s, _ = mksched(num_blocks=4, block_size=16, preemption="none")
    for i in range(2):
        s.add(mkseq(i, prompt_len=32, max_new=200))
    with pytest.raises(RuntimeError, match="preemption disabled"):
        for _ in range(60):
            out = s.schedule()
            assert not out.preempted
            if not out:
                break
            fake_step(s, out)


def test_request_larger_than_the_whole_cache_is_rejected():
    s, _ = mksched(num_blocks=2, block_size=16, max_batched=10_000)
    s.add(mkseq(0, prompt_len=16, max_new=200))
    with pytest.raises(ValueError, match="KV blocks"):
        for _ in range(400):
            out = s.schedule()
            if not out:
                break
            fake_step(s, out)


def test_system_makes_progress_under_heavy_pressure():
    """The livelock check: constant preemption must still drain the queue."""
    s, bm = mksched(num_blocks=12, block_size=16, max_batched=512)
    total = bm.num_free_blocks
    for i in range(8):
        s.add(mkseq(i, prompt_len=48, max_new=24))
    finished = []
    for _ in range(20_000):
        out = s.schedule()
        if not out:
            break
        finished.extend(fake_step(s, out))
    assert len(finished) == 8
    assert bm.num_free_blocks == total


def test_running_sequences_have_priority_over_new_ones():
    s, _ = mksched(num_blocks=64, block_size=16, max_batched=4)
    s.add(mkseq(0, prompt_len=4, max_new=10))
    fake_step(s, s.schedule())
    s.add(mkseq(1, prompt_len=64, max_new=10))
    out = s.schedule()
    assert out.scheduled[0][0].seq_id == 0       # decode of the running seq first


# ---- stats ---------------------------------------------------------------
def test_stats_account_for_every_token():
    s, _ = mksched(num_blocks=256, block_size=16)
    for i in range(4):
        s.add(mkseq(i, prompt_len=32, max_new=6))
    while True:
        out = s.schedule()
        if not out:
            break
        fake_step(s, out)
    assert s.stats.total_prefill_tokens == 4 * 32
    assert s.stats.total_decode_tokens == 4 * 5      # last token needs no step
    assert s.stats.mean_batch_size > 1
