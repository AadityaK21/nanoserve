"""Continuous batching scheduler.

Static batching decides the batch once and runs it to completion. If one
request in a batch of 32 wants 900 tokens and the rest want 40, the GPU spends
860 steps computing a batch of 1 while 31 rows sit there burning FLOPs on
finished sequences and 31 queued requests wait. Under realistic (lognormal)
output lengths that is most of the machine.

Continuous batching schedules per *iteration* instead of per batch. Every step,
finished sequences leave and queued ones join, so the batch stays full. Three
mechanisms make it work:

  Token budget. Prefill and decode share one max_num_batched_tokens budget.
  Decode is memory-bandwidth bound and barely uses the tensor cores; prefill is
  compute bound. Mixing them in one step fills both.

  Chunked prefill. A 2000-token prompt admitted whole stalls every decoding
  request for the length of that prefill, which is what puts the long tail in
  p99 TPOT. Split into budget-sized chunks it interleaves instead, trading a
  little TTFT for a lot of tail latency.

  Preemption. Sequences grow, so the batch can run out of blocks mid-flight.
  Rather than crash or deadlock, evict the newest running sequence, hand its
  blocks back, and requeue it. Its generated tokens are kept and its KV is
  recomputed on retry -- we spend compute to buy memory, which is the correct
  direction when memory is the binding constraint.

Deliberately pure Python: no torch import. The tricky logic is all bookkeeping,
and it is fully unit-testable on a laptop with no GPU.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

from .block_manager import BlockSpaceManager
from .config import SchedulerConfig
from .sequence import SeqStatus, Sequence


@dataclass
class SchedulerOutput:
    scheduled: list = field(default_factory=list)     # [(Sequence, num_tokens)]
    preempted: list = field(default_factory=list)
    num_batched_tokens: int = 0
    num_prefill_seqs: int = 0
    num_decode_seqs: int = 0

    def __bool__(self) -> bool:
        return bool(self.scheduled)


@dataclass
class SchedulerStats:
    total_steps: int = 0
    total_preemptions: int = 0
    total_prefill_tokens: int = 0
    total_decode_tokens: int = 0
    batch_size_sum: int = 0
    mixed_steps: int = 0

    @property
    def mean_batch_size(self) -> float:
        return self.batch_size_sum / self.total_steps if self.total_steps else 0.0


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, block_manager: BlockSpaceManager) -> None:
        self.cfg = cfg
        self.block_manager = block_manager
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self.stats = SchedulerStats()

    # ---- queue management ------------------------------------------------
    def add(self, seq: Sequence) -> None:
        seq.status = SeqStatus.WAITING
        self.waiting.append(seq)

    @property
    def has_work(self) -> bool:
        return bool(self.waiting or self.running)

    def num_unfinished(self) -> int:
        return len(self.waiting) + len(self.running)

    # ---- the step --------------------------------------------------------
    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput()
        budget = self.cfg.max_num_batched_tokens

        budget = self._schedule_running(out, budget)
        self._schedule_waiting(out, budget)

        # Classified before the runner advances num_computed_tokens, so
        # is_prefilling still describes what this step is doing.
        for seq, q in out.scheduled:
            if seq.is_prefilling:
                out.num_prefill_seqs += 1
                self.stats.total_prefill_tokens += q
            else:
                out.num_decode_seqs += 1
                self.stats.total_decode_tokens += q
            out.num_batched_tokens += q

        if out.scheduled:
            self.stats.total_steps += 1
            self.stats.batch_size_sum += len(out.scheduled)
            if out.num_prefill_seqs and out.num_decode_seqs:
                self.stats.mixed_steps += 1
        return out

    def _schedule_running(self, out: SchedulerOutput, budget: int) -> int:
        """Running sequences get the budget first.

        Priority order matters. Admitting a new request ahead of a running one
        that then cannot get a block is how you build a scheduler that
        livelocks under load: everything gets admitted, nothing finishes,
        nothing frees memory. Serve what you already started.
        """
        snapshot = list(self.running)     # self.running is rebuilt at the end
        still_running: list[Sequence] = []
        protected: set[int] = set()       # already granted tokens this step

        for seq in snapshot:
            if seq.status is not SeqStatus.RUNNING:
                continue                  # evicted earlier in this same step
            if budget <= 0:
                still_running.append(seq)
                continue

            want = min(seq.num_uncomputed_tokens, budget)
            if not self.cfg.enable_chunked_prefill and seq.is_prefilling:
                want = seq.num_uncomputed_tokens
                if want > budget:
                    still_running.append(seq)
                    continue

            need_tokens = seq.num_computed_tokens + want
            admitted = True
            while not self.block_manager.allocate(seq, need_tokens):
                victim = self._pick_victim(snapshot, protected, seq)
                if victim is None:
                    self._no_victim(seq, need_tokens, out)
                    admitted = False
                    break
                self._preempt(victim, out)
                if victim in still_running:
                    still_running.remove(victim)

            if admitted:
                out.scheduled.append((seq, want))
                protected.add(seq.seq_id)
                budget -= want
                still_running.append(seq)

        self.running = still_running
        return budget

    def _schedule_waiting(self, out: SchedulerOutput, budget: int) -> int:
        """Admit queued requests into whatever budget and memory is left."""
        just_preempted = {s.seq_id for s in out.preempted}
        while self.waiting and budget > 0:
            if len(self.running) >= self.cfg.max_num_seqs:
                break
            seq = self.waiting[0]
            if seq.seq_id in just_preempted:
                # Re-admitting something we evicted this step would immediately
                # take back the blocks we freed and loop forever.
                break

            want = min(seq.num_uncomputed_tokens, budget)
            if not self.cfg.enable_chunked_prefill:
                want = seq.num_uncomputed_tokens
                # Order matters: a prompt longer than the entire budget can
                # never be scheduled without chunking, so it has to fail here
                # rather than fall through to `break` and spin forever.
                if want > self.cfg.max_num_batched_tokens:
                    raise ValueError(
                        f"prompt of {want} tokens exceeds max_num_batched_tokens="
                        f"{self.cfg.max_num_batched_tokens}; enable chunked prefill"
                    )
                if want > budget:
                    break

            need_tokens = seq.num_computed_tokens + want
            if not self.block_manager.can_allocate(seq, need_tokens):
                break
            if not self.block_manager.allocate(seq, need_tokens):
                break

            self.waiting.popleft()
            seq.status = SeqStatus.RUNNING
            if seq.metrics.start == 0.0:
                seq.metrics.start = time.perf_counter()
            self.running.append(seq)
            out.scheduled.append((seq, want))
            budget -= want
        return budget

    # ---- preemption ------------------------------------------------------
    def _pick_victim(self, candidates: list[Sequence], protected: set[int], requester: Sequence):
        """Newest-first eviction.

        The youngest running sequence has done the least work, so recomputing
        it is the cheapest recovery. Anything already granted tokens this step
        is off limits -- yanking it would leave the batch referencing blocks
        that no longer belong to it.
        """
        if self.cfg.preemption_mode == "none":
            return None
        for v in reversed(candidates):
            if v is requester or v.seq_id in protected:
                continue
            if v.status is SeqStatus.RUNNING:
                return v
        return None

    def _no_victim(self, seq: Sequence, need_tokens: int, out: SchedulerOutput) -> None:
        """Blocks exhausted and nothing evictable left.

        Three distinct situations, and conflating them is how you get a server
        that hangs instead of telling you what is wrong:

          - the request is simply too big for the cache. No scheduling policy
            fixes that; fail it.
          - preemption is disabled (the ablation). Without a recovery mechanism
            the engine has nowhere to go, which is precisely the point the
            ablation is making, so say so and stop.
          - otherwise this sequence is the last one standing; send it back to
            the queue and retry with a clean allocator.
        """
        capacity = self.block_manager.num_total_blocks - self.block_manager.watermark_blocks
        if self.block_manager.blocks_for(need_tokens) > capacity:
            raise ValueError(
                f"request {seq.seq_id} needs "
                f"{self.block_manager.blocks_for(need_tokens)} KV blocks but the "
                f"cache holds {capacity} usable blocks "
                f"({capacity * self.block_manager.block_size} tokens). "
                f"Shorten the request or raise gpu_memory_utilization."
            )
        if self.cfg.preemption_mode == "none":
            raise RuntimeError(
                f"out of KV blocks at seq {seq.seq_id} with preemption disabled. "
                f"This is the failure mode preemption exists to prevent: "
                f"{len(self.running)} running sequences hold all "
                f"{self.block_manager.num_total_blocks} blocks and none can grow."
            )
        self._preempt(seq, out)

    def _preempt(self, seq: Sequence, out: SchedulerOutput) -> None:
        self.block_manager.free(seq)
        seq.reset_for_recompute()
        # Front of the queue: a preempted request has already waited once, and
        # sending it to the back is how you turn a memory-pressure blip into a
        # starved request with an unbounded tail latency.
        self.waiting.appendleft(seq)
        out.preempted.append(seq)
        self.stats.total_preemptions += 1

    # ---- completion ------------------------------------------------------
    def free_finished(self) -> list[Sequence]:
        done = [s for s in self.running if s.status.finished]
        for s in done:
            self.block_manager.free(s)
            self.running.remove(s)
            self.finished.append(s)
        return done
