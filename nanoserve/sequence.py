"""Request and sequence state.

The scheduler operates on Sequence objects, not on tensors. Keeping all mutable
per-request state in one place is what makes iteration-level scheduling
tractable: at any point between steps, a Sequence fully describes where that
request is, and the engine can add, drop, or preempt one without touching the
others.

The central invariant is num_computed_tokens:

    token_ids[:num_computed_tokens]  -- KV is in the cache
    token_ids[num_computed_tokens:]  -- KV is not, must be computed this step

Prefill, chunked prefill and decode are all just "advance num_computed_tokens
by q". Decode is the special case q == 1. Once you see it that way the
scheduler stops needing separate prefill and decode code paths.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from .config import SamplingParams
from .metrics import RequestMetrics


class SeqStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_LENGTH = enum.auto()
    FINISHED_EOS = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @property
    def finished(self) -> bool:
        return self in (
            SeqStatus.FINISHED_LENGTH,
            SeqStatus.FINISHED_EOS,
            SeqStatus.FINISHED_ABORTED,
        )


@dataclass(eq=False)
class Sequence:
    """eq=False on purpose: identity, not value.

    The scheduler does `seq in running` and `running.remove(victim)` constantly.
    With generated __eq__ those compare whole token lists -- expensive, and
    worse, two sequences with the same prompt would compare equal and the
    scheduler would evict the wrong one.
    """

    seq_id: int
    prompt_token_ids: list[int]
    sampling: SamplingParams
    arrival: float = field(default_factory=time.perf_counter)

    output_token_ids: list[int] = field(default_factory=list)
    status: SeqStatus = SeqStatus.WAITING

    # Paged KV state. block_table[i] is the physical block holding logical
    # tokens [i * block_size, (i+1) * block_size).
    block_table: list[int] = field(default_factory=list)
    num_computed_tokens: int = 0

    # Bookkeeping for the report.
    num_preemptions: int = 0
    metrics: RequestMetrics = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.metrics is None:
            self.metrics = RequestMetrics(
                request_id=self.seq_id,
                prompt_tokens=len(self.prompt_token_ids),
                arrival=self.arrival,
            )

    # ---- lengths -------------------------------------------------------
    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self) -> int:
        """Total tokens that exist, computed or not."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_uncomputed_tokens(self) -> int:
        return self.num_tokens - self.num_computed_tokens

    @property
    def is_prefilling(self) -> bool:
        """True while any prompt token still lacks a KV entry."""
        return self.num_computed_tokens < self.prompt_len

    def token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    def slice_token_ids(self, start: int, end: int) -> list[int]:
        return self.token_ids()[start:end]

    # ---- mutation ------------------------------------------------------
    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def advance_computed(self, n: int) -> None:
        self.num_computed_tokens += n
        assert self.num_computed_tokens <= self.num_tokens

    def reset_for_recompute(self) -> None:
        """Preemption by recompute.

        Blocks are handed back to the allocator and the request re-enters the
        queue. Generated tokens are kept -- on the retry its "prompt" is
        prompt + everything generated so far, so no user-visible work is lost.
        We trade compute for memory, which is the right trade when memory is
        the binding constraint.
        """
        self.block_table = []
        self.num_computed_tokens = 0
        self.status = SeqStatus.PREEMPTED
        self.num_preemptions += 1

    # ---- stopping ------------------------------------------------------
    def check_stop(self, eos_token_id: int | None) -> bool:
        if self.num_output_tokens >= self.sampling.max_new_tokens:
            self.status = SeqStatus.FINISHED_LENGTH
            return True
        if (
            not self.sampling.ignore_eos
            and eos_token_id is not None
            and self.output_token_ids
            and self.output_token_ids[-1] == eos_token_id
        ):
            self.status = SeqStatus.FINISHED_EOS
            return True
        return False


@dataclass
class RequestOutput:
    """What the engine hands back once a request is done."""

    request_id: int
    prompt_token_ids: list[int]
    output_token_ids: list[int]
    finish_reason: str
    metrics: RequestMetrics
    num_preemptions: int = 0
    text: str | None = None
