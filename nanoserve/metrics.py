"""Per-request timing records and percentile summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class RequestMetrics:
    request_id: int
    prompt_tokens: int
    arrival: float
    start: float = 0.0            # when the scheduler first ran it
    first_token: float = 0.0
    end: float = 0.0
    output_tokens: int = 0
    step_times: list = field(default_factory=list)

    @property
    def ttft(self) -> float:
        return self.first_token - self.arrival

    @property
    def queue_delay(self) -> float:
        return self.start - self.arrival

    @property
    def e2e(self) -> float:
        return self.end - self.arrival

    @property
    def tpot(self) -> float:
        """Mean seconds per output token after the first."""
        if self.output_tokens <= 1:
            return 0.0
        return (self.end - self.first_token) / (self.output_tokens - 1)


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else 0.0


def summarize(records: list[RequestMetrics], wall_time: float) -> dict:
    """Collapse a run into the numbers that go in the report.

    Throughput is computed against wall time for the whole run, not the sum of
    per-request times. Summing per-request times double-counts overlapped work
    and inflates the result.
    """
    ttfts = [r.ttft * 1000 for r in records]
    e2es = [r.e2e * 1000 for r in records]
    tpots = [r.tpot * 1000 for r in records if r.output_tokens > 1]

    total_out = sum(r.output_tokens for r in records)
    total_in = sum(r.prompt_tokens for r in records)

    def block(name, vals):
        return {
            f"{name}_mean_ms": float(np.mean(vals)) if vals else 0.0,
            f"{name}_p50_ms": _pct(vals, 50),
            f"{name}_p90_ms": _pct(vals, 90),
            f"{name}_p99_ms": _pct(vals, 99),
        }

    out = {
        "num_requests": len(records),
        "wall_time_s": wall_time,
        "prompt_tokens": total_in,
        "output_tokens": total_out,
        "output_tok_per_s": total_out / wall_time if wall_time else 0.0,
        "total_tok_per_s": (total_in + total_out) / wall_time if wall_time else 0.0,
    }
    out.update(block("ttft", ttfts))
    out.update(block("tpot", tpots))
    out.update(block("e2e", e2es))
    return out


def print_summary(tag: str, s: dict) -> None:
    print(f"\n=== {tag} ===")
    print(f"  requests            {s['num_requests']}")
    print(f"  wall time           {s['wall_time_s']:.2f} s")
    print(f"  output throughput   {s['output_tok_per_s']:.1f} tok/s")
    print(f"  total throughput    {s['total_tok_per_s']:.1f} tok/s")
    print(f"  TTFT  p50/p99       {s['ttft_p50_ms']:.1f} / {s['ttft_p99_ms']:.1f} ms")
    print(f"  TPOT  p50/p99       {s['tpot_p50_ms']:.2f} / {s['tpot_p99_ms']:.2f} ms")
    print(f"  E2E   p50/p99       {s['e2e_p50_ms']:.1f} / {s['e2e_p99_ms']:.1f} ms")
