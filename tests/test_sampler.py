"""Batched sampling with per-request parameters."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanoserve.config import SamplingParams  # noqa: E402
from nanoserve.sampler import Sampler  # noqa: E402


def greedy(n=1):
    return [SamplingParams(temperature=0.0) for _ in range(n)]


def test_greedy_is_argmax():
    s = Sampler("cpu")
    logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.0, 0.0]])
    assert s.sample(logits, greedy(2)) == [1, 0]


def test_greedy_is_deterministic_across_calls():
    s = Sampler("cpu")
    logits = torch.randn(4, 50)
    assert s.sample(logits, greedy(4)) == s.sample(logits, greedy(4))


def test_seeded_sampling_is_reproducible():
    logits = torch.randn(4, 50)
    params = [SamplingParams(temperature=1.0) for _ in range(4)]
    a = Sampler("cpu", seed=7).sample(logits, params)
    b = Sampler("cpu", seed=7).sample(logits, params)
    assert a == b


def test_top_k_1_collapses_to_greedy():
    s = Sampler("cpu", seed=0)
    logits = torch.randn(6, 100)
    params = [SamplingParams(temperature=1.0, top_k=1) for _ in range(6)]
    assert s.sample(logits, params) == logits.argmax(-1).tolist()


def test_top_p_never_picks_a_masked_token():
    """With p tiny only the top token survives, so sampling must return it."""
    s = Sampler("cpu", seed=3)
    logits = torch.tensor([[10.0, 1.0, 1.0, 1.0]])
    params = [SamplingParams(temperature=1.0, top_p=0.01)]
    assert s.sample(logits, params) == [0]


def test_per_request_params_are_honoured_in_one_batch():
    """A greedy request in a random batch must still be greedy.

    This is the mixed-batch case a real server hits constantly, and a sampler
    that quietly applies row 0's parameters to the whole batch passes every
    single-request test.
    """
    s = Sampler("cpu", seed=1)
    logits = torch.tensor([
        [10.0, 1.0, 1.0, 1.0],
        [1.0, 1.0, 1.0, 1.0],
    ])
    params = [SamplingParams(temperature=0.0), SamplingParams(temperature=1.0)]
    for _ in range(20):
        assert s.sample(logits, params)[0] == 0


def test_temperature_widens_the_distribution():
    torch.manual_seed(0)
    logits = torch.tensor([[3.0, 2.0, 1.0, 0.0]]).repeat(400, 1)
    cold = Sampler("cpu", seed=0).sample(logits, [SamplingParams(temperature=0.1)] * 400)
    hot = Sampler("cpu", seed=0).sample(logits, [SamplingParams(temperature=5.0)] * 400)
    assert len(set(cold)) < len(set(hot))
