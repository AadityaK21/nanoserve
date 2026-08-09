"""Quantisation: packing, round-trip error, and drop-in equivalence.

The claim the project makes is "int4 frees ~700 MiB, which becomes KV cache".
That claim is only worth making if the packing is real (two nibbles per byte,
not int8 in disguise) and the error is small enough that the model still works.
Both are asserted here.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from nanoserve.quant import (  # noqa: E402
    QuantLinear,
    _weight_bytes,
    dequantize_int4,
    dequantize_int8,
    quantize_int4_groupwise,
    quantize_int8_per_channel,
    quantize_model,
)


# ---- int8 ---------------------------------------------------------------
def test_int8_roundtrip_is_close():
    torch.manual_seed(0)
    w = torch.randn(64, 128)
    q, s = quantize_int8_per_channel(w)
    assert q.dtype is torch.int8
    err = (dequantize_int8(q, s, torch.float32) - w).abs().max()
    assert err < w.abs().max() / 127 * 1.01      # one quantisation step


def test_int8_handles_an_outlier_row():
    """Per-channel scales mean one huge row cannot poison the others."""
    w = torch.randn(8, 64) * 0.01
    w[3] *= 10_000
    q, s = quantize_int8_per_channel(w)
    deq = dequantize_int8(q, s, torch.float32)
    quiet = (deq[[0, 1, 2, 4, 5, 6, 7]] - w[[0, 1, 2, 4, 5, 6, 7]]).abs().max()
    assert quiet < 1e-3


def test_int8_is_exact_on_representable_values():
    scale = 0.5
    w = (torch.arange(-127, 128, dtype=torch.float32) * scale).unsqueeze(0)
    q, s = quantize_int8_per_channel(w)
    torch.testing.assert_close(dequantize_int8(q, s, torch.float32), w, atol=1e-4, rtol=1e-4)


# ---- int4 ---------------------------------------------------------------
def test_int4_packing_halves_the_bytes():
    w = torch.randn(32, 256)
    packed, s, z = quantize_int4_groupwise(w, group_size=128)
    assert packed.dtype is torch.uint8
    assert packed.shape == (32, 128)              # two weights per byte
    assert s.shape == (32, 2) and z.shape == (32, 2)
    assert packed.numel() * 1 == w.numel() // 2


def test_int4_nibbles_unpack_in_the_right_order():
    """Even index in the low nibble, odd in the high one. Swap them and the
    error looks like noise instead of like a bug."""
    w = torch.randn(4, 128)
    packed, s, z = quantize_int4_groupwise(w, group_size=128)
    q = ((w.reshape(4, 1, 128) / s.unsqueeze(2)).round() + z.unsqueeze(2)).clamp(0, 15)
    q = q.reshape(4, 128).to(torch.uint8)
    torch.testing.assert_close((packed & 0x0F), q[:, 0::2])
    torch.testing.assert_close((packed >> 4), q[:, 1::2])


def test_int4_roundtrip_error_is_bounded_by_the_group_step():
    torch.manual_seed(0)
    w = torch.randn(16, 256)
    g = 128
    packed, s, z = quantize_int4_groupwise(w, g)
    deq = dequantize_int4(packed, s, z, g, 256, torch.float32)
    per_group_step = s.repeat_interleave(g, dim=1)
    assert ((deq - w).abs() <= per_group_step * 0.51 + 1e-6).all()


def test_grouping_beats_per_tensor_on_outliers():
    """The reason group-wise exists: one wild column must not flatten the rest."""
    torch.manual_seed(0)
    w = torch.randn(8, 256) * 0.01
    w[:, 0] = 50.0
    fine = dequantize_int4(*quantize_int4_groupwise(w, 128), 128, 256, torch.float32)
    coarse = dequantize_int4(*quantize_int4_groupwise(w, 256), 256, 256, torch.float32)
    tail = slice(1, None)
    assert (fine[:, tail] - w[:, tail]).abs().mean() < (coarse[:, tail] - w[:, tail]).abs().mean()


def test_int4_rejects_a_bad_group_size():
    with pytest.raises(ValueError):
        quantize_int4_groupwise(torch.randn(4, 100), group_size=128)


# ---- QuantLinear ---------------------------------------------------------
@pytest.mark.parametrize("mode,tol", [("int8", 5e-2), ("int4", 4e-1)])
def test_quantlinear_approximates_linear(mode, tol):
    torch.manual_seed(0)
    lin = torch.nn.Linear(256, 64, bias=True)
    ql = QuantLinear.from_linear(lin, mode, 128)
    x = torch.randn(8, 256)
    rel = (ql(x) - lin(x)).abs().mean() / lin(x).abs().mean()
    assert rel < tol, f"{mode} relative error {rel:.4f}"


def test_quantlinear_preserves_bias_exactly():
    lin = torch.nn.Linear(128, 32, bias=True)
    ql = QuantLinear.from_linear(lin, "int4", 128)
    torch.testing.assert_close(ql.bias.data, lin.bias.data)
    zero_in = torch.zeros(1, 128)
    torch.testing.assert_close(ql(zero_in), lin.bias.unsqueeze(0))


# ---- whole-model driver ---------------------------------------------------
def _tiny_stack():
    return torch.nn.Sequential(
        torch.nn.Linear(256, 256, bias=False),
        torch.nn.SiLU(),
        torch.nn.Linear(256, 256, bias=False),
    )


@pytest.mark.parametrize("mode,expect", [("int8", 1.9), ("int4", 3.4)])
def test_quantize_model_reports_real_compression(mode, expect):
    m = _tiny_stack()
    report = quantize_model(m, mode, 128, skip=())
    assert report["layers_quantized"] == 2
    assert report["compression"] > expect
    assert _weight_bytes(m) < report["weight_mib_before"] * 2**20


def test_quantize_model_honours_skip():
    m = torch.nn.Sequential()
    m.add_module("keep_me", torch.nn.Linear(128, 128, bias=False))
    m.add_module("other", torch.nn.Linear(128, 128, bias=False))
    quantize_model(m, "int8", 128, skip=("keep_me",))
    assert isinstance(m.keep_me, torch.nn.Linear)
    assert isinstance(m.other, QuantLinear)


def test_quantised_model_still_produces_sane_outputs():
    torch.manual_seed(0)
    m = _tiny_stack()
    x = torch.randn(4, 256)
    ref = m(x)
    quantize_model(m, "int4", 128, skip=())
    got = m(x)
    cos = torch.nn.functional.cosine_similarity(got.flatten(), ref.flatten(), dim=0)
    assert cos > 0.98, f"int4 output cosine similarity {cos:.4f}"


def test_tied_weights_are_not_double_counted():
    a = torch.nn.Linear(64, 64, bias=False)
    b = torch.nn.Linear(64, 64, bias=False)
    b.weight = a.weight
    m = torch.nn.Sequential(a, b)
    assert _weight_bytes(m) == 64 * 64 * 4
