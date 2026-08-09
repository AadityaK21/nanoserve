"""Weight-only post-training quantisation.

Why weight-only, on a serving project: the win is not FLOPs, it is VRAM. On
8 GB, Qwen2.5-0.5B in fp16 spends ~950 MiB on weights. Every byte returned by
shrinking the weights becomes a KV block, and KV blocks are what set maximum
concurrency, and concurrency is what sets throughput on a memory-bandwidth
bound decode workload. INT4 hands back ~700 MiB, which at 12 KiB/token is
~58k more tokens of context, or roughly 100 more concurrent 512-token
sequences.

Be precise about what this does and does not buy, because it is the first thing
an interviewer will push on:

  it does     cut resident weight memory ~2x (int8) / ~3.7x (int4 g128)
  it does     raise achievable batch size, and therefore throughput
  it does not cut matmul FLOPs -- weights are dequantised to fp16 before the
              GEMM. Getting the compute win needs a fused dequant-GEMM kernel;
              that is listed as future work rather than claimed here.

Two schemes:

  int8  per-output-channel symmetric. One scale per row. Nearly free in
        accuracy, trivial to implement, 2x.

  int4  group-wise asymmetric, group_size along the input dim (128 default).
        Per-tensor int4 is unusable -- one outlier channel sets a scale that
        quantises everything else to zero. Grouping bounds the damage to 128
        weights, which is the difference between "slightly worse perplexity"
        and "outputs word salad".
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------- int8 ----
def quantize_int8_per_channel(w: torch.Tensor):
    """w: [out, in] -> (int8 weights, fp32 scales [out, 1])."""
    w = w.to(torch.float32)
    scale = w.abs().amax(dim=1, keepdim=True) / 127.0
    scale = scale.clamp(min=1e-8)
    q = (w / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale


def dequantize_int8(q: torch.Tensor, scale: torch.Tensor, dtype: torch.dtype):
    return (q.to(torch.float32) * scale).to(dtype)


# ---------------------------------------------------------------- int4 ----
def quantize_int4_groupwise(w: torch.Tensor, group_size: int = 128):
    """w: [out, in] -> (packed uint8 [out, in//2], scales, zeros) per group.

    Asymmetric: values map to [0, 15] via q = round(w / s) + z. Asymmetric
    matters more at 4 bits than at 8 because weight distributions are not
    centred, and a symmetric grid throws away half its levels on a sign the
    data barely uses.
    """
    out_f, in_f = w.shape
    if in_f % group_size != 0:
        raise ValueError(f"in_features {in_f} not divisible by group_size {group_size}")
    if in_f % 2 != 0:
        raise ValueError("in_features must be even to pack two nibbles per byte")

    g = in_f // group_size
    wg = w.to(torch.float32).reshape(out_f, g, group_size)

    w_max = wg.amax(dim=2, keepdim=True)
    w_min = wg.amin(dim=2, keepdim=True)
    scale = ((w_max - w_min) / 15.0).clamp(min=1e-8)
    zero = (-w_min / scale).round().clamp(0, 15)

    q = ((wg / scale).round() + zero).clamp(0, 15).to(torch.uint8)
    q = q.reshape(out_f, in_f)

    # Two nibbles per byte: even index in the low half, odd in the high half.
    packed = (q[:, 0::2] | (q[:, 1::2] << 4)).contiguous()
    return packed, scale.squeeze(2), zero.squeeze(2).to(torch.uint8)


def dequantize_int4(packed, scale, zero, group_size: int, in_features: int, dtype: torch.dtype):
    out_f = packed.shape[0]
    low = (packed & 0x0F).to(torch.float32)
    high = (packed >> 4).to(torch.float32)

    q = torch.empty(out_f, in_features, device=packed.device, dtype=torch.float32)
    q[:, 0::2] = low
    q[:, 1::2] = high

    g = in_features // group_size
    q = q.reshape(out_f, g, group_size)
    w = (q - zero.to(torch.float32).unsqueeze(2)) * scale.to(torch.float32).unsqueeze(2)
    return w.reshape(out_f, in_features).to(dtype)


# ------------------------------------------------------------- module ----
class QuantLinear(nn.Module):
    """Drop-in nn.Linear whose weight lives quantised and is expanded per call.

    The dequantised copy is a transient of size [out, in]; the persistent
    tensor is the small one. That is the whole point -- peak allocation goes up
    slightly during a matmul, resident allocation goes down a lot and stays
    down, and it is resident allocation that competes with the KV cache.
    """

    def __init__(self, in_features, out_features, bias: bool, mode: str, group_size: int, dtype, device):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mode = mode
        self.group_size = group_size
        self.compute_dtype = dtype

        if mode == "int8":
            self.register_buffer("qweight", torch.zeros(out_features, in_features, dtype=torch.int8, device=device))
            self.register_buffer("scales", torch.zeros(out_features, 1, dtype=torch.float32, device=device))
        elif mode == "int4":
            g = in_features // group_size
            self.register_buffer("qweight", torch.zeros(out_features, in_features // 2, dtype=torch.uint8, device=device))
            self.register_buffer("scales", torch.zeros(out_features, g, dtype=torch.float32, device=device))
            self.register_buffer("zeros", torch.zeros(out_features, g, dtype=torch.uint8, device=device))
        else:
            raise ValueError(f"unknown quantisation mode {mode!r}")

        self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device)) if bias else None

    @classmethod
    def from_linear(cls, lin: nn.Linear, mode: str, group_size: int):
        w = lin.weight.data
        m = cls(
            lin.in_features, lin.out_features, lin.bias is not None,
            mode, group_size, w.dtype, w.device,
        )
        if mode == "int8":
            q, s = quantize_int8_per_channel(w)
            m.qweight.copy_(q)
            m.scales.copy_(s)
        else:
            packed, s, z = quantize_int4_groupwise(w, group_size)
            m.qweight.copy_(packed)
            m.scales.copy_(s)
            m.zeros.copy_(z)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data)
        return m

    def dequantized_weight(self) -> torch.Tensor:
        if self.mode == "int8":
            return dequantize_int8(self.qweight, self.scales, self.compute_dtype)
        return dequantize_int4(
            self.qweight, self.scales, self.zeros,
            self.group_size, self.in_features, self.compute_dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.linear(x, self.dequantized_weight(), self.bias)

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, mode={self.mode}, group={self.group_size}"


# ------------------------------------------------------------ driver ----
def quantize_model(model: nn.Module, mode: str, group_size: int = 128, skip: tuple[str, ...] = ("lm_head",)) -> dict:
    """Replace every eligible nn.Linear in place. Returns a memory report."""
    if mode not in ("int8", "int4"):
        raise ValueError(f"unknown quantisation mode {mode!r}")

    before = _weight_bytes(model)
    replaced = 0

    def visit(module: nn.Module, prefix: str = ""):
        nonlocal replaced
        for name, child in list(module.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if any(s in full for s in skip):
                continue
            if isinstance(child, nn.Linear):
                if mode == "int4" and child.in_features % group_size != 0:
                    # Fall back rather than silently changing the group size and
                    # reporting a compression ratio that does not match reality.
                    continue
                setattr(module, name, QuantLinear.from_linear(child, mode, group_size))
                replaced += 1
            else:
                visit(child, full)

    visit(model)
    after = _weight_bytes(model)
    return {
        "mode": mode,
        "group_size": group_size if mode == "int4" else None,
        "layers_quantized": replaced,
        "weight_mib_before": before / 2**20,
        "weight_mib_after": after / 2**20,
        "compression": before / after if after else 0.0,
        "freed_mib": (before - after) / 2**20,
    }


def _weight_bytes(model: nn.Module) -> int:
    seen = set()
    total = 0
    for t in list(model.parameters()) + list(model.buffers()):
        if id(t) in seen:      # tied embeddings must not be double counted
            continue
        seen.add(id(t))
        total += t.numel() * t.element_size()
    return total
