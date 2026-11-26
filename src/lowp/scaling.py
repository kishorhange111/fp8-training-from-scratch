"""Scaling recipes: map a tensor into a format's range, quantise, map back ("fake quant").

An FP8 element has only 3 mantissa bits. Its usable range, from the smallest normal 2^−6
to 448 for E4M3, spans about 2^15, so every tensor has to be **scaled** into it first.
The recipes differ in how many values share one scale, and in how the scale is chosen:

- **per-tensor, current** (``tensor``): one fp32 scale ``max / amax(x)``. A single
  outlier squeezes every other value toward the subnormals.
- **per-tensor, delayed** (``DelayedScaler``; Transformer Engine's default): the scale
  comes from the max amax of the last H steps, so the GEMM need not wait for an amax
  reduction. Values above the stale amax saturate.
- **block** (``block``; DeepSeek-V3): fp32 scales per 1 × 128 tile of activations and
  per 128 × 128 block of weights. An outlier only affects its own tile.
- **MX** (``mx``; OCP Microscaling v1.0): blocks of 32 along the reduction axis share
  one E8M0 power-of-two scale ``2^(⌊log₂ amax⌋ − emax_elem)``. That rounding can push the
  block maximum *above* the element format's max, and the spec clamps it. ``mode="ceil"``
  rounds the scale up instead, so nothing clips.
- **NVFP4** (``nvfp4``): blocks of 16 E2M1 values, each with an E4M3 scale, under one fp32
  per-tensor scale. The E4M3 block scale is finer than a power of two, at one extra level.

Every function returns float32 values of the same shape as x. A GEMM on them gives the
numerics of the low-precision GEMM up to the fp32 accumulation order.
"""

from __future__ import annotations

import torch

from .formats import E2M1, E4M3, Format, quantize, round_e8m0

_TINY = torch.finfo(torch.float32).tiny


def _safe(amax):
    return torch.where(amax > 0, amax, torch.ones_like(amax))


def tensor(x, fmt: Format, amax=None, stochastic=False):
    amax = x.abs().max() if amax is None else torch.as_tensor(amax, dtype=torch.float32)
    s = fmt.max / _safe(amax.float())
    return quantize(x * s, fmt, saturate=True, stochastic=stochastic) / s


def _to_blocks(x, axis, size):
    """Moves ``axis`` last and splits it into blocks of ``size`` (zero-padded)."""
    x = x.float().movedim(axis, -1)
    n = x.shape[-1]
    pad = (-n) % size
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x.reshape(*x.shape[:-1], -1, size), n


def _from_blocks(b, n, axis):
    b = b.reshape(*b.shape[:-2], -1)[..., :n]
    return b.movedim(-1, axis)


def block(x, fmt: Format, tile=(1, 128), stochastic=False):
    """fp32 scale per ``tile`` of a 2-D tensor (rows × cols), as in DeepSeek-V3."""
    r, c = x.shape
    tr, tc = tile
    pr, pc = (-r) % tr, (-c) % tc
    xp = torch.nn.functional.pad(x.float(), (0, pc, 0, pr))
    R, C = xp.shape
    t = xp.reshape(R // tr, tr, C // tc, tc)
    amax = t.abs().amax(dim=(1, 3), keepdim=True)
    s = fmt.max / _safe(amax)
    q = quantize(t * s, fmt, saturate=True, stochastic=stochastic) / s
    return q.reshape(R, C)[:r, :c]


def mx(x, fmt: Format, size=32, axis=-1, mode="floor", stochastic=False):
    """OCP MX: a power-of-two (E8M0) scale per ``size`` values along ``axis``."""
    b, n = _to_blocks(x, axis, size)
    amax = b.abs().amax(-1, keepdim=True)
    if mode == "floor":  # the spec: 2^(⌊log₂ amax⌋ − emax_elem); the largest may clip
        X = round_e8m0(_safe(amax)) * 2.0**-fmt.emax
    else:  # smallest power of two with amax / X ≤ max: never clips
        X = round_e8m0(_safe(amax) / fmt.max, "ceil")
    q = quantize(b / X, fmt, saturate=True, stochastic=stochastic) * X
    return _from_blocks(q, n, axis)


def nvfp4(x, size=16, axis=-1, stochastic=False):
    """Blocks of 16 E2M1 values with E4M3 block scales under an fp32 tensor scale."""
    b, n = _to_blocks(x, axis, size)
    g = _safe(b.abs().max()) / (E4M3.max * E2M1.max)  # global scale: block scales fit E4M3
    amax = b.abs().amax(-1, keepdim=True)
    sb = quantize(amax / E2M1.max / g, E4M3, saturate=True)  # E4M3-encoded block scale
    s = torch.where(sb > 0, sb, torch.ones_like(sb)) * g
    q = quantize(b / s, E2M1, saturate=True, stochastic=stochastic) * s
    return _from_blocks(q, n, axis)


class DelayedScaler:
    """Per-tensor scale from the max amax of the previous ``history`` calls."""

    def __init__(self, fmt: Format, history: int = 16):
        self.fmt, self.history, self.amaxes = fmt, history, []
        self.saturated = 0  # values above the stale range, clipped to ±max
        self.total = 0

    def __call__(self, x, stochastic=False):
        cur = x.detach().abs().max().float()
        amax = max(self.amaxes) if self.amaxes else cur
        s = self.fmt.max / _safe(torch.as_tensor(amax))
        self.saturated += int((x.detach().abs() * s > self.fmt.max).sum())
        self.total += x.numel()
        self.amaxes = (self.amaxes + [cur])[-self.history :]
        return quantize(x * s, self.fmt, saturate=True, stochastic=stochastic) / s


def sqnr_db(x, q):
    """Signal-to-quantisation-noise ratio in dB."""
    x, q = x.double(), q.double()
    return float(10 * torch.log10((x**2).sum() / ((x - q) ** 2).sum().clamp_min(_TINY)))
