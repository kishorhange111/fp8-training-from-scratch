"""Minifloat formats, quantised by arithmetic instead of lookup tables.

A format with ``e`` exponent bits, ``m`` mantissa bits and bias ``2^(e−1) − 1`` represents
``±2^E · (1 + f/2^m)`` for normal exponents ``E ≥ 1 − bias``, and ``±2^(1−bias) · f/2^m``
below that (subnormals). In each binade [2^E, 2^(E+1)) the values are spaced
``2^(E − m)`` apart, and the subnormal range keeps the spacing of the lowest binade. So
for any x, with ``E = max(⌊log₂|x|⌋, 1 − bias)``:

    q(x) = sign(x) · round(|x| / 2^(E − m)) · 2^(E − m)

``round`` is round-half-to-even, or ``⌊· + U⌋`` for stochastic rounding. A value that
rounds into the next binade is still exact. ⌊log₂|x|⌋ comes from ``frexp`` exactly, never
from a floating-point log.

The formats differ only in what lies past the largest finite value:

| format | e, m | largest finite | beyond it | note |
|---|---|---|---|---|
| E4M3FN | 4, 3 | 448 | NaN | no infinities; S.1111.111 is NaN (OCP FP8) |
| E5M2 | 5, 2 | 57344 | ±inf | IEEE-style |
| E3M2, E2M3 | 3, 2 / 2, 3 | 28 / 7.5 | (saturate) | OCP MX FP6 |
| E2M1 | 2, 1 | 6 | (saturate) | OCP MX FP4 / NVFP4 element |

With ``saturate=False``, an overflowing result becomes inf or NaN exactly as PyTorch's
``float8_e4m3fn`` / ``float8_e5m2`` casts do. The tests check this bit for bit. Training
recipes use ``saturate=True``, which clamps to ± the largest finite value.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Format:
    name: str
    ebits: int
    mbits: int
    max: float  # largest finite magnitude
    has_inf: bool = False

    @property
    def bias(self) -> int:
        return 2 ** (self.ebits - 1) - 1

    @property
    def emin(self) -> int:  # exponent of the smallest normal
        return 1 - self.bias

    @property
    def emax(self) -> int:  # exponent of the largest normal binade
        return int(torch.frexp(torch.tensor(self.max))[1]) - 1

    @property
    def min_subnormal(self) -> float:
        return 2.0 ** (self.emin - self.mbits)


E4M3 = Format("e4m3", 4, 3, 448.0)
E5M2 = Format("e5m2", 5, 2, 57344.0, has_inf=True)
E3M2 = Format("e3m2", 3, 2, 28.0)
E2M3 = Format("e2m3", 2, 3, 7.5)
E2M1 = Format("e2m1", 2, 1, 6.0)
FORMATS = {f.name: f for f in (E4M3, E5M2, E3M2, E2M3, E2M1)}


def quantize(x: torch.Tensor, fmt: Format, saturate: bool = True, stochastic=False, generator=None):
    """Round x (any float dtype) to the nearest value of ``fmt``; returns float32."""
    x = x.float()
    a = x.abs()
    _, e = torch.frexp(a)  # a = m · 2^e with m in [0.5, 1)  ⇒  ⌊log₂ a⌋ = e − 1
    E = torch.clamp(e - 1, min=fmt.emin)
    step = torch.ldexp(torch.ones_like(a), E - fmt.mbits)
    t = a / step  # exact: dividing by a power of two
    if stochastic:
        u = torch.rand(t.shape, generator=generator, device=t.device)
        r = torch.floor(t + u)
    else:
        r = torch.round(t)  # half to even
    q = r * step
    if saturate:
        q = torch.clamp(q, max=fmt.max)
    else:
        over = q > fmt.max
        q = torch.where(over, torch.full_like(q, float("inf") if fmt.has_inf else float("nan")), q)
        if not fmt.has_inf:
            q = torch.where(torch.isinf(a), torch.full_like(q, float("nan")), q)
    q = torch.where(torch.isnan(x), x, q)
    return torch.copysign(q, x)


def round_e8m0(s: torch.Tensor, mode: str = "floor") -> torch.Tensor:
    """A positive scale rounded to a power of two in [2^−127, 2^127] (OCP E8M0).
    ``floor`` is the OCP MX rule for shared scales, ``ceil`` never lets a block overflow,
    and ``nearest`` matches PyTorch's ``float8_e8m0fnu`` cast."""
    _, e = torch.frexp(s.float())
    k = e - 1  # ⌊log₂ s⌋
    p = torch.ldexp(torch.ones_like(s.float()), k)
    if mode == "ceil":
        k = torch.where(p < s, k + 1, k)
    elif mode == "nearest":  # linear nearest, ties up: what a float32 → E8M0 cast does
        k = torch.where(s >= 1.5 * p, k + 1, k)
    elif mode != "floor":
        raise ValueError(mode)
    return torch.ldexp(torch.ones_like(s.float()), torch.clamp(k, -127, 127))


def values(fmt: Format) -> torch.Tensor:
    """All non-negative finite values of ``fmt``, by enumerating the bit patterns."""
    out = []
    for E in range(0, 2**fmt.ebits):
        for f in range(2**fmt.mbits):
            frac = f / 2**fmt.mbits
            v = frac * 2.0**fmt.emin if E == 0 else (1 + frac) * 2.0 ** (E - fmt.bias)
            if v <= fmt.max:
                out.append(v)
    return torch.tensor(sorted(set(out)))
