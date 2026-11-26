"""Linear layers whose three GEMMs see low-precision operands.

For ``y = x Wᵀ`` with x (T × K) and W (N × K), training runs three GEMMs:

| GEMM | operands | reduction axis |
|---|---|---|
| forward ``y = x Wᵀ`` | x, W | K: x's last axis, W's last axis |
| data grad ``dx = dy W`` | dy, W | N: dy's last axis, **W's first axis** |
| weight grad ``dW = dyᵀ x`` | dy, x | T: **the first axis of both** |

A scale must be constant along the reduction axis of each block, or the GEMM could not
apply it outside the inner loop. So block and MX recipes quantise each operand
**separately for each GEMM**, along that GEMM's reduction axis. W is quantised once along
K and once along N, the transposed copy MXFP8 training keeps. Per-tensor recipes need no
such care.

A ``Recipe`` maps (role, tensor, axis) to fake-quantised fp32. Roles are ``x``, ``w`` and
``g`` (the output gradient). The matmuls run in fp32 on the quantised values, which is
what an FP8/FP4 tensor core with fp32 accumulation computes, up to summation order.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from . import scaling as S
from .formats import E2M1, E4M3, E5M2


def _bf16(t):
    return t.bfloat16().float()


def hadamard(n: int, device=None) -> torch.Tensor:
    """Orthonormal Sylvester Hadamard matrix, n a power of two."""
    H = torch.ones(1, 1, device=device)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H / math.sqrt(n)


class Recipe:
    """``name`` selects the formats and scaling; ``sr_grad`` uses stochastic rounding for
    gradients. ``rht`` applies a random Hadamard transform of size 16 along the token axis
    to both wgrad operands (NVFP4 training, NVIDIA 2025). ``H·dy`` and ``H·x`` give the
    same product because H is orthogonal, but outliers are spread before quantisation."""

    NAMES = ("fp32", "bf16", "fp8_tensor", "fp8_delayed", "fp8_block", "mxfp8", "mxfp4", "nvfp4")

    def __init__(self, name: str, sr_grad: bool = False, rht: bool = False, seed: int = 0):
        if name not in self.NAMES:
            raise ValueError(name)
        self.name, self.sr_grad, self.rht = name, sr_grad, rht
        self.delayed: dict = {}
        self.g = torch.Generator().manual_seed(seed)
        self._signs = None

    def q(self, role, t, axis, key=None):
        n = self.name
        sr = self.sr_grad and role == "g"
        if n == "fp32":
            return t.float()
        if n == "bf16":
            return _bf16(t)
        fmt8 = E5M2 if role == "g" else E4M3
        if n == "fp8_tensor":  # the "hybrid" recipe: E4M3 forward, E5M2 gradients
            return S.tensor(t, fmt8, stochastic=sr)
        if n == "fp8_delayed":
            k = (key, role, axis)
            if k not in self.delayed:
                self.delayed[k] = S.DelayedScaler(fmt8)
            return self.delayed[k](t, stochastic=sr)
        if n == "fp8_block":  # DeepSeek-V3: E4M3 everywhere, 1×128 / 128×1 / 128×128 tiles
            if role == "w":
                return S.block(t, E4M3, (128, 128))
            tile = (1, 128) if axis in (-1, 1) else (128, 1)
            return S.block(t, E4M3, tile, stochastic=sr)
        if n == "mxfp8":
            return S.mx(t, E4M3, 32, axis, stochastic=sr)
        if n == "mxfp4":
            return S.mx(t, E2M1, 32, axis, stochastic=sr)
        return S.nvfp4(t, 16, axis, stochastic=sr)  # nvfp4

    def wgrad_rotate(self, a, b):
        """Applies the same random signed Hadamard along the token axis (dim 0) of both."""
        if not self.rht:
            return a, b
        T, n = a.shape[0], 16
        if T % n:
            return a, b
        if self._signs is None:
            self._signs = (torch.randint(0, 2, (n,), generator=self.g) * 2 - 1).float()
        H = hadamard(n, a.device) * self._signs.to(a.device)  # random-sign Hadamard
        rot = lambda t: (H @ t.reshape(T // n, n, -1)).reshape(T, -1)
        return rot(a), rot(b)


class _QMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, recipe, key):
        ctx.save_for_backward(x, w)
        ctx.recipe, ctx.key = recipe, key
        return recipe.q("x", x, -1, key) @ recipe.q("w", w, -1, key).t()

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        r, key = ctx.recipe, ctx.key
        dx = r.q("g", g, -1, key) @ r.q("w", w, 0, key)
        gt, xt = r.wgrad_rotate(g, x)
        dw = r.q("g", gt, 0, key).t() @ r.q("x", xt, 0, key)
        return dx, dw, None, None


class QLinear(nn.Linear):
    """``nn.Linear`` whose matmuls go through ``recipe`` (the bias stays in fp32)."""

    def __init__(self, i, o, bias=True, recipe: Recipe | None = None, key=None):
        super().__init__(i, o, bias)
        self.recipe, self.key = recipe, key

    def forward(self, x):
        if self.recipe is None:
            return super().forward(x)
        shp = x.shape
        y = _QMatmul.apply(x.reshape(-1, shp[-1]), self.weight, self.recipe, self.key)
        y = y.reshape(*shp[:-1], -1)
        return y + self.bias if self.bias is not None else y
