# fp8-training-from-scratch

[![CI](https://github.com/kishorhange111/fp8-training-from-scratch/actions/workflows/ci.yml/badge.svg)](https://github.com/kishorhange111/fp8-training-from-scratch/actions/workflows/ci.yml)

Low-precision training from scratch: the FP8 / FP6 / FP4 number formats and the scaling
recipes that make them trainable, from per-tensor FP8 to DeepSeek-V3's block scaling and
the OCP MX / NVFP4 microscaling formats of Blackwell GPUs. Everything is written in
PyTorch, and the formats are **bit-exact against PyTorch's own `float8_e4m3fn`,
`float8_e5m2` and `float8_e8m0fnu` casts**.

## Formats ([`formats.py`](src/lowp/formats.py))

One arithmetic quantiser covers every minifloat:
`q(x) = sign · round(|x| / 2^(E − m)) · 2^(E − m)` with `E = max(⌊log₂|x|⌋, 1 − bias)`.
It uses round-half-to-even (or stochastic rounding), and ⌊log₂⌋ comes from `frexp`, so
it is exact. Subnormals fall out of the clamp on E.

| format | role | largest | beyond it |
|---|---|---|---|
| E4M3 (FN) | FP8 forward, DeepSeek-V3 everywhere, MXFP8 | 448 | NaN, no infinities |
| E5M2 | FP8 gradients (the "hybrid" recipe) | 57,344 | ±inf |
| E3M2, E2M3 | MX FP6 | 28, 7.5 | saturate |
| E2M1 | MXFP4 / NVFP4 elements | 6 | saturate |
| E8M0 | MX shared scales (powers of two) | 2^127 | |

The tests compare against PyTorch on every representable value, every rounding tie and
its float32 neighbours, overflow, subnormals, ±0, ±inf, NaN and 200k random inputs. One
detail they pinned down is that PyTorch's E8M0 cast rounds to the *linearly* nearest
power of two, with ties going up, while the OCP MX spec's shared-scale rule is floor.

## Scaling recipes ([`scaling.py`](src/lowp/scaling.py))

| recipe | values per scale | scale type | reference |
|---|---|---|---|
| per-tensor, current | whole tensor | fp32 | |
| per-tensor, delayed | whole tensor | fp32, max amax of last 16 steps | Transformer Engine |
| block | 1 × 128 activations, 128 × 128 weights | fp32 | DeepSeek-V3 |
| MX | 32 along the reduction axis | E8M0 (power of two) | OCP MX v1.0 |
| NVFP4 | 16 | E4M3, under one fp32 tensor scale | NVIDIA Blackwell |

## Quantised GEMMs ([`linear.py`](src/lowp/linear.py))

Training a linear layer runs three GEMMs: forward `x Wᵀ` reduces over K, data-grad
`dy W` over N, and weight-grad `dyᵀ x` over tokens. A block scale must be constant along
the reduction axis, so each operand is **quantised separately for each GEMM, along that
GEMM's reduction axis**. W is quantised once along K and once along N, the transposed
copy that MXFP8 training keeps. Optional stochastic rounding of gradients and a random
Hadamard transform of the weight-gradient operands follow the NVFP4 training recipe. A
test checks that each GEMM uses exactly the right per-axis quantisation.

RESULTS

## Tests

```
pip install -e ".[dev]" && pytest -q
```
