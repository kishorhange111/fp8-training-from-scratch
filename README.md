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
power of two, with ties going up, while the OCP MX spec's shared-scale rule is floor. They also caught a change in PyTorch itself: an
overflowing E4M3FN cast returns NaN in 2.12 but saturates to ±448 in 2.15. The
quantiser offers both conventions (`saturate=`), and the test checks whichever one the
installed version follows.

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

## Results: where the error goes ([`errors.py`](src/lowp/errors.py))

A byte-level GPT (4 layers, d = 128) is trained in fp32 for 300 steps on the source of
Python's standard library. For one batch, every projection's input x, weight W and output
gradient dy is captured: QKV, attention out, MLP up and MLP down in each layer, 16 in all.
Each recipe then recomputes the three GEMMs from its quantised operands. The table gives
the SQNR of the GEMM output against fp32, as the median over the 16 projections with the
worst one in brackets ([results/errors.md](results/errors.md)). 30 dB is about 3%
relative error, and each 6 dB halves it.

| recipe | forward x·Wᵀ | data grad dy·W | weight grad dyᵀ·x |
|---|---|---|---|
| bf16 | 63.6 (61.5) | 61.3 (59.1) | 59.5 (56.5) |
| fp8_tensor | 39.6 (37.1) | 33.4 (31.2) | 31.2 (28.0) |
| fp8_block | 39.7 (37.5) | 37.6 (35.3) | 35.9 (33.0) |
| mxfp8 | 37.1 (35.9) | 35.5 (33.9) | 31.9 (29.5) |
| mxfp4 | 24.0 (22.9) | 22.6 (21.6) | 17.8 (14.4) |
| mxfp4 + RHT | 24.0 (22.9) | 22.6 (21.6) | 20.8 (19.0) |
| nvfp4 | 28.3 (26.1) | 26.1 (23.9) | 22.4 (18.8) |
| nvfp4 + RHT | 28.3 (26.1) | 26.1 (23.9) | 23.8 (21.0) |

- **Gradients are the hard operand.** Their amax / rms reaches 164 in QKV and 22–33 in
  MLP-up, while activations stay between 3 and 15 and weights are Gaussian (about 5).
  So the weight-gradient GEMM, which reduces over tokens with gradients on both sides,
  is the worst for every recipe.
- **Fine-grained E4M3 beats per-tensor E5M2 on gradients.** The usual "hybrid" recipe
  gives gradients E5M2 for range. DeepSeek-V3 instead keeps E4M3's extra mantissa bit
  and gets range from 1 × 128 tiles: +4.2 dB on data-grad and +4.7 dB on weight-grad,
  at the same forward accuracy.
- **Power-of-two scales cost precision.** MXFP8 is 2.6 dB below DeepSeek-style fp32
  block scales on the forward GEMM, because flooring the scale to 2^k leaves up to one
  binade of E4M3's range unused. On gradients its 32-value blocks still beat per-tensor
  E5M2, by 2.1 dB on data-grad.
- **NVFP4 beats MXFP4 by 3.5–4.6 dB.** It uses blocks of 16 instead of 32, and E4M3
  block scales instead of powers of two.
- **The random Hadamard transform helps exactly where the tails are.** Rotating the token
  axis of both weight-gradient operands spreads outliers before quantisation. That gives
  +3.0 dB for MXFP4 (worst layer +4.6) and +1.4 dB for NVFP4, while leaving the other
  two GEMMs unchanged. The product is the same in exact arithmetic, since H is orthogonal.

## Tests

`pytest -q` runs 29 tests. They check:
- E4M3 and E5M2 quantisation is **bit-identical to PyTorch's casts** (ties, subnormals,
  overflow to NaN / inf, ±0), and E8M0 matches `float8_e8m0fnu`;
- E2M1 / E3M2 / E2M3 round to the nearest representable value, ties go to even, and
  stochastic rounding is unbiased;
- per-tensor scaling maps amax to the format max with at most half an ulp of error;
  block and MX scales confine an outlier to its own tile;
- per-tensor E4M3 absorbs a 10⁴× outlier but collapses at 10⁶×, matching its normal
  range of 2^14.8;
- the MX floor rule can clip a block's maximum, and the ceil variant cannot;
- MX elements and NVFP4's E4M3 block scales lie on their grids;
- delayed scaling saturates when amax grows;
- each GEMM quantises along its own reduction axis;
- the Hadamard transform leaves the fp32 weight gradient unchanged;
- error orders bf16 < FP8 < NVFP4 < MXFP4, and every recipe trains a step.

```
pip install -e ".[dev]" && pytest -q
python -m lowp.errors --out results     # the table above (CPU)
python -m lowp.train --out results      # every recipe; use a GPU
```
