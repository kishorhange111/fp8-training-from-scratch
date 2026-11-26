import torch

from lowp import scaling as S
from lowp.formats import E2M1, E4M3, E5M2, quantize, values


def _rand(*shape, seed=0):
    return torch.randn(*shape, generator=torch.Generator().manual_seed(seed))


def test_tensor_scaling_maps_amax_to_format_max_and_bounds_relative_error():
    x = _rand(4096)
    q = S.tensor(x, E4M3)
    i = x.abs().argmax()
    assert torch.isclose(q[i], x[i])  # the largest value lands exactly on ±448 / s
    s = E4M3.max / x.abs().max()
    normal = x.abs() * s >= 2.0**E4M3.emin
    rel = ((q - x).abs() / x.abs())[normal]
    assert rel.max() <= 2.0 ** -(E4M3.mbits + 1) + 1e-7  # half an ulp of 3 mantissa bits


def test_one_outlier_hurts_per_tensor_but_only_its_own_tile_with_block_scaling():
    x = _rand(64, 512)
    y = x.clone()
    y[3, 7] = 1e4
    for fn in (lambda t: S.block(t, E4M3, (1, 128)), lambda t: S.mx(t, E4M3)):
        a, b = fn(x), fn(y)
        same = torch.ones_like(x, dtype=torch.bool)
        same[3, :128] = False  # the outlier's 1×128 tile (MX: its 32-block, a subset)
        assert torch.equal(a[same], b[same])


def test_fp8_exponent_absorbs_moderate_outliers_but_not_extreme_ones():
    """Unlike INT8, per-tensor E4M3 hardly notices a 10^4× outlier: the other values stay
    in the normal range, which spans 448 / 2^−6 ≈ 2^14.8. At 10^6× they fall into the
    subnormals or to zero."""
    x = _rand(64, 512)
    keep = torch.ones_like(x, dtype=torch.bool)
    keep[3, 7] = False
    other_tiles = torch.ones_like(x, dtype=torch.bool)
    other_tiles[3, :128] = False
    clean = S.sqnr_db(x, S.tensor(x, E4M3))
    for big, lo, hi in ((1e4, clean - 1, clean + 1), (1e6, -100, clean - 15)):
        y = x.clone()
        y[3, 7] = big
        got = S.sqnr_db(x[keep], S.tensor(y, E4M3)[keep])
        assert lo < got < hi, (big, got, clean)
        blk = S.block(y, E4M3)
        assert S.sqnr_db(x[other_tiles], blk[other_tiles]) > clean - 1  # tiles isolate it


def test_mx_floor_scale_can_clip_the_block_max_and_ceil_cannot():
    # amax = 1.9 · 2^3: floor scale 2^(3 − 8) maps it to 486 > 448, which the spec clamps
    x = torch.zeros(32)
    x[0] = 1.9 * 8
    x[1:] = torch.linspace(-1, 1, 31)
    fl, ce = S.mx(x, E4M3, mode="floor"), S.mx(x, E4M3, mode="ceil")
    assert fl[0].item() == 448 * 2.0**-5 and fl[0] < x[0]
    assert torch.isclose(ce[0], x[0], rtol=2**-4)


def test_mx_scales_are_powers_of_two_and_elements_lie_on_the_grid():
    x = _rand(8, 96)
    for fmt in (E4M3, E2M1):
        b, _ = S._to_blocks(x, -1, 32)
        amax = b.abs().amax(-1, keepdim=True)
        X = torch.exp2(torch.floor(torch.log2(amax))) * 2.0**-fmt.emax
        q, _ = S._to_blocks(S.mx(x, fmt), -1, 32)
        grid = torch.cat([-values(fmt), values(fmt)])
        assert torch.isin(q / X, grid).all()


def test_nvfp4_block_scales_are_e4m3_and_elements_e2m1():
    x = _rand(4, 64)
    q = S.nvfp4(x)
    g = x.abs().max() / (E4M3.max * E2M1.max)
    b, _ = S._to_blocks(x, -1, 16)
    sb = quantize(b.abs().amax(-1, keepdim=True) / E2M1.max / g, E4M3)
    qb, _ = S._to_blocks(q, -1, 16)
    grid = torch.cat([-values(E2M1), values(E2M1)])
    r = qb / (sb * g)  # g is not a power of two, so allow float32 rounding
    assert ((r[..., None] - grid).abs().min(-1).values < 1e-5).all()
    assert S.sqnr_db(x, q) > S.sqnr_db(x, S.tensor(x, E2M1))  # finer scales help


def test_delayed_scaling_saturates_when_amax_grows_and_equals_current_when_stable():
    d = S.DelayedScaler(E5M2, history=4)
    x = _rand(1000)
    for _ in range(3):
        assert torch.equal(d(x), S.tensor(x, E5M2))
    assert d.saturated == 0
    d(10 * x)  # stale amax: the big values clip
    assert d.saturated > 0


def test_mx_handles_padding_and_any_axis():
    x = _rand(5, 70)
    q0 = S.mx(x, E4M3, axis=0)
    q1 = S.mx(x.t().contiguous(), E4M3, axis=-1).t()
    assert q0.shape == x.shape and torch.equal(q0, q1)
