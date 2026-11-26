import pytest
import torch

from lowp.formats import E2M1, E2M3, E3M2, E4M3, E5M2, quantize, round_e8m0, values


def probe_inputs(fmt, n_random=200_000):
    """Every representable value, every midpoint (the ties), the float32 neighbours of
    each midpoint, overflow, subnormal and special values, and log-uniform randoms."""
    v = values(fmt).double()
    mids = (v[1:] + v[:-1]) / 2
    nxt = torch.cat([v, torch.tensor([fmt.max * 2])])
    over = (fmt.max + torch.cat([v[-1:] + (nxt[-1] - v[-1]) / 4, nxt[-1:]])).double()
    base = torch.cat([v, mids, over, torch.tensor([fmt.max * 1.0625, 1e30, 2.0**-30])]).float()
    near = torch.cat([torch.nextafter(base, torch.tensor(float("inf"))),
                      torch.nextafter(base, torch.tensor(0.0))])  # fmt: skip
    g = torch.Generator().manual_seed(0)
    rnd = torch.exp2(torch.empty(n_random).uniform_(-24, 20, generator=g))
    x = torch.cat([base, near, rnd])
    x = torch.cat([x, -x, torch.tensor([0.0, -0.0, float("inf"), -float("inf"), float("nan")])])
    return x


@pytest.mark.parametrize("fmt,dtype", [(E4M3, torch.float8_e4m3fn), (E5M2, torch.float8_e5m2)])
def test_bit_exact_against_torch_casts(fmt, dtype):
    x = probe_inputs(fmt)
    want = x.to(dtype).float()
    got = quantize(x, fmt, saturate=False)
    same = (got == want) | (got.isnan() & want.isnan())
    assert same.all(), x[~same][:10]
    assert (torch.signbit(got) == torch.signbit(want))[~want.isnan()].all()  # incl. −0


def test_e8m0_matches_torch():
    g = torch.Generator().manual_seed(1)
    s = torch.exp2(torch.empty(10_000).uniform_(-120, 120, generator=g))
    s = torch.cat([s, 1.5 * torch.exp2(torch.arange(-100.0, 100.0))])  # exact ties
    assert torch.equal(round_e8m0(s, "nearest"), s.to(torch.float8_e8m0fnu).float())
    fl, ce = round_e8m0(s, "floor"), round_e8m0(s, "ceil")
    assert (fl <= s).all() and (s < 2 * fl).all()
    assert (ce >= s).all() and (ce < 2 * s).all()


@pytest.mark.parametrize(
    "fmt,table",
    [
        (E2M1, [0, 0.5, 1, 1.5, 2, 3, 4, 6]),
        (E3M2, None),
        (E2M3, None),
    ],
)
def test_small_formats_round_to_nearest_representable(fmt, table):
    v = values(fmt)
    if table is not None:
        assert v.tolist() == table
    assert v.max().item() == fmt.max
    x = torch.linspace(-1.2 * fmt.max, 1.2 * fmt.max, 100_001)
    q = quantize(x, fmt)
    grid = torch.cat([-v.flip(0), v])
    d = (x[:, None] - grid[None]).abs()
    best = d.min(1).values
    assert ((q - x).abs() <= best + 1e-6).all()  # nearest (ties may go either way here)
    assert torch.isin(q, grid).all()


def test_ties_go_to_even_mantissa():
    # 1.5 lies between 1 (mantissa 0, even) and 2 (mantissa 0 of the next binade): for E2M1
    # 1.25 lies between 1.0 (m = 0) and 1.5 (m = 1) → 1.0; 1.75 → 2.0
    q = quantize(torch.tensor([1.25, 1.75, 2.5, 3.5, 5.0]), E2M1)
    assert q.tolist() == [1.0, 2.0, 2.0, 4.0, 4.0]


def test_stochastic_rounding_is_unbiased_and_lands_on_neighbours():
    x = torch.full((200_000,), 1.3)
    g = torch.Generator().manual_seed(0)
    q = quantize(x, E2M1, stochastic=True, generator=g)
    assert set(q.unique().tolist()) == {1.0, 1.5}
    assert abs(q.mean().item() - 1.3) < 3e-3
