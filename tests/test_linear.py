import pytest
import torch

from lowp import scaling as S
from lowp.formats import E4M3
from lowp.linear import QLinear, Recipe, hadamard


def _layer(recipe, seed=0):
    torch.manual_seed(seed)
    return QLinear(256, 128, recipe=recipe, key="l")


def _grads(layer, x, dy):
    x = x.clone().requires_grad_()
    y = layer(x)
    y.backward(dy)
    return y.detach(), x.grad, layer.weight.grad


def _data(seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(64, 256, generator=g), torch.randn(64, 128, generator=g)


def test_fp32_recipe_equals_nn_linear():
    x, dy = _data()
    ref = _grads(_layer(None), x, dy)
    got = _grads(_layer(Recipe("fp32")), x, dy)
    for a, b in zip(ref, got, strict=True):
        torch.testing.assert_close(a, b, atol=1e-5, rtol=1e-5)


def test_random_hadamard_transform_leaves_the_weight_gradient_unchanged():
    H = hadamard(16)
    torch.testing.assert_close(H @ H.t(), torch.eye(16))
    x, dy = _data()
    ref = _grads(_layer(Recipe("fp32")), x, dy)[2]
    got = _grads(_layer(Recipe("fp32", rht=True)), x, dy)[2]
    torch.testing.assert_close(ref, got, atol=1e-4, rtol=1e-4)


def test_each_gemm_quantises_along_its_own_reduction_axis():
    x, dy = _data()
    layer = _layer(Recipe("mxfp8"))
    y, dx, dw = _grads(layer, x, dy)
    w = layer.weight.detach()
    q = lambda t, ax: S.mx(t, E4M3, 32, ax)
    torch.testing.assert_close(y, q(x, -1) @ q(w, -1).t() + layer.bias.detach())
    torch.testing.assert_close(dx, q(dy, -1) @ q(w, 0))
    torch.testing.assert_close(dw, q(dy, 0).t() @ q(x, 0))


def _rel(a, b):
    return float((a - b).norm() / b.norm())


def test_error_grows_as_precision_drops():
    x, dy = _data()
    ref = _grads(_layer(Recipe("fp32")), x, dy)
    err = {}
    for name in ("bf16", "fp8_tensor", "fp8_block", "mxfp8", "mxfp4", "nvfp4"):
        got = _grads(_layer(Recipe(name)), x, dy)
        err[name] = max(_rel(a, b) for a, b in zip(got, ref, strict=True))
    assert err["bf16"] < 0.01
    assert max(err["fp8_tensor"], err["fp8_block"], err["mxfp8"]) < 0.1
    assert err["bf16"] < min(err["fp8_block"], err["mxfp8"])
    assert max(err["fp8_block"], err["mxfp8"]) < min(err["mxfp4"], err["nvfp4"])
    assert err["nvfp4"] < err["mxfp4"]  # finer E4M3 block scales over 16 values


@pytest.mark.parametrize("name", Recipe.NAMES)
def test_every_recipe_trains_a_step(name):
    x, dy = _data()
    layer = _layer(Recipe(name, sr_grad=name.endswith("fp4"), rht=name == "nvfp4"))
    _, dx, dw = _grads(layer, x, dy)
    assert torch.isfinite(dx).all() and torch.isfinite(dw).all()
