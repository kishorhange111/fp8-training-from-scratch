"""Quantisation error of each recipe on each GEMM, measured on real training tensors.

    python -m lowp.errors --out results

Trains the fp32 GPT for ``--steps`` steps, then captures, for one batch and every
projection (QKV, attention out, MLP up, MLP down in every layer), the input x, the weight
W and the output gradient dy. Each recipe then recomputes the three GEMMs from its
quantised operands, and the error is reported as the SQNR of the GEMM *output* in dB
against fp32. An SQNR of 30 dB is a relative error of about 3%. Also reported: how
heavy-tailed each operand is (amax / rms), which decides how much a shared scale costs.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

import torch

from .linear import Recipe
from .model import GPT, GPTConfig
from .scaling import sqnr_db
from .train import batches, load_text

RECIPES = [
    ("bf16", False), ("fp8_tensor", False), ("fp8_block", False), ("mxfp8", False),
    ("mxfp4", False), ("mxfp4", True), ("nvfp4", False), ("nvfp4", True),
]  # fmt: skip


def capture(model, x, y):
    store = {}
    hooks = []
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.Linear) and name != "head":

            def fwd(mod, inp, out, name=name):
                store.setdefault(name, {})["x"] = inp[0].detach().reshape(-1, inp[0].shape[-1])
                store[name]["w"] = mod.weight.detach().clone()

            def bwd(mod, gin, gout, name=name):
                store[name]["g"] = gout[0].detach().reshape(-1, gout[0].shape[-1])

            hooks += [m.register_forward_hook(fwd), m.register_full_backward_hook(bwd)]
    model.zero_grad()
    model(x, y).backward()
    for h in hooks:
        h.remove()
    return store


def gemm_errors(t, rec: Recipe):
    x, w, g = t["x"], t["w"], t["g"]
    out = {"fwd": sqnr_db(x @ w.t(), rec.q("x", x, -1) @ rec.q("w", w, -1).t())}
    out["dgrad"] = sqnr_db(g @ w, rec.q("g", g, -1) @ rec.q("w", w, 0))
    gt, xt = rec.wgrad_rotate(g, x)
    out["wgrad"] = sqnr_db(g.t() @ x, rec.q("g", gt, 0).t() @ rec.q("x", xt, 0))
    return out


def tail(t):
    return float(t.abs().max() / t.pow(2).mean().sqrt())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    c = GPTConfig()
    model = GPT(c)
    data = load_text()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95), weight_decay=0.1)
    it = batches(data, 32, c.ctx, a.steps + 1, a.seed)
    for _, (x, y) in zip(range(a.steps), it, strict=False):
        loss = model(x, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    x, y = next(it)
    store = capture(model, x, y)
    rows = []
    for name, rht in RECIPES:
        rec = Recipe(name, rht=rht)
        per = {k: gemm_errors(t, rec) for k, t in store.items()}
        row = {"recipe": name + (" + RHT" if rht else "")}
        for gemm in ("fwd", "dgrad", "wgrad"):
            v = [p[gemm] for p in per.values()]
            row[gemm] = statistics.median(v)
            row[gemm + "_worst"] = min(v)
            row[gemm + "_worst_layer"] = min(per, key=lambda k, gm=gemm: per[k][gm])
        rows.append(row)
        print(json.dumps(row), flush=True)
    tails = {k: {r: tail(t[r]) for r in ("x", "w", "g")} for k, t in store.items()}
    res = {"steps": a.steps, "final_loss": float(loss), "rows": rows, "tails": tails}
    with open(os.path.join(a.out, "errors.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    L = [
        f"fp32 GPT ({c.layers} layers, d = {c.d}) after {a.steps} steps (loss {float(loss):.2f}). "
        f"SQNR in dB of each GEMM's output against fp32, median over the {len(store)} "
        "projections (worst in brackets). Higher is better; 30 dB ≈ 3% relative error.",
        "",
        "| recipe | forward x·Wᵀ | data grad dy·W | weight grad dyᵀ·x |",
        "|---|---|---|---|",
    ]
    for r in rows:
        L.append(
            f"| {r['recipe']} | {r['fwd']:.1f} ({r['fwd_worst']:.1f}) "
            f"| {r['dgrad']:.1f} ({r['dgrad_worst']:.1f}) "
            f"| {r['wgrad']:.1f} ({r['wgrad_worst']:.1f}) |"
        )
    L += ["", "amax / rms of each operand (a Gaussian of this size gives about 4–5):", "",
          "| projection | x | W | dy |", "|---|---|---|---|"]  # fmt: skip
    for k, t in tails.items():
        L.append(f"| {k} | {t['x']:.1f} | {t['w']:.1f} | {t['g']:.1f} |")
    with open(os.path.join(a.out, "errors.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
