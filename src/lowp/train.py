"""Train the same GPT under each recipe from the same initialisation and data order.

    python -m lowp.train --recipes fp32 bf16 fp8_tensor fp8_block mxfp8 mxfp4 nvfp4 --out results

The data is byte-level text that needs no download: the source of Python's standard
library (``--data`` may point anywhere else). Every recipe sees the same batches in the
same order, so loss differences come from the arithmetic alone.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
import sysconfig
import time

import numpy as np
import torch

from .linear import Recipe
from .model import GPT, GPTConfig


def load_text(root=None, mb=8.0) -> np.ndarray:
    root = root or sysconfig.get_paths()["stdlib"]
    out, total = [], 0
    for f in sorted(glob.glob(os.path.join(root, "*.py"))):  # top level only: deterministic
        with open(f, "rb") as fh:
            b = fh.read()
        out.append(b)
        total += len(b)
        if total >= mb * 2**20:
            break
    return np.frombuffer(b"".join(out)[: int(mb * 2**20)], dtype=np.uint8)


def batches(data, B, T, steps, seed):
    g = np.random.default_rng(seed)
    for _ in range(steps):
        i = g.integers(0, len(data) - T - 1, B)
        x = np.stack([data[j : j + T] for j in i]).astype(np.int64)
        y = np.stack([data[j + 1 : j + T + 1] for j in i]).astype(np.int64)
        yield torch.from_numpy(x), torch.from_numpy(y)


@torch.no_grad()
def evaluate(model, data, B, T, n, dev):
    model.eval()
    losses = [float(model(x.to(dev), y.to(dev))) for x, y in batches(data, B, T, n, seed=123)]
    model.train()
    return sum(losses) / len(losses)


def train(recipe_name, a, train_data, val_data, dev):
    torch.manual_seed(a.seed)
    rec = Recipe(recipe_name, sr_grad=a.sr_grad, rht=a.rht, seed=a.seed)
    c = GPTConfig(d=a.d, layers=a.layers, heads=a.heads, ctx=a.ctx)
    model = GPT(c, rec).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.1)
    sched = lambda s: min(1.0, (s + 1) / a.warmup) * 0.5 * (1 + math.cos(math.pi * s / a.steps))
    hist, t0 = [], time.perf_counter()
    for step, (x, y) in enumerate(batches(train_data, a.batch, a.ctx, a.steps, a.seed)):
        for gr in opt.param_groups:
            gr["lr"] = a.lr * sched(step)
        loss = model(x.to(dev), y.to(dev))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if not math.isfinite(float(loss)):
            hist.append({"step": step, "loss": float("nan"), "diverged": True})
            break
        if step % a.log_every == 0 or step == a.steps - 1:
            row = {"step": step, "loss": float(loss)}
            if step % a.eval_every == 0 or step == a.steps - 1:
                row["val"] = evaluate(model, val_data, a.batch, a.ctx, a.eval_batches, dev)
            hist.append(row)
            print(json.dumps({"recipe": recipe_name, **row}), flush=True)
    sat = sum(d.saturated for d in rec.delayed.values())
    tot = sum(d.total for d in rec.delayed.values())
    return {
        "recipe": recipe_name, "sr_grad": a.sr_grad, "rht": a.rht,
        "seconds": time.perf_counter() - t0,
        "final_val": hist[-1].get("val"), "history": hist,
        "delayed_saturated_frac": sat / tot if tot else None,
    }  # fmt: skip


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--recipes", nargs="+", default=list(Recipe.NAMES))
    ap.add_argument("--data", default=None)
    ap.add_argument("--mb", type=float, default=8.0)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=128)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sr-grad", action="store_true", help="stochastic rounding for gradients")
    ap.add_argument("--rht", action="store_true", help="random Hadamard transform for wgrad")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(a.out, exist_ok=True)
    data = load_text(a.data, a.mb)
    n_val = len(data) // 20
    train_data, val_data = data[:-n_val], data[-n_val:]
    path = os.path.join(a.out, f"train{a.tag}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {(r["recipe"], r["sr_grad"], r["rht"]) for r in map(json.loads, open(path))}
    for name in a.recipes:
        if (name, a.sr_grad, a.rht) in done:
            continue
        res = train(name, a, train_data, val_data, a.device)
        res["config"] = {k: v for k, v in vars(a).items() if k != "recipes"}
        with open(path, "a") as f:
            f.write(json.dumps(res) + "\n")


if __name__ == "__main__":
    main()
