"""A small byte-level GPT whose projection and MLP GEMMs run through a ``Recipe``.

As in production FP8/FP4 recipes, the embedding, the LM head, LayerNorms and attention's
softmax(QKᵀ)V stay in high precision. The quantised GEMMs are the ones holding almost
all the FLOPs: QKV, attention out, MLP up and MLP down.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .linear import QLinear, Recipe


@dataclass
class GPTConfig:
    vocab: int = 256
    d: int = 128
    layers: int = 4
    heads: int = 4
    ctx: int = 128


class Block(nn.Module):
    def __init__(self, c: GPTConfig, recipe, i):
        super().__init__()
        self.h = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv = QLinear(c.d, 3 * c.d, recipe=recipe, key=f"{i}.qkv")
        self.proj = QLinear(c.d, c.d, recipe=recipe, key=f"{i}.proj")
        self.fc1 = QLinear(c.d, 4 * c.d, recipe=recipe, key=f"{i}.fc1")
        self.fc2 = QLinear(4 * c.d, c.d, recipe=recipe, key=f"{i}.fc2")

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, -1)
        q, k, v = (t.view(B, T, self.h, -1).transpose(1, 2) for t in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, T, D))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class GPT(nn.Module):
    def __init__(self, c: GPTConfig, recipe: Recipe | None = None):
        super().__init__()
        self.c = c
        self.tok = nn.Embedding(c.vocab, c.d)
        self.pos = nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c, recipe, i) for i in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, c.vocab, bias=False)
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        T = idx.shape[1]
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        for b in self.blocks:
            x = b(x)
        logits = self.head(self.ln(x))
        if targets is None:
            return logits
        return F.cross_entropy(logits.flatten(0, 1), targets.flatten())
