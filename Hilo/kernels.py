"""Triton kernels for the HiloNet forward on CUDA in bf16 (backend="triton").

Everything a coordinate does between two attention matmuls is one kernel: the out projection and residual of
the previous attention, the MLP at the end of a block, then the RMSNorm and the q / k / v projections of the
next attention, stored straight in the token-major layout that attention needs ([heads, B, tokens, features,
d/heads], tokens being rows or columns). The attention matmuls are batched cuBLAS calls. The function is
hilo_net with dt=bfloat16 up to rounding: the residual stream, norms and MLP are float32 (the MLP matmuls in
tf32x3, about float32 accuracy), q / k / v / scores / attention outputs bf16, and the out projection accumulates
the heads in float32.
"""
from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor

BLOCK = 128


@triton.jit
def _rmsnorm(h, D: tl.constexpr):
    return h * tl.rsqrt(tl.sum(h * h, axis=1)[:, None] / D + 1e-6)


@triton.jit
def _token_major(b, i, j, B, m, n, COL: tl.constexpr, DK: tl.constexpr, D: tl.constexpr):
    """[BLOCK, D] offsets of coordinates (b, i, j) in a [heads, B, T, F, DK] buffer: rows are the tokens
    (T = m, F = n) or, with COL, columns (T = n, F = m); entry c of a coordinate's vector is head c // DK."""
    c = tl.arange(0, D)
    tok = (b * n + j) * m + i if COL else (b * m + i) * n + j
    return (c // DK)[None, :] * (B * m * n * DK) + tok[:, None] * DK + (c % DK)[None, :]


@triton.jit
def _coord_kernel(H, CH, GB, O, Q, K, V, X, E, WO, M1, MB1, M2, MB2, WQ, WK, WV, WOUT,
                  B, m, n, C,
                  D: tl.constexpr, HID: tl.constexpr, DK: tl.constexpr, CP: tl.constexpr,
                  EMBED: tl.constexpr, O_COL: tl.constexpr, MLP: tl.constexpr, QKV: tl.constexpr,
                  QKV_COL: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < B * m * n)[:, None]
    b = offs // (m * n)
    i = (offs % (m * n)) // n
    j = offs % n
    d = tl.arange(0, D)
    hp = H + offs[:, None] * D + d[None, :]
    if EMBED:
        c = tl.arange(0, CP)
        ch = tl.load(CH + offs[:, None] * C + c[None, :], mask=mask & (c[None, :] < C), other=0.0)
        e = tl.load(E + c[:, None] * D + d[None, :], mask=c[:, None] < C, other=0.0)
        h = tl.dot(ch, e, input_precision="ieee") + tl.load(GB + b[:, None] * D + d[None, :], mask=mask, other=0.0)
    else:
        h = tl.load(hp, mask=mask, other=0.0)
        o = tl.load(O + _token_major(b, i, j, B, m, n, O_COL, DK, D), mask=mask, other=0.0)
        h += tl.dot(o, tl.load(WO + d[:, None] * D + d[None, :]))
    if MLP:
        k = tl.arange(0, HID)
        z = tl.dot(_rmsnorm(h, D), tl.load(M1 + d[:, None] * HID + k[None, :]), input_precision="tf32x3")
        z += tl.load(MB1 + k)[None, :]
        z = z * tl.sigmoid(z)
        h += tl.dot(z, tl.load(M2 + k[:, None] * D + d[None, :]), input_precision="tf32x3")
        h += tl.load(MB2 + d)[None, :]
    if QKV:
        tl.store(hp, h, mask=mask)
        x = _rmsnorm(h, D).to(tl.bfloat16)
        po = _token_major(b, i, j, B, m, n, QKV_COL, DK, D)
        w = d[:, None] * D + d[None, :]
        tl.store(Q + po, tl.dot(x, tl.load(WQ + w)).to(tl.bfloat16), mask=mask)
        tl.store(K + po, tl.dot(x, tl.load(WK + w)).to(tl.bfloat16), mask=mask)
        tl.store(V + po, tl.dot(x, tl.load(WV + w)).to(tl.bfloat16), mask=mask)
    else:
        tl.store(X + offs, tl.sum(h * tl.load(WOUT + d)[None, :], axis=1), mask=offs < B * m * n)


def supported(cfg) -> bool:
    """The kernels need power-of-two widths of at least 16 (tl.dot tiles)."""
    def p2(x):
        return x >= 16 and x & (x - 1) == 0
    return p2(cfg.d) and p2(cfg.mlp_hidden) and cfg.d % cfg.heads == 0


def hilo_net_triton(chan: Tensor, glob: Tensor, phi: dict, *, iters: int, tie: bool, heads: int,
                    attn: str) -> Tensor:
    """hilo_net(chan, glob, phi, ..., dt=torch.bfloat16) on CUDA: chan [B, m, n, C], glob [B, n_glob] -> X [B, m, n]."""
    B, m, n, C = chan.shape
    D, HID = phi["E"].shape[1], phi["0m1"].shape[1]
    DK, dev, bf = D // heads, chan.device, torch.bfloat16
    chan = chan.contiguous()
    h = torch.empty(B, m, n, D, device=dev)
    X = torch.empty(B, m, n, device=dev)
    gb = (glob @ phi["Eg"]).contiguous()
    Q, K, V = torch.empty(3, heads * B, m * n * DK, device=dev, dtype=bf)
    O = torch.empty(heads * B, m * n * DK, device=dev, dtype=bf)
    linear = attn == "linear"
    scale = {"r": 1 / (n * math.sqrt(DK)) / (m if linear else 1), "c": 1 / (m * math.sqrt(DK)) / (n if linear else 1)}
    w = {}
    for name, v in phi.items():
        if name[-2:-1] in ("r", "c") and name[-1] in "qkvo":
            w[name] = ((v * scale[name[-2]]) if name[-1] == "q" else v).to(bf).contiguous()
    grid = (triton.cdiv(B * m * n, BLOCK),)
    CP = max(16, triton.next_power_of_2(C))

    def launch(embed=False, o_col=False, wo=None, mlp=None, qkv=None, qkv_col=False):
        p = (lambda k: phi[f"{mlp}{k}"]) if mlp is not None else (lambda k: phi["out"])
        q = (lambda k: w[f"{qkv}{k}"]) if qkv is not None else (lambda k: O)
        _coord_kernel[grid](h, chan, gb, O, Q, K, V, X, phi["E"], w[wo] if wo else O,
                            p("m1"), p("mb1"), p("m2"), p("mb2"), q("q"), q("k"), q("v"), phi["out"],
                            B, m, n, C, D=D, HID=HID, DK=DK, CP=CP, EMBED=embed, O_COL=o_col,
                            MLP=mlp is not None, QKV=qkv is not None, QKV_COL=qkv_col, BLOCK=BLOCK,
                            num_warps=4)

    def attend(T, F):
        q, k, v = (t.view(heads * B, T, F * DK) for t in (Q, K, V))
        A = torch.bmm(q, k.mT)
        if not linear:
            A = torch.softmax(A.float(), dim=-1).to(bf)
        torch.bmm(A, v, out=O.view(heads * B, T, F * DK))

    def blk(it):
        return 0 if tie else it

    launch(embed=True, qkv=f"{blk(0)}r")
    for it in range(iters):
        b, last = blk(it), it == iters - 1
        attend(m, n)
        launch(wo=f"{b}ro", qkv=f"{b}c", qkv_col=True)
        attend(n, m)
        launch(o_col=True, wo=f"{b}co", mlp=b, qkv=None if last else f"{blk(it + 1)}r")
    return X
