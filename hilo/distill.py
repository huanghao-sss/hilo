"""Initialize a HiloNet by distilling a hand-designed update rule: an optional starting point for meta-training.

    net = hilo.init_net("muon")                    # default config, approximates Muon's update direction
    net = hilo.distill.muon(cfg, steps=4000, device="cuda", out="hilo_muon.pt")
    python -m hilo.distill --out hilo_muon.pt [--config '{"tie": 6}'] [--steps 4000]

The HiloNet is regressed onto teacher(G, ema) on synthetic gradient statistics: per matrix a signal S
(random singular vectors, power-law spectrum, random row / column scales) observed through i.i.d. Gaussian
noise of random strength, g_t = S + noise_t. G, the gradient EMAs of the HiloNet (cfg.betas) and the EMAs the
teacher needs are drawn from their exact stationary joint distribution (EMA_a and EMA_b of the noise have
covariance (1-a)(1-b) / (1-ab)). Weights W and progress t/T are random; the teacher ignores them.

Targets have the output norm of the net (RMS 1 for output_norm="none"), so `lr` of Hilo is the RMS of a
step. Muon scales its orthogonalized update by sqrt(max(1, m/n)), an RMS of about 1/sqrt(n) for an m x n
weight (n = input dimension), so a Hilo lr of about muon_lr / sqrt(n) matches its step size.
"""
from __future__ import annotations

import argparse
import json
import math
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .net import HiloNet, HiloNetConfig, features, head_weights, hilo_net
from .norms import normalize

Teacher = Callable[[Tensor, dict], Tensor]  # (G [B, m, n], {beta: EMA [B, m, n]}) -> update direction
SHAPES = ((64, 64), (128, 128), (256, 256), (128, 32), (32, 128), (256, 64), (64, 256), (192, 96), (96, 192))
NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(X: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Muon's quintic Newton-Schulz orthogonalization of a batch [B, m, n] (float32)."""
    a, b, c = NS_COEFFS
    tall = X.size(-2) > X.size(-1)
    if tall:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for _ in range(steps):
        A = X @ X.mT
        X = a * X + (b * A + c * A @ A) @ X
    return X.mT if tall else X


def muon_teacher(momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5) -> tuple[Teacher, tuple]:
    """Muon's direction: orthogonalized momentum (Nesterov: lerp(G, EMA, momentum)); returns (teacher, betas)."""
    def teacher(G, ema):
        u = G.lerp(ema[momentum], momentum) if nesterov else ema[momentum]
        return newton_schulz(u, ns_steps)
    return teacher, (momentum,)


def sample(B: int, m: int, n: int, betas: tuple, gen: torch.Generator, device) -> tuple[Tensor, dict, Tensor, Tensor]:
    """Synthetic (G, {beta: EMA}, W, progress) for B matrices of shape m x n (see the module docstring)."""
    def U(lo, hi, *shape):
        return lo + (hi - lo) * torch.rand(*shape, generator=gen, device=device)

    def randn(*shape):
        return torch.randn(*shape, generator=gen, device=device)

    r = min(m, n)
    Qm, Qn = torch.linalg.qr(randn(B, m, r))[0], torch.linalg.qr(randn(B, n, r))[0]
    s = torch.arange(1, r + 1, device=device) ** -U(0.0, 2.0, B, 1)
    rows = torch.exp(U(0.0, 1.0, B, 1, 1) * randn(B, m, 1))
    cols = torch.exp(U(0.0, 1.0, B, 1, 1) * randn(B, 1, n))
    S = Qm * s[:, None, :] @ Qn.mT
    S = S / S.square().mean(dim=(1, 2), keepdim=True).sqrt()
    sigma = torch.exp(U(math.log(0.1), math.log(10.0), B, 1, 1))
    bs = torch.tensor((0.0,) + tuple(betas), dtype=torch.float64)
    cov = (1 - bs[:, None]) * (1 - bs[None, :]) / (1 - bs[:, None] * bs[None, :])
    L = torch.linalg.cholesky(cov + 1e-12 * torch.eye(len(bs), dtype=torch.float64)).float().to(device)
    noise = randn(B, m, n, len(bs)) @ L.T
    chan = (S[..., None] + sigma[..., None] * noise) * (rows * cols)[..., None]
    W = randn(B, m, n) * torch.exp(U(math.log(0.01), math.log(3.0), B, 1, 1))
    return chan[..., 0], {b: chan[..., i + 1] for i, b in enumerate(betas)}, W, U(0.0, 1.0, ())


def _target_norm(cfg: HiloNetConfig) -> str:
    return "rms" if cfg.output_norm == "none" else cfg.output_norm


def _forward(cfg: HiloNetConfig, phi: dict, G: Tensor, M: Tensor, W: Tensor, progress: float) -> Tensor:
    chan, glob = features(G, M, W, progress, cfg.clip)
    X = hilo_net(chan, glob, phi, head_weights(phi, cfg.heads), iters=cfg.iters, tie=bool(cfg.tie),
                 heads=cfg.heads, attn=cfg.attn)
    return normalize(X, cfg.output_norm)


def _batch(cfg, teacher, extra, shape, B, gen, device):
    betas = tuple(dict.fromkeys(cfg.betas + tuple(extra)))
    G, ema, W, progress = sample(B, *shape, betas, gen, device)
    M = torch.stack([ema[b] for b in cfg.betas], -1)
    return G, M, W, progress.item(), normalize(teacher(G, ema), _target_norm(cfg))


def cosine(U: Tensor, T: Tensor) -> Tensor:
    return F.cosine_similarity(U.flatten(1), T.flatten(1), dim=1)


def distill(net: HiloNet, teacher: Teacher, extra_betas: tuple = (), steps: int = 3000, batch: int = 8,
            shapes: tuple = SHAPES, lr: float = 1e-3, seed: int = 0, device: str | None = None,
            log_every: int = 250) -> HiloNet:
    """Adam on the relative squared error between the HiloNet's update and the teacher's (both normalized to
    the net's output norm), one random shape per step; returns the distilled HiloNet."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = net.cfg
    phi = {k: v.detach().to(device).clone().requires_grad_() for k, v in net.phi.items()}
    opt = torch.optim.Adam(phi.values(), lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.5 * (1 + math.cos(math.pi * i / steps)))
    gen = torch.Generator(device=device).manual_seed(seed)
    tf32, torch.backends.cuda.matmul.allow_tf32 = torch.backends.cuda.matmul.allow_tf32, True
    try:
        _fit(cfg, phi, opt, sched, teacher, extra_betas, steps, batch, shapes, gen, device, log_every)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return HiloNet(cfg, phi=torch.cat([v.detach().reshape(-1).cpu() for v in phi.values()]))


def _fit(cfg, phi, opt, sched, teacher, extra_betas, steps, batch, shapes, gen, device, log_every):
    for it in range(steps):
        shape = shapes[int(torch.randint(len(shapes), (), generator=gen, device=device))]
        G, M, W, progress, T = _batch(cfg, teacher, extra_betas, shape, batch, gen, device)
        U = _forward(cfg, phi, G, M, W, progress)
        loss = (U - T).square().sum(dim=(1, 2)).div(T.square().sum(dim=(1, 2))).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if log_every and (it % log_every == 0 or it == steps - 1):
            print(json.dumps({"it": it, "shape": shape, "loss": loss.item(),
                              "cosine": cosine(U, T).mean().item()}), flush=True)


@torch.no_grad()
def evaluate(net: HiloNet, teacher: Teacher, extra_betas: tuple = (), shapes: tuple = SHAPES, batch: int = 8,
             seed: int = 1, device: str | None = None) -> dict:
    """Mean cosine similarity between the HiloNet's and the teacher's update per shape, on fresh samples."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    gen = torch.Generator(device=device).manual_seed(seed)
    out = {}
    for shape in shapes:
        G, M, W, progress, T = _batch(net.cfg, teacher, extra_betas, shape, batch, gen, device)
        out[f"{shape[0]}x{shape[1]}"] = cosine(net(G, M, W, progress), T).mean().item()
    return out


def muon(cfg: HiloNetConfig | None = None, momentum: float = 0.95, nesterov: bool = True, seed: int = 0,
         out: str | None = None, **kwargs) -> HiloNet:
    """A HiloNet initialized from the random init of `seed` and distilled toward Muon's update direction.
    kwargs: see distill. With `out`, saves the net there."""
    teacher, extra = muon_teacher(momentum, nesterov)
    net = distill(HiloNet(cfg, seed=seed), teacher, extra, seed=seed, **kwargs)
    if out:
        net.save(out, note=f"hilo.distill.muon momentum={momentum} nesterov={nesterov}")
    return net


def main():
    ap = argparse.ArgumentParser(description="Distill a HiloNet toward Muon's update direction.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", help="HiloNetConfig as JSON")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--momentum", type=float, default=0.95)
    ap.add_argument("--no-nesterov", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    a = ap.parse_args()
    cfg = HiloNetConfig(**json.loads(a.config)) if a.config else None
    net = muon(cfg, a.momentum, not a.no_nesterov, a.seed, a.out, steps=a.steps, batch=a.batch, lr=a.lr,
               device=a.device)
    teacher, extra = muon_teacher(a.momentum, not a.no_nesterov)
    print(json.dumps({"cosine": evaluate(net, teacher, extra, device=a.device)}))


if __name__ == "__main__":
    main()
