"""The outer loop shared by every estimator: fitness shaping, the outer optimizer and logging.

An estimator maps theta to a search direction d in noise units (d ~ sigma * grad / 2 for shaping="none");
the outer optimizer steps theta -= alpha * sigma * step(d), so alpha is measured in units of sigma whatever
the optimizer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

import torch

SHAPINGS = ("rms", "rank", "none")
OPTIMIZERS = ("sgdm", "adam")


class Estimator(Protocol):
    def __call__(self, theta: torch.Tensor, it: int) -> tuple[torch.Tensor, dict]: ...


def penalize(L: torch.Tensor, diverged_penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-finite losses -> worst finite loss + penalty. Returns (losses, finite mask)."""
    finite = torch.isfinite(L)
    pen = (L[finite].max().item() + diverged_penalty) if finite.any() else 1e3
    return torch.where(finite, L, torch.full_like(L, pen)), finite


def shape(L: torch.Tensor, mode: str) -> torch.Tensor:
    """Antithetic losses [pairs, 2] (+, -) -> per-pair weights z [pairs].
    rms: (L+ - L-) / rms over pairs (scale-free, sign and relative size kept);
    rank: difference of centered ranks of all 2 * pairs losses in [-0.5, 0.5] (OpenAI-ES fitness shaping);
    none: L+ - L- (unbiased; d = mean z * noise estimates 2 * sigma * gradient)."""
    if mode == "rms":
        diff = L[:, 0] - L[:, 1]
        return diff / diff.square().mean().sqrt().clamp_min(1e-8)
    if mode == "rank":
        r = torch.empty(L.numel(), dtype=L.dtype)
        r[L.flatten().argsort()] = torch.arange(L.numel(), dtype=L.dtype)
        r = (r / max(L.numel() - 1, 1) - 0.5).view_as(L)
        return r[:, 0] - r[:, 1]
    if mode == "none":
        return L[:, 0] - L[:, 1]
    raise ValueError(f"shaping must be one of {SHAPINGS}, got {mode!r}")


class Outer:
    """sgdm: mom = beta mom + (1 - beta) d, step = mom.
    adam: bias-corrected m / (sqrt(v) + 1e-8) with beta1 = momentum, beta2 = 0.999."""

    def __init__(self, kind: str, dim: int, alpha: float, momentum: float):
        if kind not in OPTIMIZERS:
            raise ValueError(f"optimizer must be one of {OPTIMIZERS}, got {kind!r}")
        self.kind, self.alpha, self.momentum = kind, alpha, momentum
        self.state = {"t": 0, "m": torch.zeros(dim)} | ({"v": torch.zeros(dim)} if kind == "adam" else {})

    def step(self, theta: torch.Tensor, d: torch.Tensor, sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        st, b1 = self.state, self.momentum
        st["t"] += 1
        st["m"] = b1 * st["m"] + (1 - b1) * d
        if self.kind == "sgdm":
            step = st["m"]
        else:
            st["v"] = 0.999 * st["v"] + 0.001 * d.square()
            step = (st["m"] / (1 - b1 ** st["t"])) / ((st["v"] / (1 - 0.999 ** st["t"])).sqrt() + 1e-8)
        delta = self.alpha * sigma * step
        return theta - delta, delta


def minimize(estimator: Estimator, theta0: torch.Tensor, sigma: torch.Tensor, iters: int, outer: Outer,
             out: str | Path | None = None, on_iter: Callable[[int, torch.Tensor], None] | None = None,
             it0: int = 0, theta: torch.Tensor | None = None) -> torch.Tensor:
    """theta <- outer step on estimator(theta, it) for it in [it0, iters). out: log.jsonl + resumable state.pt."""
    theta = (theta0 if theta is None else theta).clone().float()
    out = Path(out) if out else None
    for it in range(it0, iters):
        d, rec = estimator(theta, it)
        theta, delta = outer.step(theta, d, sigma)
        rec = {"it": it} | rec | {"step_norm": delta.norm().item(), "drift": ((theta - theta0) / sigma).norm().item()}
        print(json.dumps(rec), flush=True)
        if out:
            with open(out / "log.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
            torch.save({"theta": theta, "theta0": theta0, "outer": outer.state, "it": it + 1}, out / "state.pt")
        if on_iter:
            on_iter(it, theta)
    return theta


def resume(out: str | Path | None, theta0: torch.Tensor, outer: Outer) -> tuple[torch.Tensor, torch.Tensor, int]:
    """(theta0, theta, first iteration) from out/state.pt if present."""
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        if (Path(out) / "state.pt").exists():
            st = torch.load(Path(out) / "state.pt")
            outer.state = st["outer"]
            return st["theta0"], st["theta"], st["it"]
    return theta0, theta0, 0
