"""ES (evolution strategies): one meta-gradient estimator.

Each iteration evaluates `pairs` antithetic pairs phi +/- sigma * eps. Both members of a pair train on the
same task (same init seed and data order: common random numbers), so their difference measures the
perturbation and not the task. Only the final loss enters; the cost is 2 * pairs full training runs per
iteration. The returned gradient is the noise-unit estimate multiplied by sigma, so any
``torch.optim`` optimizer can take it from ``theta.grad``. Another estimator with the same call
replaces ES in ``Hilo.meta.minimize``.
"""
from __future__ import annotations

import math
from typing import Callable

import torch

from ..hilonet import HiloNetConfig, param_spec

SHAPINGS = ("rms", "rank", "none")
Evaluate = Callable[[list[tuple[torch.Tensor, int]]], list[float]]


def es_sigma(cfg: HiloNetConfig, matrix: float, vector: float) -> torch.Tensor:
    """Per-parameter ES scale: `matrix` on 2-D tensors, `vector` on 1-D."""
    return torch.cat([torch.full((math.prod(s),), float(matrix if len(s) == 2 else vector))
                      for _, s in param_spec(cfg)])


def penalize(losses: torch.Tensor, diverged_penalty: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Non-finite losses -> worst finite loss + penalty. Returns (losses, finite mask)."""
    finite = torch.isfinite(losses)
    penalty = (losses[finite].max().item() + diverged_penalty) if finite.any() else 1e3
    return torch.where(finite, losses, torch.full_like(losses, penalty)), finite


def shape(losses: torch.Tensor, mode: str) -> torch.Tensor:
    """Antithetic losses [pairs, 2] (+, -) -> per-pair weights z [pairs].
    rms: (L+ - L-) / rms over pairs (scale-free, sign and relative size kept);
    rank: difference of centered ranks of all 2 * pairs losses in [-0.5, 0.5] (OpenAI-ES fitness shaping);
    none: L+ - L- (unbiased; d = mean z * noise estimates 2 * sigma * gradient)."""
    if mode == "rms":
        diff = losses[:, 0] - losses[:, 1]
        return diff / diff.square().mean().sqrt().clamp_min(1e-8)
    if mode == "rank":
        ranks = torch.empty(losses.numel(), dtype=losses.dtype)
        ranks[losses.flatten().argsort()] = torch.arange(losses.numel(), dtype=losses.dtype)
        ranks = (ranks / max(losses.numel() - 1, 1) - 0.5).view_as(losses)
        return ranks[:, 0] - ranks[:, 1]
    if mode == "none":
        return losses[:, 0] - losses[:, 1]
    raise ValueError(f"shaping must be one of {SHAPINGS}, got {mode!r}")


class ES:
    """Meta-gradient from antithetic pairs. ``evaluate([(theta, task_seed), ...])`` -> final losses.

    ``__call__(theta, it)`` -> ``(gradient, log)``. ``gradient`` has the shape of theta and is the noise-unit
    estimate multiplied by sigma.
    """

    def __init__(self, evaluate: Evaluate, sigma: torch.Tensor, pairs: int = 8, seed: int = 0,
                 shaping: str = "rms", diverged_penalty: float = 0.5):
        self.evaluate, self.sigma, self.pairs, self.seed = evaluate, sigma, pairs, seed
        self.shaping, self.diverged_penalty = shaping, diverged_penalty

    def __call__(self, theta: torch.Tensor, it: int) -> tuple[torch.Tensor, dict]:
        generator = torch.Generator().manual_seed(self.seed * 1_000_003 + it)
        epsilon = torch.randn(self.pairs, theta.numel(), generator=generator)
        jobs = [(theta + sign * self.sigma * epsilon[i], 10_000_000 * (self.seed + 1) + 1000 * it + i)
                for i in range(self.pairs) for sign in (1, -1)]
        losses, finite = penalize(torch.tensor(self.evaluate(jobs), dtype=torch.float64).view(self.pairs, 2),
                                   self.diverged_penalty)
        weights = shape(losses, self.shaping).float()
        gradient = (weights[:, None] * epsilon).mean(0) * self.sigma
        return gradient, {"pair_mean": losses.mean().item(),
                          "pair_absdiff": (losses[:, 0] - losses[:, 1]).abs().mean().item(),
                          "dead": int((~finite).sum())}
