"""ES (evolution strategies) on the loss at the end of training.

Each iteration evaluates `pairs` antithetic pairs phi +/- sigma * eps. Both members of a pair train on the
same task (same init seed and data order: common random numbers), so their difference measures the
perturbation and not the task. Only the final loss enters; the cost is 2 * pairs full training runs per outer
step. Hilo.meta.train runs it.
"""
from __future__ import annotations

from typing import Callable

import torch

from .core import penalize, shape

Evaluate = Callable[[list[tuple[torch.Tensor, int]]], list[float]]


class ES:
    """d = mean_i z_i eps_i over antithetic pairs; evaluate([(theta, task_seed), ...]) -> final losses."""

    def __init__(self, evaluate: Evaluate, sigma: torch.Tensor, pairs: int = 8, seed: int = 0,
                 shaping: str = "rms", diverged_penalty: float = 0.5):
        self.evaluate, self.sigma, self.pairs, self.seed = evaluate, sigma, pairs, seed
        self.shaping, self.diverged_penalty = shaping, diverged_penalty

    def __call__(self, theta: torch.Tensor, it: int) -> tuple[torch.Tensor, dict]:
        gen = torch.Generator().manual_seed(self.seed * 1_000_003 + it)
        eps = torch.randn(self.pairs, theta.numel(), generator=gen)
        jobs = [(theta + sgn * self.sigma * eps[i], 10_000_000 * (self.seed + 1) + 1000 * it + i)
                for i in range(self.pairs) for sgn in (1, -1)]
        L, finite = penalize(torch.tensor(self.evaluate(jobs), dtype=torch.float64).view(self.pairs, 2),
                             self.diverged_penalty)
        z = shape(L, self.shaping).float()
        return (z[:, None] * eps).mean(0), {"pair_mean": L.mean().item(),
                                            "pair_absdiff": (L[:, 0] - L[:, 1]).abs().mean().item(),
                                            "dead": int((~finite).sum())}

