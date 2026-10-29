"""Full-unroll antithetic ES on the loss at the end of training.

Each iteration evaluates `pairs` antithetic pairs phi +/- sigma * eps. Both members of a pair train on the
same task (same init seed and data order: common random numbers), so their difference measures the
perturbation and not the task. Only the final loss enters, so there is no truncation bias however long the
inner run is; the cost is 2 * pairs full training runs per outer step.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch

from ..net import HiloNet, es_sigma
from .core import Outer, minimize, penalize, resume, shape
from .pool import EvalPool

Evaluate = Callable[[list[tuple[torch.Tensor, int]]], list[float]]


class FullES:
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


def meta_train(objective: Callable[[HiloNet, int], float] | None = None, net: HiloNet | None = None,
               iters: int = 20, pairs: int = 8, sigma_mult: float = 1.0, alpha: float = 1.0, momentum: float = 0.9,
               optimizer: str = "sgdm", shaping: str = "rms", seed: int = 0, out: str | Path | None = None,
               save_every: int = 10, make_objective: Callable[[str], Callable[[HiloNet, int], float]] | None = None,
               devices: list[str] | None = None, net_options: dict | None = None) -> HiloNet:
    """Meta-train a HiloNet (default: random init with the default config) with full-unroll ES.

    Serial: objective(net, task_seed) -> final loss, evaluated in this process.
    Parallel: make_objective(device) -> objective, built once in each worker process of `devices`
        (e.g. ["cuda:0", "cuda:1"]).
    optimizer / shaping / alpha / momentum: see hilo.meta.core (alpha is in units of sigma).
    net_options: HiloNet execution options for the evaluated nets (precision, compile, max_coords).
    Returns the trained net; with `out`, writes log.jsonl, a resumable state.pt, out/hilo_it{k}.pt every
    `save_every` iterations and out/hilo_final.pt.
    """
    assert (objective is None) != (make_objective is None), "pass objective or make_objective (+ devices)"
    net = net or HiloNet()
    cfg, net_options = net.cfg, net_options or {}
    if objective is not None:
        def evaluate(jobs):
            return [objective(HiloNet(cfg, phi=theta, **net_options), task) for theta, task in jobs]
    else:
        evaluate = EvalPool(make_objective, devices or ["cuda:0"], cfg, net_options)
    sigma = es_sigma(cfg) * sigma_mult
    outer = Outer(optimizer, sigma.numel(), alpha, momentum)
    theta0, theta, it0 = resume(out, net.flat().float(), outer)
    try:
        theta = minimize(FullES(evaluate, sigma, pairs, seed, shaping), theta0, sigma, iters, outer, out,
                         _saver(cfg, out, save_every), it0, theta)
    finally:
        if isinstance(evaluate, EvalPool):
            evaluate.close()
    return _finish(cfg, theta, out, f"hilo.meta ES iters={iters} pairs={pairs}")


def _saver(cfg, out, save_every):
    def save(it, theta):
        if out and (it + 1) % save_every == 0:
            HiloNet(cfg, phi=theta).save(Path(out) / f"hilo_it{it + 1:04d}.pt", note=f"hilo.meta it={it + 1}")
    return save


def _finish(cfg, theta, out, note):
    trained = HiloNet(cfg, phi=theta)
    if out:
        trained.save(Path(out) / "hilo_final.pt", note=note)
    return trained
