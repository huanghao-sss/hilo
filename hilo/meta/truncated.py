"""Online ES on truncated unrolls: PES (Vicol et al., 2021) and NRES (Li et al., 2023).

The inner problems are never restarted for an outer step: every particle keeps its training run and advances
it by `trunc` steps per outer iteration, so the HiloNet is updated horizon / trunc times per training run
instead of once (full-unroll ES). Particles are antithetic pairs on a shared task, as in full ES. Both
estimators are unbiased for the Gaussian-smoothed sum of segment losses over the whole run:

    pes:  fresh noise eps_k every segment, applied as +eps_k / -eps_k; the direction is the accumulated
          xi = sum_k eps_k since the start of the run (a perturbation also moves the later segments).
    nres: one noise eps per run, reused for every segment; the direction is eps. Lower variance than PES.

    d = mean_i z_i * (xi_i | eps_i),   z = shaping(L+ - L-)   (hilo.meta.core)

A run ends after its horizon (fixed, or log-uniform between two bounds per run) or on divergence, and the
pair starts a new task. With stagger, the first run of each pair ends after a random number of segments, so
the pairs are spread over the horizon instead of all reaching the same phase together.

The task: init(task_seed, horizon) -> state starts a training run of `horizon` steps (Hilo's total_steps);
unroll(net, state, steps) -> (state, loss) trains `steps` more steps with `net` and returns the segment's
meta-loss (the sum or mean of the per-step training losses). State persists in this process (serial) or in
the worker owning the particle (parallel); resuming from out/state.pt starts new runs.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Callable, Protocol

import torch

from ..net import HiloNet, es_sigma
from .core import Outer, minimize, penalize, resume, shape
from .es import _finish, _saver
from .pool import Job, Unroller, UnrollPool

ESTIMATORS = ("pes", "nres")


class TruncatedTask(Protocol):
    def init(self, task_seed: int, horizon: int) -> Any: ...

    def unroll(self, net: HiloNet, state: Any, steps: int) -> tuple[Any, float]: ...


class TruncatedES:
    """unroll([(particle, theta, steps, reset), ...]) -> segment losses; reset = (task_seed, horizon) starts a
    new run for that particle first. Particle 2i is the + member of pair i, 2i + 1 the - member."""

    def __init__(self, unroll: Callable[[list[Job]], list[float]], sigma: torch.Tensor, pairs: int = 8,
                 trunc: int = 50, horizon: int | tuple[int, int] = 1000, estimator: str = "nres",
                 stagger: bool = True, seed: int = 0, shaping: str = "rms", diverged_penalty: float = 0.5):
        if estimator not in ESTIMATORS:
            raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
        self.unroll, self.sigma, self.pairs, self.trunc = unroll, sigma, pairs, trunc
        self.horizon, self.estimator, self.stagger, self.seed = horizon, estimator, stagger, seed
        self.shaping, self.diverged_penalty = shaping, diverged_penalty
        self.gen = None

    def _horizon(self) -> int:
        if isinstance(self.horizon, int):
            return self.horizon
        lo, hi = self.horizon
        u = torch.rand((), generator=self.gen, dtype=torch.float64).item()
        return int(round(math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))))

    def _start(self, i: int, first: bool):
        H = self._horizon()
        end = H
        if first and self.stagger:
            end = min(H, self.trunc * int(torch.randint(1, math.ceil(H / self.trunc) + 1, (),
                                                        generator=self.gen)))
        self.task[i] = 500_000_000 + 1_000_000 * self.seed + self.runs
        self.runs += 1
        self.H[i], self.end[i], self.t[i] = H, end, 0
        self.xi[i].zero_()
        self.eps[i] = torch.randn(self.dim, generator=self.gen)

    def __call__(self, theta: torch.Tensor, it: int) -> tuple[torch.Tensor, dict]:
        P = self.pairs
        if self.gen is None:
            self.gen = torch.Generator().manual_seed(self.seed * 1_000_003 + it)
            self.dim, self.runs = theta.numel(), 0
            self.task, self.H, self.end, self.t = [0] * P, [0] * P, [0] * P, [0] * P
            self.xi, self.eps = torch.zeros(P, self.dim), torch.zeros(P, self.dim)
            for i in range(P):
                self._start(i, first=True)
        jobs, steps, used = [], [], torch.empty(P, self.dim)
        for i in range(P):
            reset = (self.task[i], self.H[i]) if self.t[i] == 0 else None
            used[i] = torch.randn(self.dim, generator=self.gen) if self.estimator == "pes" else self.eps[i]
            self.xi[i] += used[i]
            K = min(self.trunc, self.end[i] - self.t[i])
            steps.append(K)
            jobs += [(2 * i, theta + self.sigma * used[i], K, reset), (2 * i + 1, theta - self.sigma * used[i], K, reset)]
        L, finite = penalize(torch.tensor(self.unroll(jobs), dtype=torch.float64).view(P, 2), self.diverged_penalty)
        z = shape(L, self.shaping).float()
        d = (z[:, None] * (self.xi if self.estimator == "pes" else used)).mean(0)
        progress = sum(self.t[i] / self.H[i] for i in range(P)) / P
        resets = 0
        for i in range(P):
            self.t[i] += steps[i]
            if self.t[i] >= self.end[i] or not finite[i].all():
                self._start(i, first=False)
                resets += 1
        return d, {"seg_mean": L.mean().item(), "pair_absdiff": (L[:, 0] - L[:, 1]).abs().mean().item(),
                   "dead": int((~finite).sum()), "progress": progress, "resets": resets}


def meta_train_truncated(task: TruncatedTask | None = None, net: HiloNet | None = None, *,
                         horizon: int | tuple[int, int], trunc: int = 50, estimator: str = "nres",
                         stagger: bool = True, iters: int = 1000, pairs: int = 8, sigma_mult: float = 1.0,
                         alpha: float = 0.3, momentum: float = 0.9, optimizer: str = "sgdm",
                         shaping: str = "rms", seed: int = 0, out: str | Path | None = None, save_every: int = 100,
                         make_task: Callable[[str], TruncatedTask] | None = None, devices: list[str] | None = None,
                         net_options: dict | None = None) -> HiloNet:
    """Meta-train a HiloNet with NRES (default) or PES on truncated unrolls (see the module docstring).

    Serial: `task` runs in this process. Parallel: make_task(device) builds one task in each worker process
        of `devices`; particle p (2 per pair) lives in worker p % len(devices).
    horizon: training-run length, or (min, max) for a log-uniform length per run. trunc: steps per segment
        (one outer step per segment). Other arguments as in hilo.meta.meta_train.
    """
    assert (task is None) != (make_task is None), "pass task or make_task (+ devices)"
    net = net or HiloNet()
    cfg, net_options = net.cfg, net_options or {}
    unroll = Unroller(task, cfg, net_options) if task is not None else UnrollPool(
        make_task, devices or ["cuda:0"], cfg, net_options)
    sigma = es_sigma(cfg) * sigma_mult
    outer = Outer(optimizer, sigma.numel(), alpha, momentum)
    theta0, theta, it0 = resume(out, net.flat().float(), outer)
    est = TruncatedES(unroll, sigma, pairs, trunc, horizon, estimator, stagger, seed, shaping)
    try:
        theta = minimize(est, theta0, sigma, iters, outer, out, _saver(cfg, out, save_every), it0, theta)
    finally:
        if isinstance(unroll, UnrollPool):
            unroll.close()
    return _finish(cfg, theta, out, f"hilo.meta {estimator} iters={iters} pairs={pairs} trunc={trunc}")
