"""Meta-training.

An estimator maps the current parameters and the iteration to a meta-gradient in parameter space.
The objective that has a descent direction after a long unroll is the Gaussian-smoothed family loss
``E[L(phi + sigma * eps)]``. ES estimates it from final losses.
Any ``torch.optim`` optimizer consumes that gradient from ``theta.grad``.

    theta = torch.nn.Parameter(net.flat())
    estimator = Hilo.meta.ES(evaluate, sigma, pairs=8)
    optimizer = torch.optim.SGD([theta], lr=1e-3)
    Hilo.meta.minimize(estimator, optimizer, theta, iters=100, sigma=sigma, out="runs/exp")

``Hilo.meta.train`` builds this loop with ES and ``torch.optim.SGD`` (no momentum). Pass another
``torch.optim.Optimizer`` class or a ``params -> Optimizer`` callable to use a different one.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
from torch import nn
from torch.optim import Optimizer

from ..hilonet import HiloNet, n_params
from .es import ES, SHAPINGS, es_sigma, shape
from .loop import Estimator, minimize, resume
from .pool import EvalPool

__all__ = ["train", "minimize", "Estimator", "ES", "es_sigma", "SHAPINGS", "shape", "EvalPool"]


def _sigma(sigma: torch.Tensor | float, n: int) -> torch.Tensor:
    scale = torch.as_tensor(sigma, dtype=torch.float32).reshape(-1)
    if scale.numel() == 1:
        return scale.expand(n).contiguous()
    if scale.numel() != n:
        raise ValueError(f"sigma has {scale.numel()} elements, HiloNet has {n}")
    return scale


def _make_optimizer(optimizer: Callable[[list[nn.Parameter]], Optimizer] | type[Optimizer] | None,
                    theta: nn.Parameter, lr: float) -> Optimizer:
    params = [theta]
    if optimizer is None:
        return torch.optim.SGD(params, lr=lr)
    if isinstance(optimizer, type) and issubclass(optimizer, Optimizer):
        return optimizer(params, lr=lr)
    return optimizer(params)


def train(objective: Callable[[HiloNet, int], float] | None = None, net: HiloNet | None = None, *,
          sigma: torch.Tensor | float, iters: int = 100, pairs: int = 8, alpha: float = 1e-3,
          optimizer: Callable[[list[nn.Parameter]], Optimizer] | type[Optimizer] | None = None,
          shaping: str = "rms", seed: int = 0, out: str | Path | None = None,
          save_every: int = 50, make_task: Callable[[str], Callable[[HiloNet, int], float]] | None = None,
          devices: list[str] | None = None, net_options: dict | None = None) -> HiloNet:
    """Meta-train a HiloNet (default: Hilo.net(), the random init with the default config) with ES.

    objective(net, task_seed) -> the validation loss at the end of one training run. You write this function.
    Pass `net` to Hilo.optimizer, and seed the model and its data order with task_seed. The same net and
    task_seed must return the same loss.
    Serial: `objective` runs in this process. Parallel: make_task(device) builds one such function in each
    worker process of `devices` (e.g. ["cuda:0", "cuda:1"]).
    sigma: ES perturbation, a float (same on every parameter) or a tensor of length `n_params`.
    `Hilo.meta.es_sigma(cfg, matrix, vector)` builds the per-parameter tensor.
    alpha: learning rate passed to the optimizer class (default 1e-3). For SGD it counts steps in units of sigma.
    optimizer: a torch.optim.Optimizer subclass, constructed as `optimizer([theta], lr=alpha)`, or a callable
    `params -> Optimizer`. Default: torch.optim.SGD with no momentum.
    shaping is the fitness shaping of ES.
    A different estimator uses Hilo.meta.minimize.
    net_options: execution options of the evaluated nets (precision, compile, max_coords).
    Returns the trained net; with `out`, writes log.jsonl, a resumable state.pt, out/hilo_it{k}.pt every
    `save_every` iterations and out/hilo_final.pt.
    """
    if (objective is None) == (make_task is None):
        raise ValueError("pass objective (serial) or make_task + devices (parallel)")
    net = net or HiloNet()
    cfg, net_options = net.cfg, net_options or {}
    sigma = _sigma(sigma, n_params(cfg))
    theta = nn.Parameter(net.flat().float())
    opt = _make_optimizer(optimizer, theta, alpha)
    theta0, it0 = resume(out, theta, opt, sigma)
    if objective is not None:
        def run(jobs):
            return [objective(HiloNet(cfg, phi=phi, **net_options), task_seed) for phi, task_seed in jobs]
    else:
        run = EvalPool(make_task, devices or ["cuda:0"], cfg, net_options)
    estimator = ES(run, sigma, pairs, seed, shaping)

    def save(it, phi):
        if out and (it + 1) % save_every == 0:
            HiloNet(cfg, phi=phi).save(Path(out) / f"hilo_it{it + 1:04d}.pt", note=f"Hilo.meta it={it + 1}")

    try:
        phi = minimize(estimator, opt, theta, iters, out, save, it0, theta0, sigma)
    finally:
        if isinstance(run, EvalPool):
            run.close()
    trained = HiloNet(cfg, phi=phi)
    if out:
        trained.save(Path(out) / "hilo_final.pt", note=f"Hilo.meta iters={iters} pairs={pairs}")
    return trained
