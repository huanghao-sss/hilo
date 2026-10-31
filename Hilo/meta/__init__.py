"""Meta-training of the HiloNet with ES (evolution strategies; no backpropagation through training).

    Hilo.meta.train(objective, net)

The outer optimizer (sgdm | adam), fitness shaping (rms | rank | none) and worker processes (make_task +
devices) are options of train. The building blocks (ES, Outer, minimize, EvalPool) compose for other loops.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from ..hilonet import HiloNet, es_sigma
from .core import OPTIMIZERS, SHAPINGS, Outer, minimize, resume, shape
from .es import ES
from .pool import EvalPool

__all__ = ["train", "OPTIMIZERS", "SHAPINGS", "ES", "Outer", "minimize", "shape", "EvalPool"]


def train(objective: Callable[[HiloNet, int], float] | None = None, net: HiloNet | None = None, *,
          iters: int = 100, pairs: int = 8, sigma_mult: float = 1.0, alpha: float = 1.0, momentum: float = 0.9,
          optimizer: str = "sgdm", shaping: str = "rms", seed: int = 0, out: str | Path | None = None,
          save_every: int = 50, make_task: Callable[[str], Callable[[HiloNet, int], float]] | None = None,
          devices: list[str] | None = None, net_options: dict | None = None) -> HiloNet:
    """Meta-train a HiloNet (default: Hilo.net(), the random init with the default config) with ES (evolution strategies).

    objective(net, task_seed) -> the validation loss at the end of one training run. You write this function.
    Pass `net` to Hilo.optimizer, and seed the model and its data order with task_seed. The same net and
    task_seed must return the same loss.
    Serial: `objective` runs in this process. Parallel: make_task(device) builds one such function in each
    worker process of `devices` (e.g. ["cuda:0", "cuda:1"]).
    alpha: outer step in units of sigma.
    optimizer / shaping / momentum: see Hilo.meta.core.
    net_options: execution options of the evaluated nets (precision, compile, max_coords).
    Returns the trained net; with `out`, writes log.jsonl, a resumable state.pt, out/hilo_it{k}.pt every
    `save_every` iterations and out/hilo_final.pt.
    """
    if (objective is None) == (make_task is None):
        raise ValueError("pass objective (serial) or make_task + devices (parallel)")
    net = net or HiloNet()
    cfg, net_options = net.cfg, net_options or {}
    sigma = es_sigma(cfg) * sigma_mult
    outer = Outer(optimizer, sigma.numel(), alpha, momentum)
    theta0, theta, it0 = resume(out, net.flat().float(), outer)
    if objective is not None:
        def run(jobs):
            return [objective(HiloNet(cfg, phi=theta, **net_options), task_seed) for theta, task_seed in jobs]
    else:
        run = EvalPool(make_task, devices or ["cuda:0"], cfg, net_options)
    est = ES(run, sigma, pairs, seed, shaping)

    def save(it, theta):
        if out and (it + 1) % save_every == 0:
            HiloNet(cfg, phi=theta).save(Path(out) / f"hilo_it{it + 1:04d}.pt", note=f"Hilo.meta it={it + 1}")

    try:
        theta = minimize(est, theta0, sigma, iters, outer, out, save, it0, theta)
    finally:
        if isinstance(run, EvalPool):
            run.close()
    trained = HiloNet(cfg, phi=theta)
    if out:
        trained.save(Path(out) / "hilo_final.pt", note=f"Hilo.meta es iters={iters} pairs={pairs}")
    return trained
