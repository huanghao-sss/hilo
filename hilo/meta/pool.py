"""Worker processes for meta-training, one per entry of `devices` (spawned; list a GPU several times to run
several small evaluations on it at once). The factory make_objective(device) / make_task(device) runs once
in each worker and must be picklable: a module-level function or functools.partial."""
from __future__ import annotations

import traceback
from typing import Any, Callable

import torch
import torch.multiprocessing as mp

from ..net import HiloNet, HiloNetConfig

# (particle, theta, steps, reset); reset = (task_seed, horizon) starts a new run first, None continues
Job = tuple[int, torch.Tensor, int, Any]


def _setup(device):
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))


def _eval_worker(make_objective, device, cfg, net_options, tasks, results):
    try:
        _setup(device)
        objective = make_objective(device)
    except Exception:
        results.put((-1, traceback.format_exc()))
        return
    while (job := tasks.get()) is not None:
        idx, theta, task = job
        try:
            results.put((idx, float(objective(HiloNet(cfg, phi=theta, **net_options), task))))
        except Exception:
            results.put((idx, traceback.format_exc()))


class Unroller:
    """Runs truncated-unroll jobs against a TruncatedTask, keeping each particle's inner state between calls."""

    def __init__(self, task, cfg: HiloNetConfig, net_options: dict | None = None):
        self.task, self.cfg, self.net_options, self.states = task, cfg, net_options or {}, {}

    def run(self, pid: int, theta: torch.Tensor, steps: int, reset) -> float:
        if reset is not None:
            self.states[pid] = self.task.init(*reset)
        self.states[pid], loss = self.task.unroll(HiloNet(self.cfg, phi=theta, **self.net_options),
                                                  self.states[pid], steps)
        return float(loss)

    def __call__(self, jobs: list[Job]) -> list[float]:
        return [self.run(*job) for job in jobs]


def _unroll_worker(make_task, device, cfg, net_options, tasks, results):
    try:
        _setup(device)
        unroller = Unroller(make_task(device), cfg, net_options)
    except Exception:
        results.put((-1, traceback.format_exc()))
        return
    while (job := tasks.get()) is not None:
        idx, *job = job
        try:
            results.put((idx, unroller.run(*job)))
        except Exception:
            results.put((idx, traceback.format_exc()))


class _Pool:
    def __init__(self, target, factory, devices, cfg, net_options, shared_queue: bool):
        ctx = mp.get_context("spawn")
        self.results = ctx.Queue()
        self.queues = [ctx.Queue()] if shared_queue else [ctx.Queue() for _ in devices]
        self.procs = [ctx.Process(target=target, daemon=True,
                                  args=(factory, d, cfg, net_options or {}, self.queues[i % len(self.queues)],
                                        self.results))
                      for i, d in enumerate(devices)]
        for p in self.procs:
            p.start()

    def _gather(self, n: int) -> list[Any]:
        out: list[Any] = [None] * n
        for _ in range(n):
            idx, val = self.results.get()
            if isinstance(val, str):
                self.close()
                raise RuntimeError(f"hilo.meta worker failed:\n{val}")
            out[idx] = val
        return out

    def close(self):
        for i in range(len(self.procs)):
            self.queues[i % len(self.queues)].put(None)
        for p in self.procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()


class EvalPool(_Pool):
    """Full-run evaluations [(theta, task_seed), ...] -> final losses, from a shared queue (dynamic load
    balancing)."""

    def __init__(self, make_objective: Callable[[str], Callable[[HiloNet, int], float]], devices: list[str],
                 cfg: HiloNetConfig, net_options: dict | None = None):
        super().__init__(_eval_worker, make_objective, devices, cfg, net_options, shared_queue=True)

    def __call__(self, jobs: list[tuple[torch.Tensor, int]]) -> list[float]:
        for i, (theta, task) in enumerate(jobs):
            self.queues[0].put((i, theta.cpu(), task))
        return self._gather(len(jobs))


class UnrollPool(_Pool):
    """Truncated unrolls [(particle, theta, steps, reset), ...] -> segment losses. A particle's inner state
    lives in one worker (particle % len(devices)), so its jobs always go there."""

    def __init__(self, make_task: Callable[[str], Any], devices: list[str], cfg: HiloNetConfig,
                 net_options: dict | None = None):
        super().__init__(_unroll_worker, make_task, devices, cfg, net_options, shared_queue=False)

    def __call__(self, jobs: list[Job]) -> list[float]:
        for i, (pid, theta, steps, reset) in enumerate(jobs):
            self.queues[pid % len(self.queues)].put((i, pid, theta.cpu(), steps, reset))
        return self._gather(len(jobs))
