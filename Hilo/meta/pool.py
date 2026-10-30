"""Worker processes for meta-training, one per entry of `devices` (spawned; list a GPU several times to run
several small evaluations on it at once). The factory make_task(device) runs once in each
worker and must be picklable: a module-level function or functools.partial."""
from __future__ import annotations

import queue
import traceback
from typing import Callable

import torch
import torch.multiprocessing as mp

from ..hilonet import HiloNet, HiloNetConfig


def _eval_worker(make_task, device, cfg, net_options, tasks, results):
    try:
        if str(device).startswith("cuda"):
            torch.cuda.set_device(torch.device(device))
        objective = make_task(device)
    except Exception:
        results.put((-1, traceback.format_exc()))
        return
    while (job := tasks.get()) is not None:
        idx, theta, task = job
        try:
            results.put((idx, float(objective(HiloNet(cfg, phi=theta, **net_options), task))))
        except Exception:
            results.put((idx, traceback.format_exc()))


class EvalPool:
    """Full-run evaluations [(theta, task_seed), ...] -> final losses, from a shared queue (dynamic load
    balancing)."""

    def __init__(self, make_task: Callable[[str], Callable[[HiloNet, int], float]], devices: list[str],
                 cfg: HiloNetConfig, net_options: dict | None = None):
        ctx = mp.get_context("spawn")
        self.tasks, self.results = ctx.Queue(), ctx.Queue()
        self.procs = [ctx.Process(target=_eval_worker, daemon=True,
                                  args=(make_task, d, cfg, net_options or {}, self.tasks, self.results))
                      for d in devices]
        for p in self.procs:
            p.start()

    def __call__(self, jobs: list[tuple[torch.Tensor, int]]) -> list[float]:
        for i, (theta, task) in enumerate(jobs):
            self.tasks.put((i, theta.cpu(), task))
        out: list[float] = [0.0] * len(jobs)
        for _ in jobs:
            while True:
                try:
                    idx, val = self.results.get(timeout=5)
                    break
                except queue.Empty:
                    dead = [p for p in self.procs if not p.is_alive()]
                    if dead:
                        self.close()
                        raise RuntimeError(f"Hilo.meta worker exited with code {dead[0].exitcode} (a script that "
                                           "starts workers needs an `if __name__ == '__main__':` guard)")
            if isinstance(val, str):
                self.close()
                raise RuntimeError(f"Hilo.meta worker failed:\n{val}")
            out[idx] = val
        return out

    def close(self):
        for _ in self.procs:
            self.tasks.put(None)
        for p in self.procs:
            p.join(timeout=60)
            if p.is_alive():
                p.terminate()
