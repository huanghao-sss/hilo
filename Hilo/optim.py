"""Hilo.optimizer: W <- (1 - lr * weight_decay) * W - lr * U for 2-D parameters, U from a HiloNet."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from .hilonet import HiloNet


def assign_owners(shapes: list[tuple[int, int]], world: int) -> list[int]:
    """Rank computing each matrix: greedy longest-processing-time on the estimated HiloNet cost
    m*n*(m + n + 4096) (attention matmuls ~ m*n*(m+n), elementwise work ~ m*n)."""
    load, owner = [0] * world, [0] * len(shapes)
    cost = [m * n * (m + n + 4096) for m, n in shapes]
    for i in sorted(range(len(shapes)), key=lambda i: (-cost[i], i)):
        r = min(range(world), key=lambda r: (load[r], r))
        owner[i] = r
        load[r] += cost[i]
    return owner


class Optimizer(torch.optim.Optimizer):
    """Hilo.optimizer, HiHiTech's Learned Optimizer for matrix parameters:

        M_beta <- beta * M_beta + (1 - beta) * G     for every beta in net.cfg.betas
        W <- (1 - lr * weight_decay) * W - lr * U,   U = net(G, M, W, t / total_steps)

    Args:
        params: 2-D tensors or param groups. Parameters that are not matrices need another optimizer
            (see `split_params`).
        total_steps: training length T; HiloNet is conditioned on progress t / T.
        lr: step size multiplying U. Like any torch optimizer, change group["lr"] as you like.
        weight_decay: decoupled weight decay.
        net: a HiloNet, a path to a saved one (.pt), or None for a randomly initialized HiloNet.
        shard: under torch.distributed, the matrices are split across ranks by HiloNet cost (each rank
            computes and keeps the state of its share) and every updated matrix is broadcast from its
            owner. Gradients must already be identical on all ranks (all-reduced). shard=False: every
            rank computes every update (identical results, no communication).
    """

    def __init__(self, params: Iterable[Tensor] | Iterable[dict], total_steps: int, lr: float = 1e-3,
                 weight_decay: float = 0.0, net: HiloNet | str | Path | None = None, shard: bool = True):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    raise ValueError(f"Hilo.optimizer handles 2-D parameters only, got shape {tuple(p.shape)}")
        self.net = net if isinstance(net, HiloNet) else HiloNet.load(net) if net is not None else HiloNet()
        dist = torch.distributed
        self.rank, self.world = (dist.get_rank(), dist.get_world_size()) if (
            shard and dist.is_available() and dist.is_initialized()) else (0, 1)
        self.total_steps, self.step_count = total_steps, 0
        self._owners: dict[tuple, list[int]] = {}
        self._bufs: dict[tuple, Tensor] = {}
        self._betas: dict[torch.device, Tensor] = {}

    @property
    def progress(self) -> float:
        return min(self.step_count / self.total_steps, 1.0)

    def _momenta(self, ps: list[Tensor]) -> Tensor:
        """The gradient EMAs of same-shape matrices as one contiguous [B, m, n, k] buffer; state["M"] of each
        matrix is a view into it (rebuilt after load_state_dict replaced the state tensors)."""
        key = tuple(id(p) for p in ps)
        buf = self._bufs.get(key)
        states = [self.state[p] for p in ps]
        if buf is None or any("M" not in st or st["M"].data_ptr() != buf[j].data_ptr()
                               for j, st in enumerate(states)):
            buf = torch.zeros(len(ps), *ps[0].shape, len(self.net.cfg.betas), device=ps[0].device)
            for j, st in enumerate(states):
                if "M" in st:
                    buf[j].copy_(st["M"])
                st["M"] = buf[j]
            self._bufs[key] = buf
        return buf

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None]
            key = tuple(id(p) for p in params)
            if key not in self._owners:
                self._owners[key] = assign_owners([tuple(p.shape) for p in params], self.world)
            owners = self._owners[key]
            by_shape: dict[tuple, list[Tensor]] = {}
            for p, r in zip(params, owners):
                if r == self.rank:
                    by_shape.setdefault(tuple(p.shape), []).append(p)
            for ps in by_shape.values():
                dev = ps[0].device
                if dev not in self._betas:
                    self._betas[dev] = torch.tensor(self.net.cfg.betas, device=dev)
                G = torch.nan_to_num(torch.stack([p.grad.float() for p in ps]), 0.0, 0.0, 0.0)
                M = self._momenta(ps).lerp_(G[..., None], 1 - self._betas[dev])
                U = self.net(G, M, torch.stack([p.float() for p in ps]), self.progress)
                for p, u in zip(ps, U):
                    if group["weight_decay"]:
                        p.mul_(1 - group["lr"] * group["weight_decay"])
                    p.add_(u.to(p.dtype), alpha=-group["lr"])
            if self.world > 1:
                works = [torch.distributed.broadcast(p.data, src=r, async_op=True) for p, r in zip(params, owners)]
                for w in works:
                    w.wait()
        self.step_count += 1
        return loss

    def state_dict(self):
        sd = super().state_dict()
        sd["step_count"] = self.step_count
        return sd

    def load_state_dict(self, state_dict):
        state_dict = dict(state_dict)
        self.step_count = state_dict.pop("step_count", 0)
        super().load_state_dict(state_dict)


def split_params(model: nn.Module, exclude: Callable[[str, Tensor], bool] | None = None
                 ) -> tuple[list[Tensor], list[Tensor]]:
    """(matrices for Hilo.optimizer, everything else). By default 2-D parameters whose name contains "embed",
    "lm_head", "wte" or "wpe" go with the non-matrix ones; pass `exclude(name, p)` to choose yourself."""
    if exclude is None:
        exclude = lambda n, p: any(k in n for k in ("embed", "lm_head", "wte", "wpe"))
    mats, rest = [], []
    for n, p in model.named_parameters():
        (mats if p.ndim == 2 and not exclude(n, p) else rest).append(p)
    return mats, rest
