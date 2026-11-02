"""The meta-training loop: an estimator writes ``theta.grad``, then any PyTorch optimizer steps.

Any estimator that implements ``(theta, it) -> (gradient, log)`` runs in this loop; ES is one.
The gradient is in parameter space. ``sigma`` is used only to log the drift in units of the ES scale.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Protocol

import torch
from torch import nn
from torch.optim import Optimizer


class Estimator(Protocol):
    """``(theta, it) -> (gradient shaped like theta, log record)``."""

    def __call__(self, theta: torch.Tensor, it: int) -> tuple[torch.Tensor, dict]: ...


def _load_legacy(optimizer: Optimizer, state: dict, sigma: torch.Tensor | None) -> None:
    """Momentum from an older checkpoint. ``m`` is in noise units; the buffer is ``sigma * m``."""
    param = optimizer.param_groups[0]["params"][0]
    slot = optimizer.state[param]
    moment = state["m"].detach().to(device=param.device, dtype=param.dtype)
    scale = torch.ones_like(moment) if sigma is None else torch.as_tensor(sigma, device=param.device, dtype=param.dtype)
    if isinstance(optimizer, torch.optim.Adam):
        second = state["v"].detach().to(device=param.device, dtype=param.dtype) if "v" in state else torch.zeros_like(moment)
        slot["step"] = torch.tensor(float(state.get("t", 0)))
        slot["exp_avg"] = scale * moment
        slot["exp_avg_sq"] = scale.square() * second
    else:
        slot["momentum_buffer"] = scale * moment


def minimize(estimator: Estimator, optimizer: Optimizer, theta: nn.Parameter, iters: int,
             out: str | Path | None = None, on_iter: Callable[[int, torch.Tensor], None] | None = None,
             it0: int = 0, theta0: torch.Tensor | None = None, sigma: torch.Tensor | None = None) -> torch.Tensor:
    """Run iterations ``[it0, iters)``: ``theta.grad`` from the estimator, then ``optimizer.step()``."""
    theta0 = theta.detach().clone() if theta0 is None else theta0.detach()
    out = Path(out) if out else None
    for it in range(it0, iters):
        optimizer.zero_grad(set_to_none=True)
        gradient, record = estimator(theta.detach(), it)
        if not torch.isfinite(gradient.norm()):
            raise FloatingPointError(f"iteration {it}: non-finite gradient estimate {json.dumps(record)}")
        theta.grad = gradient.detach()
        before = theta.detach().clone()
        optimizer.step()
        delta = before - theta.detach()
        moved = theta.detach() - theta0
        drift = (moved / sigma).norm().item() if sigma is not None else moved.norm().item()
        record = {"it": it} | record | {"step_norm": delta.norm().item(), "drift": drift}
        print(json.dumps(record), flush=True)
        if out:
            with open(out / "log.jsonl", "a") as log:
                log.write(json.dumps(record) + "\n")
            torch.save({"theta": theta.detach().cpu(), "theta0": theta0.cpu(),
                        "optimizer": optimizer.state_dict(), "it": it + 1}, out / "state.pt")
        if on_iter:
            on_iter(it, theta.detach())
    return theta.detach()


def resume(out: str | Path | None, theta: nn.Parameter, optimizer: Optimizer,
           sigma: torch.Tensor | None = None) -> tuple[torch.Tensor, int]:
    """Load ``out/state.pt`` into ``theta`` and ``optimizer``. Returns ``(theta0, next iteration)``."""
    if out:
        path = Path(out)
        path.mkdir(parents=True, exist_ok=True)
        if (path / "state.pt").exists():
            state = torch.load(path / "state.pt", weights_only=False)
            theta.data.copy_(state["theta"].float())
            if "optimizer" in state:
                optimizer.load_state_dict(state["optimizer"])
            else:
                _load_legacy(optimizer, state["outer"], sigma)
            return state["theta0"].float(), int(state["it"])
    return theta.detach().clone(), 0
