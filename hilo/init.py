"""The starting HiloNet of meta-training: a random init, a Muon-distilled one (hilo.distill) or a saved one."""
from __future__ import annotations

from pathlib import Path

from .net import HiloNet, HiloNetConfig

INITS = ("random", "muon")


def init_net(init: str = "random", cfg: HiloNetConfig | None = None, seed: int = 0, out: str | None = None,
             **kwargs) -> HiloNet:
    """"random" (the random init of `seed`), "muon" (that init distilled toward Muon; kwargs go to
    hilo.distill.distill; with `out`, loaded from there if it exists, else saved there) or the path of a saved
    HiloNet (cfg is then ignored)."""
    if init == "random":
        return HiloNet(cfg, seed=seed)
    if init == "muon":
        if out and Path(out).exists():
            return HiloNet.load(out)
        from .distill import muon
        return muon(cfg, seed=seed, out=out, **kwargs)
    return HiloNet.load(init)
