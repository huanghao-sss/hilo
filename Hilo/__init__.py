"""Hilo: HiHiTech's Learned Optimizer.

A framework for learned matrix optimizers: the update of every matrix parameter is computed by a small
Transformer (the HiloNet, Hilo.net) that you meta-train on your own task family (Hilo.meta.train).

    opt = Hilo.optimizer(matrices, total_steps=T, lr=1e-3, net="runs/my_net/hilo_final.pt")
"""
from .hilonet import HiloNet as net
from .hilonet import HiloNetConfig as config
from .norms import OUTPUT_NORMS, normalize
from .optim import Optimizer as optimizer
from .optim import split_params

__all__ = ["optimizer", "net", "config", "split_params", "OUTPUT_NORMS", "normalize"]
__version__ = "0.1.0"
