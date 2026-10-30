"""hilo: HiHiTech's Learned Optimizer.

A framework for learned matrix optimizers: the update of every matrix parameter is computed by a small
Transformer (HiloNet) that you meta-train on your own task family (hilo.meta), starting from a random init or
from a HiloNet distilled toward a hand-designed rule (hilo.distill).
"""
from .init import INITS, init_net
from .net import HiloNet, HiloNetConfig
from .norms import OUTPUT_NORMS, normalize
from .optim import Hilo, split_params

__all__ = ["Hilo", "HiloNet", "HiloNetConfig", "init_net", "INITS", "split_params", "OUTPUT_NORMS", "normalize"]
__version__ = "0.1.0"
