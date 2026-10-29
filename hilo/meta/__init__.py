"""Meta-training of HiloNet with zeroth-order estimators (no backpropagation through training).

    meta_train            full-unroll antithetic ES on the final loss: objective(net, task_seed) -> loss
    meta_train_truncated  online ES on truncated unrolls, NRES or PES: a TruncatedTask with
                          init(task_seed, horizon) -> state and unroll(net, state, steps) -> (state, loss)

Both take the outer optimizer (sgdm | adam), fitness shaping (rms | rank | none), and worker processes
(make_objective / make_task + devices). The building blocks (FullES, TruncatedES, Outer, minimize, the pools)
compose for other loops.
"""
from ..net import HiloNetConfig, init_flat
from .core import OPTIMIZERS, SHAPINGS, Outer, minimize, shape
from .es import FullES, meta_train
from .pool import EvalPool, Unroller, UnrollPool
from .truncated import ESTIMATORS, TruncatedES, TruncatedTask, meta_train_truncated

__all__ = ["meta_train", "meta_train_truncated", "FullES", "TruncatedES", "TruncatedTask", "Outer", "minimize",
           "shape", "EvalPool", "UnrollPool", "Unroller", "ESTIMATORS", "OPTIMIZERS", "SHAPINGS",
           "HiloNetConfig", "init_flat"]
