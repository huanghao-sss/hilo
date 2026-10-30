"""Output normalization of an update matrix: the network computes X, the optimizer applies U = normalize(X).

    "none"       U = X
    "rms"        RMS(U) = 1
    "frobenius"  ||U||_F = 1
    "row"        ||U||_F = sqrt(m): the mean squared row norm is 1

All-zero matrices stay zero.
"""
import math

import torch
from torch import Tensor

OUTPUT_NORMS = ("none", "rms", "frobenius", "row")


def target_norm(mode: str, m: int, n: int) -> float:
    """Frobenius norm of the normalized update of an m x n matrix (for the fixed-norm modes)."""
    if mode == "rms":
        return math.sqrt(m * n)
    if mode == "frobenius":
        return 1.0
    if mode == "row":
        return math.sqrt(m)
    raise ValueError(f"output_norm {mode!r} has no fixed norm; choose from {OUTPUT_NORMS}")


def normalize(X: Tensor, mode: str) -> Tensor:
    """Normalize each matrix of a batch [..., m, n]."""
    if mode == "none":
        return X
    ss = X.square().sum(dim=(-2, -1), keepdim=True)
    return X * (target_norm(mode, *X.shape[-2:]) * torch.rsqrt(ss.clamp_min(1e-30))) * (ss > 0)
