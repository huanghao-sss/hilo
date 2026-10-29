"""HiloNet: the matrix Transformer at the core of hilo (HiHiTech's Learned Optimizer).

Per matrix W (m x n) every coordinate carries the channels [G, M_beta_1, ..., M_beta_k, W] (M_beta are EMAs
of the gradient), each divided by its tensor RMS and clipped at +-20 (scale equivariance). A global context
vector carries log(rms M_beta / rms G), log(rms W + 1e-4), progress t/T, 1 - t/T and log(m/n).
Coordinates are embedded to d dimensions and mixed by blocks of [row attention -> column attention -> MLP]
with residuals and RMSNorm. Row attention treats rows as tokens and scores row pairs by channel bilinear
forms summed over all columns (with one channel and linear attention the score matrix is X X^T), so the
network is equivariant to row and column permutations and applies to any matrix shape. A linear read-out
gives X, and the update is U = normalize(X, output_norm) (see hilo.norms). The parameters are
meta-trained with hilo.meta.

A new HiloNet is randomly initialized with the residual branches at zero, so it starts as a per-coordinate
linear map of the normalized channels.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor

from .norms import OUTPUT_NORMS, normalize


@dataclass
class HiloNetConfig:
    d: int = 16                      # embedding width per coordinate
    heads: int = 2
    mlp_hidden: int = 32
    blocks: int = 2                  # number of distinct blocks (ignored when tie > 0)
    tie: int = 12                    # > 0: one shared block applied `tie` times
    attn: str = "linear"             # "linear" (scores / m) or "softmax"
    betas: tuple = (0.5, 0.9, 0.99, 0.999)
    clip: float = 20.0
    output_norm: str = "none"        # "none" | "rms" | "frobenius" | "row", see hilo.norms
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.betas = tuple(float(b) for b in self.betas)
        assert self.d % self.heads == 0 and self.attn in ("linear", "softmax")
        assert self.output_norm in OUTPUT_NORMS, (self.output_norm, OUTPUT_NORMS)

    @property
    def iters(self) -> int:
        return self.tie or self.blocks

    @property
    def channels(self) -> list[str]:
        return ["G"] + [f"M{b:g}" for b in self.betas] + ["W"]

    @property
    def n_glob(self) -> int:
        return len(self.betas) + 4


def param_spec(cfg: HiloNetConfig) -> list[tuple[str, tuple]]:
    """Names and shapes of the flat parameter vector (order matters)."""
    d, h = cfg.d, cfg.mlp_hidden
    out = [("E", (len(cfg.channels), d)), ("Eg", (cfg.n_glob, d))]
    for b in range(1 if cfg.tie else cfg.blocks):
        for ax in ("r", "c"):
            out += [(f"{b}{ax}q", (d, d)), (f"{b}{ax}k", (d, d)), (f"{b}{ax}v", (d, d)), (f"{b}{ax}o", (d, d))]
        out += [(f"{b}m1", (d, h)), (f"{b}mb1", (h,)), (f"{b}m2", (h, d)), (f"{b}mb2", (d,))]
    out += [("out", (d,))]
    return out


def n_params(cfg: HiloNetConfig | None = None) -> int:
    return sum(math.prod(s) for _, s in param_spec(cfg or HiloNetConfig()))


def init_flat(cfg: HiloNetConfig | None = None, seed: int = 0) -> Tensor:
    """Random init; attention and MLP output projections start at zero (residual branches off)."""
    cfg = cfg or HiloNetConfig()
    g = torch.Generator().manual_seed(seed)
    parts = []
    for name, shp in param_spec(cfg):
        if len(shp) == 2:
            t = torch.randn(shp, generator=g) / math.sqrt(shp[0])
            if name.endswith(("o", "m2")):
                t.zero_()
        elif name == "out":
            t = torch.randn(shp, generator=g) / math.sqrt(cfg.d)
        else:
            t = torch.zeros(shp)
        parts.append(t.reshape(-1))
    return torch.cat(parts)


def es_sigma(cfg: HiloNetConfig | None = None) -> Tensor:
    """Default per-parameter ES noise scale: 0.02 for matrices, 0.01 for vectors (scale with sigma_mult)."""
    return torch.cat([torch.full((math.prod(s),), 0.02 if len(s) == 2 else 0.01)
                      for _, s in param_spec(cfg or HiloNetConfig())])


def _cn(h: Tensor) -> Tensor:
    return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + 1e-6)


def _attn(a: Tensor, w: dict, pre: str, heads: int, attn: str, dt: torch.dtype) -> Tensor:
    """a [B, T, F, d]: T tokens (rows, or columns after a transpose) whose features are their F coordinates.
    Each head projects to [B, T, F, d/heads] (contiguous), so a token is a flat vector of length F*d/heads.
    Returns [B, T, F, d] in dtype dt."""
    B, T, Fn, d = a.shape
    dk = d // heads
    c = 1.0 / (Fn * math.sqrt(dk)) / (T if attn == "linear" else 1)
    a = a.to(dt, memory_format=torch.contiguous_format)
    out = None
    for h in range(heads):
        q = (a @ (w[pre + "q"][h] * c)).view(B, T, Fn * dk)
        k = (a @ w[pre + "k"][h]).view(B, T, Fn * dk)
        v = (a @ w[pre + "v"][h]).view(B, T, Fn * dk)
        A = q @ k.mT
        if attn == "softmax":
            A = torch.softmax(A.float(), dim=-1).to(dt)
        o = (A @ v).view(-1, dk)
        out = o @ w[pre + "o"][h] if out is None else torch.addmm(out, o, w[pre + "o"][h])
    return out.view(B, T, Fn, d)


def hilo_net(chan: Tensor, glob: Tensor, phi: dict, w: dict, *, iters: int, tie: bool, heads: int, attn: str,
             dt: torch.dtype = torch.float32) -> Tensor:
    """chan [B, m, n, C] normalized channels, glob [B, n_glob] -> raw update X [B, m, n].
    phi: parameters; w: per-head attention weights in dtype dt (see HiloNet._to). The residual stream,
    norms and MLP stay in float32; attention matmuls run in dt."""
    h = chan @ phi["E"] + (glob @ phi["Eg"])[:, None, None, :]
    for i in range(iters):
        b = 0 if tie else i
        h += _attn(_cn(h), w, f"{b}r", heads, attn, dt)
        h += _attn(_cn(h).transpose(1, 2), w, f"{b}c", heads, attn, dt).transpose(1, 2)
        z = F.silu(_cn(h) @ phi[f"{b}m1"] + phi[f"{b}mb1"])
        h += z @ phi[f"{b}m2"] + phi[f"{b}mb2"]
    return h @ phi["out"]


def channels(G: Tensor, M: Tensor, W: Tensor, clip: float = 20.0) -> tuple[Tensor, Tensor]:
    """G, W [B, m, n]; M [B, m, n, k] -> (RMS-normalized clipped channels [B, m, n, k + 2], their RMS [B, k + 2])."""
    chan = torch.cat([G[..., None], M, W[..., None]], -1)
    r = chan.square().mean(dim=(1, 2)).sqrt()
    return (chan / r.clamp_min(1e-30)[:, None, None, :]).clamp(-clip, clip), r


def features(G: Tensor, M: Tensor, W: Tensor, progress: float, clip: float = 20.0) -> tuple[Tensor, Tensor]:
    """-> (chan [B, m, n, k + 2], glob [B, k + 4])."""
    B, m, n = G.shape
    k = M.shape[-1]
    chan, r = channels(G, M, W, clip)
    lg = torch.log(r[:, 1:1 + k].clamp_min(1e-30) / r[:, :1].clamp_min(1e-30))
    lw = torch.log(r[:, -1:] + 1e-4)
    f = torch.tensor(progress, device=G.device, dtype=torch.float32)
    ctx = torch.stack([f, 1 - f, torch.full_like(f, math.log(m / n))]).expand(B, 3)
    return chan, torch.cat([lg, lw, ctx], -1)


_COMPILED = None


def _compiled_net():
    """One compiled hilo_net shared by all HiloNets (parameters are graph inputs, so meta-training's many
    instances reuse it); each matrix shape / batch size is one specialization."""
    global _COMPILED
    if _COMPILED is None:
        import torch._dynamo.config as dc
        dc.recompile_limit = max(dc.recompile_limit, 64)
        _COMPILED = torch.compile(hilo_net, dynamic=False)
    return _COMPILED


PRECISIONS = {"auto": None, "fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16}


class HiloNet:
    """The hilo network: (G, M, W, progress) of a batch of same-shape matrices -> updates U.

        HiloNet()                         random init (seed 0), default config
        HiloNet(HiloNetConfig(...), seed) random init, custom architecture
        HiloNet.load("hilo.pt")           parameters saved with .save() or by hilo.meta
        HiloNet(cfg, phi=flat)            explicit flat parameters (used by meta-training)

    Execution options (not part of the saved network; meta-train and train with the same precision):
        precision: dtype of the attention matmuls: "bf16", "tf32", "fp32" (exact reference) or "auto"
            (default: bf16 on CUDA, fp32 on CPU). The residual stream, norms and MLP are float32 in all modes.
        compile: torch.compile the network (one compilation per matrix shape).
        max_coords: matrices of one shape are processed in chunks of at most this many coordinates
            (bounds peak memory, about 0.5 KB per coordinate).

    Cost: every attention layer forms m x m and n x n score matrices with inner dimension n*d/heads
    (m*d/heads), i.e. O(iters * m n (m + n) * d) flops per matrix.
    """

    def __init__(self, config: HiloNetConfig | None = None, seed: int = 0, phi: Tensor | None = None,
                 precision: str = "auto", compile: bool = False, max_coords: int = 2 ** 23):
        self.cfg = config or HiloNetConfig()
        assert precision in PRECISIONS, (precision, list(PRECISIONS))
        spec = param_spec(self.cfg)
        flat = (init_flat(self.cfg, seed) if phi is None else phi).detach().float().reshape(-1)
        assert flat.numel() == sum(math.prod(s) for _, s in spec), (flat.numel(), n_params(self.cfg))
        self.phi, i = {}, 0
        for name, shp in spec:
            k = math.prod(shp)
            self.phi[name] = flat[i:i + k].reshape(shp)
            i += k
        self.precision, self.max_coords = precision, max_coords
        self._w = None
        self.compile = compile
        self._net = _compiled_net() if compile else hilo_net

    def __repr__(self):
        return f"HiloNet({self.cfg}, n_params={n_params(self.cfg)}, precision={self.precision!r})"

    def flat(self) -> Tensor:
        return torch.cat([v.reshape(-1).cpu() for v in self.phi.values()])

    def save(self, path: str | Path, note: str = "") -> None:
        torch.save({"format": "hilo-net-v1", "phi": self.flat(), "config": json.dumps(asdict(self.cfg)),
                    "note": note}, path)

    @classmethod
    def load(cls, path: str | Path, **options) -> "HiloNet":
        """options: precision, compile, max_coords."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = json.loads(ckpt["config"]) if isinstance(ckpt["config"], str) else ckpt["config"]
        return cls(HiloNetConfig(**cfg), phi=ckpt["phi"], **options)

    def _to(self, device: torch.device) -> None:
        """Move parameters to `device` and split the attention weights per head (contiguous, in the matmul dtype)."""
        self.phi = {k: v.to(device) for k, v in self.phi.items()}
        self._mode = self.precision if self.precision != "auto" else "bf16" if device.type == "cuda" else "fp32"
        dt, dk, H = PRECISIONS[self._mode], self.cfg.d // self.cfg.heads, self.cfg.heads
        self._w = {}
        for name, v in self.phi.items():
            if name[-1] in "qkv" and name[-2] in "rc":
                self._w[name] = [v[:, h * dk:(h + 1) * dk].contiguous().to(dt) for h in range(H)]
            elif name[-1] == "o" and name[-2] in "rc":
                self._w[name] = [v[h * dk:(h + 1) * dk].contiguous().to(dt) for h in range(H)]

    @torch.no_grad()
    def __call__(self, G: Tensor, M: Tensor, W: Tensor, progress: float) -> Tensor:
        """G, W [B, m, n]; M [B, m, n, len(betas)] (gradient EMAs) -> U [B, m, n]."""
        if self._w is None or self.phi["E"].device != G.device:
            self._to(G.device)
        B, m, n = G.shape
        step = max(1, self.max_coords // (m * n))
        tf32 = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = self._mode != "fp32" or tf32
        try:
            Us = []
            for i in range(0, B, step):
                chan, glob = features(G[i:i + step], M[i:i + step], W[i:i + step], progress, self.cfg.clip)
                X = self._net(chan, glob, self.phi, self._w, iters=self.cfg.iters, tie=bool(self.cfg.tie),
                              heads=self.cfg.heads, attn=self.cfg.attn, dt=PRECISIONS[self._mode])
                Us.append(normalize(X, self.cfg.output_norm))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return Us[0] if len(Us) == 1 else torch.cat(Us)
