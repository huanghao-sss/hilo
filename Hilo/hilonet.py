"""HiloNet: the matrix Transformer at the core of Hilo (HiHiTech's Learned Optimizer).

Per matrix W (m x n) every coordinate carries the channels [G, M_beta_1, ..., M_beta_k, W] (M_beta are EMAs
of the gradient), each divided by its tensor RMS and clipped at +-20 (scale equivariance). A global context
vector carries log(rms M_beta / rms G), log(rms W + 1e-4), progress t/T and log(m/n).
Coordinates are embedded to d dimensions and mixed by blocks of [row attention -> column attention -> MLP]
with residuals and RMSNorm. Row attention treats rows as tokens and scores row pairs by channel bilinear
forms summed over all columns (with one channel and linear attention the score matrix is X X^T), so the
network is equivariant to row and column permutations and applies to any matrix shape. A linear read-out
gives X, and the update is U = normalize(X, output_norm) (see Hilo.norms). The parameters are
meta-trained with Hilo.meta.

The random init has the residual branches at zero, so a new HiloNet is a per-coordinate linear map of the
normalized channels.
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
    """Hilo.config: the HiloNet architecture (saved with the network)."""
    d: int = 16                      # embedding width per coordinate
    heads: int = 2
    mlp_hidden: int = 32
    blocks: int = 2                  # distinct blocks in one pass
    tie: int = 2                     # the pass is repeated this many times (0: once)
    attn: str = "linear"             # "linear" (scores / m) or "softmax"
    betas: tuple = (0.5, 0.9, 0.99, 0.999)
    clip: float = 20.0
    output_norm: str = "none"        # "none" | "rms" | "frobenius" | "row", see Hilo.norms
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        self.betas = tuple(float(b) for b in self.betas)
        assert self.d % self.heads == 0 and self.attn in ("linear", "softmax")
        assert self.blocks >= 1 and self.tie >= 0, (self.blocks, self.tie)
        assert self.output_norm in OUTPUT_NORMS, (self.output_norm, OUTPUT_NORMS)

    @property
    def iters(self) -> int:
        """Applications of a block: `blocks` distinct blocks, the pass repeated `tie` times (once if tie is 0)."""
        return self.blocks * (self.tie or 1)

    @property
    def channels(self) -> list[str]:
        return ["G"] + [f"M{b:g}" for b in self.betas] + ["W"]

    @property
    def n_glob(self) -> int:
        return len(self.betas) + 3


def param_spec(cfg: HiloNetConfig) -> list[tuple[str, tuple]]:
    """Names and shapes of the flat parameter vector (order matters)."""
    d, h = cfg.d, cfg.mlp_hidden
    out = [("E", (len(cfg.channels), d)), ("Eg", (cfg.n_glob, d))]
    for b in range(cfg.blocks):
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


def head_weights(phi: dict, heads: int, dt: torch.dtype = torch.float32) -> dict:
    """The attention weights of phi split per head (contiguous, in dtype dt), as hilo_net takes them."""
    w = {}
    for name, v in phi.items():
        if name[-2:-1] in ("r", "c") and name[-1] in "qkvo":
            parts = v.chunk(heads, dim=0 if name[-1] == "o" else 1)
            w[name] = [p.contiguous().to(dt) for p in parts]
    return w


def hilo_net(chan: Tensor, glob: Tensor, phi: dict, w: dict, *, blocks: int, tie: int, heads: int, attn: str,
             dt: torch.dtype = torch.float32) -> Tensor:
    """chan [B, m, n, C] normalized channels, glob [B, n_glob] -> raw update X [B, m, n].
    phi: parameters; w: per-head attention weights in dtype dt (head_weights). `blocks` distinct blocks are
    applied in order, and that pass is repeated `tie` times (once if tie is 0). The residual stream, norms
    and MLP stay in float32; attention matmuls run in dt."""
    h = chan @ phi["E"] + (glob @ phi["Eg"])[:, None, None, :]
    for i in range(blocks * (tie or 1)):
        b = i % blocks
        h = h + _attn(_cn(h), w, f"{b}r", heads, attn, dt)
        h = h + _attn(_cn(h).transpose(1, 2), w, f"{b}c", heads, attn, dt).transpose(1, 2)
        z = F.silu(_cn(h) @ phi[f"{b}m1"] + phi[f"{b}mb1"])
        h = h + z @ phi[f"{b}m2"] + phi[f"{b}mb2"]
    return h @ phi["out"]


def channels(G: Tensor, M: Tensor, W: Tensor, clip: float = 20.0) -> tuple[Tensor, Tensor]:
    """G, W [B, m, n]; M [B, m, n, k] -> (RMS-normalized clipped channels [B, m, n, k + 2], their RMS [B, k + 2])."""
    chan = torch.cat([G[..., None], M, W[..., None]], -1)
    r = chan.square().mean(dim=(1, 2)).sqrt()
    return (chan / r.clamp_min(1e-30)[:, None, None, :]).clamp(-clip, clip), r


def features(G: Tensor, M: Tensor, W: Tensor, progress: float, clip: float = 20.0) -> tuple[Tensor, Tensor]:
    """-> (chan [B, m, n, k + 2], glob [B, k + 3])."""
    B, m, n = G.shape
    k = M.shape[-1]
    chan, r = channels(G, M, W, clip)
    lg = torch.log(r[:, 1:1 + k].clamp_min(1e-30) / r[:, :1].clamp_min(1e-30))
    lw = torch.log(r[:, -1:] + 1e-4)
    f = torch.tensor(progress, device=G.device, dtype=torch.float32)
    ctx = torch.stack([f, torch.full_like(f, math.log(m / n))]).expand(B, 2)
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


BACKENDS = ("auto", "triton", "torch")
PRECISIONS = {"auto": None, "fp32": torch.float32, "tf32": torch.float32, "bf16": torch.bfloat16}


class HiloNet:
    """Hilo.net, the HiloNet: (G, M, W, progress) of a batch of same-shape matrices -> updates U.

        Hilo.net()                          random init (seed 0), default config
        Hilo.net(Hilo.config(...), seed)    random init, custom architecture
        Hilo.net.load("net.pt")             parameters saved with .save() or by Hilo.meta
        Hilo.net(cfg, phi=flat)             explicit flat parameters (used by meta-training)

    Execution options (not part of the saved network; meta-train and train with the same precision):
        precision: dtype of the attention matmuls: "bf16", "tf32", "fp32" (exact reference) or "auto"
            (default: bf16 on CUDA, fp32 on CPU). The residual stream, norms and MLP are float32 in all modes.
        backend: "auto" (default) runs the fused Triton kernels (Hilo.kernels) on CUDA in bf16 when Triton is
            installed and d, mlp_hidden are powers of two >= 16, otherwise the PyTorch implementation;
            "triton" requires the kernels, "torch" never uses them.
        compile: torch.compile the PyTorch implementation (one compilation per matrix shape).
        max_coords: matrices of one shape are processed in chunks of at most this many coordinates
            (bounds peak memory, about 0.5 KB per coordinate).

    Cost: every attention layer forms m x m and n x n score matrices with inner dimension n*d/heads
    (m*d/heads), i.e. O(iters * m n (m + n) * d) flops per matrix.
    """

    def __init__(self, config: HiloNetConfig | None = None, seed: int = 0, phi: Tensor | None = None,
                 precision: str = "auto", backend: str = "auto", compile: bool = False,
                 max_coords: int = 2 ** 23):
        self.cfg = config or HiloNetConfig()
        assert precision in PRECISIONS, (precision, list(PRECISIONS))
        assert backend in BACKENDS, (backend, BACKENDS)
        spec = param_spec(self.cfg)
        if phi is None:
            phi = init_flat(self.cfg, seed)
        flat = phi.detach().float().reshape(-1)
        assert flat.numel() == sum(math.prod(s) for _, s in spec), (flat.numel(), n_params(self.cfg))
        self.phi, i = {}, 0
        for name, shp in spec:
            k = math.prod(shp)
            self.phi[name] = flat[i:i + k].reshape(shp)
            i += k
        self.precision, self.backend, self.max_coords = precision, backend, max_coords
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
        """options: precision, backend, compile, max_coords."""
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = json.loads(ckpt["config"]) if isinstance(ckpt["config"], str) else ckpt["config"]
        return cls(HiloNetConfig(**cfg), phi=ckpt["phi"], **options)

    def _to(self, device: torch.device) -> None:
        """Move parameters to `device` and split the attention weights per head (in the matmul dtype)."""
        self.phi = {k: v.to(device) for k, v in self.phi.items()}
        self._mode = self.precision if self.precision != "auto" else "bf16" if device.type == "cuda" else "fp32"
        self._w = head_weights(self.phi, self.cfg.heads, PRECISIONS[self._mode])
        self._triton = None
        if self.backend != "torch" and device.type == "cuda" and self._mode == "bf16":
            try:
                from . import kernels
                if kernels.supported(self.cfg):
                    self._triton = kernels.hilo_net_triton
            except ImportError:
                pass
        if self.backend == "triton" and self._triton is None:
            raise RuntimeError("backend='triton' needs CUDA, precision bf16, Triton, and d / mlp_hidden powers of "
                               f"two >= 16 (device {device}, precision {self._mode}, {self.cfg})")

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
                kw = dict(blocks=self.cfg.blocks, tie=self.cfg.tie, heads=self.cfg.heads, attn=self.cfg.attn)
                if self._triton is not None:
                    X = self._triton(chan, glob, self.phi, **kw)
                else:
                    X = self._net(chan, glob, self.phi, self._w, dt=PRECISIONS[self._mode], **kw)
                Us.append(normalize(X, self.cfg.output_norm))
        finally:
            torch.backends.cuda.matmul.allow_tf32 = tf32
        return Us[0] if len(Us) == 1 else torch.cat(Us)
