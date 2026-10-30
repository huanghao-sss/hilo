"""Meta-train a HiloNet for modded-nanogpt on a scaled-down copy of the #36 baseline (one GPU).

The proxy is train_gpt_hilo.py shrunk: same architecture, init, AdamW groups and schedule, loss, and a
constant Hilo lr; smaller width, depth, batch and steps. With full ES (default) every evaluation trains one
proxy model from scratch with `Hilo(net=net)` and returns its final validation loss; with PES / NRES the
runs advance --trunc steps per outer step and return their mean training loss over the segment. Pairs of a
task share init and data order.

    # from a modded-nanogpt checkout (needs data/fineweb10B, see the modded-nanogpt README)
    python meta_train_proxy.py --out runs/proxy --iters 100 --pairs 8
    python meta_train_proxy.py --out runs/proxy --devices cuda:0,cuda:1,cuda:2,cuda:3   # parallel evaluations
    python meta_train_proxy.py --out runs/proxy_nres --estimator nres --trunc 50 --min-steps 100 --iters 2000
    python meta_train_proxy.py --eval runs/proxy/hilo_final.pt --against runs/proxy/hilo_it0010.pt
    HILO_NET=runs/proxy/hilo_final.pt torchrun --nproc_per_node=8 train_gpt_hilo.py

Proxy defaults (4 x 256, 300 steps of 32k tokens) take tens of seconds per evaluation on an H100-class GPU;
scale them toward the target as your budget allows: the closer the proxy, the better the transfer.
Evaluations run with deterministic CUDA kernels, so the two members of an ES pair differ only by the
perturbation (without it, two identical runs of a random-init HiloNet can end ~1 nat apart).
"""
import argparse
import functools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # needed by deterministic cuBLAS

import torch
import torch.nn.functional as F
from torch import Tensor, nn

import hilo
import hilo.meta
from hilo.net import HiloNetConfig

# ----------------------------------------------------------------- architecture of #36 (unchanged)


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gains = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), weight=self.gains.type_as(x))


class Linear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features, bias=True)

    def forward(self, x):
        return F.linear(x, self.weight.type_as(x), self.bias.type_as(x))


class Rotary(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        angular_freq = (1 / 1024) ** torch.linspace(0, 1, steps=dim // 4, dtype=torch.float32)
        self.register_buffer("angular_freq", torch.cat([angular_freq, angular_freq.new_zeros(dim // 4)]))

    def forward(self, x_BTHD: Tensor):
        pos = torch.arange(x_BTHD.size(1), dtype=torch.float32, device=x_BTHD.device)
        theta = torch.outer(pos, self.angular_freq)[None, :, None, :]
        cos, sin = theta.cos(), theta.sin()
        x1, x2 = x_BTHD.to(dtype=torch.float32).chunk(2, dim=-1)
        return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), 3).type_as(x_BTHD)


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, head_dim=128):
        super().__init__()
        self.num_heads, self.head_dim = dim // head_dim, head_dim
        hdim = self.num_heads * self.head_dim
        self.q, self.k, self.v, self.proj = Linear(dim, hdim), Linear(dim, hdim), Linear(dim, hdim), Linear(hdim, dim)
        self.rotary = Rotary(head_dim)

    def forward(self, x: Tensor):
        B, T = x.size(0), x.size(1)
        q = self.q(x).view(B, T, self.num_heads, self.head_dim)
        k = self.k(x).view(B, T, self.num_heads, self.head_dim)
        v = self.v(x).view(B, T, self.num_heads, self.head_dim)
        q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                           scale=0.12, is_causal=True).transpose(1, 2)
        return self.proj(y.contiguous().view(B, T, self.num_heads * self.head_dim))


class MLP(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc, self.proj = Linear(dim, 4 * dim), Linear(4 * dim, dim)

    def forward(self, x: Tensor):
        return self.proj(self.fc(x).relu().square())


class Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.attn, self.mlp, self.norm1, self.norm2 = CausalSelfAttention(dim), MLP(dim), RMSNorm(dim), RMSNorm(dim)

    def forward(self, x: Tensor):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, vocab_size: int, num_layers: int, model_dim: int):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, model_dim).bfloat16()
        self.blocks = nn.ModuleList([Block(model_dim) for _ in range(num_layers)])
        self.proj = Linear(model_dim, vocab_size)
        self.norm1, self.norm2 = RMSNorm(model_dim), RMSNorm(model_dim)

    def forward(self, inputs: Tensor, targets: Tensor):
        x = self.norm1(self.embed(inputs))
        for block in self.blocks:
            x = block(x)
        logits = self.proj(self.norm2(x)).float()
        logits = 15 * logits * (logits.square() + 15 ** 2).rsqrt()
        return F.cross_entropy(logits.view(targets.numel(), -1), targets.view(-1), reduction="sum")


def init_params(model: nn.Module, gen: torch.Generator) -> None:
    """The #36 init, drawn from `gen` so that both members of an ES pair start from the same model."""
    for name, p in model.named_parameters():
        w = p.data
        if name.endswith("weight"):
            if "proj" in name:
                w.zero_()
            else:
                std = 1.0 if "embed" in name else 0.33 ** 0.5 / w.size(-1) ** 0.5
                w.copy_(torch.randn(w.shape, generator=gen, device=w.device) * std)
        elif name.endswith("bias"):
            w.zero_()
        elif name.endswith("gains"):
            w.fill_(1.0)
        else:
            raise Exception(f"Uninitialized parameter: {name}")

# ----------------------------------------------------------------- proxy task


def load_shard(path: Path) -> Tensor:
    header = torch.from_file(str(path), False, 256, dtype=torch.int32)
    assert header[0] == 20240520 and header[1] == 1, f"bad shard {path}"
    with path.open("rb", buffering=0) as f:
        tokens = torch.empty(int(header[2]), dtype=torch.uint16)
        f.seek(256 * 4)
        f.readinto(tokens.numpy())
    return tokens


class Proxy:
    def __init__(self, data: str = "data/fineweb10B", layers: int = 4, dim: int = 256, steps: int = 300,
                 batch_seqs: int = 32, seq_len: int = 1024, mbs: int = 32, val_seqs: int = 64,
                 lr_hidden: float = 1e-3, cooldown_frac: float = 0.7, device: str = "cuda",
                 deterministic: bool = True, compile: bool = False):
        self.layers, self.dim, self.steps = layers, dim, steps
        self.batch_seqs, self.seq_len, self.mbs = batch_seqs, seq_len, mbs
        self.lr_hidden, self.cooldown_frac = lr_hidden, cooldown_frac
        if deterministic:
            torch.use_deterministic_algorithms(True)
        self.dev = dev = torch.device(device)
        self.train = load_shard(sorted(Path(data).glob("fineweb_train_*.bin"))[0]).to(dev)
        val = load_shard(sorted(Path(data).glob("fineweb_val_*.bin"))[0])[:val_seqs * seq_len + 1].to(dev)
        self.val = (val[:-1].int().view(-1, seq_len), val[1:].long().view(-1, seq_len))
        self.model = GPT(50304, layers, dim).to(dev)
        if compile:
            self.model.compile(dynamic=False)

    def _start(self, task_seed: int, horizon: int) -> Tensor:
        """#36 init and the data order of a run, both drawn from the task seed; returns the batch offsets."""
        gen = torch.Generator(device=self.dev).manual_seed(task_seed)
        init_params(self.model, gen)
        n_tok = self.batch_seqs * self.seq_len
        return torch.randint(0, self.train.numel() - n_tok - 1, (horizon,), generator=gen, device=self.dev)

    def _optimizers(self, net, horizon: int):
        model = self.model
        adam = torch.optim.AdamW([dict(params=[model.embed.weight], lr=0.7),
                                  dict(params=[model.proj.weight], lr=0.004),
                                  dict(params=[p for p in model.parameters() if p.ndim < 2], lr=0.015)],
                                 betas=(0.8, 0.95), eps=1e-10, weight_decay=0.001, fused=True)
        opt = hilo.Hilo([p for p in model.blocks.parameters() if p.ndim >= 2], total_steps=horizon,
                        lr=self.lr_hidden, net=net)
        for g in adam.param_groups:
            g["initial_lr"] = g["lr"]
        return adam, opt

    def _train(self, adam, opt, starts: Tensor, step0: int, steps: int, horizon: int) -> float:
        """Steps [step0, step0 + steps) of a run of `horizon` steps; returns their mean training loss."""
        model, n_tok = self.model, self.batch_seqs * self.seq_len
        total = torch.zeros((), device=self.dev)
        for step in range(step0, step0 + steps):
            buf = self.train[starts[step]:starts[step] + n_tok + 1]
            x, y = buf[:-1].int().view(-1, self.seq_len), buf[1:].long().view(-1, self.seq_len)
            for i in range(0, len(x), self.mbs):
                loss = model(x[i:i + self.mbs], y[i:i + self.mbs])
                loss.backward()
                total += loss.detach()
            f = step / horizon
            eta = 1.0 if f < 1 - self.cooldown_frac else (1 - f) / self.cooldown_frac
            for g in adam.param_groups:
                g["lr"] = g["initial_lr"] * eta
            adam.step()
            opt.step()
            model.zero_grad(set_to_none=True)
        return _finite(total.item() / (steps * n_tok))

    def __call__(self, net, task_seed: int) -> float:
        """Train the proxy from scratch with `net` on the hidden matrices; returns final val loss (full ES)."""
        starts = self._start(task_seed, self.steps)
        self._train(*self._optimizers(net, self.steps), starts, 0, self.steps, self.steps)
        with torch.no_grad():
            loss = sum(self.model(self.val[0][i:i + self.mbs], self.val[1][i:i + self.mbs])
                       for i in range(0, len(self.val[0]), self.mbs)).item() / self.val[1].numel()
        return _finite(loss)

    def init(self, task_seed: int, horizon: int) -> dict:
        """A run for PES / NRES (hilo.meta.TruncatedTask); the run's state is kept off the shared model."""
        starts = self._start(task_seed, horizon)
        return dict(model=_clone(self.model.state_dict()), adam=None, hilo=None, starts=starts, step=0,
                    horizon=horizon)

    def unroll(self, net, st: dict, steps: int) -> tuple[dict, float]:
        """`steps` more steps of the run with `net`; returns the new state and their mean training loss."""
        self.model.load_state_dict(st["model"])
        adam, opt = self._optimizers(net, st["horizon"])
        if st["adam"] is not None:
            adam.load_state_dict(st["adam"])
            opt.load_state_dict(st["hilo"])
        loss = self._train(adam, opt, st["starts"], st["step"], steps, st["horizon"])
        return dict(st, model=_clone(self.model.state_dict()), adam=adam.state_dict(), hilo=opt.state_dict(),
                    step=st["step"] + steps), loss


def _clone(sd: dict) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}


def _finite(x: float) -> float:
    return x if math.isfinite(x) else float("nan")


def make_proxy(device: str, **kwargs) -> Proxy:
    """Builds the proxy inside a meta-training worker process (hilo.meta.EvalPool / UnrollPool)."""
    return Proxy(device=device, **kwargs)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/fineweb10B")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-seqs", type=int, default=32)
    ap.add_argument("--mbs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3, help="Hilo lr (use the same one in train_gpt_hilo.py)")
    ap.add_argument("--out", default="runs/proxy")
    ap.add_argument("--init", default="random", help="starting HiloNet: random, muon (distilled toward Muon "
                    "first, cached in OUT/init_muon.pt; see hilo.distill) or a saved HiloNet (.pt)")
    ap.add_argument("--config", help="HiloNetConfig as JSON, e.g. '{\"tie\": 6, \"output_norm\": \"rms\"}'")
    ap.add_argument("--seed", type=int, default=0, help="random init of the HiloNet and ES noise")
    ap.add_argument("--estimator", default="es", choices=("es",) + hilo.meta.ESTIMATORS,
                    help="es: full runs, final val loss; pes / nres: truncated unrolls, mean training loss")
    ap.add_argument("--trunc", type=int, default=50, help="pes / nres: steps per segment (one outer step each)")
    ap.add_argument("--min-steps", type=int, help="pes / nres: run lengths log-uniform in [min-steps, steps]")
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--sigma-mult", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, help="outer step in units of sigma (default: 1.0 for es, 0.3 for "
                    "pes / nres)")
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--optimizer", default="sgdm", choices=hilo.meta.OPTIMIZERS)
    ap.add_argument("--shaping", default="rms", choices=hilo.meta.SHAPINGS)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--eval", help="evaluate this HiloNet (.pt) on held-out tasks instead of meta-training")
    ap.add_argument("--against", help="with --eval: a second HiloNet, compared pairwise on the same tasks")
    ap.add_argument("--seeds", type=int, default=8, help="held-out tasks for --eval")
    ap.add_argument("--devices", help="comma-separated worker devices for parallel ES evaluations, e.g. "
                    "cuda:0,cuda:1 (repeat a GPU to run several evaluations on it at once)")
    ap.add_argument("--precision", default="auto", help="HiloNet matmul precision: auto | bf16 | tf32 | fp32")
    ap.add_argument("--compile-net", action="store_true", help="torch.compile the HiloNet")
    ap.add_argument("--compile-model", action="store_true", help="torch.compile the proxy GPT")
    ap.add_argument("--nondeterministic", action="store_true", help="allow nondeterministic CUDA kernels")
    a = ap.parse_args()
    proxy_kw = dict(data=a.data, layers=a.layers, dim=a.dim, steps=a.steps, batch_seqs=a.batch_seqs, mbs=a.mbs,
                    lr_hidden=a.lr, deterministic=not a.nondeterministic, compile=a.compile_model)
    net_options = dict(precision=a.precision, compile=a.compile_net)

    if a.eval:
        proxy = Proxy(**proxy_kw)
        nets = {"eval": hilo.HiloNet.load(a.eval, **net_options)}
        if a.against:
            nets["against"] = hilo.HiloNet.load(a.against, **net_options)
        losses = {k: [] for k in nets}
        for s in range(a.seeds):
            task = 999_000_000 + s  # disjoint from the task seeds used by meta-training
            rec = {"task": task} | {k: proxy(net, task) for k, net in nets.items()}
            for k in nets:
                losses[k].append(rec[k])
            print(json.dumps(rec), flush=True)
        L = {k: torch.tensor(v) for k, v in losses.items()}
        summary = {f"{k}_mean": v.mean().item() for k, v in L.items()}
        if a.against:
            d = L["eval"] - L["against"]
            summary |= {"mean_diff": d.mean().item(), "sem": (d.std() / len(d) ** 0.5).item()}
        print(json.dumps(summary))
        return

    Path(a.out).mkdir(parents=True, exist_ok=True)
    net = hilo.init_net(a.init, HiloNetConfig(**json.loads(a.config)) if a.config else None, a.seed,
                        out=str(Path(a.out) / "init_muon.pt"))
    print(f"meta-training {net!r} on proxy {a.layers}x{a.dim}, {a.steps} steps, {a.estimator}", flush=True)
    meta = dict(iters=a.iters, pairs=a.pairs, sigma_mult=a.sigma_mult, momentum=a.momentum, optimizer=a.optimizer,
                shaping=a.shaping, seed=a.seed, out=a.out, save_every=a.save_every, net_options=net_options)
    if a.alpha is not None:
        meta["alpha"] = a.alpha
    factory = dict(devices=a.devices.split(",")) if a.devices else {}
    if a.estimator == "es":
        if a.devices:
            factory["make_objective"] = functools.partial(make_proxy, **proxy_kw)
        hilo.meta.meta_train(None if a.devices else Proxy(**proxy_kw), net, **factory, **meta)
    else:
        if a.devices:
            factory["make_task"] = functools.partial(make_proxy, **proxy_kw)
        hilo.meta.meta_train_truncated(None if a.devices else Proxy(**proxy_kw), net, estimator=a.estimator,
                                       horizon=(a.min_steps, a.steps) if a.min_steps else a.steps,
                                       trunc=a.trunc, **factory, **meta)


if __name__ == "__main__":
    main()
