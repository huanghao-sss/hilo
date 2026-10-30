"""Meta-train a HiloNet on your own task family (hilo.meta), from a random init.

The task is a small teacher-student MLP regression; replace it with your training loop
(examples/modded_nanogpt/meta_train_proxy.py does this for a GPT).
    es:        objective(net, task_seed) trains a fresh model and returns the final validation loss.
    pes, nres: Task.init(task_seed, horizon) starts a run, Task.unroll(net, state, steps) continues it and
               returns the segment's mean training loss; the HiloNet is updated every --trunc steps.

    python examples/meta_train.py --iters 10 --pairs 4 --out runs/meta_mlp
    python examples/meta_train.py --estimator nres --trunc 20 --iters 100 --out runs/meta_mlp_nres
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hilo  # noqa: E402
import hilo.meta  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--estimator", default="es", choices=("es",) + hilo.meta.ESTIMATORS)
ap.add_argument("--iters", type=int, default=10)
ap.add_argument("--pairs", type=int, default=4)
ap.add_argument("--steps", type=int, default=200, help="inner training steps (the meta-objective horizon)")
ap.add_argument("--trunc", type=int, default=20, help="pes / nres: steps per outer step")
ap.add_argument("--lr", type=float, default=1e-3, help="Hilo lr during meta-training (keep it when using the net)")
ap.add_argument("--output-norm", default="none", choices=hilo.OUTPUT_NORMS)
ap.add_argument("--sigma-mult", type=float, default=1.0)
ap.add_argument("--init", default="random", help="starting HiloNet: random, muon (distilled toward Muon, "
                "see hilo.distill) or a saved HiloNet (.pt)")
ap.add_argument("--seed", type=int, default=0, help="random init of the HiloNet and ES noise")
ap.add_argument("--out", default="runs/meta_mlp")
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
args = ap.parse_args()
dev = torch.device(args.device)


class Task:
    def init(self, task_seed: int, horizon: int) -> dict:
        g = torch.Generator(device="cpu").manual_seed(task_seed)
        torch.manual_seed(task_seed)
        teacher = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 8)).to(dev)
        model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 8)).to(dev)
        mats, rest = hilo.split_params(model)
        return dict(teacher=teacher, model=model, gen=g, adam=torch.optim.Adam(rest, lr=1e-3),
                    opt=hilo.Hilo(mats, total_steps=horizon, lr=args.lr))

    def unroll(self, net: hilo.HiloNet, st: dict, steps: int) -> tuple[dict, float]:
        st["opt"].net = net
        total = torch.zeros((), device=dev)
        for _ in range(steps):
            x = torch.randn(256, 32, generator=st["gen"]).to(dev)
            loss = F.mse_loss(st["model"](x), st["teacher"](x).detach())
            loss.backward()
            total += loss.detach()
            st["opt"].step()
            st["adam"].step()
            st["model"].zero_grad(set_to_none=True)
        return st, total.item() / steps


def objective(net: hilo.HiloNet, task_seed: int) -> float:
    st = Task().init(task_seed, args.steps)
    xval = torch.randn(1024, 32, generator=st["gen"]).to(dev)
    st, _ = Task().unroll(net, st, args.steps)
    with torch.no_grad():
        return F.mse_loss(st["model"](xval), st["teacher"](xval)).item()


net = hilo.init_net(args.init, hilo.HiloNetConfig(output_norm=args.output_norm), args.seed, device=args.device)
meta = dict(iters=args.iters, pairs=args.pairs, sigma_mult=args.sigma_mult, seed=args.seed, out=args.out)
if args.estimator == "es":
    net = hilo.meta.meta_train(objective, net, **meta)
else:
    net = hilo.meta.meta_train_truncated(Task(), net, horizon=args.steps, trunc=args.trunc, estimator=args.estimator,
                                         **meta)
print(f"meta-trained HiloNet: {args.out}/hilo_final.pt  (use: hilo.Hilo(..., lr={args.lr}, "
      f"net='{args.out}/hilo_final.pt'))")
