"""Meta-train a HiloNet on your own task family (Hilo.meta.train), from a random init.

The task is a small teacher-student MLP regression; replace it with your training loop
(examples/modded_nanogpt/meta_train_proxy.py does this for a GPT): task(net, task_seed) trains a fresh model
with `net` and returns its final validation loss.

    python examples/meta_train.py --iters 10 --pairs 4 --out runs/meta_mlp
"""
import argparse
import functools
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Hilo  # noqa: E402
import Hilo.meta  # noqa: E402


class Task:
    def __init__(self, device="cpu", steps=200, lr=1e-3):
        self.device, self.steps, self.lr = torch.device(device), steps, lr

    def __call__(self, net: Hilo.net, task_seed: int) -> float:
        g = torch.Generator().manual_seed(task_seed)
        torch.manual_seed(task_seed)
        teacher = nn.Sequential(nn.Linear(32, 64), nn.Tanh(), nn.Linear(64, 8)).to(self.device)
        model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                              nn.Linear(128, 8)).to(self.device)
        mats, rest = Hilo.split_params(model)
        opt = Hilo.optimizer(mats, total_steps=self.steps, lr=self.lr, net=net)
        adam = torch.optim.Adam(rest, lr=1e-3)
        for _ in range(self.steps):
            x = torch.randn(256, 32, generator=g).to(self.device)
            F.mse_loss(model(x), teacher(x).detach()).backward()
            opt.step()
            adam.step()
            model.zero_grad(set_to_none=True)
        xval = torch.randn(1024, 32, generator=g).to(self.device)
        with torch.no_grad():
            return F.mse_loss(model(xval), teacher(xval)).item()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--steps", type=int, default=200, help="inner training steps (the meta-objective horizon)")
    ap.add_argument("--lr", type=float, default=1e-3, help="Hilo lr during meta-training (keep it when using the net)")
    ap.add_argument("--init", help="a saved HiloNet (.pt) to continue from (default: a random init)")
    ap.add_argument("--output-norm", default="none", choices=Hilo.OUTPUT_NORMS, help="for the random init")
    ap.add_argument("--seed", type=int, default=0, help="random init of the HiloNet and ES noise")
    ap.add_argument("--out", default="runs/meta_mlp")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--devices", help="comma-separated worker devices, e.g. cuda:0,cuda:0,cuda:1 (default: serial)")
    args = ap.parse_args()

    net = Hilo.net.load(args.init) if args.init else Hilo.net(Hilo.config(output_norm=args.output_norm), args.seed)
    task = functools.partial(Task, steps=args.steps, lr=args.lr)
    meta = dict(iters=args.iters, pairs=args.pairs, seed=args.seed, out=args.out)
    if args.devices:
        meta |= dict(make_task=task, devices=args.devices.split(","))
    Hilo.meta.train(None if args.devices else task(args.device), net, **meta)
    print(f"meta-trained HiloNet: {args.out}/hilo_final.pt  (use: Hilo.optimizer(..., lr={args.lr}, "
          f"net='{args.out}/hilo_final.pt'))")


if __name__ == "__main__":
    main()
