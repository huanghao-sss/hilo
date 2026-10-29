"""Minimal hilo usage: matrices with Hilo, everything else with Adam. Runs on CPU in seconds.

    python examples/quickstart.py [--net <hilonet.pt>]

Without --net this uses a randomly initialized HiloNet; examples/meta_train.py meta-trains one.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hilo  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--net", help="a saved HiloNet .pt (default: random init)")
ap.add_argument("--lr", type=float, default=1e-3, help="use the lr the HiloNet was meta-trained with")
ap.add_argument("--steps", type=int, default=300)
args = ap.parse_args()

torch.manual_seed(0)
# teacher-student regression with a 3-layer MLP
teacher = nn.Sequential(nn.Linear(32, 128), nn.Tanh(), nn.Linear(128, 10))
model = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 10))

mats, rest = hilo.split_params(model)  # 2-D weights -> Hilo, biases -> Adam
opt_mat = hilo.Hilo(mats, total_steps=args.steps, lr=args.lr, net=args.net)
opt_rest = torch.optim.Adam(rest, lr=1e-3)

for step in range(args.steps):
    x = torch.randn(256, 32)
    with torch.no_grad():
        y = teacher(x)
    loss = F.mse_loss(model(x), y)
    loss.backward()
    opt_mat.step()
    opt_rest.step()
    model.zero_grad(set_to_none=True)
    if step % 50 == 0 or step == args.steps - 1:
        print(f"step {step:4d}  loss {loss.item():.5f}")
