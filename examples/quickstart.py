"""Meta-train a HiloNet, then train a model with it.

    python examples/quickstart.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import Hilo  # noqa: E402
import Hilo.meta  # noqa: E402

steps, lr = 50, 1e-3
gen = torch.Generator().manual_seed(0)
train_features = torch.randn(256, 32, generator=gen)
val_features = torch.randn(256, 32, generator=gen)
true_weight = torch.randn(32, 1, generator=gen)
train_targets = train_features @ true_weight + 0.1 * torch.randn(256, 1, generator=gen)
val_targets = val_features @ true_weight + 0.1 * torch.randn(256, 1, generator=gen)


def objective(net, task_seed):
    """Return the validation loss."""
    torch.manual_seed(task_seed)
    model = nn.Linear(32, 1)
    mats, rest = Hilo.split_params(model)
    opt = Hilo.optimizer(mats, total_steps=steps, lr=lr, net=net)
    adam = torch.optim.Adam(rest, lr=1e-3)
    for _ in range(steps):
        F.mse_loss(model(train_features), train_targets).backward()
        opt.step()
        adam.step()
        model.zero_grad(set_to_none=True)
    with torch.no_grad():
        return F.mse_loss(model(val_features), val_targets).item()


def main():
    trained = Hilo.meta.train(objective, Hilo.net(seed=0), iters=2, pairs=2, out="runs/quickstart")
    print(objective(trained, task_seed=1))


if __name__ == "__main__":
    main()
