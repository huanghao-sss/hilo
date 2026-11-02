"""Train the CIFAR-10 airbench94 network with Hilo on the convolutional filters.

The procedure is Keller Jordan's airbench94 Muon speedrun
(https://github.com/KellerJordan/cifar10-airbench, MIT): eight epochs, batch 2000, whitening,
the same augmentation and BatchNorm, and SGD on the biases and the classifier. Muon updates the six
convolution matrices there; here those matrices go to ``Hilo.optimizer``. A convolution is stored as
``(out_channels, in_channels * 9)``, the matrix Muon orthogonalizes. The objective is the test
cross-entropy. The published Muon baseline is 94.01% test-time-augmentation accuracy.

    python examples/cifar10_airbench/train.py --once
    python examples/cifar10_airbench/train.py --out runs/airbench --iters 100 --pairs 8
    python examples/cifar10_airbench/train.py --eval runs/airbench/hilo_final.pt --seeds 8
"""
import argparse
import functools
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
import torch.nn.functional as F
import torchvision
from torch import nn

import Hilo
import Hilo.meta

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2470, 0.2435, 0.2616)
EPOCHS = 8
BIAS_EPOCHS = 3
# One gradient EMA at Muon's momentum, and an update of Frobenius norm sqrt(out_channels), the norm of
# Muon's orthogonalized step for these wide matrices.
CONFIG = dict(betas=(0.6,), output_norm="row")


def load_split(path: str, train: bool, device: torch.device, dtype: torch.dtype):
    """CIFAR-10 images in [0, 1] as ``(N, 3, 32, 32)`` and integer labels, cached next to the download."""
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    file = root / ("train.pt" if train else "test.pt")
    if not file.exists():
        dataset = torchvision.datasets.CIFAR10(root, download=True, train=train)
        torch.save({"images": torch.tensor(dataset.data), "labels": torch.tensor(dataset.targets)}, file)
    data = torch.load(file, map_location="cpu", weights_only=True)
    images = data["images"].to(device=device, dtype=dtype).div_(255).permute(0, 3, 1, 2).contiguous()
    if device.type == "cuda":
        images = images.contiguous(memory_format=torch.channels_last)
    return images, data["labels"].to(device)


def normalize(images: torch.Tensor) -> torch.Tensor:
    mean = images.new_tensor(CIFAR_MEAN).view(1, 3, 1, 1)
    std = images.new_tensor(CIFAR_STD).view(1, 3, 1, 1)
    return (images - mean) / std


def batch_flip_lr(images: torch.Tensor) -> torch.Tensor:
    """Flip each image independently with probability 1/2. Airbench reuses this for the whole epoch."""
    flip = (torch.rand(len(images), device=images.device) < 0.5).view(-1, 1, 1, 1)
    return torch.where(flip, images.flip(-1), images)


def batch_crop(images: torch.Tensor, crop_size: int) -> torch.Tensor:
    """The airbench random translation: one integer shift per image, shared by its channels."""
    radius = (images.size(-1) - crop_size) // 2
    shifts = torch.randint(-radius, radius + 1, size=(len(images), 2), device=images.device)
    out = torch.empty((len(images), 3, crop_size, crop_size), device=images.device, dtype=images.dtype)
    if images.device.type == "cuda":
        out = out.contiguous(memory_format=torch.channels_last)
    if radius <= 2:
        for shift_y in range(-radius, radius + 1):
            for shift_x in range(-radius, radius + 1):
                mask = (shifts[:, 0] == shift_y) & (shifts[:, 1] == shift_x)
                out[mask] = images[mask, :, radius + shift_y:radius + shift_y + crop_size,
                                   radius + shift_x:radius + shift_x + crop_size]
    else:
        wide = torch.empty((len(images), 3, crop_size, crop_size + 2 * radius), device=images.device, dtype=images.dtype)
        for shift in range(-radius, radius + 1):
            mask = shifts[:, 0] == shift
            wide[mask] = images[mask, :, radius + shift:radius + shift + crop_size, :]
        for shift in range(-radius, radius + 1):
            mask = shifts[:, 1] == shift
            out[mask] = wide[mask, :, :, radius + shift:radius + shift + crop_size]
    return out


class BatchNorm(nn.BatchNorm2d):
    """Airbench uses a low running-stat momentum and does not train the scale."""

    def __init__(self, num_features):
        super().__init__(num_features, eps=1e-12, momentum=0.4)
        self.weight.requires_grad = False


class Conv(nn.Module):
    """A 3×3 convolution whose weight is the matrix ``(out_channels, in_channels * 9)``."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels * 9))
        self.reset_parameters()

    def reset_parameters(self):
        # Kaiming, then Dirac on the first ``in_channels`` filters, as in airbench94's Conv.reset_parameters.
        weight = torch.empty(self.out_channels, self.in_channels, 3, 3)
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
        nn.init.dirac_(weight[:self.in_channels])
        self.weight.data.copy_(weight.flatten(1))

    def forward(self, x):
        weight = self.weight.view(self.out_channels, self.in_channels, 3, 3)
        return F.conv2d(x, weight, padding=1)


class ConvGroup(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.conv1 = Conv(channels_in, channels_out)
        self.pool = nn.MaxPool2d(2)
        self.norm1 = BatchNorm(channels_out)
        self.conv2 = Conv(channels_out, channels_out)
        self.norm2 = BatchNorm(channels_out)
        self.activ = nn.GELU()

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.activ(self.norm1(x))
        x = self.activ(self.norm2(self.conv2(x)))
        return x


class CifarNet(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        widths = {"block1": 64, "block2": 256, "block3": 256}
        whiten_width = 24  # two signed copies of a 2×2×3 patch basis
        self.whiten = nn.Conv2d(3, whiten_width, 2, padding=0, bias=True)
        self.whiten.weight.requires_grad = False
        self.layers = nn.Sequential(
            nn.GELU(),
            ConvGroup(whiten_width, widths["block1"]),
            ConvGroup(widths["block1"], widths["block2"]),
            ConvGroup(widths["block2"], widths["block3"]),
            nn.MaxPool2d(3),
        )
        self.head = nn.Linear(widths["block3"], 10, bias=False)
        for module in self.modules():
            if isinstance(module, BatchNorm):
                module.float()
            else:
                module.to(dtype=dtype)

    def reset(self):
        for module in self.modules():
            if type(module) in (nn.Conv2d, Conv, BatchNorm, nn.Linear):
                module.reset_parameters()
        self.head.weight.data *= 1 / self.head.weight.data.std()

    def init_whiten(self, train_images, eps=5e-4):
        channels, (height, width) = train_images.shape[1], self.whiten.weight.shape[2:]
        patches = train_images.unfold(2, height, 1).unfold(3, width, 1).transpose(1, 3).reshape(-1, channels, height, width).float()
        flat = patches.view(len(patches), -1)
        covariance = (flat.T @ flat) / len(flat)
        eigenvalues, eigenvectors = torch.linalg.eigh(covariance, UPLO="U")
        basis = eigenvectors.T.reshape(-1, channels, height, width) / torch.sqrt(eigenvalues.view(-1, 1, 1, 1) + eps)
        self.whiten.weight.data[:] = torch.cat((basis, -basis)).to(self.whiten.weight.dtype)

    def forward(self, x, whiten_bias_grad=True):
        bias = self.whiten.bias if whiten_bias_grad else self.whiten.bias.detach()
        x = F.conv2d(x, self.whiten.weight, bias)
        x = self.layers(x)
        x = x.view(len(x), -1)
        return self.head(x) / x.size(-1)


def convolution_matrices(model: CifarNet) -> list[torch.Tensor]:
    return [p for name, p in model.named_parameters() if name.endswith(("conv1.weight", "conv2.weight"))]


class Airbench:
    """One airbench94 run. ``task_seed`` draws the initialization, the augmentation and the data order."""

    def __init__(self, data: str = "cifar10", device: str = "cuda", steps: int | None = None,
                 batch_size: int = 2000, lr: float = 0.24, deterministic: bool = True, compile: bool = False):
        if deterministic:
            torch.use_deterministic_algorithms(True)
        elif device.startswith("cuda"):
            torch.backends.cudnn.benchmark = True
        self.device = torch.device(device)
        self.dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.batch_size, self.lr = batch_size, lr
        self.train_images, self.train_labels = load_split(data, True, self.device, self.dtype)
        self.test_images, self.test_labels = load_split(data, False, self.device, self.dtype)
        self.test_images = normalize(self.test_images)
        per_epoch = len(self.train_images) // batch_size
        self.steps = steps if steps is not None else EPOCHS * per_epoch
        self.bias_steps = max(1, round(self.steps * BIAS_EPOCHS / EPOCHS)) if steps is not None else BIAS_EPOCHS * per_epoch
        self.model = CifarNet(self.dtype).to(self.device)
        if self.device.type == "cuda":
            self.model = self.model.to(memory_format=torch.channels_last)
        if compile:
            self.model.compile(mode="max-autotune")

    def batches(self):
        images = batch_flip_lr(normalize(self.train_images))
        padded = F.pad(images, (2, 2, 2, 2), "reflect")
        produced = 0
        epoch = 0
        while produced < self.steps:
            batch_images = batch_crop(padded, self.train_images.shape[-1])
            if epoch % 2 == 1:
                batch_images = batch_images.flip(-1)
            indices = torch.randperm(len(batch_images), device=batch_images.device)
            for start in range(0, len(indices) - self.batch_size + 1, self.batch_size):
                choice = indices[start:start + self.batch_size]
                yield batch_images[choice], self.train_labels[choice]
                produced += 1
                if produced >= self.steps:
                    return
            epoch += 1

    def test_metrics(self, tta: bool) -> dict:
        model, images, labels = self.model, self.test_images, self.test_labels
        model.eval()
        loss = correct = 0.0
        with torch.no_grad():
            for start in range(0, len(labels), 2000):
                inputs, targets = images[start:start + 2000], labels[start:start + 2000]
                logits = (tta_logits(model, inputs) if tta else model(inputs)).float()
                loss += F.cross_entropy(logits, targets, reduction="sum").item()
                correct += (logits.argmax(1) == targets).sum().item()
        return {"loss": loss / len(labels), "accuracy": correct / len(labels)}

    def __call__(self, net, task_seed: int) -> float:
        """Train from scratch with ``net`` on the convolutions. Return the test cross-entropy."""
        torch.manual_seed(task_seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(task_seed)
        model = self.model
        model.reset()
        model.init_whiten(normalize(self.train_images[:5000]))
        weight_decay = 2e-6 * self.batch_size
        bias_lr, head_lr = 0.053, 0.67
        norm_biases = [p for name, p in model.named_parameters() if "norm" in name and p.requires_grad]
        sgd = torch.optim.SGD([
            dict(params=[model.whiten.bias], lr=bias_lr, weight_decay=weight_decay / bias_lr),
            dict(params=norm_biases, lr=bias_lr, weight_decay=weight_decay / bias_lr),
            dict(params=[model.head.weight], lr=head_lr, weight_decay=weight_decay / head_lr),
        ], momentum=0.85, nesterov=True, fused=self.device.type == "cuda")
        for group in sgd.param_groups:
            group["initial_lr"] = group["lr"]
        hilo = Hilo.optimizer(convolution_matrices(model), total_steps=self.steps, lr=self.lr, net=net)
        for step, (inputs, labels) in enumerate(self.batches()):
            model.train()
            F.cross_entropy(model(inputs, whiten_bias_grad=step < self.bias_steps), labels,
                            label_smoothing=0.2, reduction="sum").backward()
            sgd.param_groups[0]["lr"] = bias_lr * (1 - step / self.bias_steps)
            for group in sgd.param_groups[1:]:
                group["lr"] = group["initial_lr"] * (1 - step / self.steps)
            hilo.param_groups[0]["lr"] = self.lr * (1 - step / self.steps)
            sgd.step()
            # Muon rescales each filter onto sqrt(out_channels) before the orthogonal step. That rescaling
            # stays here; Hilo replaces the orthogonal update.
            for weight in convolution_matrices(model):
                weight.data.mul_(weight.shape[0] ** 0.5 / weight.detach().float().norm().clamp_min(1e-12))
            hilo.step()
            model.zero_grad(set_to_none=True)
        loss = self.test_metrics(tta=False)["loss"]
        return loss if math.isfinite(loss) else float("nan")


def tta_logits(model: CifarNet, inputs: torch.Tensor) -> torch.Tensor:
    """Airbench test-time augmentation: mirror, plus a one-pixel diagonal shift, each with weight 1/2."""
    def mirror(batch):
        return 0.5 * model(batch) + 0.5 * model(batch.flip(-1))

    padded = F.pad(inputs, (1, 1, 1, 1), "reflect")
    shifted = 0.5 * mirror(padded[:, :, 0:32, 0:32]) + 0.5 * mirror(padded[:, :, 2:34, 2:34])
    return 0.5 * mirror(inputs) + 0.5 * shifted


def make_airbench(device: str, **kwargs) -> Airbench:
    """Builds the task inside a meta-training worker process."""
    return Airbench(device=device, **kwargs)


def report(task: Airbench, net, task_seed: int) -> dict:
    loss = task(net, task_seed)
    metrics = task.test_metrics(tta=False)
    metrics["tta_accuracy"] = task.test_metrics(tta=True)["accuracy"]
    metrics["task"] = task_seed
    # ``task`` already evaluated the test loss; keep that value so a non-finite run stays non-finite.
    metrics["loss"] = loss
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="cifar10")
    parser.add_argument("--steps", type=int, default=None, help="training steps (default: 8 epochs of airbench94)")
    parser.add_argument("--batch-size", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=0.24,
                        help="Hilo learning rate, decayed linearly to zero (Muon's 0.24)")
    parser.add_argument("--out", default="runs/airbench")
    parser.add_argument("--init", help="a saved HiloNet (.pt) to continue from, or to train with under --once")
    parser.add_argument("--config", help="Hilo.config as JSON, e.g. '{\"tie\": 2}'")
    parser.add_argument("--seed", type=int, default=0, help="random init of the HiloNet and ES noise; also the --once task")
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--pairs", type=int, default=8)
    parser.add_argument("--es-matrix", type=float, default=0.02, help="ES scale on 2-D HiloNet tensors")
    parser.add_argument("--es-vector", type=float, default=0.01, help="ES scale on 1-D HiloNet tensors")
    parser.add_argument("--alpha", type=float, default=1e-3,
                        help="outer learning rate (for SGD, in units of sigma)")
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--optimizer", default="sgd", choices=("sgd", "adam"),
                        help="torch.optim class passed to Hilo.meta.train")
    parser.add_argument("--shaping", default="rms", choices=Hilo.meta.SHAPINGS)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--once", action="store_true", help="one training run, then print test loss and accuracy")
    parser.add_argument("--eval", help="evaluate this HiloNet (.pt) on held-out tasks instead of meta-training")
    parser.add_argument("--against", help="with --eval: a second HiloNet, compared pairwise on the same tasks")
    parser.add_argument("--seeds", type=int, default=8, help="held-out tasks for --eval")
    parser.add_argument("--devices", help="comma-separated worker devices, e.g. cuda:0,cuda:0,cuda:0")
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--backend", default="auto", choices=("auto", "triton", "torch"))
    parser.add_argument("--compile-net", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--nondeterministic", action="store_true")
    args = parser.parse_args()
    task_kw = dict(data=args.data, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                   deterministic=not args.nondeterministic, compile=args.compile_model)
    net_options = dict(precision=args.precision, backend=args.backend, compile=args.compile_net)

    if args.once or args.eval:
        task = Airbench(device="cuda" if torch.cuda.is_available() else "cpu", **task_kw)
        if args.once:
            net = Hilo.net.load(args.init, **net_options) if args.init else Hilo.net(Hilo.config(**CONFIG), args.seed)
            print(json.dumps(report(task, net, args.seed)), flush=True)
            return
        nets = {"eval": Hilo.net.load(args.eval, **net_options)}
        if args.against:
            nets["against"] = Hilo.net.load(args.against, **net_options)
        losses = {name: [] for name in nets}
        for index in range(args.seeds):
            task_seed = 999_000_000 + index
            record = {"task": task_seed}
            for name, net in nets.items():
                metrics = report(task, net, task_seed)
                record[name] = metrics["loss"]
                record[f"{name}_accuracy"] = metrics["accuracy"]
                record[f"{name}_tta_accuracy"] = metrics["tta_accuracy"]
                losses[name].append(metrics["loss"])
            print(json.dumps(record), flush=True)
        summary = {f"{name}_mean": torch.tensor(values).mean().item() for name, values in losses.items()}
        if args.against:
            difference = torch.tensor(losses["eval"]) - torch.tensor(losses["against"])
            summary |= {"mean_diff": difference.mean().item(),
                        "sem": (difference.std() / len(difference) ** 0.5).item()}
        print(json.dumps(summary), flush=True)
        return

    config = Hilo.config(**(CONFIG | json.loads(args.config or "{}")))
    net = Hilo.net.load(args.init) if args.init else Hilo.net(config, args.seed)
    print(f"meta-training {net!r} on airbench94, {args.steps or '8 epochs'} steps", flush=True)
    def outer(params):
        if args.optimizer == "adam":
            return torch.optim.Adam(params, lr=args.alpha, betas=(args.momentum or 0.9, 0.999))
        return torch.optim.SGD(params, lr=args.alpha, momentum=args.momentum, dampening=args.momentum)

    meta = dict(iters=args.iters, pairs=args.pairs,
                sigma=Hilo.meta.es_sigma(net.cfg, args.es_matrix, args.es_vector),
                alpha=args.alpha, optimizer=outer, shaping=args.shaping, seed=args.seed,
                out=args.out, save_every=args.save_every, net_options=net_options)
    if args.devices:
        meta |= dict(make_task=functools.partial(make_airbench, **task_kw), devices=args.devices.split(","))
    Hilo.meta.train(None if args.devices else Airbench(device="cuda" if torch.cuda.is_available() else "cpu", **task_kw),
                    net, **meta)


if __name__ == "__main__":
    main()
