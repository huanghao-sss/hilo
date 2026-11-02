# Using Hilo on CIFAR-10 airbench

The example is Keller Jordan's [airbench94 Muon speedrun](https://github.com/KellerJordan/cifar10-airbench/blob/master/airbench94_muon.py)
([repo](https://github.com/KellerJordan/cifar10-airbench), MIT). On one A100 the published Muon run reaches
94.01% test-time-augmentation accuracy in 2.59 seconds. The network is a small convolutional net on CIFAR-10:
eight epochs, batch 2000, patch whitening, flip and a two-pixel translation, BatchNorm, and SGD with Nesterov
momentum on the whitening bias, the BatchNorm biases and the linear classifier.

`train.py` keeps that procedure. The six convolution matrices, which `airbench94_muon.py` updates with Muon,
go to `Hilo.optimizer`. Each convolution is the matrix `(out_channels, in_channels * 9)`, the same flattening
Muon orthogonalizes. Before every Hilo step the matrix is rescaled to Frobenius norm `sqrt(out_channels)`,
the same rescaling Muon applies, and Hilo replaces the orthogonal update. The HiloNet config is
`betas=(0.6,)`, Muon's momentum, and `output_norm="row"`, which gives the update Muon's norm. The learning
rate is Muon's 0.24 with its linear decay, so the network chooses the direction of each step. SGD keeps the
original schedule. The Muon baseline itself is the official script.

One run is the whole airbench94 schedule, so meta-training does not need a scaled-down proxy. The objective
is the test cross-entropy. `--once` and the logs also report accuracy; the published number to beat is the
test-time-augmentation accuracy.

Requires `torchvision` in addition to `pip install -e <Hilo repo>`.

```bash
# one run with a meta-trained net. With no --init the readout is zero, so the convolutions stay at init.
python examples/cifar10_airbench/train.py --once --init runs/airbench/hilo_final.pt

# meta-train with ES. Each outer step trains 2 * pairs networks from scratch. --init starts from a saved net,
# for example one fitted to the Muon step.
python examples/cifar10_airbench/train.py --out runs/airbench --init hilo_muon.pt --iters 100 --pairs 8

# several evaluations at once on one GPU (the network is small); repeat the device
python examples/cifar10_airbench/train.py --out runs/airbench --devices cuda:0,cuda:0,cuda:0

python examples/cifar10_airbench/train.py --eval runs/airbench/hilo_final.pt --seeds 8
python examples/cifar10_airbench/train.py --eval runs/airbench/hilo_final.pt --against runs/airbench/hilo_it0010.pt
```

`--es-matrix` and `--es-vector` are the ES scales on 2-D and 1-D HiloNet tensors. `--optimizer` selects the `torch.optim` class the script passes to `Hilo.meta.train`
(`sgd`, the default, or `adam`). `--alpha` is that optimizer's learning rate (default `1e-3`). For SGD it counts steps in units of the ES
perturbation scale. `--momentum` is forwarded to that optimizer. `--shaping` is the other argument of `Hilo.meta.train`. Each
iteration appends one line to `out/log.jsonl`. `--eval` uses held-out task seeds, disjoint from meta-training. With
`--against`, the two HiloNets share each task seed.

`--steps` shortens a run; the default is eight epochs. Evaluations use deterministic CUDA kernels, so the same
net and task seed return the same loss. `--nondeterministic` turns that off and enables cuDNN autotuning.
`--compile-model` compiles the network the way the speedrun does.
