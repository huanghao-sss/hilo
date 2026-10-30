# Hilo: HiHiTech's Learned Optimizer

English | [简体中文](README.zh-CN.md)

Hilo is a **general framework for learned matrix optimizers**. The update U of every matrix parameter W is computed
by a small matrix Transformer (the HiloNet, `Hilo.net`), and then `W ← W − lr·U`. The parameters of the HiloNet
are found by meta-training: on a family of training tasks you care about, Hilo directly optimizes "the validation
loss after training with it".

| Module | Purpose |
|---|---|
| `Hilo.optimizer` | The optimizer (a `torch.optim.Optimizer`), with sharded multi-GPU computation |
| `Hilo.net` / `Hilo.config` | Configurable HiloNet architecture; random init; save / load |
| `Hilo.meta` | Meta-training (`Hilo.meta.train`): full-unroll antithetic ES; the objective is your own training loop |

A HiloNet starts from a random init. After meta-training on your task family, pass the saved `.pt` to
`Hilo.optimizer`.

## Why learned optimizers

In [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html) (2019), Rich Sutton summarizes
seventy years of AI research: building human knowledge into a method helps in the short term but soon plateaus;
what wins in the long run are general methods that scale with compute, namely search and learning. Chess, Go,
speech recognition and computer vision all went this way: hand-designed features and rules were eventually
overtaken by what more compute could learn.

The optimizers we train neural networks with are still hand-designed, and each generation encodes a human
hypothesis about the loss surface:

- Momentum: gradient directions are correlated across steps.
- Adam: diagonal preconditioning by per-coordinate second moments.
- Shampoo, SOAP: Kronecker-structured approximations of curvature.
- Muon: orthogonalize the momentum so that all singular directions of the update have equal size.

On top of these come learning-rate warmup and decay, weight decay, gradient clipping, and a set of hyperparameters
to tune per task. These designs were adopted partly because people can derive and analyze them, not because they
are the best possible update rules.

An update rule is just a function: it takes the gradient, its history, the weights and the training progress, and
returns this step's update. A learned optimizer represents this function by a network and optimizes it directly
for "how well it trains a family of tasks". Progress then comes from more meta-training tasks, proxy tasks closer to
the target, and longer meta-training, that is, from compute, instead of from the next hypothesis someone comes up
with. This line of work is moving fast:

- VeLO (2022) trained a general-purpose learned optimizer with 4000 TPU-months.
- Celo2 (2026) meta-trained with only 4.5 GPU-hours, scaled stably to GPT-3 1.3B, and beat a tuned AdamW.
- ELO (2026) went further, beating AdamW across language models and ImageNet and matching Muon on language models.

Hilo follows this idea:

- **Raw signals only.** The inputs of a HiloNet are the gradient, gradient EMAs at a few time scales, the weights
  and the training progress t/T. There is no orthogonalization, no Adam-style preconditioning and no learning-rate
  schedule; how to use these signals is left entirely to meta-training.
- **Only the structure of the problem.** The structure that is kept is about the shape of the problem: equivariance
  to permutations of matrix rows and columns, so one network applies to matrices of any shape, and invariance to an
  overall scaling of the inputs. Like the translation equivariance of a convolution, it describes the problem and
  does not prescribe a solution.

The price is running a network at every step, plus the compute for meta-training.

## Installation

```bash
pip install -e .                 # depends on torch only
```

## Quick start

```python
import Hilo

mats, rest = Hilo.split_params(model)       # 2-D matrices -> Hilo; embeddings, heads, gains, biases -> another optimizer
opt = Hilo.optimizer(mats, total_steps=T, lr=1e-3, net="runs/my_net/hilo_final.pt")
adam = torch.optim.Adam(rest, lr=1e-3)

for step in range(T):
    model(batch).backward()
    opt.step(); adam.step()
    model.zero_grad(set_to_none=True)
```

- `net`: a `Hilo.net`, a saved `*.pt`, or `None` (a randomly initialized HiloNet).
- `lr`: the multiplier of U; use the value the HiloNet was meta-trained with. As with any PyTorch optimizer, you can
  also change `group["lr"]` yourself.
- `total_steps`: the HiloNet takes the training progress t/T as input, so it must be given.
- `weight_decay`: optional decoupled weight decay, default 0.
- Distributed: after the gradient all-reduce, ranks split the matrices by estimated HiloNet cost (each keeps the
  momenta of its share only) and broadcast the results (`shard=True`, default); no extra communication code needed.

An example that runs on CPU in seconds: `python examples/quickstart.py`.

## HiloNet

```python
cfg = Hilo.config(d=16, heads=2, mlp_hidden=32, tie=12, attn="linear",
                  betas=(0.5, 0.9, 0.99, 0.999), output_norm="none")
net = Hilo.net(cfg, seed=0)                 # random init
net.save("net.pt");  net = Hilo.net.load("net.pt")
```

Every coordinate of a matrix W (m×n) carries the channels `[G, M_β1, …, M_βk, W]` (M_β are EMAs of the gradient
with the coefficients `betas`), each divided by its tensor RMS and clipped at ±`clip`, so the network is invariant
to an overall scaling of gradients and weights. The global features are `log(rms M_β / rms G)`, `log(rms W)`, t/T,
1−t/T and `log(m/n)`. Coordinates are embedded in d dimensions and pass through blocks of
[row attention → column attention → MLP] (residual + RMSNorm). Row attention treats rows as tokens and scores
them by channel bilinear forms summed over all columns (with one channel and linear attention this is XXᵀ), so the
network is equivariant to row and column permutations and applies to any matrix shape. A linear read-out gives X,
and `output_norm` turns it into U. The random init has the residual branches at zero, so a new HiloNet is a
per-coordinate linear map of the normalized channels.

| Field | Meaning |
|---|---|
| `d`, `heads`, `mlp_hidden` | Coordinate embedding width, attention heads, MLP width |
| `blocks` / `tie` | Number of distinct blocks; or, with `tie > 0`, one shared block applied `tie` times |
| `attn` | `linear` (scores / m) or `softmax` |
| `betas` | Coefficients of the input gradient EMAs |
| `clip` | Clipping threshold of the normalized channels |
| `output_norm` | Norm of the update, see below |

Update norm `output_norm` (`Hilo.norms`):

| Option | U |
|---|---|
| `none` (default) | U = X |
| `rms` | RMS(U) = 1 |
| `frobenius` | ‖U‖_F = 1 |
| `row` | ‖U‖_F = √m (mean squared row norm 1) |

All options except `none` fix the magnitude, so the step size is set by `lr` alone. The right `lr` scale differs
between options, so keep the `lr` of meta-training.

Execution options (not part of the network and not saved in the `.pt`; keep them the same in meta-training and use):

```python
net = Hilo.net.load("net.pt", precision="auto", backend="auto", compile=False, max_coords=2**23)
```

- `precision`: precision of the attention matmuls, `bf16`, `tf32`, `fp32` or `auto` (default: bf16 on CUDA, fp32 on
  CPU). The residual stream, norms and MLP are always float32. Output error relative to fp32: about 0.4% for bf16,
  0.02% for tf32.
- `backend`: `auto` (default) runs fused Triton kernels on CUDA in bf16 (`Hilo/kernels.py`; needs Triton and `d`,
  `mlp_hidden` powers of two ≥ 16), otherwise the PyTorch implementation; `triton` requires the kernels, `torch`
  never uses them. Each kernel does everything a coordinate computes between two attention matmuls (out projection
  and residual, MLP, RMSNorm, q / k / v written in the layout the next attention reads); the attention matmuls are
  cuBLAS. Its error relative to fp32 is the same as the PyTorch bf16 path.
- `compile`: `torch.compile` the PyTorch implementation; the first call per matrix shape compiles for about 25 s.
- `max_coords`: matrices of one shape are processed in chunks of at most this many coordinates, which bounds peak
  memory (about 0.5 KB per coordinate).

Cost: every attention layer forms m×m and n×n score matrices, about O(iters · mn(m+n) · d) per matrix, plus
O(iters · mn · d) elementwise work. Time per matrix in bf16 for the default config on an A100 (measured while sharing
the GPU, which roughly halves the available bandwidth and matmul throughput):

| Shape | PyTorch | PyTorch + compile | Triton (default) |
|---|---|---|---|
| 256×256 | 9.3 ms | 6.2 ms | 1.4 ms |
| 1024×256 | 38 ms | 26 ms | 6.4 ms |
| 768×768 | 73 ms | 40 ms | 16 ms |
| 3072×768 | 380 ms | 276 ms | 114 ms |

A full step over the 72 hidden matrices of modded-nanogpt #36 takes 12.7 s with PyTorch and 3.4 s with Triton on one
GPU (sharded over 8 GPUs, about 1/8 of that). With Triton the attention matmuls take most of the time and run near
the GPU's matmul throughput, so the remaining cost is set by the O(iters · mn(m+n) · d) flops of the architecture.

## Meta-training (`Hilo.meta`)

**Full tutorial: [docs/meta_training.md](docs/meta_training.md)** ([简体中文](docs/meta_training.zh-CN.md)). It
covers the two levels of optimization and their terms, why ES estimates gradients from loss values alone,
hyperparameters, log fields, and the end-to-end workflow.

The HiloNet parameters φ are trained with full-unroll antithetic ES: each outer step trains 2 × pairs models from
scratch with perturbed HiloNets φ ± σε and moves φ toward the side with the lower final validation loss. It only
runs your training loop and looks at loss values, with no backpropagation through training, and the meta-objective
is the final loss itself.

```python
import Hilo, Hilo.meta

# objective(net, task_seed): train a model from task_seed (with Hilo.optimizer(..., net=net)), return its final val loss
net = Hilo.meta.train(objective, Hilo.net(seed=0), iters=100, pairs=8, out="runs/my_net")
# -> out/hilo_final.pt, plus hilo_itXXXX.pt checkpoints, log.jsonl and a resumable state.pt
```

Also available, see the tutorial:

- **Outer optimizer and fitness shaping**: `optimizer`, `shaping`, `alpha`.
- **Worker processes**: `make_task` + `devices`.

Runnable example: `python examples/meta_train.py`, on a small teacher-student MLP.

## modded-nanogpt example

[`examples/modded_nanogpt/`](examples/modded_nanogpt/README.md): the official baseline of modded-nanogpt Track 3
(#36, `train_gpt_simple.py`) with its hidden matrices trained by `Hilo.optimizer` (changes in
`train_gpt_hilo.diff`), plus a script that meta-trains a HiloNet on a scaled-down copy of the baseline.

## Repository layout

```
hilo/
├── Hilo/                       # the Python package
│   ├── __init__.py             # public API
│   ├── optim.py                # Hilo.optimizer (momentum state, sharding, weight decay), split_params
│   ├── hilonet.py              # Hilo.net, Hilo.config: (G, M, W, t/T) -> U; random init, save / load
│   ├── kernels.py              # Triton kernels of the HiloNet forward (CUDA, bf16)
│   ├── norms.py                # update norms: none / rms / frobenius / row
│   └── meta/                   # meta-training
│       ├── __init__.py         #   Hilo.meta.train
│       ├── es.py               #   full-unroll ES: FullES
│       ├── core.py             #   outer loop: shaping (rms / rank / none), outer optimizers (sgdm / adam)
│       └── pool.py             #   worker processes: EvalPool
├── docs/
│   ├── meta_training.md        # meta-training tutorial: ES, hyperparameters, log, workflow
│   └── meta_training.zh-CN.md
├── examples/
│   ├── quickstart.py           # minimal usage (CPU)
│   ├── meta_train.py           # meta-training on your own task (a teacher-student MLP)
│   └── modded_nanogpt/         # the official modded-nanogpt #36 baseline
│       ├── README.md
│       ├── README.zh-CN.md
│       ├── train_gpt_hilo.py   #   the official train_gpt_simple.py: Muon -> Hilo.optimizer
│       ├── train_gpt_hilo.diff #   the changes against the official script
│       └── meta_train_proxy.py #   meta-train / evaluate a HiloNet on the scaled-down baseline
├── tests/
│   ├── test_hilo.py
│   ├── test_distributed.py
│   └── fixtures/hilonet_ref.pt
├── pyproject.toml
├── LICENSE
├── README.md
└── README.zh-CN.md
```

## Tests

```bash
python tests/test_hilo.py          # reference implementation, save / load, update norms, weight decay, state_dict, meta-training
python tests/test_distributed.py   # 2-process sharding matches a single process
```

## License

MIT. The training scripts in `examples/modded_nanogpt/` are adapted from
[modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt) (MIT).
