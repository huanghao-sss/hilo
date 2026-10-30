# Meta-training tutorial: training a HiloNet with ES

English | [简体中文](meta_training.zh-CN.md)

This tutorial explains from scratch what `Hilo.meta` does, how full-unroll evolution strategies (ES) estimate the
meta-gradient, and how to tune and monitor meta-training. It assumes gradient descent and PyTorch, and no prior
knowledge of evolution strategies.

## 1. The problem: two levels of optimization

Meta-training has two nested loops:

- **Inner**: an ordinary training run. The model parameters $w$ start from their init and are trained for $T$
  steps with `Hilo.optimizer`, each update given by the HiloNet:
  $w_{t+1} = w_t - \text{lr}\cdot U_\phi(G_t, M_t, w_t, t/T)$.
- **Outer**: adjust the HiloNet parameters $\phi$ so that inner training ends better.

The outer loop minimizes the **meta-objective** $\mathcal{L}(\phi)$: the final loss of an inner training run with
$\phi$, in expectation over the task distribution. A "task" is fixed by a task seed: the model
init and the data order.

In principle the gradient of $\mathcal{L}(\phi)$ could be backpropagated through the whole inner run, but that
means second-order differentiation through hundreds or thousands of training steps: memory grows linearly with the
number of steps, and gradients through long unrolls explode or vanish. The learned-optimizer literature therefore
mostly uses **zeroth-order estimates**: run the inner training, look only at loss values, and estimate the gradient
direction from random perturbations. `Hilo.meta.train` does this with ES.

Terms:

| Term | Meaning |
|---|---|
| Particle | One perturbed copy of the HiloNet parameters $\phi \pm \sigma\varepsilon$ and the inner run it drives |
| Antithetic pair | Two particles sharing one task: $\phi + \sigma\varepsilon$ and $\phi - \sigma\varepsilon$ |
| Horizon | Total number of steps $T$ of one inner run (also the `total_steps` of `Hilo.optimizer`) |
| Outer step / iteration | One update of $\phi$; one line of the log |

## 2. Evolution strategies (ES): gradients from loss values

### Gaussian smoothing

ES does not optimize $\mathcal{L}(\phi)$ directly but its **Gaussian smoothing**:

$$J_\sigma(\phi) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\big[\mathcal{L}(\phi + \sigma\varepsilon)\big]$$

$J_\sigma$ is the average of $\mathcal{L}$ over a neighborhood of radius about $\sigma$. The loss of an inner run is
often very rugged in $\phi$ (chaos, divergence, discrete jumps); only after smoothing is there a usable gradient. A
larger $\sigma$ smooths more but moves further from the original objective.

The gradient of $J_\sigma$ has an expression that needs only loss values:

$$\nabla J_\sigma(\phi) = \frac{1}{\sigma}\,\mathbb{E}\big[\mathcal{L}(\phi + \sigma\varepsilon)\,\varepsilon\big]
= \frac{1}{2\sigma}\,\mathbb{E}\big[\big(\mathcal{L}(\phi + \sigma\varepsilon) - \mathcal{L}(\phi - \sigma\varepsilon)\big)\,\varepsilon\big]$$

Intuitively: try both sides of a random direction $\varepsilon$, move toward the side with the lower loss, by an
amount set by the loss difference. Averaging over $P$ samples gives the estimate

$$\hat g = \frac{1}{2\sigma P}\sum_{i=1}^{P}\big(L_i^+ - L_i^-\big)\,\varepsilon_i$$

### Two variance-reduction tricks

- **Antithetic sampling**: every $\varepsilon$ is used with both $+$ and $-$. The difference cancels the even-order
  terms of $\mathcal{L}$ (including the constant), which greatly reduces the variance.
- **Common random numbers**: the two particles of a pair use **the same task seed**, so their init and data order
  are identical. Then $L^+ - L^-$ reflects only the perturbation, without the noise of "this task is just hard".
  This is also why the inner training should be deterministic (section 6): the same $\phi$ on the same task twice
  should give exactly the same loss.

### Full-unroll ES

Each outer step samples $P$ directions $\varepsilon_i$, runs $2P$ **complete** inner trainings, takes their final
validation losses as $L_i^\pm$, and updates $\phi$.

- The meta-objective is the final loss you actually care about; nothing about the inner run other than that loss
  enters the estimate.
- The $2P$ runs are fully independent and parallelize trivially, so more compute directly buys more samples.
- $\phi$ is updated once per $2P$ full runs.

## 3. Writing your own task

`Hilo.meta.train` takes a function that trains a model from a task seed with the given HiloNet and returns its final
loss:

```python
def objective(net, task_seed):
    torch.manual_seed(task_seed)                       # init and data order are fixed by task_seed
    model = build_model()
    mats, rest = Hilo.split_params(model)
    opt = Hilo.optimizer(mats, total_steps=T, lr=1e-3, net=net)
    adam = torch.optim.Adam(rest, lr=1e-3)
    for step in range(T):
        loss_fn(model, next_batch()).backward()
        opt.step(); adam.step(); model.zero_grad(set_to_none=True)
    return validation_loss(model)

net = Hilo.meta.train(objective, Hilo.net(seed=0), iters=100, pairs=8, out="runs/my_net")
```

## 4. Hyperparameters

| Parameter | Role | Advice |
|---|---|---|
| `sigma_mult` | Multiplier of the perturbation size $\sigma$ (base: 0.02 for matrix parameters, 0.01 for vectors) | Start at 1.0. Increase it if `pair_absdiff` is tiny and `drift` barely grows (the perturbation has no visible effect); decrease it if many runs diverge (`dead` > 0) |
| `pairs` | Antithetic pairs $P$ per outer step; compute $\propto 2P$ | Start at 8; more if the direction is noisy |
| `alpha` | Outer step in units of $\sigma$: $\phi \leftarrow \phi - \alpha\,\sigma\cdot\text{step}$ | `sgdm`: 1.0 (the default); `adam` moves each coordinate by about $\alpha\sigma$ per step, use 0.01–0.1 |
| `optimizer` | `sgdm` (momentum averaging) or `adam` | `sgdm` |
| `momentum` | Momentum of `sgdm` / β1 of `adam` | 0.9 |
| `shaping` | How loss differences become weights: `rms` divides by the RMS of the pair differences (insensitive to the loss scale); `rank` uses centered ranks (robust to outliers); `none` the raw difference (unbiased, but the step scales with the loss) | `rms`; `rank` if losses occasionally take extreme values |

The `lr` of `Hilo.optimizer` and the number of inner steps are not parameters of `Hilo.meta` but part of the task:
use the same lr in meta-training and in use, and make the inner runs as close to the target training length as
the budget allows.

## 5. Reading the log

Every outer step writes one line to the terminal and to `out/log.jsonl`:

| Field | Meaning | What to look for |
|---|---|---|
| `it` | Outer step index | |
| `pair_mean` | Mean final loss of all particles in this step | Should fall over meta-training, but tasks change every step; look at the trend |
| `pair_absdiff` | Mean absolute loss difference of the pairs | Strength of the signal; near 0 means $\sigma$ is too small or inner nondeterminism hides the signal |
| `dead` | Particles whose loss is not finite | Persistently above 0 means the perturbation or the Hilo `lr` is too large |
| `step_norm` | Norm of this step's change of $\phi$ | Should be steady; spikes often precede instability |
| `drift` | Distance of $\phi$ from its start, in units of $\sigma$ | Steady growth means progress in a consistent direction |

All losses during meta-training mix in task differences; to tell whether a HiloNet got better, use the held-out
evaluation of section 6.

## 6. End-to-end workflow

1. **Write the task**: an `objective(net, task_seed)`. Make it deterministic:
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` and `torch.use_deterministic_algorithms(True)`; the same HiloNet on the same
   task seed twice should give bitwise-identical results.
2. **Choose the starting point**: a random init `Hilo.net(cfg, seed)`, or `Hilo.net.load("net.pt")` to continue
   from an earlier HiloNet.
3. **Small trial run**: a few outer steps; check that `pair_absdiff` is clearly above 0 and `dead` is 0.
4. **Meta-train**: save the log and checkpoints with `out=` (a `hilo_itXXXX.pt` every `save_every` steps); after an
   interruption, resume with the same `out`. With several GPUs, parallelize with `make_task` + `devices`.
5. **Held-out evaluation**: on task seeds not seen in meta-training, compare the new HiloNet pairwise with a
   reference (random init, an earlier checkpoint or a hand-designed optimizer), looking at the mean loss difference
   and its standard error (`meta_train_proxy.py --eval A --against B`).
6. **Scale up step by step**: start the next stage from `Hilo.net.load(...)` and continue meta-training on wider,
   longer tasks closer to the target.
7. **Use it**: `Hilo.optimizer(mats, total_steps=T, lr=<the meta-training lr>, net="runs/.../hilo_final.pt")`, with
   the same execution options (`precision`, ...) as in meta-training.

## 7. Parallelism

The particles' runs within one outer step are independent and can be split over worker processes: one process per
`devices` entry; listing a GPU several times runs several on it at once. Each worker builds the objective
once with `make_task(device)`, which must be picklable (a module-level function or a `functools.partial`). Workers
take jobs from a shared queue, balancing the load automatically. With deterministic tasks, parallel and serial results are bitwise identical.

```python
net = Hilo.meta.train(make_task=build_objective, devices=["cuda:0", "cuda:1"], net=Hilo.net(seed=0),
                      net_options=dict(precision="bf16"), out="runs/my_net")
```

`FullES`, `Outer`, `minimize` and `EvalPool` are exported as well and compose into other meta-training loops.

## References

- Salimans et al., 2017. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.*
- Metz et al., 2019. *Understanding and Correcting Pathologies in the Training of Learned Optimizers.*
- Metz et al., 2022. *VeLO: Training Versatile Learned Optimizers by Scaling Up.*
- Thérien et al., 2024. *μLO: Compute-Efficient Meta-Generalization of Learned Optimizers.*
- Moudgil et al., 2025. *Celo: Training Versatile Learned Optimizers on a Compute Diet.*
- Moudgil et al., 2026. *Celo2: Towards Learned Optimization Free Lunch.*
- Huang et al., 2026. *Efficient Long-Horizon Learning for Learned Optimization.*
