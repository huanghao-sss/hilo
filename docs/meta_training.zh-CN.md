# 元训练教程：用 ES 训练 HiloNet

[English](meta_training.md) | 简体中文

这份教程从零讲清楚 `Hilo.meta` 在做什么、全程进化策略（ES）怎样估计元梯度、元训练怎么调、怎么看日志。
只需要知道梯度下降和 PyTorch，不需要事先了解进化策略。

## 1. 问题：两层优化

元训练有两层循环：

- **内层**：一次普通的模型训练。模型参数 $w$ 从初始化开始，用 `Hilo.optimizer` 训练 $T$ 步，每一步的更新由 HiloNet 给出：
  $w_{t+1} = w_t - \text{lr}\cdot U_\phi(G_t, M_t, w_t, t/T)$。
- **外层**：调整 HiloNet 的参数 $\phi$，让内层训练的结果更好。

外层要最小化的是**元目标** $\mathcal{L}(\phi)$：用 $\phi$ 跑完一次内层训练后的终点损失，再对任务分布取期望。
一个"任务"由任务种子决定：模型初始化和数据顺序。

$\mathcal{L}(\phi)$ 对 $\phi$ 的梯度原则上可以反向传播穿过整个内层训练得到，但这要对几百上千步训练做二阶求导：
显存随步数线性增长，长展开的梯度还会爆炸或消失。学习型优化器领域因此普遍改用**零阶估计**：只运行内层训练、
只看损失值，用随机扰动估计梯度方向。`Hilo.meta.train` 用的 ES 就是这样做的。

术语：

| 术语 | 含义 |
|---|---|
| 粒子 | 一份带扰动的 HiloNet 参数 $\phi \pm \sigma\varepsilon$ 及它驱动的那次内层训练 |
| 对偶配对（pair） | 共享同一任务的两个粒子：$\phi + \sigma\varepsilon$ 和 $\phi - \sigma\varepsilon$ |
| horizon | 一次内层训练的总步数 $T$（也是 `Hilo.optimizer` 的 `total_steps`） |
| 外层更新 / iteration | $\phi$ 的一次更新，日志里的一行 |

## 2. 进化策略（ES）：只用损失值估计梯度

### 高斯平滑

ES 不直接优化 $\mathcal{L}(\phi)$，而是优化它的**高斯平滑**版本：

$$J_\sigma(\phi) = \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, I)}\big[\mathcal{L}(\phi + \sigma\varepsilon)\big]$$

$J_\sigma$ 是 $\mathcal{L}$ 在半径约 $\sigma$ 的邻域里的平均。内层训练的损失对 $\phi$ 往往很崎岖（混沌、发散、
离散的跳变），平滑之后才有可用的梯度；$\sigma$ 越大越平滑，但离原目标越远。

$J_\sigma$ 的梯度有一个只需要损失值的表达式：

$$\nabla J_\sigma(\phi) = \frac{1}{\sigma}\,\mathbb{E}\big[\mathcal{L}(\phi + \sigma\varepsilon)\,\varepsilon\big]
= \frac{1}{2\sigma}\,\mathbb{E}\big[\big(\mathcal{L}(\phi + \sigma\varepsilon) - \mathcal{L}(\phi - \sigma\varepsilon)\big)\,\varepsilon\big]$$

直观地说：沿随机方向 $\varepsilon$ 两边各试一下，哪边损失低就往哪边走，走多少由损失差决定。用 $P$ 对样本平均就得到估计：

$$\hat g = \frac{1}{2\sigma P}\sum_{i=1}^{P}\big(L_i^+ - L_i^-\big)\,\varepsilon_i$$

### 两个降噪技巧

- **对偶采样**：每个 $\varepsilon$ 同时用 $+$ 和 $-$。损失差抵消了 $\mathcal{L}$ 的偶数阶项（包括常数项），方差小得多。
- **公共随机数**：一对的两个粒子用**同一个任务种子**，初始化和数据顺序完全相同。于是 $L^+ - L^-$ 只反映扰动的影响，
  不混入"这个任务本来就难"的噪声。这也是为什么内层训练最好是确定性的（见第 6 节）：同一个 $\phi$、同一个任务跑两次
  应该得到完全相同的损失。

### 全程 ES

每次外层更新：采样 $P$ 个 $\varepsilon_i$，跑 $2P$ 次**完整的**内层训练，取训练终点的验证损失作为 $L_i^\pm$，然后更新 $\phi$。

- 元目标就是你真正关心的终点损失；除了这个损失，内层训练的其他信息都不进入估计。
- $2P$ 次训练完全独立，容易并行，多出来的算力直接变成更多样本。
- 每 $2P$ 次完整训练更新一次 $\phi$。

## 3. 写你自己的任务

`Hilo.meta.train` 接受一个函数：用给定的 HiloNet 从任务种子训练一个模型，返回终点损失。

```python
def objective(net, task_seed):
    torch.manual_seed(task_seed)                       # 初始化、数据顺序都由 task_seed 决定
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

## 4. 超参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `sigma_mult` | 扰动大小 $\sigma$ 的倍数（基准：矩阵参数 0.02，向量参数 0.01） | 1.0 起步。`pair_absdiff` 很小、`drift` 几乎不增长时，扰动看不出效果，增大；发散多（`dead` > 0）时减小 |
| `pairs` | 每次外层更新的配对数 $P$，计算量 $\propto 2P$ | 8 起步；方向噪声大时增加 |
| `alpha` | 外层步长，以 $\sigma$ 为单位：$\phi \leftarrow \phi - \alpha\,\sigma\cdot\text{step}$ | `sgdm`：1.0（默认值）；`adam` 每个坐标每步约移动 $\alpha\sigma$，取 0.01–0.1 |
| `optimizer` | `sgdm`（动量平均）或 `adam` | `sgdm` |
| `momentum` | `sgdm` 的动量 / `adam` 的 β1 | 0.9 |
| `shaping` | 损失差怎样变成权重：`rms` 除以各对差的 RMS（对损失尺度不敏感）；`rank` 中心化排名（对离群值稳健）；`none` 原始差（无偏，但步长随损失尺度变化） | `rms`；损失偶尔出现极端值时 `rank` |

`lr`（`Hilo.optimizer` 的步长）和内层训练步数不是 `Hilo.meta` 的参数，而是任务的一部分：元训练时 lr 用多少，使用时就用多少；
内层训练长度在预算允许时尽量接近目标训练长度。

## 5. 读日志

每次外层更新向终端和 `out/log.jsonl` 写一行：

| 字段 | 含义 | 怎么看 |
|---|---|---|
| `it` | 外层更新序号 | |
| `pair_mean` | 这次所有粒子终点损失的平均 | 应随训练下降，但每次的任务不同，只看趋势 |
| `pair_absdiff` | 配对损失差的平均绝对值 | 估计信号的强度；接近 0 说明 $\sigma$ 太小或内层不确定性掩盖了信号 |
| `dead` | 损失不是有限值的粒子数 | 持续不为 0 说明扰动或 Hilo `lr` 太大 |
| `step_norm` | 这次 $\phi$ 改变量的范数 | 应平稳；突增常预示不稳定 |
| `drift` | $\phi$ 离起点的距离，以 $\sigma$ 为单位 | 持续增长说明在稳定地往某个方向走 |

元训练过程中的损失都混有任务差异，判断 HiloNet 有没有变好，要用第 6 节的留出评估。

## 6. 端到端流程

1. **写任务**：写一个 `objective(net, task_seed)`。设置确定性：
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `torch.use_deterministic_algorithms(True)`；
   同一个 HiloNet、同一个任务种子跑两次，结果应逐位相同。
2. **选起点**：随机初始化 `Hilo.net(cfg, seed)`，或用 `Hilo.net.load("net.pt")` 从之前的 HiloNet 继续。
3. **小规模试跑**：几次外层更新，看 `pair_absdiff` 是否明显大于 0、`dead` 是否为 0。
4. **元训练**：用 `out=` 保存日志与检查点（每 `save_every` 次一个 `hilo_itXXXX.pt`），中断后用同一个 `out` 续跑。
   多张卡时用 `make_task` + `devices` 并行。
5. **留出评估**：在元训练没见过的任务种子上，把新 HiloNet 和对照（随机初始化、上一个检查点或手工优化器）配对比较，
   看平均损失差和标准误（`meta_train_proxy.py --eval A --against B`）。
6. **逐步放大**：用 `Hilo.net.load(...)` 作为下一阶段的起点，在更宽、更长、更接近目标的任务上继续元训练。
7. **使用**：`Hilo.optimizer(mats, total_steps=T, lr=<元训练时的 lr>, net="runs/.../hilo_final.pt")`，
   执行选项（`precision` 等）与元训练保持一致。

## 7. 并行

一次外层更新里各粒子的训练互相独立，可以分给多个工作进程：每个 `devices` 条目一个进程，同一张卡写多次就在这张卡上
同时跑多个。工作进程里用 `make_task(device)` 构建一次目标函数，`make_task` 必须能被
pickle（模块级函数或 `functools.partial`）。各进程从共享队列取任务，自动均衡负载。确定性的任务下，并行与串行的结果逐位相同。

```python
net = Hilo.meta.train(make_task=build_objective, devices=["cuda:0", "cuda:1"], net=Hilo.net(seed=0),
                      net_options=dict(precision="bf16"), out="runs/my_net")
```

`FullES`、`Outer`、`minimize`、`EvalPool` 也单独导出，可以组合成别的元训练循环。

## 参考文献

- Salimans et al., 2017. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.*
- Metz et al., 2019. *Understanding and Correcting Pathologies in the Training of Learned Optimizers.*
- Metz et al., 2022. *VeLO: Training Versatile Learned Optimizers by Scaling Up.*
- Thérien et al., 2024. *μLO: Compute-Efficient Meta-Generalization of Learned Optimizers.*
- Moudgil et al., 2025. *Celo: Training Versatile Learned Optimizers on a Compute Diet.*
- Moudgil et al., 2026. *Celo2: Towards Learned Optimization Free Lunch.*
- Huang et al., 2026. *Efficient Long-Horizon Learning for Learned Optimization.*
