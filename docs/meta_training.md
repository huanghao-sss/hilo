# 元训练教程：用 ES、PES、NRES 训练 HiloNet

这份教程从零讲清楚 `hilo.meta` 在做什么、三种估计器（全程 ES、PES、NRES）各自是什么、实测效果如何、该怎么选、怎么调、怎么看日志。
只需要知道梯度下降和 PyTorch，不需要事先了解进化策略。

## 1. 问题：两层优化

元训练有两层循环：

- **内层**：一次普通的模型训练。模型参数 $w$ 从初始化开始，用 `hilo.Hilo` 训练 $T$ 步，每一步的更新由 HiloNet 给出：
  $w_{t+1} = w_t - \text{lr}\cdot U_\phi(G_t, M_t, w_t, t/T)$。
- **外层**：调整 HiloNet 的参数 $\phi$，让内层训练的结果更好。

外层要最小化的是**元目标** $\mathcal{L}(\phi)$：用 $\phi$ 跑完（一段或全部）内层训练后得到的损失，再对任务分布取期望。
一个"任务"由任务种子决定：模型初始化和数据顺序。

$\mathcal{L}(\phi)$ 对 $\phi$ 的梯度原则上可以反向传播穿过整个内层训练得到，但这要对几百上千步训练做二阶求导：
显存随步数线性增长，长展开的梯度还会爆炸或消失。学习型优化器领域因此普遍改用**零阶估计**：只运行内层训练、
只看损失值，用随机扰动估计梯度方向。`hilo.meta` 里的三种估计器都属于这一类。

术语：

| 术语 | 含义 |
|---|---|
| 粒子 | 一份带扰动的 HiloNet 参数 $\phi \pm \sigma\varepsilon$ 及它驱动的那次内层训练 |
| 对偶配对（pair） | 共享同一任务的两个粒子：$\phi + \sigma\varepsilon$ 和 $\phi - \sigma\varepsilon$ |
| horizon | 一次内层训练的总步数 $T$（也是 `Hilo` 的 `total_steps`） |
| 段（segment） | 截断估计器里一次外层更新推进的 `trunc` 步内层训练 |
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
  不混入"这个任务本来就难"的噪声。这也是为什么内层训练最好是确定性的（见第 11 节）：同一个 $\phi$、同一个任务跑两次
  应该得到完全相同的损失。

### 全程 ES（`meta_train`）

每次外层更新：采样 $P$ 个 $\varepsilon_i$，跑 $2P$ 次**完整的**内层训练，取训练终点的验证损失作为 $L_i^\pm$，然后更新 $\phi$。

- 优点：元目标就是你真正关心的终点损失；没有截断偏差；实现和推理都最简单；$2P$ 次训练完全独立，容易并行。
- 缺点：每 $2P$ 次完整训练才更新一次 $\phi$。horizon 很长时，外层更新非常慢。

## 3. 截断：为什么不能简单地"每段更新一次"

想让 $\phi$ 更新得更频繁，自然的想法是把一次训练切成长度为 `trunc` 的段，每跑完一段就更新一次 $\phi$。
此时元目标变成整次训练里各段损失之和：

$$\mathcal{L}(\phi) = \sum_{k} L_k, \qquad L_k = \text{第 } k \text{ 段的（平均）训练损失}$$

**普通截断 ES** 的做法是：每段开始时，两个粒子都从同一个未扰动的状态出发，只在这一段内加扰动，用这一段的损失差估计梯度。
问题在于，$\phi$ 不只影响当前这一段：它在前面各段里塑造了模型当前的权重和动量，而这些又决定了这一段的损失。
普通截断 ES 只看到"这一段内 $\phi$ 的作用"，看不到"$\phi$ 怎样把模型带到了现在这个状态"，所以是**有偏**的。
这种偏差会让元训练偏向贪心、短视的更新（例如学出过小的步长：短期内损失下降稳，但长期训练慢），即所谓短视偏差
（short-horizon bias，Wu et al., 2018）。`tests/test_hilo.py` 的线性系统上，普通截断 ES 的梯度偏差约 40%。

PES 和 NRES 都通过**让每个粒子保持自己的扰动轨迹**来消除这个偏差：粒子不回到共享状态，而是一直沿着自己被扰动过的
训练继续跑，每次外层更新只把它推进一段。区别在于噪声怎么采、梯度方向怎么算。

## 4. PES：每段新噪声，方向用累计噪声

PES（Persistent Evolution Strategies，Vicol et al., 2021）在第 $k$ 段采样**新的**噪声 $\varepsilon_k$，
配对两边分别用 $\phi + \sigma\varepsilon_k$ 和 $\phi - \sigma\varepsilon_k$ 继续各自的训练。

第 $k$ 段的损失 $L_k$ 由这次训练里第 $1$ 到第 $k$ 段的所有扰动共同决定：前面的扰动改变了模型状态，当前的扰动改变了这一段的更新。
由于各段噪声独立，$\mathbb{E}[L_k\,\varepsilon_j]/\sigma$ 恰好是"第 $j$ 段的 $\phi$"对 $L_k$ 的偏导；把 $j \le k$ 的贡献加起来就是
$\phi$ 对 $L_k$ 的全导数。所以 PES 用**累计噪声**作为方向：

$$\xi_k = \sum_{j \le k}\varepsilon_j, \qquad
\hat g_k = \frac{1}{2\sigma P}\sum_{i=1}^{P}\big(L_{k,i}^+ - L_{k,i}^-\big)\,\xi_{k,i}$$

把一次训练里所有段的 $\hat g_k$ 加起来，就是整次训练（各段损失之和）高斯平滑后梯度的无偏估计。
一次训练结束（跑满 horizon 或发散）后 $\xi$ 清零，这一对换新任务重新开始。

特点：

- 无偏，并且每次训练里 $\phi$ 更新 horizon / trunc 次。
- 每段都换一个新的随机方向，探索的方向多。
- $\xi_k$ 是 $k$ 个独立噪声之和，范数按 $\sqrt{k}$ 增长：训练后期的段估计方差大。horizon / trunc 越大，这个问题越明显。

## 5. NRES：一次训练只用一个噪声

NRES（Noise-Reuse Evolution Strategies，Li et al., 2023）在一次训练开始时采样**一个** $\varepsilon$，之后每一段都复用它；
梯度方向就是 $\varepsilon$ 本身：

$$\hat g_k = \frac{1}{2\sigma P}\sum_{i=1}^{P}\big(L_{k,i}^+ - L_{k,i}^-\big)\,\varepsilon_i$$

因为整次训练都用同一个扰动 $\phi \pm \sigma\varepsilon$，第 $k$ 段的损失就是"用扰动后的 $\phi$ 训练到第 $k$ 段"的损失，
上式直接是它的平滑梯度，同样无偏。可以把 NRES 理解为**把全程 ES 拆成一段一段在线计算**：平滑方式和全程 ES 一样，
但不用等整次训练结束才更新 $\phi$。

特点：

- 无偏；Li et al. 证明在一定假设下，它在这一族在线估计器（包括 PES）里方差最低：不再引入每段的额外随机性，
  也没有 $\sqrt{k}$ 增长。
- 同一对粒子在一次训练里始终沿同一个方向探索，相邻几次外层更新从这对粒子拿到的方向高度相关。
  因此各对粒子需要**错开**处在训练的不同阶段（默认的 `stagger=True`），并且 `pairs` 不宜太少。

## 6. 三种估计器对比与选择

| | 全程 ES | 普通截断 ES | PES | NRES |
|---|---|---|---|---|
| 元目标 | 终点验证损失 | 各段训练损失之和 | 各段训练损失之和 | 各段训练损失之和 |
| 每次训练的 $\phi$ 更新次数 | 1 / (2P 次训练) | horizon / trunc | horizon / trunc | horizon / trunc |
| 偏差 | 无 | 有（短视） | 无 | 无 |
| 方差 | 低 | 低 | 较高，随段数增长 | 低 |
| 需要保存的内层状态 | 无 | 有 | 有 | 有 |
| hilo 中 | `meta_train` | 不提供 | `meta_train_truncated(estimator="pes")` | `estimator="nres"`（默认） |

两点补充：

- **元目标不同。** 全程 ES 优化终点损失；PES / NRES 优化整个训练过程里训练损失的总和，也会奖励"前期降得快"。
  单轮遍历数据的语言模型里，每个 batch 第一次被看到时的训练损失和验证损失几乎一样，所以二者差别主要在于是否关心训练中途。
- **滞后（hysteresis）。** 截断估计器里 $\phi$ 在一次训练中途就被更新，所以粒子当前的状态是由稍早的 $\phi$ 产生的。
  这带来轻微的偏差，外层步长小时可以忽略。

怎么选：默认用 **NRES**，它在第 7 节的实测里明显好于另外两种。全程 ES 实现最简单、直接优化终点损失，
适合内层训练很短、外层可以更新很多次的情况。PES 方差更高，用它时取更小的 `alpha` 或更多 `pairs`。

## 7. 实测：同样算力下的三种估计器

设置：

- 内层任务是 `examples/modded_nanogpt/meta_train_proxy.py` 的缩小版 #36：2 层、宽 128、每步 8 × 512 token、horizon 100 步。
- 三种方案都从同一个随机初始化的 HiloNet（`seed=0`）出发，都花 25,600 步内层训练：
  全程 ES 是 16 次外层更新；PES / NRES 取 `trunc=10`，是 160 次外层更新。
- 都用 `pairs=8`、`sgdm`、`rms`；`alpha` 全程 ES 取 1.0，PES / NRES 取 0.3。
- 评估：在 8 个元训练没见过的任务种子上从头训练，取终点验证损失（越低越好）。
  "更宽"把宽度加到 256，"更长"把训练加到 200 步，这两种设置都没在元训练中出现。

| 终点验证损失 | 训练分布内 | 更宽（256） | 更长（200 步） |
|---|---|---|---|
| 随机初始化 | 9.44 | 9.48 | 8.81 |
| 全程 ES | 7.94 | 7.95 | 7.90 |
| PES | 7.82 | 7.82 | 7.63 |
| NRES | **7.22** | **7.37** | **6.94** |

不同任务的损失差别很大，所以按任务配对比较：三种设置的 8 个任务上，NRES 都好于另外两种。
它比全程 ES 低 0.72 / 0.58 / 0.96，比 PES 低 0.60 / 0.45 / 0.69，标准误都在 0.05 以内。

训练过程中的检查点（训练分布内）：

- 全程 ES 只更新了 16 次 $\phi$，提升几乎都发生在前 4 次（7.98），之后停滞。
- PES 在一半预算时最好（7.50），之后变差。它的方向 $\xi$ 方差大，同样的 `alpha` 对它偏大。
- NRES 在四分之三预算时最好（7.16），最后略回升到 7.22。

这组实验用的是单个元训练种子；`alpha` 没有按方案分别调；horizon 只有 100 步。

## 8. 写你自己的任务

### 全程 ES：一个函数

```python
def objective(net, task_seed):
    torch.manual_seed(task_seed)                       # 初始化、数据顺序都由 task_seed 决定
    model = build_model()
    mats, rest = hilo.split_params(model)
    opt = hilo.Hilo(mats, total_steps=T, lr=1e-3, net=net)
    adam = torch.optim.Adam(rest, lr=1e-3)
    for step in range(T):
        loss_fn(model, next_batch()).backward()
        opt.step(); adam.step(); model.zero_grad(set_to_none=True)
    return validation_loss(model)

net = hilo.meta.meta_train(objective, hilo.HiloNet(seed=0), iters=100, pairs=8, out="runs/my_hilo")
```

### PES / NRES：一个带状态的任务

```python
class Task:
    def init(self, task_seed, horizon):                # 开始一次 horizon 步的训练，返回它的全部状态
        torch.manual_seed(task_seed)
        model = build_model()
        mats, rest = hilo.split_params(model)
        return dict(model=model, opt=hilo.Hilo(mats, total_steps=horizon, lr=1e-3),
                    adam=torch.optim.Adam(rest, lr=1e-3), data=make_loader(task_seed))

    def unroll(self, net, st, steps):                  # 用 net 把这次训练再推进 steps 步
        st["opt"].net = net                            # 每段换成当前粒子的 HiloNet
        total = 0.0
        for _ in range(steps):
            loss = loss_fn(st["model"], next(st["data"]))
            loss.backward()
            total += loss.item()
            st["opt"].step(); st["adam"].step(); st["model"].zero_grad(set_to_none=True)
        return st, total / steps                       # 这一段的平均训练损失

net = hilo.meta.meta_train_truncated(Task(), hilo.HiloNet(seed=0), estimator="nres",
                                     horizon=(100, 2000), trunc=50, iters=5000, pairs=8, out="runs/my_hilo_nres")
```

要点：

- 状态里要包含继续训练所需的一切：模型、两个优化器的状态、数据位置、学习率日程用到的步数。
  "分段跑完"必须和"一口气跑完"完全等价，否则估计器看到的就不是同一个训练过程。
- `horizon` 必须传给 `Hilo(total_steps=...)`（以及你自己的学习率日程），HiloNet 以 t/T 为输入。
- 状态一直留在内存里：串行时在本进程，并行时在负责该粒子的工作进程。显存约为
  `2 × pairs × 一份训练状态`。
- 可运行的完整例子：`examples/meta_train.py`（MLP 回归，`--estimator es|pes|nres`）和
  `examples/modded_nanogpt/meta_train_proxy.py`（缩小版 GPT，状态以 `state_dict` 形式保存在共享模型之外）。

## 9. 超参数

| 参数 | 作用 | 建议 |
|---|---|---|
| `sigma_mult` | 扰动大小 $\sigma$ 的倍数（基准：矩阵参数 0.02，向量参数 0.01） | 1.0 起步。`pair_absdiff` 很小、`drift` 几乎不增长时，扰动看不出效果，增大；发散多（`dead` > 0）时减小 |
| `pairs` | 每次外层更新的配对数 $P$，计算量 $\propto 2P$ | 8 起步；方向噪声大时增加 |
| `alpha` | 外层步长，以 $\sigma$ 为单位：$\phi \leftarrow \phi - \alpha\,\sigma\cdot\text{step}$ | `sgdm`：全程 ES 取 1.0，PES / NRES 外层更新多，取 0.3（默认）；`adam` 每个坐标每步约移动 $\alpha\sigma$，取 0.01–0.1 |
| `optimizer` | `sgdm`（动量平均）或 `adam` | `sgdm`（第 7 节的设置）；Celo 等在截断估计器上用 Adam |
| `momentum` | `sgdm` 的动量 / `adam` 的 β1 | 0.9 |
| `shaping` | 损失差怎样变成权重：`rms` 除以各对差的 RMS（对损失尺度不敏感）；`rank` 中心化排名（对离群值稳健）；`none` 原始差（无偏，但步长随损失尺度变化） | `rms`；损失偶尔出现极端值时 `rank` |
| `trunc` | 截断估计器每段步数 | 50（Celo / Celo2 / ELO 的设置）；越小 $\phi$ 更新越频繁，PES 的 $\xi$ 也增长越快 |
| `horizon` | 一次训练的步数，或 `(最小, 最大)` 对数均匀 | 尽量接近目标训练长度；Celo 用 (100, 2000) |
| `stagger` | 首次训练在随机段数后结束，使各对错开 | 保持 True |

`lr`（Hilo 的步长）不是 `hilo.meta` 的参数，而是任务的一部分：元训练时用多少，使用时就用多少。

## 10. 读日志

每次外层更新向终端和 `out/log.jsonl` 写一行：

| 字段 | 含义 | 怎么看 |
|---|---|---|
| `it` | 外层更新序号 | |
| `pair_mean`（全程 ES） | 这次所有粒子终点损失的平均 | 应随训练下降，但每次的任务不同，只看趋势 |
| `seg_mean`（PES / NRES） | 这次所有粒子当前段训练损失的平均 | 混合了不同训练阶段，本身起伏很大；看长时间的平滑趋势 |
| `pair_absdiff` | 配对损失差的平均绝对值 | 估计信号的强度；接近 0 说明 $\sigma$ 太小或内层不确定性掩盖了信号 |
| `dead` | 损失不是有限值的粒子数 | 持续不为 0 说明扰动或 Hilo `lr` 太大 |
| `progress`（PES / NRES） | 各对粒子当前在各自训练中的平均进度 t/T | 错开正常时在 0.5 附近波动 |
| `resets`（PES / NRES） | 这次结束训练、换新任务的对数 | |
| `step_norm` | 这次 $\phi$ 改变量的范数 | 应平稳；突增常预示不稳定 |
| `drift` | $\phi$ 离起点的距离，以 $\sigma$ 为单位 | 持续增长说明在稳定地往某个方向走 |

元训练过程中的损失都混有任务差异，判断 HiloNet 有没有变好，要用第 11 节的留出评估。

## 11. 端到端流程

1. **写任务**：先写全程 ES 的 `objective`，或 PES / NRES 的 `Task`。设置确定性：
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` 和 `torch.use_deterministic_algorithms(True)`；
   同一个 HiloNet、同一个任务种子跑两次，结果应逐位相同。截断任务再检查一次"分段跑"和"一口气跑"一致。
2. **小规模试跑**：几次外层更新，看 `pair_absdiff` 是否明显大于 0、`dead` 是否为 0。
3. **元训练**：用 `out=` 保存日志与检查点（每 `save_every` 次一个 `hilo_itXXXX.pt`），中断后用同一个 `out` 续跑。
   多张卡时用 `make_objective` / `make_task` + `devices` 并行。
4. **留出评估**：在元训练没见过的任务种子上，把新 HiloNet 和对照（随机初始化、上一个检查点或手工优化器）配对比较，
   看平均损失差和标准误（`meta_train_proxy.py --eval A --against B`）。
5. **逐步放大**：用 `HiloNet.load(...)` 作为下一阶段的起点，在更宽、更长、更接近目标的任务上继续元训练。
6. **使用**：`hilo.Hilo(mats, total_steps=T, lr=<元训练时的 lr>, net="runs/.../hilo_final.pt")`，
   执行选项（`precision` 等）与元训练保持一致。

## 12. 并行

一次外层更新里各粒子的训练互相独立，可以分给多个工作进程：每个 `devices` 条目一个进程，同一张卡写多次就在这张卡上
同时跑多个。工作进程里用 `make_objective(device)` / `make_task(device)` 构建一次目标函数或任务，它们必须能被 pickle
（模块级函数或 `functools.partial`）。全程 ES 从共享队列取任务，自动均衡负载；截断估计器里粒子 $p$ 的状态固定在第
`p % len(devices)` 个进程，它的每一段都在那里跑。确定性的任务下，并行与串行的结果逐位相同。

```python
net = hilo.meta.meta_train(make_objective=build_objective, devices=["cuda:0", "cuda:1"], net=hilo.HiloNet(seed=0),
                           net_options=dict(precision="bf16", compile=True), out="runs/my_hilo")
net = hilo.meta.meta_train_truncated(make_task=build_task, devices=["cuda:0", "cuda:1"], horizon=1000, trunc=50)
```

`FullES`、`TruncatedES`、`Outer`、`minimize`、`Unroller`、`EvalPool`、`UnrollPool` 也单独导出，可以组合成别的元训练循环。

## 参考文献

- Salimans et al., 2017. *Evolution Strategies as a Scalable Alternative to Reinforcement Learning.*
- Wu et al., 2018. *Understanding Short-Horizon Bias in Stochastic Meta-Optimization.*
- Metz et al., 2019. *Understanding and Correcting Pathologies in the Training of Learned Optimizers.*
- Vicol, Metz, Sohl-Dickstein, 2021. *Unbiased Gradient Estimation in Unrolled Computation Graphs with Persistent Evolution Strategies.*
- Metz et al., 2022. *VeLO: Training Versatile Learned Optimizers by Scaling Up.*
- Li et al., 2023. *Variance-Reduced Gradient Estimation via Noise-Reuse in Online Evolution Strategies.*
- Thérien et al., 2024. *μLO: Compute-Efficient Meta-Generalization of Learned Optimizers.*
- Moudgil et al., 2025. *Celo: Training Versatile Learned Optimizers on a Compute Diet.*
- Moudgil et al., 2026. *Celo2: Towards Learned Optimization Free Lunch.*
- Huang et al., 2026. *Efficient Long-Horizon Learning for Learned Optimization.*
