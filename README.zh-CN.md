# Hilo：HiHiTech's Learned Optimizer

[English](README.md) | 简体中文

Hilo 是一个**学习型矩阵优化器（learned optimizer）的通用框架**。网络中每个矩阵参数 W 的更新 U 由一个小型
矩阵 Transformer（HiloNet，即 `Hilo.net`）计算，然后 `W ← W − lr·U`。HiloNet 的参数通过元训练得到：
在你关心的一族训练任务上，直接优化"用它训练完之后的验证损失"。

| 模块 | 作用 |
|---|---|
| `Hilo.optimizer` | 优化器（`torch.optim.Optimizer` 接口），支持多卡分片计算 |
| `Hilo.net` / `Hilo.config` | 可配置的 HiloNet 架构；随机初始化；保存 / 加载 |
| `Hilo.meta` | 元训练（`Hilo.meta.train`）：全程对偶 ES；目标函数就是你自己的训练循环 |

HiloNet 从随机初始化开始。在你的任务族上元训练后，把保存的 `.pt` 交给 `Hilo.optimizer`。

## 为什么用学习型优化器

Rich Sutton 在 [The Bitter Lesson](http://www.incompleteideas.net/IncIdeas/BitterLesson.html)（2019）里总结了 AI 研究七十年的经验：
把人的知识直接写进方法，短期内有效，但很快会遇到瓶颈；长期胜出的总是能随算力扩展的通用方法，也就是搜索和学习。
国际象棋、围棋、语音识别、计算机视觉都走过这条路：手工设计的特征和规则最终被用更多算力学出来的东西超过。

今天训练神经网络的优化器仍然是手工设计的，每一代都是一个关于损失曲面的人为假设：

- 动量：梯度方向在相邻步之间相关。
- Adam：按坐标的二阶矩做对角预条件。
- Shampoo、SOAP：用 Kronecker 结构近似曲率。
- Muon：把动量正交化，让更新的各奇异方向等幅。

此外还有学习率预热与衰减、权重衰减、梯度裁剪，以及一组需要逐个任务调的超参数。这些设计之所以被采用，部分原因是人能推导、
能分析，而不是因为它们就是最好的更新规则。

更新规则本质上只是一个函数：输入梯度、梯度历史、权重和训练进度，输出这一步的更新。学习型优化器用一个网络表示这个函数，
直接以"在一族任务上训练得有多好"为目标来优化它。这样，优化器的进步来自更多的元训练任务、更接近目标的代理任务和更长的
元训练，也就是来自算力，而不再受限于人能想出的下一个假设。这条路线正在快速推进：

- VeLO（2022）用 4000 TPU 月训练出通用的学习型优化器。
- Celo2（2026）只用 4.5 GPU 小时元训练，就稳定扩展到 GPT-3 1.3B 规模，并超过调好的 AdamW。
- ELO（2026）进一步在语言模型和 ImageNet 上全面超过 AdamW，在语言模型上与 Muon 相当。

Hilo 按这个思路设计：

- **只给原始信号。** HiloNet 的输入只有梯度、几个时间尺度的梯度 EMA、权重和训练进度 t/T；没有正交化、没有 Adam 式预条件、
  也没有学习率日程，怎样利用这些信号完全由元训练决定。
- **只保留问题结构。** 保留的结构只与问题的形状有关：对矩阵行、列置换等变，因此一个网络适用于任意形状的矩阵；
  对输入整体缩放不变。这就像卷积对平移的等变：它描述问题的形状，而不规定解法。

代价是每步要运行一个网络，并且需要元训练的算力。

## 安装

```bash
pip install -e .                 # 只依赖 torch
```

## 快速开始

```python
import Hilo

mats, rest = Hilo.split_params(model)       # 2-D 矩阵 -> Hilo；嵌入、输出头、增益、偏置 -> 其他优化器
opt = Hilo.optimizer(mats, total_steps=T, lr=1e-3, net="runs/my_net/hilo_final.pt")
adam = torch.optim.Adam(rest, lr=1e-3)

for step in range(T):
    model(batch).backward()
    opt.step(); adam.step()
    model.zero_grad(set_to_none=True)
```

- `net`：`Hilo.net` 实例、保存的 `*.pt`，或 `None`（随机初始化的 HiloNet）。
- `lr`：U 的乘子，使用元训练时的那个值。和任何 PyTorch 优化器一样，也可以自己修改 `group["lr"]`。
- `total_steps`：HiloNet 以训练进度 t/T 为输入，因此必须给出。
- `weight_decay`：可选的解耦权重衰减，默认 0。
- 分布式：梯度 all-reduce 之后，各卡按估计的 HiloNet 计算量均衡分摊矩阵（只保存自己那部分的动量），算完后广播
  （`shard=True`，默认），不需要额外的通信代码。

CPU 上几秒跑完的示例：`python examples/quickstart.py`。

## HiloNet

```python
cfg = Hilo.config(d=16, heads=2, mlp_hidden=32, tie=12, attn="linear",
                  betas=(0.5, 0.9, 0.99, 0.999), output_norm="none")
net = Hilo.net(cfg, seed=0)                 # 随机初始化
net.save("net.pt");  net = Hilo.net.load("net.pt")
```

每个矩阵 W（m×n）的每个坐标带通道 `[G, M_β1, …, M_βk, W]`（M_β 为梯度的 EMA，β 由 `betas` 指定），
各自除以张量 RMS 并截断到 ±`clip`，因此对梯度和权重的整体缩放不变。全局特征为 `log(rms M_β / rms G)`、
`log(rms W)`、t/T、1−t/T、`log(m/n)`。坐标嵌入到 d 维后经过若干个 [行注意力 → 列注意力 → MLP] 块
（残差 + RMSNorm）。行注意力把行当作 token，打分是所有列上通道双线性型之和（单通道、线性注意力时就是 XXᵀ），
因此网络对行、列置换等变，可以用于任意形状的矩阵。线性读出得到 X，再按 `output_norm` 得到 U。
随机初始化时残差分支为零，所以新建的 HiloNet 是归一化通道的逐坐标线性组合。

| 配置项 | 含义 |
|---|---|
| `d`, `heads`, `mlp_hidden` | 坐标嵌入宽度、注意力头数、MLP 宽度 |
| `blocks` / `tie` | 不共享的块数；或 `tie > 0` 时一个共享块迭代 `tie` 次 |
| `attn` | `linear`（打分 / m）或 `softmax` |
| `betas` | 输入的梯度 EMA 系数 |
| `clip` | 归一化通道的截断阈值 |
| `output_norm` | 更新范数，见下表 |

更新范数 `output_norm`（`Hilo.norms`）：

| 选项 | U |
|---|---|
| `none`（默认） | U = X |
| `rms` | RMS(U) = 1 |
| `frobenius` | ‖U‖_F = 1 |
| `row` | ‖U‖_F = √m（各行范数的平方平均为 1） |

除 `none` 外，其余选项都固定了幅度，此时步长完全由 `lr` 决定。不同选项下合适的 `lr` 量级不同，所以 `lr` 应与元训练时一致。

执行选项（不属于网络本身，不随 `.pt` 保存；元训练和使用时应保持一致）：

```python
net = Hilo.net.load("net.pt", precision="auto", backend="auto", compile=False, max_coords=2**23)
```

- `precision`：注意力矩阵乘的精度，`bf16`、`tf32`、`fp32` 或 `auto`（默认：CUDA 上 bf16，CPU 上 fp32）。
  残差流、归一化和 MLP 始终是 float32。相对 fp32 的输出误差约为 bf16 0.4%、tf32 0.02%。
- `backend`：`auto`（默认）在 CUDA、bf16 下使用融合的 Triton kernel（`Hilo/kernels.py`；需要安装 Triton，且 `d`、
  `mlp_hidden` 是不小于 16 的 2 的幂），否则使用 PyTorch 实现；`triton` 要求必须用 kernel，`torch` 从不使用。
  每个 kernel 完成一个坐标在两次注意力矩阵乘之间的全部计算（输出投影与残差、MLP、RMSNorm，并按下一次注意力读取的
  布局写出 q / k / v）；注意力矩阵乘本身用 cuBLAS。相对 fp32 的误差与 PyTorch 的 bf16 实现相同。
- `compile`：用 `torch.compile` 编译 PyTorch 实现，每种矩阵形状首次调用约需 25 s 编译。
- `max_coords`：同形状矩阵按块处理，每块最多这么多个坐标，用来限制峰值显存（约 0.5 KB / 坐标）。

计算量：每层注意力要形成 m×m 和 n×n 的打分矩阵，单个矩阵约 O(iters · mn(m+n) · d)，逐元素运算约 O(iters · mn · d)。
A100 上默认配置、bf16 下单个矩阵的耗时（与其他任务共用 GPU 时测得，可用带宽和矩阵乘吞吐大约减半）：

| 形状 | PyTorch | PyTorch + compile | Triton（默认） |
|---|---|---|---|
| 256×256 | 9.3 ms | 6.2 ms | 1.4 ms |
| 1024×256 | 38 ms | 26 ms | 6.4 ms |
| 768×768 | 73 ms | 40 ms | 16 ms |
| 3072×768 | 380 ms | 276 ms | 114 ms |

modded-nanogpt #36 的 72 个隐藏矩阵完整更新一步，单卡上 PyTorch 需 12.7 s，Triton 需 3.4 s（8 卡分片约为其 1/8）。
用 Triton 时大部分时间花在注意力矩阵乘上，且已接近 GPU 的矩阵乘吞吐，剩余开销由架构本身 O(iters · mn(m+n) · d)
的计算量决定。

## 元训练（`Hilo.meta`）

**完整教程：[docs/meta_training.zh-CN.md](docs/meta_training.zh-CN.md)**（[English](docs/meta_training.md)）。
内容包括：两层优化与术语、ES 为什么只用损失值就能估计梯度、超参数、日志字段和端到端流程。

HiloNet 的参数 φ 用全程对偶 ES 训练：每次外层更新用扰动后的 HiloNet φ ± σε 从头训练 2 × pairs 个模型，
让 φ 朝终点验证损失更低的一侧移动。它只运行你的训练循环、只看损失值，不需要反向传播穿过训练过程，元目标就是终点损失本身。

```python
import Hilo, Hilo.meta

# objective(net, task_seed)：从 task_seed 训练一个模型（用 Hilo.optimizer(..., net=net)），返回终点验证损失
net = Hilo.meta.train(objective, Hilo.net(seed=0), iters=100, pairs=8, out="runs/my_net")
# -> out/hilo_final.pt，以及 hilo_itXXXX.pt 检查点、log.jsonl、可续跑的 state.pt
```

此外还提供以下功能，用法见教程：

- **外层优化器与打分方式**：`optimizer`、`shaping`、`alpha`。
- **多进程并行**：`make_task` + `devices`。

可运行示例：`python examples/meta_train.py`，在一个小型师生 MLP 上元训练。

## modded-nanogpt 接入示例

[`examples/modded_nanogpt/`](examples/modded_nanogpt/README.zh-CN.md)：以 modded-nanogpt Track 3 的官方基线
（#36，`train_gpt_simple.py`）为例，用 `Hilo.optimizer` 训练隐藏矩阵（改动见 `train_gpt_hilo.diff`），
并给出在缩小版基线上元训练 HiloNet 的脚本。

## 目录结构

```
hilo/
├── Hilo/                       # Python 包
│   ├── __init__.py             # 公共 API
│   ├── optim.py                # Hilo.optimizer（动量状态、分片、权重衰减）、split_params
│   ├── hilonet.py              # Hilo.net、Hilo.config：(G, M, W, t/T) -> U；随机初始化、保存/加载
│   ├── kernels.py              # HiloNet 前向的 Triton kernel（CUDA、bf16）
│   ├── norms.py                # 更新范数选项：none / rms / frobenius / row
│   └── meta/                   # 元训练
│       ├── __init__.py         #   Hilo.meta.train
│       ├── es.py               #   全程 ES：FullES
│       ├── core.py             #   外层循环：打分（rms / rank / none）、外层优化器（sgdm / adam）
│       └── pool.py             #   多进程：EvalPool
├── docs/
│   ├── meta_training.md        # 元训练教程：ES 原理、超参数、日志与流程
│   └── meta_training.zh-CN.md
├── examples/
│   ├── quickstart.py           # 最小用法（CPU）
│   ├── meta_train.py           # 在自己的任务（师生 MLP）上元训练
│   └── modded_nanogpt/         # modded-nanogpt #36 官方基线接入
│       ├── README.md
│       ├── README.zh-CN.md
│       ├── train_gpt_hilo.py   #   官方 train_gpt_simple.py：Muon -> Hilo.optimizer
│       ├── train_gpt_hilo.diff #   与官方脚本的差异
│       └── meta_train_proxy.py #   在缩小版基线上元训练 / 评估 HiloNet
├── tests/
│   ├── test_hilo.py
│   ├── test_distributed.py
│   └── fixtures/hilonet_ref.pt
├── pyproject.toml
├── LICENSE
├── README.md
└── README.zh-CN.md
```

## 测试

```bash
python tests/test_hilo.py          # 与参考实现逐步对比、保存/加载、各更新范数、权重衰减、state_dict、元训练
python tests/test_distributed.py   # 2 进程分片与单进程结果一致
```

## 许可

MIT。`examples/modded_nanogpt/` 中的训练脚本改自 [modded-nanogpt](https://github.com/KellerJordan/modded-nanogpt)（MIT）。
