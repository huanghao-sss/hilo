# 在 modded-nanogpt 中使用 Hilo

[English](README.md) | 简体中文

以 modded-nanogpt Track 3（优化器赛道）的**官方基线** #36
[`records/track_3_optimization/train_gpt_simple.py`](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py)
为例：GPT 12 层 × 768 维，每步 524288 token，3250 步；嵌入 / 输出头 / 增益与偏置用 AdamW，隐藏矩阵原本用 Muon。

`train_gpt_hilo.py` 就是这个官方脚本，改动只有三处（全部见 [`train_gpt_hilo.diff`](train_gpt_hilo.diff)）：

1. 删除 Muon（Newton–Schulz 与 `Muon` 类）。
2. 隐藏矩阵交给 `Hilo.optimizer`：

   ```python
   import Hilo

   optimizer2 = Hilo.optimizer([p for p in model.blocks.parameters() if p.ndim >= 2], total_steps=train_steps,
                               lr=float(os.environ.get("HILO_LR", 1e-3)), net=os.environ.get("HILO_NET") or None)
   ```

3. 原脚本的"先平稳后线性衰减"学习率日程只作用于 AdamW，`Hilo.optimizer` 的 lr 保持常数。

模型、数据、初始化、AdamW、验证都不变。梯度 all-reduce 之后 `optimizer2.step()` 会按卡分摊 HiloNet 的计算并广播结果。

## 运行

在 modded-nanogpt 仓库根目录下运行（需要 `data/fineweb10B`，用 `python data/cached_fineweb10B.py 9` 下载），
并 `pip install -e <Hilo 仓库>`。

```bash
# 用元训练好的 HiloNet 训练（HILO_LR 与元训练时一致）
HILO_NET=runs/proxy/hilo_final.pt HILO_LR=1e-3 torchrun --standalone --nproc_per_node=8 train_gpt_hilo.py

# 不设置 HILO_NET：随机初始化的 HiloNet
torchrun --standalone --nproc_per_node=8 train_gpt_hilo.py

# 基线对照：直接运行官方的 train_gpt_simple.py
```

和原脚本一样，命令行最后一个参数是重复试验次数（`num_trials`）。

## 元训练 HiloNet：`meta_train_proxy.py`

在完整规模上做 ES 太贵，所以在 `train_gpt_hilo.py` 的缩小版上元训练：架构、初始化、AdamW 分组与日程、损失、
常数 Hilo lr 都与之相同，只缩小宽度、深度、批大小和步数（默认 4 层 × 256 维、300 步、每步 32k token，单卡）。
每次 ES 评估从头训练一个代理模型，返回终点验证损失。

元训练从随机初始化开始，也可以用 `--init net.pt` 从保存的 HiloNet 继续。

```bash
python meta_train_proxy.py --out runs/proxy --iters 100 --pairs 8                       # 从随机初始化开始
python meta_train_proxy.py --config '{"output_norm": "rms", "tie": 6}' --out runs/proxy_rms  # 换架构 / 更新范数
python meta_train_proxy.py --init runs/proxy/hilo_final.pt --steps 600 --out runs/proxy2  # 加长代理继续训练
python meta_train_proxy.py --eval runs/proxy2/hilo_final.pt --against runs/proxy/hilo_final.pt --seeds 8
```

外层优化参数 `--optimizer sgdm|adam`、`--shaping rms|rank|none`、`--alpha`、`--momentum` 的含义和取值、
日志怎么读，见[元训练教程](../../docs/meta_training.zh-CN.md)。
`--eval` 在与元训练不重叠的留出任务上评估（终点验证损失）；加 `--against` 时两个 HiloNet 在同一组任务上配对比较。

效率相关的参数：

- `--devices cuda:0,cuda:1,...`：一次外层更新的各次评估分给多个工作进程并行跑；同一张卡可以写多次。
- `--precision`、`--backend`、`--compile-net`：HiloNet 的执行选项，见主 README。4 层 × 256 维时，用 Triton kernel
  （CUDA 上的默认）HiloNet 每个代理步约 75 ms（PyTorch 实现约 450 ms），而模型前向加反向约 160 ms。
- `--compile-model`：编译代理 GPT。
- 评估默认使用确定性 CUDA kernel，同一个网络、同一个任务重复运行的结果完全相同；`--nondeterministic` 可以关掉。

代理越接近目标设置，迁移越好；常见做法是先在小代理上训练，再逐步加宽、加长代理继续训练。

## 开销

HiloNet 的开销随矩阵大小增长。12 × 768 的模型共 72 个矩阵：用 Triton kernel 单卡每步 3.4 s（A100，与其他任务共用
GPU 时测得；PyTorch 的 bf16 实现为 12.7 s），8 卡分片约 0.45 s。其中大部分是 3072 × 768 矩阵的注意力矩阵乘，已接近
GPU 的矩阵乘吞吐；要再快，需要减小架构的计算量（`Hilo.config` 的 `tie`、`d`），并为该配置重新元训练。
