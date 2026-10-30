# 在 modded-nanogpt 中使用 hilo

以 modded-nanogpt Track 3（优化器赛道）的**官方基线** #36
[`records/track_3_optimization/train_gpt_simple.py`](https://github.com/KellerJordan/modded-nanogpt/blob/master/records/track_3_optimization/train_gpt_simple.py)
为例：GPT 12 层 × 768 维，每步 524288 token，3250 步；嵌入 / 输出头 / 增益与偏置用 AdamW，隐藏矩阵原本用 Muon。

`train_gpt_hilo.py` 就是这个官方脚本，改动只有三处（全部见 [`train_gpt_hilo.diff`](train_gpt_hilo.diff)）：

1. 删除 Muon（Newton–Schulz 与 `Muon` 类）。
2. 隐藏矩阵交给 `hilo.Hilo`：

   ```python
   import hilo

   optimizer2 = hilo.Hilo([p for p in model.blocks.parameters() if p.ndim >= 2], total_steps=train_steps,
                          lr=float(os.environ.get("HILO_LR", 1e-3)), net=os.environ.get("HILO_NET") or None)
   ```

3. 原脚本的"先平稳后线性衰减"学习率日程只作用于 AdamW，`Hilo` 的 lr 保持常数。

模型、数据、初始化、AdamW、验证都不变。梯度 all-reduce 之后 `optimizer2.step()` 会按卡分摊 HiloNet 的计算并广播结果。

## 运行

在 modded-nanogpt 仓库根目录下运行（需要 `data/fineweb10B`，用 `python data/cached_fineweb10B.py 9` 下载），
并 `pip install -e <hilo 仓库>`。

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
默认用全程 ES：每次评估从头训练一个代理模型，返回终点验证损失。`--estimator pes` / `nres` 改用截断展开：
每个粒子的代理训练每次外层更新只前进 `--trunc` 步，返回这一段的平均训练损失；`--min-steps` 让每次训练的长度在
`[min-steps, steps]` 内对数均匀采样。分段续跑与一口气跑完逐位一致。

```bash
python meta_train_proxy.py --out runs/proxy --iters 100 --pairs 8                       # 从随机初始化开始
python meta_train_proxy.py --config '{"output_norm": "rms", "tie": 6}' --out runs/proxy_rms  # 换架构 / 更新范数
python meta_train_proxy.py --init muon --estimator nres --trunc 50 --iters 2000 --out runs/proxy_muon  # 从 Muon 蒸馏起步
python meta_train_proxy.py --init runs/proxy/hilo_final.pt --steps 600 --out runs/proxy2  # 加长代理继续训练
python meta_train_proxy.py --estimator nres --trunc 50 --min-steps 100 --steps 1000 --iters 2000 \
    --out runs/proxy_nres
python meta_train_proxy.py --eval runs/proxy2/hilo_final.pt --against runs/proxy/hilo_final.pt --seeds 8
```

估计器怎么选、外层优化参数 `--optimizer sgdm|adam`、`--shaping rms|rank|none`、`--alpha`、`--momentum` 的含义和取值、
日志怎么读，见[元训练教程](../../docs/meta_training.md)。
`--eval` 在与元训练不重叠的留出任务上评估（终点验证损失）；加 `--against` 时两个 HiloNet 在同一组任务上配对比较。

效率相关的参数：

- `--devices cuda:0,cuda:1,...`：一次外层更新的各次评估分给多个工作进程并行跑；同一张卡可以写多次。
  截断模式下每个粒子的训练状态（模型、AdamW 与 Hilo 状态）常驻在它的工作进程里，4 层 × 256 维时每个粒子约 0.3 GB。
- `--precision`、`--compile-net`：HiloNet 的执行选项，见主 README。HiloNet 是代理任务每步的主要开销：
  4 层 × 256 维时，bf16 下 HiloNet 每步约 450 ms，编译后约 300 ms，而模型前向加反向约 160 ms。
- `--compile-model`：编译代理 GPT。
- 评估默认使用确定性 CUDA kernel，同一个网络、同一个任务重复运行的结果完全相同；`--nondeterministic` 可以关掉。
代理越接近目标设置，迁移越好；常见做法是先在小代理上训练，再逐步加宽、加长代理继续训练。

## 开销

HiloNet 的开销随矩阵大小增长。12 × 768 的模型共 72 个矩阵：fp32 下单卡每步约 17 s（独占 A100）；
bf16 对 768 维矩阵快 2.4–3.4 倍，按此估算单卡约 6 s、8 卡分片约 1 s。需要更快时，可以把 `net` 换成 `hilo.HiloNet.load(path, compile=True)`，或减小 `HiloNetConfig` 的 `tie` / `d`。
