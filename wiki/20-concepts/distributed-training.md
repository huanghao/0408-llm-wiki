# 分布式训练

## 为什么需要分布式训练

单张 GPU/TPU 的显存有限（A100 = 80GB），计算速度有限。当模型参数量或训练数据规模超出单张卡的承载能力时，需要把训练任务分散到多张卡上并行完成。

---

## 两种主要并行方式

### 数据并行（Data Parallel）

**最常见的方式**，适合模型能放进单张卡但数据量大的情况。

**原理**：

```
卡 0：持有完整模型参数，处理 batch[0:16]，计算梯度 g_0
卡 1：持有完整模型参数，处理 batch[16:32]，计算梯度 g_1
卡 2：持有完整模型参数，处理 batch[32:48]，计算梯度 g_2
...
        ↓ AllReduce：所有卡交换并平均梯度
        g_avg = (g_0 + g_1 + g_2 + ...) / N
        ↓ 每张卡用 g_avg 更新参数
卡 0 参数 ← 卡 0 参数 - lr * g_avg  （和卡 1、卡 2 完全一样）
```

- 每张卡看到不同的数据（数据切分），但持有相同的参数
- 每步结束后通过 **AllReduce** 同步梯度，确保所有卡的参数始终一致
- N 张卡 = effective batch size × N，训练速度近似线性加速

**数据切分提升读取效率**：N 张卡可以同时从存储系统读取不同的数据，I/O 并行度提升 N 倍。单卡训练时，加载数据往往是瓶颈（CPU 往 GPU 喂数据跟不上）；多卡时每张卡各自的 DataLoader 并行读取，整体吞吐量更高。

**Wayformer 的配置**：16 TPU v3 core，每卡 batch=16，等效 global batch=256。数据切分，参数共享同步，训练结束取任意一张卡的参数（它们完全一样）。

### 模型并行（Model Parallel）

**适合模型太大无法放进单张卡**（如千亿参数 LLM）。

把模型的不同层分配到不同的卡：卡 0 跑前几层，输出传给卡 1 跑中间层，依此类推（Pipeline Parallel）；或者把同一层的参数矩阵按行/列分割（Tensor Parallel）。

Wayformer（20M 参数）远小于单卡容量，不需要模型并行。

---

## AllReduce：梯度同步的核心操作

AllReduce 是多卡之间汇总梯度的通信操作：

```
输入：每张卡各自的梯度向量 g_i（和模型参数同形状）
操作：所有卡交换数据，每张卡都得到 sum(g_i) / N
输出：每张卡都持有相同的平均梯度
```

常见算法：Ring-AllReduce（每张卡依次传给下一张，一圈后每张卡都有所有信息），通信量为 $O(M)$（M=参数量），和卡数无关——这是数据并行近线性扩展的关键。

**PyTorch 里的使用**：

```python
# 单机多卡
model = torch.nn.DataParallel(model)          # 简单版，自动切分 batch

# 多机多卡（推荐）
torch.distributed.init_process_group(backend='nccl')
model = torch.nn.parallel.DistributedDataParallel(model)
# DDP 在反向传播时自动插入 AllReduce，效率更高
```

---

## 数据并行的完整训练循环

```python
# 初始化（每个进程在一张卡上运行）
rank = torch.distributed.get_rank()   # 当前卡的编号，0~N-1
world_size = torch.distributed.get_world_size()  # 总卡数 N

# 数据切分：每张卡只看 1/N 的数据
sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)

model = DDP(model)   # 包装后，梯度自动同步

for batch in dataloader:
    # 前向：每张卡处理各自的 16 个样本
    loss = model(batch)

    # 反向：计算本卡的梯度
    loss.backward()
    # DDP 在这里自动触发 AllReduce，把所有卡的梯度平均

    # 更新：所有卡用相同的平均梯度更新参数
    optimizer.step()
    optimizer.zero_grad()

# 训练结束：所有卡的参数完全一致，保存任意一张即可
if rank == 0:
    torch.save(model.state_dict(), 'checkpoint.pt')
```

> **关于数据切分的三个常见问题：**
>
> **Q1：每张卡是否每个 epoch 都只看固定的那 1/N 的数据？**
>
> 不是。`DistributedSampler` 每个 epoch 开始时**重新打乱全部数据**，再重新切分。代码通常写成：
> ```python
> for epoch in range(num_epochs):
>     sampler.set_epoch(epoch)   # 每 epoch 换不同的随机种子
>     for batch in dataloader:
>         ...
> ```
> 如果不调用 `set_epoch(epoch)`，每个 epoch 每张卡看到的数据顺序相同，等于没有打乱——这是一个常见 bug。多个 epoch 训练下来，每张卡合计也会看遍整个数据集（只是在时间上分散开了），所以不存在"某张卡永远看不到某些样本"的问题。
>
> **Q2：把大数据集切成多个独立文件，能提升 I/O 效率吗？**
>
> 完全正确，这正是工业界的标准做法。将数据集预先切成 N 份（或更多份）独立文件，每张卡只读自己负责的那一份，各卡的 I/O 请求打到不同的文件或不同的存储节点上，实现真正的 I/O 并行。常见格式：WebDataset（每个 shard 是一个 tar 包）、TFRecord（TensorFlow 生态）。`DistributedSampler` 的逻辑切分只是软件层面的"分配"，数据文件本身如果都在同一个文件里，依然存在单点 I/O 瓶颈；物理上把文件切开，才能让 I/O 和计算一样并行扩展。
>
> **Q3：数据切分（每卡只看 1/N）和全局大随机（每卡从全部数据随机采样），训练效果一样吗？**
>
> 理论上等价，实践中切分版略有优势：
>
> - **等价的原因**：AllReduce 将各卡梯度平均，等效于用 global batch size = 单卡 batch × N 的大 batch 做一步 SGD。无论每张卡拿到的是"确定切分"还是"独立随机采样"的 batch，只要 epoch 级别下每张卡覆盖的数据分布和全局一致，最终梯度平均的统计特性是相同的。
>
> - **切分版的 I/O 优势**：全局随机采样意味着每张卡在整个数据集上随机跳读，产生大量小的随机 I/O，在机械硬盘或分布式文件系统上非常慢（随机读吞吐远低于顺序读）。切分版每张卡顺序读自己那一份文件，I/O 效率远高于随机跳读。
>
> - **不完全等价的细节**：切分版在单个 epoch 内每张卡看到的样本完全不重叠（互补），全局随机版存在不同卡看到同一个样本的概率（重复采样）。数据量足够大时这个差异可忽略；数据量很小时切分版能保证全覆盖，全局随机版有遗漏风险。

---

## 关键概念汇总

| 概念 | 含义 |
|------|------|
| **Data Parallel** | 多卡持有完整模型，各处理不同数据，梯度同步 |
| **Model Parallel** | 模型切分到多卡，每卡只持有部分参数 |
| **AllReduce** | 多卡间汇总梯度的通信操作，结果是平均梯度 |
| **Effective Batch Size** | 单卡 batch × 卡数，等价于更大的 batch 训练 |
| **Rank** | 当前进程/卡的编号（0~N-1） |
| **World Size** | 总进程/卡的数量 N |
| **DistributedSampler** | 确保不同卡读取不同数据的采样器 |

---

## 和 wiki 内其他概念的关联

- [Wayformer](../30-papers/wayformer-2207.05844.md)：16 TPU v3 数据并行训练的具体配置和 epoch 分析
- [PNC 模型架构](../00-overview/pnc-model-architecture.md)：64× A100 数据并行，显存接近 80GB 上限时的 I/O 与计算瓶颈分析
