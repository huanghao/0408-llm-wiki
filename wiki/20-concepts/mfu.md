# Model FLOPs Utilization（MFU）

MFU 是 PaLM 论文（Chowdhery et al., 2022）提出的指标，用来衡量大模型训练时对硬件算力的实际利用率。

## 是什么

MFU 的核心定义只有一句话：**实际算力 / 硬件峰值算力，与模型实现细节无关**。

$$\text{MFU} = \frac{\text{实际观测到的 token 吞吐量（tokens/s）}}{\text{理论最大 token 吞吐量（tokens/s）}}$$

"理论最大 token 吞吐量"是假设硬件 100% 跑满时能达到的速度，由硬件峰值 FLOPS 和模型每 token 需要多少 FLOPS 决定：

$$\text{理论最大吞吐量} = \frac{P}{C_{\text{token}}}$$

其中 $P$ 是硬件总峰值 FLOPS，$C_{\text{token}}$ 是每个 token 的理论 FLOPS。

**对 Transformer 模型，$C_{\text{token}}$ 怎么算**（来自 PaLM 论文 Appendix B）：

每个 token 的 FLOPS 由两部分组成：

**线性层**：模型参数量为 $N$，前向传播每 token 需要 $2N$ 次乘加（每个参数参与一次乘法和一次加法），反向传播需要 $4N$（梯度计算需要两次矩阵乘法），合计 $6N$。

**注意力层**：注意力是 $O(T^2)$ 的操作（每个 token 要和序列里所有 $T$ 个 token 计算相似度），每 token 的注意力 FLOPS 随序列长度 $T$ 线性增长。精确项为 $12HQT$（$H$ 个头，每头维度 $Q$，序列长度 $T$，前向 $4HQT$ + 反向 $8HQT$）。

合并：

$$C_{\text{token}} = 6N + 12HQT$$

**实践中的近似**：对于大模型，$N$ 远大于注意力项。以 GPT-3（175B）为例：$6N = 1.05 \times 10^{12}$，而 $12HQT = 12 \times 96 \times 128 \times 2048 \approx 3 \times 10^8$，差了约 3500 倍。因此通常直接用：

$$C_{\text{token}} \approx 6N \quad \Rightarrow \quad \text{MFU} \approx \frac{\text{tokens/s} \times 6N}{P}$$

**如果不是 Transformer**：MFU 定义本身是通用的，只需把 $C_{\text{token}}$ 换成该模型架构的理论 FLOPS 公式即可。

**和硬件升级的关系**：每一代新硬件（A100 → H100 → B200）的峰值 FLOPS $P$ 都在翻倍提升。MFU 不变的情况下，tokens/s 随 $P$ 等比提升——H100 BF16 峰值约 989 TFLOPS，是 A100 的 3 倍多，同样的模型和并行策略训练速度也约快 3 倍。MFU 是衡量"是否充分利用了新硬件"的标尺：升级硬件后 MFU 下降，说明通信带宽或内存带宽成了新瓶颈。

**simcluster 里的 utilization 是不同的概念**：那是 GPU 配额利用率（业务分配了多少 GPU、实际跑了多少），是调度层面的指标。MFU 是计算层面的指标，衡量的是"正在运行的 GPU 算力有多少在做有效的矩阵运算"。

## 为什么需要 MFU

**MFU 的关键设计原则：只看"模型本身需要多少 FLOPS"，与实现细节无关**，因此不同团队、不同硬件、不同实现的数字可以直接横向比较。

在 PaLM 之前，大家用的是 **HFU（Hardware FLOPs Utilization）**，它把实际执行的所有 FLOPS（包括工程优化引入的额外计算）都算进去。问题在于一个常见的工程技巧——**activation recomputation**：

训练时反向传播需要用到前向传播的中间激活值（每一层的输出）。朴素做法是把所有激活值都存在显存里，但大模型的激活值非常大（可能占几十 GB），显存装不下。Activation recomputation 的做法是：前向传播时不保存激活值，反向传播需要时重新算一遍前向。这样显存大幅减少，代价是多做了约 $+33\%$ 的计算量。

HFU 把这部分重算的 FLOPS 也算进分子，导致：

- 同样的训练系统，用了 recomputation 的 HFU 会虚高
- 不同团队的 HFU 数字不可比较（recomputation 策略不同）

MFU 的分母只用模型架构决定的理论 FLOPS（$6N + 12HQT$），不管你实际跑了多少额外计算，因此是实现无关的公平比较。

## 典型数值

PaLM 论文给出的对比（Table 3）。这里选的都是 2020–2022 年规模最大的代表性模型，PaLM 用它们来证明自己的训练系统效率更高：

| 模型 | 机构 | 规模 | MFU | 背景 |
|------|------|------|-----|------|
| GPT-3 | OpenAI | 175B | 21.3% | 2020 年，当时最大的语言模型，在 V100 集群上训练 |
| Gopher | DeepMind | 280B | 32.5% | 2021 年，DeepMind 的大规模 LM，MassiveText 数据集 |
| Megatron-Turing NLG | Microsoft/NVIDIA | 530B | 30.2% | 2021 年，当时最大的模型，专门优化了 Megatron 并行框架 |
| **PaLM** | **Google** | **540B** | **46.2%** | **2022 年，Pathways 系统 + TPU v4，MFU 显著领先** |

**用 175B × 6 能算出 21.3% 吗**：可以反推验证。GPT-3 在约 1000 台 A100 上训练（实际用的是 V100，这里用 A100 近似），A100 BF16 峰值 312 TFLOPS，1000 卡总峰值约 $312 \times 10^{12} \times 1000$ FLOPS/s。每 token 需要 $6 \times 175 \times 10^9 \approx 10^{12}$ FLOPS。若实测吞吐约 66,000 tokens/s，则 MFU = $(66000 \times 10^{12}) / (312 \times 10^{12} \times 1000) \approx 21\%$。和 21.3% 基本吻合。核心公式就是：**MFU = 实测 tokens/s × 6N / (GPU 数量 × 单卡峰值 FLOPS)**。

Llama 3 报告的 MFU 为 **38–43%**（BF16），属于业界较高水平。

## 怎么理解这个数字

MFU 不可能达到 100%，原因包括：

- **通信开销**：多机多卡训练时，梯度同步、张量并行的 all-reduce 操作会占用时间
- **内存带宽瓶颈**：大模型的 attention 和 embedding 层是 memory-bound 而非 compute-bound，FLOPS 利用率天然偏低
- **流水线气泡**：流水线并行中，每个 micro-batch 的开始和结束阶段有 GPU 空闲
- **调度和框架开销**：Python 调度、CUDA kernel launch 等

**40–50% 是目前顶级系统的水平**，能达到这个范围说明训练系统的工程优化做得很好。低于 30% 通常意味着有明显的效率问题（通信瓶颈、并行策略不当等）。

**2026 年这个结论还成立吗**：基本成立，但上限在提高。H100 时代（2023–2024）的顶级系统（如 Llama 3、DeepSeek-V3）报告的 MFU 在 38–50% 范围内，B200/GB200 时代（2025–2026）因为 NVLink 带宽和 HBM 带宽的大幅提升，通信瓶颈有所缓解，顶级系统的 MFU 正在向 50–60% 靠近。"低于 30% 有问题"这个判断仍然成立；"40–50% 是顶级水平"在新硬件上可以适当上调到 45–55%。

## 和吞吐量的关系

MFU 和 tokens/s（吞吐量）都能衡量训练效率，但侧重不同：

- **tokens/s（总集群）**：绝对值，和硬件数量直接相关，不同集群规模不可比
- **tokens/s/GPU（单卡吞吐）**：归一化到单张卡，可以跨规模比较，但不能跨硬件代际比较（A100 和 H100 的单卡峰值不同）
- **MFU**：相对值，归一化到硬件峰值，可以跨集群、跨硬件代际比较

**单卡吞吐能比较吗**：可以，但有局限。单卡 tokens/s 可以比较"同一代硬件上不同系统的效率"，比如两个团队都用 A100，单卡吞吐更高的系统更好。但跨代际（A100 vs H100）就不行了——H100 单卡快 3 倍，不代表系统工程做得更好。MFU 是最"公平"的比较，因为它除掉了硬件本身的性能差异。

实际使用时两者结合看：tokens/s 告诉你训练要花多少时间，MFU 告诉你这套系统的工程质量。

## 附录：simcluster 的 utilization 是配额利用率，不是 MFU

结论来自直接阅读 simcluster 代码。

**关键文件**：`resource_manager/server/modules/resource/resouruce_statistics.py` 和 `business_utilization.py`

`get_resource_stastics()` 函数的核心逻辑：

```python
business_statics[business_name] = {
    "quota":      ResourceQuota(quota=b_u.quota),       # 业务分配到的 GPU/CPU 配额
    "quota_used": ResourceQuota(quota=b_u.quota_used),  # 业务实际占用的 GPU/CPU 数量
}
```

`ResourceQuota` 里存的是 `gpu40_num`、`gpu80_num`、`cpu_num`——**GPU/CPU 的数量，不是 FLOPS**。整个代码库里没有任何地方计算模型的理论 FLOPS 或实测 token 吞吐量。

`BusinessUtilization.update_quota_used()` 的逻辑是遍历 running jobs，累加每个 job 占用的 GPU 数量：

```python
for job in self.running_jobs:
    res_quota = ResourceQuota(job=job)   # 这个 job 占了多少 GPU
    self.quota_used += res_quota
```

**结论**：simcluster 的 utilization = 已占用 GPU 数 / 分配配额中的 GPU 数，是**资源调度层面的占用率**，回答的是"分配给我的 GPU 有多少在跑任务"。MFU 回答的是"正在跑任务的 GPU 里有多少算力在做有效的矩阵运算"。两者是不同层级的指标，不能互相替代。

## 参考

- Chowdhery et al., 2022: *PaLM: Scaling Language Modeling with Pathways*（arxiv: 2204.02311）—— MFU 的提出论文，Section 4.1 和 Appendix B 有完整推导
- Llama 3 论文 Section 3.3：报告了 38–43% 的 BF16 MFU
