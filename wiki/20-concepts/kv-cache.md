# KV Cache

一句话总结：自回归生成时，把每一步算出的 Key/Value 缓存起来，避免每次生成新 token 时重复计算所有历史 token 的 K/V，是 LLM 推理加速的基础机制。

---

## 核心问题：自回归生成的重复计算浪费

Transformer 的自回归生成是逐 token 进行的：生成第 $t$ 个 token 时，需要对前 $t-1$ 个 token 做 attention。标准做法下，每一步都要重新计算所有历史 token 的 Q、K、V。

**浪费在哪里**：第 $t$ 步生成时，前 $t-1$ 个 token 的 K/V 在第 $t-1$ 步已经算过了，而它们的值不会变——K/V 只依赖各自 token 的输入，不依赖后续 token。重新计算是纯浪费。

**FLOPs 数量级**：对一个 $N$ 参数的模型，每个 token 前向传播约需 $2N$ FLOPs（线性层主导）。生成一段长度为 $L$ 的序列，朴素方案总计 $\sum_{t=1}^{L} 2N \cdot t \approx N L^2$ FLOPs，是序列长度的平方级别。有了 KV Cache，每步只计算新 token，总计约 $2NL$ FLOPs，降为线性。

以 7B 模型、生成 1000 个 token 为例：朴素方案约 $7 \times 10^9 \times 10^6 = 7 \times 10^{15}$ FLOPs，KV Cache 方案约 $7 \times 10^{12}$ FLOPs，差了约 1000 倍。

---

## KV Cache 原理

**直觉**：Decoder-only 模型做 causal attention，第 $t$ 步的新 token 只能 attend 到位置 $\leq t$ 的 token。历史 token 的 K/V 不受新 token 影响，可以安全缓存。

**每步的操作**：

1. 输入新 token，计算它的 Q、K、V（三个小向量，形状 `[1, D_head]`）
2. 把新 token 的 K、V **append** 到缓存（cache 从 `[t-1, D_head]` 变成 `[t, D_head]`）
3. 用新 token 的 Q 对缓存中**全部** K/V 做 attention，得到输出

Q 只需要当前 token 的（用来"查询"历史信息），K/V 则需要完整历史（提供被查询的信息）。

**Prefill 阶段**：输入 prompt 时，所有 token 并行计算，一次性填满 KV Cache，这个阶段是 compute-bound。之后的逐 token 生成（Decode 阶段）是 memory-bound——每步只算一个 token，但要读取全部缓存。

---

## 内存分析

**KV Cache 大小公式**：

$$\text{KV Cache 大小} = 2 \times L_{\text{layer}} \times H_{\text{kv}} \times D_{\text{head}} \times S \times B \times \text{bytes}$$

各项含义：
- $2$：K 和 V 各一份
- $L_{\text{layer}}$：Transformer 层数
- $H_{\text{kv}}$：KV head 数（MHA 下等于 Q head 数，GQA/MQA 下更少）
- $D_{\text{head}}$：每个 head 的维度
- $S$：序列长度（已生成的 token 数）
- $B$：batch size
- $\text{bytes}$：精度字节数（FP16 = 2，BF16 = 2，INT8 = 1）

**7B 模型具体估算**（以 LLaMA-2 7B 为例，MHA 配置）：

| 参数 | 数值 |
|------|------|
| 层数 $L_{\text{layer}}$ | 32 |
| Q/K/V head 数 $H$ | 32 |
| Head 维度 $D_{\text{head}}$ | 128 |
| 精度 | BF16（2 字节） |

单条序列（$B=1$）、序列长度 $S=1024$ 时：

$$2 \times 32 \times 32 \times 128 \times 1024 \times 1 \times 2 = 536{,}870{,}912 \approx 512 \text{ MB}$$

即约 **0.5 GB per 1K token（单条请求）**。

序列长度翻倍到 2K 就是 1 GB；batch size=4、序列 1K 就是 2 GB。

**和模型权重的对比**：7B 模型权重本身约 14 GB（BF16）。在 batch size=4、序列长度 2K 时，KV Cache 已经达到 4 GB，约占模型权重的 30%。随着序列长度增长，KV Cache 的占比会持续上升。

---

## Prompt Cache（Prefix Cache）

**问题**：多轮对话或多次请求往往共享相同的系统 prompt（system prompt），每次请求都重新计算这部分 K/V 是浪费。

**方案**：把系统 prompt 的 KV Cache 预计算好，存在服务端，所有请求共享同一份缓存。这被称为 **prefix caching** 或 **prompt caching**。

**效果**：Prefill 阶段跳过已缓存的前缀，直接从新内容开始计算。系统 prompt 越长、请求频率越高，节省越显著。对于"1000 token 系统 prompt + 50 token 用户问题"这类场景，Prefill 计算量可以减少 95%。

**工程实现要点**：缓存的 key 通常是 token 序列的哈希值，前缀完全一致才能命中。一旦用户消息插入，后续所有 token 的 KV 都需要重新计算（因为 attention 是 causal 的，后面的 token 依赖前面）。

主流推理框架（vLLM、TGI、SGLang）都支持 prefix caching。Anthropic API 的 prompt caching 功能本质上是同一机制的服务端实现。

---

## GQA / MQA 对 KV Cache 的影响

标准 Multi-Head Attention（MHA）中，每个 Q head 对应独立的一组 K/V head，KV Cache 大小和 Q head 数成正比。

**MQA（Multi-Query Attention）**：所有 Q head 共享同一组 K/V。KV head 数从 $H$ 降到 1，KV Cache 缩减 $H$ 倍（如 32 倍）。质量有所下降。

**GQA（Grouped-Query Attention）**：Q head 分组，每组共享一组 K/V。KV head 数从 $H$ 降到 $G$，KV Cache 缩减 $H/G$ 倍。LLaMA 3、Mistral、Gemma、Qwen 等主流模型均采用 GQA，典型配置 $G=8$（32 个 Q head 共享 8 组 K/V，缩减 4 倍）。

**7B 模型 GQA 估算**（$G=8$，其余同上，$S=1024$，$B=1$）：

$$2 \times 32 \times 8 \times 128 \times 1024 \times 1 \times 2 = 134{,}217{,}728 = 128 \text{ MB}$$

相比 MHA 的 512 MB，节省 4 倍，效果显著。

GQA/MQA 是目前 LLM 推理部署中减少 KV Cache 内存最直接的架构手段，详见 [Attention 机制的优化技术](./attention-optimization.md)。

---

## 长上下文的内存压力

**128K 上下文的 KV Cache 有多大**（以 LLaMA-3 8B 为例，GQA，$H_{\text{kv}}=8$，$L=32$，$D_{\text{head}}=128$，BF16）：

$$2 \times 32 \times 8 \times 128 \times 131072 \times 1 \times 2 = 17{,}179{,}869{,}184 \approx 16 \text{ GB}$$

单条 128K 请求的 KV Cache 已经超过模型权重本身（约 16 GB BF16）。

**实际影响**：
- **服务端**：batch size 受到严重压缩。一张 80 GB A100，128K 请求下 KV Cache 本身就能吃掉大半显存，有效 batch size 可能降到个位数。
- **推理吞吐**：Decode 阶段每步都要读取全部 KV Cache，内存带宽成为瓶颈，GPU 算力大量闲置（memory-bound）。
- **成本**：长上下文请求的推理成本远高于短请求，不仅因为计算量，更因为显存占用导致的并发度下降。

这是长上下文 LLM 推理的核心工程挑战之一。

---

## PagedAttention（vLLM）

**问题**：传统 KV Cache 为每个请求预先分配一块连续显存（按最大序列长度分配），导致两类浪费：
1. **内部碎片**：请求实际生成 200 token，但预分配了 2048 token 的空间，剩余空间浪费
2. **外部碎片**：不同大小的请求分配和释放后，显存出现不连续的空洞，无法被新请求利用

**PagedAttention 方案**：借鉴操作系统的虚拟内存分页思想，把 KV Cache 切分成固定大小的 **page**（block），每个 page 存若干 token 的 K/V（典型值 16 token/block）。每个请求的 KV Cache 由一系列不连续的 page 组成，通过 block table（逻辑 page → 物理 page 的映射表）寻址。

**核心效果**：
- **显存利用率大幅提升**：内部碎片仅在最后一个 block 内，外部碎片几乎消除，显存利用率从 60–70% 提升到 90%+
- **支持动态增长**：请求生成过程中按需分配新 block，不需要预先知道最终长度
- **Copy-on-Write 支持 beam search 和 prefix sharing**：多个请求可以共享同一批 block（prefix cache 的物理实现）

**vLLM** 是 PagedAttention 的原始实现，该论文（Kwon et al., SOSP 2023）是 LLM 推理系统方向的代表性工作。PagedAttention 已被 TGI、SGLang、TensorRT-LLM 等主流框架采纳或借鉴。

---

## 和 wiki 内其他概念的关联

- [Attention 机制的优化技术](./attention-optimization.md)：GQA/MQA 的完整介绍，以及 FlashAttention 在 Prefill 阶段的加速作用
- [量化](./quantization.md)：KV Cache 本身也可以量化（INT8 KV Cache），进一步减少内存占用；量化文档中提到的"是否优先量化 KV Cache"是长上下文部署的实际权衡
- [MFU](./mfu.md)：Decode 阶段因为 KV Cache 读取是 memory-bound，MFU 通常远低于 Prefill 阶段，是推理系统效率分析的重要背景
