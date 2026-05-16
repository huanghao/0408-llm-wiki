# Attention 机制的优化技术

标准 Multi-Head Self-Attention 的复杂度是 $O(L^2 \cdot D)$——序列长度平方级别，长序列时极其昂贵。

---

## 优化手段总览

这篇文档只收录**不改变 attention 语义、只改变计算方式**的加速手段。改变"Q 能 attend 哪些 K/V"的变体（GQA/MLA/Factorized/Sliding Window 等）在 [Attention 直觉](./attention-intuition.md) 里详解。

按工业采用度排序。精度列：✓ = 精确等价无损失，△ = 轻微影响，✗ = 有损失。

| 方法 | 解决什么 | 精度 | 主要用途 | 工业采用度 |
|------|---------|------|---------|----------|
| **FlashAttention** | 显存带宽瓶颈 | ✓ 无损 | 所有 Transformer 训练/推理 | ★★★ 所有主流 LLM |
| **RoPE / ALiBi** | 位置编码长度外推 | ✓ 通常更好 | LLM 位置编码 | ★★★ LLM 标准配置 |
| **Latent Queries** | 高冗余输入压缩 | △ 有压缩损失 | 多模态/传感器输入 | ★★☆ Wayformer 等 |
| **Linear Attention** | O(L²) → O(L) | ✗ 有损 | — | ★☆☆ 研究阶段 |

**阅读建议**：FlashAttention 是工程必知项，几乎所有主流框架已内置。RoPE/ALiBi 在读 LLM 系统论文时会频繁出现。

---

## 标准 Attention 的计算瓶颈

```python
# 标准 Self-Attention，序列长度 L，维度 D
Q, K, V = W_Q(x), W_K(x), W_V(x)    # [B, L, D]

attn = softmax(Q @ K.T / sqrt(D))     # [B, L, L]  <- O(L²) 的瓶颈
output = attn @ V                      # [B, L, D]
```

**为什么是 O(L²)**：`Q @ K.T` 的输出是 `[L, L]`，有 L² 个元素，每个元素是长度为 D 的向量点积（O(D) 次操作），总计 O(L²D)。通用法则：输出元素数 × 每个元素的操作量。

**为什么内存也是 O(L²)**：反向传播时需要保存前向的中间变量。计算 dL/dQ 需要用到 attn 矩阵，所以这个 `[B, L, L]` 的矩阵要在内存里保留到反向传播完成。L=2048、H=32 时约占 256MB。

---

## FlashAttention（★★★）

不改变计算量（仍然 O(L²)），改变计算顺序——把 Q/K/V 分块，在片上高速缓存（SRAM）里完成局部计算，不把完整的 `[L, L]` 矩阵写入显存（HBM）。反向传播时从 Q/K/V 重新计算，用计算换内存。

**效果**：训练 2-4× 加速，内存减少 5-20×，**精度完全一致（非近似）**。

PyTorch 2.0+ 的 `F.scaled_dot_product_attention` 自动使用，GPT-4/LLaMA/Mistral 等所有主流 LLM 都在用。

---

## RoPE / ALiBi（★★★）

标准正弦 PE 在序列长度超出训练长度时表现急剧下降。

**RoPE**：旋转矩阵作用于 Q/K，让相对位置 j-i 自然出现在点积里，外推性更好。LLaMA/GPT-NeoX/Mistral 等标准配置。

**ALiBi**：在 attention score 上加和相对距离成比例的负偏置，不需要位置 embedding，天然支持长度外推。

---

## Latent Queries（★★☆）

```python
latent = nn.Parameter(randn(M, D))              # M 个可学习向量，M << L

# 第一层：cross-attention，把 L 个 token 压缩进 M 个 latent（O(M*L)，线性）
latent = cross_attention(Q=latent, KV=tokens)

# 后续层：在短序列 M 上做 self-attention（O(M²) << O(L²)）
latent = self_attention(latent)
```

不是简单的 Linear(L→M)，而是 cross-attention 让每个 latent 向量自主决定聚合哪些信息。来源：Perceiver（Jaegle et al., ICML 2021）；Wayformer 的 Latent Query 加速模块。

---

## Linear Attention（★☆☆）

用 kernel 函数替代 softmax，利用结合律先算 phi(K)^T V（D×D 矩阵）再乘 phi(Q)，把 O(L²D) 降到 O(LD²)。但近似 softmax 的质量不足，长程任务表现差，目前仍是研究阶段。代表：Performer、Linformer、RetNet。

---

## 和 wiki 内其他概念的关联

- [Attention 直觉](./attention-intuition.md)：GQA/MLA/Factorized/Sliding Window 等改变 attend 范围的变体详解
- [位置编码（PE）](./positional-encoding.md)：RoPE/ALiBi 的完整推导和对比
- [Wayformer](../30-papers/wayformer-2207.05844.md)：Latent Queries 的具体应用场景
- [分布式训练](./distributed-training.md)：FlashAttention 节省的内存使更大 batch 的分布式训练成为可能
