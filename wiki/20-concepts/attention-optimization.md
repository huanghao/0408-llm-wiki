# Attention 机制的优化技术

标准 Multi-Head Self-Attention 的复杂度是 $O(L^2 \cdot D)$——序列长度平方级别，长序列时极其昂贵。

---

## 优化手段总览

按工业采用度排序。精度列：✓ = 精确等价无损失，△ = 轻微影响，✗ = 有损失。

| 方法 | 解决什么 | 精度 | 主要用途 | 工业采用度 |
|------|---------|------|---------|----------|
| **FlashAttention** | 显存带宽瓶颈 | ✓ 无损 | 所有 Transformer 训练/推理 | ★★★ 所有主流 LLM |
| **GQA / MQA** | 推理 KV Cache 内存 | ✓ 极小影响 | LLM 推理部署 | ★★★ LLaMA/Mistral/Gemma |
| **RoPE / ALiBi** | 位置编码长度外推 | ✓ 通常更好 | LLM 位置编码 | ★★★ LLM 标准配置 |
| **Factorized/Axial** | 多维输入的 O(L²) | △ 轻微 | 视频/图像/驾驶 | ★★☆ 领域主流 |
| **Latent Queries** | 高冗余输入压缩 | △ 有压缩损失 | 多模态/传感器输入 | ★★☆ Wayformer 等 |
| **Sparse Attention** | 长序列的 O(L²) | ✗ 有损 | 超长文档（>8K） | ★☆☆ BigBird/Longformer |
| **Linear Attention** | O(L²) → O(L) | ✗ 有损 | — | ★☆☆ 研究阶段 |

**阅读建议**：如果你用 LLM 或通用 Transformer，重点看前三个。如果做视频/驾驶/图像这类多维结构输入，再看 Factorized/Axial。

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

## GQA / MQA（★★★）

标准 MHA 每个 head 各有一组 K/V，GQA 让多个 head 共享同一组 K/V：

```python
# 标准 MHA（H=32 个 head）
K = W_K(x).reshape(B, L, H=32, D_head)   # 32 组 K

# GQA（G=8 组 K/V，每 4 个 head 共享一组）
K = W_K(x).reshape(B, L, G=8, D_head)    # 8 组 K
```

主要效果：减少推理时 KV Cache 内存（自回归生成时每步缓存所有历史 K/V，共享后减少 H/G 倍）。LLaMA 3/Mistral/Gemma/Qwen 都用 GQA，G=8 是典型配置。

---

## RoPE / ALiBi（★★★）

标准正弦 PE 在序列长度超出训练长度时表现急剧下降。

**RoPE**：旋转矩阵作用于 Q/K，让相对位置 j-i 自然出现在点积里，外推性更好。LLaMA/GPT-NeoX/Mistral 等标准配置。

**ALiBi**：在 attention score 上加和相对距离成比例的负偏置，不需要位置 embedding，天然支持长度外推。

---

## Factorized / Axial Attention（★★☆）

**适用场景**：输入有明确的多维结构（时序×空间、行×列、传感器×时间步）。

**问题**：把多维输入展平后复杂度爆炸。例：时序 T=10，空间 S=128，展平后序列长 1280，复杂度 O(1280²) = O(1.6M)。

**Axial 方案**：沿每个轴分别做 attention，用 for 循环理解最直观：

```python
# 输入 [T, S, D]（时序 × 空间 × 特征）
# 展平方案：O((T×S)²) 复杂度
x = self_attention(x.reshape(T*S, D))

# Axial 方案：分两步，各轴独立
# 第一步：沿 S 轴做 attention（每个时间步 t 独立，S 个 token 互相 attend）
for t in range(T):
    x[t] = self_attention(x[t])    # x[t] 的 shape 是 [S, D]，O(S²)
# 执行 T 次 O(S²)，总复杂度 O(T * S²)

# 第二步：沿 T 轴做 attention（每个空间位置 s 独立，T 个 token 互相 attend）
for s in range(S):
    x[:, s] = self_attention(x[:, s])  # x[:,s] 的 shape 是 [T, D]，O(T²)
# 执行 S 次 O(T²)，总复杂度 O(S * T²)

# 两步加起来：O(T*S² + S*T²) = O(TS(T+S))，远小于 O((TS)²) = O(T²S²)
```

for 循环版本等价于更高效的 reshape 版本（把"不参与 attention 的维度"放进 batch），两者的数学结果完全一样——详见 [Tensor 操作参考：为什么 reshape 和 for 循环等价](./tensor-operations.md)。

**有没有质量损失**：有轻微损失——全轴 attention 允许 T 和 S 方向的任意两个 token 直接交互，Axial 需要两步才能跨维度交互。实践中多堆几层可以补偿，大模型时差异消失。

**来源**：Axial-DeepLab（Wang et al., ECCV 2020），Timesformer（视频），Wayformer（驾驶）。

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

## Sparse Attention（★☆☆）

每个 query 只 attend 到局部窗口，复杂度 O(L × w)（w = 窗口大小）。变体：Local+Global（Longformer）、BigBird（Local+Global+Random）。质量有损失，适合超长序列（>8K）且局部信息为主的任务。

---

## Linear Attention（★☆☆）

用 kernel 函数替代 softmax，利用结合律先算 phi(K)^T V（D×D 矩阵）再乘 phi(Q)，把 O(L²D) 降到 O(LD²)。但近似 softmax 的质量不足，长程任务表现差，目前仍是研究阶段。代表：Performer、Linformer、RetNet。

---

## 和 wiki 内其他概念的关联

- [Tensor 操作参考](./tensor-operations.md)：reshape/permute/expand/einsum 详解，理解 Axial Attention 实现的基础
- [Wayformer](../30-papers/wayformer-2207.05844.md)：Factorized Attention 和 Latent Queries 的具体应用
- [Axial-DeepLab](../30-papers/axial-deeplab-2003.07853.md)：Axial Attention 的原始论文
- [分布式训练](./distributed-training.md)：FlashAttention 节省的内存使更大 batch 的分布式训练成为可能
