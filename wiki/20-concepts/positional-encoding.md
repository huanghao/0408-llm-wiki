# 位置编码（Positional Encoding，PE）

## 为什么需要位置编码

Transformer 的 self-attention 是排列不变的（permutation equivariant）——把输入序列打乱顺序，attention 的计算结果只是相应打乱，模型感知不到"谁在前面谁在后面"。

但位置往往很重要：
- NLP 里"猫咬狗"和"狗咬猫"意思截然不同
- 驾驶场景里"1 秒前的位置"和"现在的位置"因果关系不同
- 空间里"道路的起点"和"终点"是不一样的

PE 就是把位置信息注入 token 的方式，让模型知道"这个 token 在序列里的位置"。

---

## 一：加法式 PE（最经典）

### 原始 Transformer 的正弦/余弦 PE（Vaswani et al., 2017）

**做法**：给每个位置 $t$ 计算一个固定的 D 维向量，加到 token embedding 上：

$$\text{PE}[t][2i] = \sin\left(\frac{t}{10000^{2i/D}}\right)$$
$$\text{PE}[t][2i+1] = \cos\left(\frac{t}{10000^{2i/D}}\right)$$

```python
t = 0, 1, 2, ..., L-1     # 序列位置
i = 0, 1, ..., D/2-1      # 特征维度的索引

PE: [L, D]                 # 每个位置一个 D 维向量
x_with_pos = x + PE        # 直接加到 token embedding 上，shape 不变
```

**直觉**：不同频率的正弦波叠加，相邻位置的 PE 向量相似（连续性），远距离位置的 PE 向量差异大（区分性）。低维分量频率高（区分近距离），高维分量频率低（区分长距离）。

**优点**：不需要训练，对超过训练长度的位置也能生成 PE（外推性一般）。

**缺点**：绝对位置编码——只知道"我在第 5 位"，不知道"我和第 3 位之间有多少距离"。

---

### 可学习 PE（BERT 风格）

**做法**：直接把 PE 作为参数，从随机初始化开始训练：

```python
PE = nn.Embedding(max_len, D)   # [max_len, D]，可学习参数
x_with_pos = x + PE[position]   # 查表 + 加法
```

**优点**：灵活，模型自己学到什么位置编码最有用。

**缺点**：只能处理训练时见过长度的序列（max_len 固定）；不同位置之间没有内置的连续性。

**Wayformer/MTR 用的是 0 初始化可学习 PE**：初始为 0 意味着模型从"不关心顺序"出发，如果一个模态不需要顺序信息（比如 road graph 的段），PE 训练后仍接近 0；需要顺序的（时序轨迹），PE 学到有意义的值。

---

## 二：乘法式 / 旋转式 PE

### RoPE（Rotary Position Embedding，Su et al., 2022）

**做法**：不是把 PE 加到 embedding 上，而是在每个 attention head 里**把旋转矩阵作用于 Q 和 K**：

$$\hat{q}_m = R_m \cdot q_m, \quad \hat{k}_n = R_n \cdot k_n$$

其中 $R_m$ 是位置 $m$ 的旋转矩阵。点积 $\hat{q}_m \cdot \hat{k}_n = q_m^T R_m^T R_n k_n = q_m^T R_{n-m} k_n$——只依赖**相对位置** $n-m$。

```python
# 在每个 attention 头里，计算 Q/K 之前先做旋转
cos, sin = precompute_freqs(D_head, max_len)   # 每个位置的旋转参数

def rotate_half(x):
    x1, x2 = x[..., :D//2], x[..., D//2:]
    return torch.cat([-x2, x1], dim=-1)

Q_rotated = Q * cos + rotate_half(Q) * sin   # [B, H, L, D_head]
K_rotated = K * cos + rotate_half(K) * sin   # [B, H, L, D_head]
# 之后正常计算 attention：Q_rotated @ K_rotated.T
```

**核心优势**：
- 点积结果只依赖相对位置 $n-m$，不依赖绝对位置——模型天然感知"距离"
- 外推性好：对超出训练长度的位置，旋转矩阵仍然有意义（不像可学习 PE 会越界）
- 已成为所有主流 LLM（LLaMA/GPT-NeoX/Mistral/Qwen）的标配

**直觉**：把 Q/K 向量想象成平面上的一个点，位置 $t$ 对应旋转 $t \times \theta$ 度。两个向量的点积取决于它们的角度差，角度差就是相对位置。

---

## 三：偏置式 PE

### ALiBi（Attention with Linear Biases，Press et al., 2022）

**做法**：不修改 embedding，而是在 attention score 矩阵上**直接加偏置**：

$$\text{score}[i,j] = \frac{Q_i \cdot K_j}{\sqrt{D}} - m \cdot |i - j|$$

其中 $m$ 是每个 head 预设的斜率（不同 head 的 $m$ 不同）。距离越远，惩罚越大。

```python
# attention 计算时加入位置偏置
scores = Q @ K.T / sqrt(D)           # [B, H, L, L]，标准 attention 分数
bias = -slopes * distance_matrix     # [H, L, L]，距离越远惩罚越大
scores = scores + bias
weights = softmax(scores)
```

**直觉**：简单直接——"远的东西关注少一点"写成公式。不需要 embedding，不需要旋转，纯粹的数学偏置。

**优点**：天然外推——对超出训练长度的位置，偏置还是线性增大，物理含义清晰。

**缺点**：每个 head 的 $m$ 是手工预设的（不可学习），且强制"距离越远越不相关"可能并不总成立。

---

## 四：驾驶场景里的 PE 设计

驾驶场景有多种"位置"类型，PE 的选择更复杂：

| 模态 | "位置"含义 | PE 类型 | 说明 |
|------|-----------|---------|------|
| Agent 历史轨迹 | 时间步（0=最旧帧，t=当前帧）| 正弦/可学习 | 时序有因果关系，需要 PE |
| Road Graph 段 | 折线在空间中的方向/位置 | 通常不加或 0 初始化 | 段的空间顺序不像时序那么重要 |
| 意图锚点（MTR）| 空间坐标 $(x, y)$（米）| **正弦 PE 作用于坐标值** | 把 2D 坐标编码成 D 维向量，表示"这个意图在哪个方向" |

**MTR 的意图锚点 PE**：

```python
# intention_points: [K=64, 2]，K 个锚点的 2D 坐标（单位：米）
# 目标：把 2D 坐标 (x, y) 编码成 D 维向量

# 做法：对 x 和 y 分别做正弦位置编码，再拼接
PE_x = sinusoidal_1d(intention_points[:, 0])   # [K, D/2]，对 x 坐标编码
PE_y = sinusoidal_1d(intention_points[:, 1])   # [K, D/2]，对 y 坐标编码
PE_xy = concat([PE_x, PE_y])                    # [K, D]
Q_I = MLP(PE_xy)                               # [K, D]，再过 MLP 学习非线性变换
```

这里的 PE 作用于**坐标值**（不是序列索引）：让模型知道"这个意图锚点在空间里是直行方向（前方 30m）还是左转方向（左 20m）"，不同的空间位置对应不同的 PE 向量。

---

## 各种 PE 对比

| PE 类型 | 编码方式 | 绝对 vs 相对 | 外推性 | 参数量 | 主要用途 |
|---------|---------|------------|--------|--------|---------|
| 正弦/余弦 | 加到 embedding | 绝对 | 一般 | 0（固定）| 原始 Transformer |
| 可学习 | 加到 embedding | 绝对 | 差（有界）| max_len × D | BERT、GPT-2 |
| 0 初始化可学习 | 加到 embedding | 绝对 | 差 | max_len × D | Wayformer、MTR |
| **RoPE** | 旋转 Q/K | **相对** | **好** | 0（固定）| **LLaMA/Mistral 标配** |
| ALiBi | 偏置 attention score | 相对 | 好 | 0（固定）| 部分长文本模型 |
| 坐标正弦 | 加到 embedding | 绝对空间 | — | 0（固定）| 驾驶/图像坐标编码 |

---

## 和 wiki 内其他概念的关联

- [Attention 直觉](./attention-intuition.md)：PE 为 attention 提供位置感知，否则 self-attention 是排列不变的
- [Attention 优化技术](./attention-optimization.md)：RoPE 和 ALiBi 在「位置编码改进」节
- [MTR: Motion Transformer](../30-papers/mtr-2209.13508.md)：意图锚点用正弦 PE 编码 2D 空间坐标
- [Wayformer](../30-papers/wayformer-2207.05844.md)：用 0 初始化可学习 PE，让模型自适应决定哪些模态需要顺序信息
