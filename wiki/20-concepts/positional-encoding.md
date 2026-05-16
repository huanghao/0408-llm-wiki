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

**数值示例（D=4，即 i=0,1 两对 sin/cos）**：

分母 $10000^{2i/D}$：i=0 时为 $10000^0 = 1$，i=1 时为 $10000^{2/4} = 100$。于是：

| 位置 t | PE[t][0] = sin(t/1) | PE[t][1] = cos(t/1) | PE[t][2] = sin(t/100) | PE[t][3] = cos(t/100) |
|--------|--------------------|--------------------|----------------------|----------------------|
| t=0    | sin(0)=**0.00**    | cos(0)=**1.00**    | sin(0)=**0.000**     | cos(0)=**1.000**     |
| t=1    | sin(1)=**0.84**    | cos(1)=**0.54**    | sin(0.01)=**0.010**  | cos(0.01)=**1.000**  |
| t=3    | sin(3)=**0.14**    | cos(3)=**-0.99**   | sin(0.03)=**0.030**  | cos(0.03)=**1.000**  |
| t=10   | sin(10)=**-0.54**  | cos(10)=**-0.84**  | sin(0.1)=**0.100**   | cos(0.1)=**0.995**   |
| t=100  | sin(100)=**-0.51** | cos(100)=**0.86**  | sin(1)=**0.841**     | cos(1)=**0.540**     |

观察这张表：
- **低维（i=0）变化快**：t=0→1 时 PE[0] 从 0.00 跳到 0.84，每步变化大，能区分相邻位置
- **高维（i=1）变化慢**：t=0→100 时 PE[2] 才从 0.000 变到 0.841，能区分远距离位置
- **所有值在 [-1,1] 之间**：sin/cos 的值域，永远不会超出这个范围

**会重复吗？** 理论上 sin/cos 是周期函数，$\sin(t/1)$ 的周期是 $2\pi \approx 6.28$——t=6 和 t=0 的低维分量值很接近。但 PE 是 D 维向量（D 通常 512+），每对 sin/cos 的频率不同，所有维度同时重复的概率极低。Vaswani 等人用 $10000^{2i/D}$ 作为分母，就是为了让不同维度的周期相差极大（从 $2\pi$ 到 $20000\pi$），使得在任何实际序列长度内（通常 <10000）都不会出现两个完全相同的 PE 向量。

**作者为什么选这个形式？** 论文里的设计目标有两个：① 每个位置有唯一编码；② 相对位置可以从 PE 向量里推算出来。正弦/余弦满足一个数学性质：$\text{PE}[t+k]$ 可以表示成 $\text{PE}[t]$ 的线性变换（旋转矩阵），即模型理论上可以从两个位置的 PE 向量计算出它们之间的距离。可学习 PE 没有这个性质。此外，正弦 PE 不需要训练数据，能外推到训练时没见过的位置。

**优点**：不需要训练，对超过训练长度的位置也能生成 PE（外推性一般）。

**缺点**：绝对位置编码——只知道"我在第 5 位"，不知道"我和第 3 位之间有多少距离"。

---

### 可学习 PE（BERT 风格）

**做法**：直接把 PE 作为参数，从随机初始化开始训练：

```python
PE = nn.Embedding(max_len, D)   # [max_len, D]，可学习参数
x_with_pos = x + PE[position]   # 查表 + 加法
```

`max_len` 是训练时预先设定的最大序列长度，比如 512 或 2048。`nn.Embedding` 是一张大小固定的表，有 max_len 行，每行是一个 D 维向量，通过训练学出来。**超过 max_len 就查不到对应行了**——这和正弦 PE 不同，正弦 PE 可以代入任意 t 值计算，可学习 PE 只有 max_len 个格子，多出来的位置没有 PE 可用。

**优点**：灵活，模型自己学到什么位置编码最有用。

**缺点**：
- **无法处理超出训练长度的序列**：训练时 max_len=512，推理时来了一个 600 token 的输入，第 513 到 600 个位置没有对应的可学习 PE，需要截断或额外处理
- **位置之间没有内置的连续关系**：正弦 PE 里 PE[5] 和 PE[6] 因为公式相似所以向量也相近。可学习 PE 里 PE[5] 和 PE[6] 是完全独立训练出来的两个向量，它们的相似程度取决于训练数据碰巧如何驱动——模型没有理由主动让相邻位置的 PE 相似。如果训练数据里位置 5 和位置 6 的 token 行为很像，模型可能学出相近的 PE；如果不像，两个 PE 向量可能差距很大。这不一定是坏事（位置 5 本来就和位置 6 有不同含义），但缺少"位置是连续的"这个归纳偏置

**为什么可学习 PE 没有成为最主流的选择**：灵活性是优点，但带来了两个代价。第一是外推性差（只能处理训练时见过的长度）；第二是相对位置感知缺失（模型只知道"我在第 5 位"，不知道"我比第 3 位晚了 2 步"）。RoPE 在保持灵活性的同时解决了这两个问题，成为 LLM 的标配；可学习 PE 仍然用在 BERT、Wayformer 等对序列长度有上限、且不需要感知相对距离的场景里。

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
