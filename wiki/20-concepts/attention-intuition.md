# Attention 机制：从直觉到各种变体

## 一句话直觉

Attention 的本质是：**用一个向量（Query）去检索一组向量（Keys），找到最相关的，然后把对应的信息（Values）加权聚合回来。**

Q/K/V 的类比：图书馆检索
- **Query（查询）**：你的问题，"我想找关于猫的书"
- **Key（目录/标签）**：每本书的书签标签，"这本书讲猫"、"这本书讲狗"……
- **Value（内容）**：每本书的实际内容
- **Attention 分数**：你的问题和每本书标签的匹配程度
- **输出**：按匹配程度加权读取各本书的内容

---

## 核心计算：三步

```python
# Q: [Q_len, D]，K: [KV_len, D]，V: [KV_len, D_v]
# Q_len 和 KV_len 可以不同；D 必须相同（才能做点积）

# Step 1：计算相似度（Q 和每个 K 的点积）
scores = Q @ K.T / sqrt(D)   # [Q_len, KV_len]，Q_len × KV_len 个分数

# Step 2：归一化成权重
weights = softmax(scores, dim=-1)   # [Q_len, KV_len]，每行和为 1

# Step 3：用权重聚合 Value
output = weights @ V   # [Q_len, D_v]，输出 shape 和 Q 相同的行数
```

**速记：三个矩阵相乘 Q × K^T × V，两个乘法各有一个维度约束，结果是"Q行V列"**

用 $L_Q$ 表示 Q 的行数，$L_{KV}$ 表示 K/V 的行数（self-attention 时两者相等），$D_v$ 表示 V 的特征维度（通常等于 D，但可以不同）：

```
Q            @    K.T          =   scores        @    V            =   output
[L_Q, D]        [D, L_KV]        [L_Q, L_KV]        [L_KV, D_v]      [L_Q, D_v]
```

两个乘法的维度要求：
1. `Q @ K.T`：Q 的列数（D）= K.T 的行数（D）→ **Q 和 K 最后一维必须相同**
2. `scores @ V`：scores 的列数（$L_{KV}$）= V 的行数（$L_{KV}$）→ **K 和 V 行数必须相同**

输出 shape 是 `[L_Q, D_v]`：行数由 Q 决定，列数由 V 决定。**输出行数永远等于 Q 的行数。**

---

## Multi-Head Attention：正交维度，不是第四种变体

上面的核心计算是单头（Single-Head）版本——只用一组 $W_Q, W_K, W_V$ 做一次检索。**实际使用中几乎所有 Transformer 都用多头版本**，它和四种变体（Self/Causal/Cross/Local）是正交的维度，可以任意组合。

**多头的做法**：并行做 H 次独立的 attention，每次用不同的投影矩阵，让模型从 H 个角度同时捕获不同的关联。

```python
# 输入 x: [B, L, D]，H 个头，每头维度 D_head = D / H
H = 8
D_head = D // H

# 一次性计算所有头的 Q/K/V
Q_all = W_Q(x).reshape(B, L, H, D_head).transpose(1, 2)   # [B, H, L, D_head]
K_all = W_K(x).reshape(B, L, H, D_head).transpose(1, 2)   # [B, H, L, D_head]
V_all = W_V(x).reshape(B, L, H, D_head).transpose(1, 2)   # [B, H, L, D_head]

# 每个头独立做 attention（H 个头并行，互不干扰）
scores = Q_all @ K_all.transpose(-1, -2) / sqrt(D_head)   # [B, H, L, L]
weights = softmax(scores, dim=-1)                          # [B, H, L, L]
head_outputs = weights @ V_all                             # [B, H, L, D_head]

# 拼接所有头，再过一个线性层
concat = head_outputs.transpose(1, 2).reshape(B, L, D)    # [B, L, D]
output = W_O(concat)                                       # [B, L, D]
```

**为什么要多头**：单头只从一个角度看相关性。H 个头用 H 组不同的 $W_Q/W_K/W_V$，在训练中自动分化——头 1 可能学语法依存，头 2 学语义关联，头 3 学句法角色。最后拼接整合，获得更丰富的表示。

**参数量不变**：每头的 $D_{head} = D/H$，H 个头总参数量 $H \times D_{head} = D$，和单头相同。多头是把同样的参数量用 H 种方式分配，不是增加参数。

**复杂度**：每头序列长度仍是 L，维度缩小为 $D_h = D/H$，H 个头合计：

$$H \times O\!\left(L^2 \cdot \frac{D}{H}\right) = O(L^2 D)$$

时间复杂度和单头相同，多头不增加计算量。

**四种变体都有多头版本**（单头版只在教学时出现）：
- Multi-Head **Self**-Attention → BERT Encoder
- Multi-Head **Causal**-Attention → GPT/LLaMA Decoder
- Multi-Head **Cross**-Attention → Transformer Decoder 的 Encoder-Decoder Attention
- Multi-Head **Local Self**-Attention → MTR Encoder

---

## 各变体在著名模型中的用途

按重要性排序（越靠前越常见）：

| 变体 | 用在哪里 | 作用 |
|------|---------|------|
| **Causal Self-Attention** | GPT-2/3/4、LLaMA 1/2/3、Qwen、Mistral——**所有 Decoder-only LLM** | 自回归生成：每个 token 只能看之前的 token，防止"偷看未来" |
| **Full Self-Attention** | BERT、ViT、MTR/Wayformer 的 Encoder 部分 | 双向理解：每个 token 可以看全部 token，适合理解任务 |
| **Multi-Head（以上两者的标准配置）** | 几乎所有 Transformer | 从多角度并行提取关联，标配，单头版只在教学中出现 |
| **Cross-Attention** | 原始 Transformer（机器翻译）的 Decoder、UniAD、Wayformer Decoder | Decoder 从 Encoder 读取信息；或"意图 query 向场景 token 提问" |
| **Grouped Query Attention（GQA）** | LLaMA 3、Mistral、Gemma、Qwen 2+ | 多个 Q head 共享一组 K/V，大幅降低推理时 KV Cache 内存 |
| **Local Self-Attention** | MTR Encoder、Axial-DeepLab、TimeSformer | 大规模空间/时序数据，全局 attention 会 OOM |
| **Sliding Window + Global（Longformer）** | Longformer、BigBird | 超长文档（>8K token），大多数 token 只看局部窗口 |
| **Latent Attention（MLA）** | DeepSeek-V2/V3/R1 | K/V 先压缩到低维 latent，进一步降低 KV Cache |
| **Factorized/Axial Attention** | Wayformer、TimeSformer、Axial-DeepLab | 多维结构输入（时序×空间），分维度做 attention，避免展平后的 $O((TS)^2)$ |

**快速判断规则**：
- GPT/LLaMA/Qwen 等 LLM → Causal Self-Attention（+ GQA）
- BERT/ViT 等理解模型 → Full Self-Attention
- Encoder-Decoder（翻译/摘要）→ Encoder 用 Full，Decoder 内部用 Causal，Encoder-Decoder 之间用 Cross
- 驾驶/视频/图像大规模 token → Local Self-Attention 或 Factorized Attention

---

## 各变体详解

### 1. Full Self-Attention（双向自注意力）

**Q、K、V 全来自同一序列，每个位置可以看全部 token。**

```python
x: [L, D]

Q = W_Q(x)  # [L, D]
K = W_K(x)  # [L, D]
V = W_V(x)  # [L, D]

output = attention(Q, K, V)   # [L, D]
```

**类比**：一组人开会，每个人可以向所有人提问——充分理解整个上下文。

**用途**：BERT 类 Encoder 的特征融合，适合理解任务（分类、问答）。

**Q 是二维而不是三维**：在代码里 `Q: [L, D]`。Batch 维度隐含在外层 `[B, L, D]`，Multi-Head 时头维度另外处理。

**复杂度推导**：

| 步骤 | 操作 | Shape | 时间复杂度 |
|------|------|-------|-----------|
| Q = x W_Q | `[L,D] × [D,D]` | `[L,D]` | $O(LD^2)$ |
| K = x W_K | 同上 | `[L,D]` | $O(LD^2)$ |
| V = x W_V | 同上 | `[L,D]` | $O(LD^2)$ |
| S = Q K^T / √D | `[L,D] × [D,L]` | `[L,L]` | **$O(L^2 D)$**（主瓶颈）|
| softmax(S) | `[L,L]` | `[L,L]` | $O(L^2)$ |
| S @ V | `[L,L] × [L,D]` | `[L,D]` | $O(L^2 D)$ |

**时间复杂度**：$O(L^2 D)$；**空间复杂度**：$O(L^2)$（存 attention 矩阵，反向传播时需要）。

为什么 $L^2$ 是瓶颈：$D$ 通常几百到几千，$L$ 可以是几千到几万，$L^2$ 增长远快于 $LD$。

**精确 FLOPs 与 MFU 的关系**：大 O 忽略了系数。精确计算一层 attention 的 FLOPs（乘法+加法各算一次）：

| 操作 | FLOPs |
|------|-------|
| Q、K、V 三个投影（各 `[L,D]×[D,D]`）| $6LD^2$ |
| 输出投影 W_O | $2LD^2$ |
| Q K^T | $2L^2D$ |
| attn × V | $2L^2D$ |
| **合计（仅 attention 部分）** | $\mathbf{8LD^2 + 4L^2D}$ |

$L < 2D$ 时 $LD^2$ 项主导（大多数 LLM），$L > 2D$ 时 $L^2D$ 项主导（超长序列）。详见 [MFU](./mfu.md)。

---

### 2. Causal Attention（因果注意力 / Masked Self-Attention）

**Q、K、V 来自同一序列，但每个位置只能看自己和之前的 token。**

这是所有自回归语言模型（GPT、LLaMA、Qwen 等）的核心机制。

```python
x: [L, D]

Q = W_Q(x)  # [L, D]
K = W_K(x)  # [L, D]
V = W_V(x)  # [L, D]

scores = Q @ K.T / sqrt(D)   # [L, L]

# 把右上角（"未来"位置）设为 -∞，softmax 后权重变成 0
causal_mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float('-inf'))

weights = softmax(scores, dim=-1)   # [L, L]，右上角全是 0
output = weights @ V                 # [L, D]
```

**可视化**：attention 权重矩阵是下三角形：

```
位置:    0    1    2    3
  0   [0.9  0    0    0  ]   ← 位置0只看自己
  1   [0.3  0.7  0    0  ]   ← 位置1看0和1
  2   [0.2  0.4  0.4  0  ]   ← 位置2看0、1、2
  3   [0.1  0.2  0.3  0.4]   ← 位置3看全部
```

**为什么必须 mask**：自回归生成时，预测第 $t$ 个 token 只能用前面已生成的 token——如果 attention 能看到"未来"，训练时模型直接复制答案，什么都学不到。

**与 Full Self-Attention 的区别**：Full 用于理解（BERT，双向），Causal 用于生成（GPT，单向）。

**复杂度**：和 Full Self-Attention 相同，$O(L^2 D)$——右上角是 0，但矩阵还是 `[L, L]`，理论复杂度不变。

---

### 3. Cross-Attention（交叉注意力）

**Q 来自一个序列，K/V 来自另一个序列。**

```python
query_seq: [Q_len, D]
kv_seq:    [L, D]

Q = W_Q(query_seq)   # [Q_len, D]
K = W_K(kv_seq)      # [L, D]
V = W_V(kv_seq)      # [L, D]

output = attention(Q, K, V)   # [Q_len, D]
```

**类比**：学生（Q_len 个）向老师（L 个）提问——学生不需要和老师一样多。

**用途**：Decoder 从 Encoder 读取信息；Wayformer 的 Trajectory Decoder 里 seeds 向 scene_enc 提问。

**$L_Q \neq L_{KV}$**：Q 和 K/V 来自不同序列，行数自然可以不同。

**复杂度推导**：

| 步骤 | 操作 | Shape | 时间复杂度 |
|------|------|-------|-----------|
| S = Q K^T | `[L_Q,D] × [D,L_KV]` | `[L_Q,L_KV]` | $O(L_Q \cdot L_{KV} \cdot D)$ |
| S @ V | `[L_Q,L_KV] × [L_KV,D]` | `[L_Q,D]` | $O(L_Q \cdot L_{KV} \cdot D)$ |

**时间复杂度**：$O(L_Q \cdot L_{KV} \cdot D)$。$L_Q \ll L_{KV}$ 时效益显著：K=64 个 seed 查询 L=400 个 scene token，比 self-attention 的 $O(400^2 D)$ 小 **6.25 倍**。

---

### 4. Local Self-Attention（局部自注意力）

**Q 来自全序列，但每个 Q 的 K/V 只取最近邻的 k 个。**

```python
x: [N, D]

# 近邻按物理坐标（如地理位置、像素坐标）的欧氏距离预先计算，不是向量相似度
neighbors: [N, k, D]

Q = W_Q(x)                          # [N, D]
K = W_K(neighbors.reshape(N, k, D)) # [N, k, D]
V = W_V(neighbors.reshape(N, k, D)) # [N, k, D]

scores = Q.unsqueeze(1) @ K.transpose(-1,-2) / sqrt(D)  # [N, 1, k]
weights = softmax(scores)
output = (weights @ V).squeeze(1)   # [N, D]
```

**Q 是二维，K 是三维**：Q 每个 token 一个向量 `[N, D]`；K 每个 token 有 k 个邻居 `[N, k, D]`。用 `.unsqueeze(1)` 把 Q 扩成 `[N, 1, D]` 后做 batch 矩阵乘法。

**类比**：每个人只向身边 k 个邻居打听消息，不向所有人提问。

**用途**：MTR Encoder 处理大量 polyline（N=800+），全局 attention 会 OOM。

**复杂度推导**：

| 步骤 | 说明 | 时间复杂度 |
|------|------|-----------|
| S = Q · K_local^T | N 个 token，每个和 k 个邻居做 D 次点积 | $O(NkD)$ |
| S @ V | `[N,1,k] × [N,k,D]` | $O(NkD)$ |

**时间复杂度**：$O(N \cdot k \cdot D)$，线性于 N；**空间复杂度**：$O(N \cdot k)$，不需要 $N \times N$ 矩阵。k=16、N=800 时节省 **50 倍**（vs 全局 $O(800^2 D)$）。

---

### 4b. Deformable Attention / MSDeformAttn（可变形注意力）

**和 Local Self-Attention 容易混淆但本质不同。** 两者都是"只看少量位置"的稀疏 attention，但"少量位置"的选择方式完全不同。

| | Local Self-Attention | Deformable Attention (MSDeformAttn) |
|---|---|---|
| **采样位置** | 固定的 k 个近邻（按物理坐标预先计算）| **可学习的偏移**（由 query 特征预测）|
| **采样位置随输入变化？** | 不变（同一位置永远看同样的邻居）| 变化（不同 query 特征预测不同偏移）|
| **有无 Q K^T 点积？** | 有（标准 attention 公式）| **没有**（权重直接由线性层预测）|
| **参考点** | 每个 token 自身的物理位置 | 每个 query 的参考点（可以是自身位置或预测位置）|
| **多尺度？** | 不支持 | 天然支持（跨 L 个尺度采样）|
| **典型用途** | MTR encoder（polyline 邻居交互）| BEVFormer SCA/TSA、Deformable DETR |
| **来源** | 图注意力 / 局部窗口 attention | Deformable Convolution（Dai et al., 2017）|

**核心区别在于"可学习偏移"**：Local Attention 的邻居是固定的（如最近 16 个 polyline），不管内容是什么都看同样的位置。Deformable Attention 的采样点是 query 特征通过线性层预测的偏移量 Δp，不同的 query 会看不同的位置——模型自己学会了"对于这个 query，应该去哪里采样"。

```python
# Local Self-Attention（固定邻居）
neighbors = knn(positions, k=16)           # 预计算，和内容无关
K = W_K(features[neighbors])              # 固定位置的特征
scores = Q @ K.T / sqrt(d)               # 标准点积

# Deformable Attention（可学习偏移）
offsets = linear(query_feature)            # 从 query 内容预测偏移 [M*K*2]
weights = softmax(linear(query_feature))   # 从 query 内容预测权重 [M*K]
sampled = bilinear_sample(V, ref_point + offsets)  # 在偏移位置双线性插值
output = (weights * sampled).sum()         # 加权求和，没有 Q K^T 点积
```

**Deformable Attention 没有 Q K^T 步骤**——这是和所有其他 attention 变体最大的区别。标准 attention（包括 Local）通过 Q·K^T 点积计算"谁和谁相关"；Deformable Attention 跳过这步，直接由 query 特征预测"去哪里看"（offsets）和"看到的东西多重要"（weights）。这使得它更接近 deformable convolution 而非传统 attention。

**多尺度版本（MSDeformAttn）**：从 L=4 个尺度各采样 K=4 个点，M=8 个头，总共 L×K×M = 128 个采样点。注意力权重在所有尺度和采样点上联合归一化（Σ_{l,k} A_{mlqk} = 1），让模型自动决定从哪个尺度取信息。这使得它可以替代 FPN 做多尺度融合。

**详细架构和公式见 [Deformable DETR](../30-papers/deformable-detr-2010.04159.md) 文档。**

---

### 5. GQA（Grouped Query Attention）

**标准 MHA 里每个 Q head 各有一组对应的 K/V head（H 对 H）。GQA 把 K/V head 数量减少到 H_KV < H，让多个 Q head 共享同一组 K/V，从而大幅降低推理时 KV Cache 的内存占用。**

回顾前面 MHA 的符号：H 个头，每头维度 $D_{head} = D/H$，Q/K/V 的形状都是 `[B, H, L, D_head]`。在 GQA 里，Q 保持 H 个头，但 K/V 只有 H_KV 个头（H_KV < H），每 G = H/H_KV 个 Q head 共享一组 K/V：

```python
H    = 32   # Q head 数，和 MHA 一样
H_KV = 8    # K/V head 数，比 Q 少（GQA 常见配置）
G    = H // H_KV   # = 4，每组 4 个 Q head 共享一组 K/V
D_head = D // H    # 每头维度，和 MHA 一样

Q = W_Q(x).reshape(B, L, H,    D_head)   # [B, L, 32, D_head]，和 MHA 相同
K = W_K(x).reshape(B, L, H_KV, D_head)  # [B, L,  8, D_head]，只有 8 组
V = W_V(x).reshape(B, L, H_KV, D_head)  # [B, L,  8, D_head]

# 推理时 KV Cache：只需存 8 组 K/V，而不是 32 组
# 计算时把 K/V 广播到 32 组，再正常做 attention
K_expanded = K.repeat_interleave(G, dim=2)  # [B, L, 32, D_head]
V_expanded = V.repeat_interleave(G, dim=2)  # [B, L, 32, D_head]
```

**为什么能降低 KV Cache**：自回归生成时，每生成一个 token 都要把当前的 K/V 缓存起来供后续 token 使用。标准 MHA 需要缓存 H 组 K/V，GQA（H_KV=8）只缓存 8 组，内存减少 4 倍（32/8=4）。

**质量影响**：极小。GQA 是 MHA 和 MQA（H_KV=1，所有 Q 共享同一组 K/V）的折中——H_KV=8 时质量几乎不损失，H_KV=1（MQA）时有轻微下降。

**复杂度**：计算 attention 时 K/V 被广播到 H_Q 组，计算量和 MHA 相同；**推理时 KV Cache 内存降低 H_Q/H_KV 倍**。

**使用**：LLaMA 3（H_Q=32, H_KV=8）、Mistral 7B、Gemma、Qwen 2+。

---

### 6. Sparse Attention（稀疏注意力）

标准 Full Self-Attention 里每个 token 都 attend 所有其他 token（dense）。Sparse Attention 的思路是：**大多数 token 对之间的 attention 权重接近 0，不如一开始就不计算它们**，只保留真正有意义的连接，把复杂度从 $O(L^2)$ 降到接近线性。

Sparse Attention 有多种实现方式，最主流的是 **Sliding Window + Global（Longformer 风格）**：

**大多数 token 只看局部窗口，少数"全局 token"可以看全部 token。**

```python
# window_size=512，L=8192 的长文档

for i, token in enumerate(tokens):
    if token.is_global:
        # Global token：可以 attend 到所有 L 个 token，也被所有 token attend
        token.attends_to = all_tokens          # O(L) per global token
    else:
        # 普通 token：只 attend 到左右各 window_size/2 个 token
        start = max(0, i - window_size // 2)
        end = min(L, i + window_size // 2)
        token.attends_to = tokens[start:end]   # O(window_size) per token
```

**哪些是 Global token**：`[CLS]` token、问答任务里问题的 token、需要感知全局的特殊位置。

**复杂度**：$O(L \cdot w + n_g \cdot L)$，其中 $w$ 是窗口大小，$n_g$ 是 global token 数量。$n_g \ll L$ 时接近线性。

**直觉**：大多数信息可以通过局部传播（普通 token 从邻居获取），少数关键位置需要全局视野（global token 统筹全局）。

**BigBird 的扩展**：在 Sliding Window + Global 基础上再加一组随机连接（每个 token 还随机 attend 少量远端 token），理论上保证任意两个 token 之间的信息可以在 O(1) 层内传播。用于基因组序列（序列极长，局部窗口不够）。

**Sparse Attention 的局限**：质量有损失——全局 attention 允许任意两个 token 直接交互，Sparse 版本需要多层间接传播才能覆盖远距离依赖，在需要精细全局推理的任务（如复杂问答）上会有差距。超长文档（>8K token）且局部信息为主时才值得用。

**使用**：Longformer（文档级 NLP）、BigBird（基因组序列）、超长上下文任务。

---

### 7. Factorized / Axial Attention

**对于有多个维度结构的输入（时序×空间、行×列），沿各维度分别做 attention，而不是展平成一个长序列。**

```python
# 输入 x: [T, S, D]（时序 T × 空间 S × 特征 D）

# 全局 attention：展平后复杂度 O((T×S)²)
x_flat = x.reshape(T*S, D)
x_flat = self_attention(x_flat)   # O((T·S)²)

# Factorized/Axial attention：分两步，复杂度 O(T·S² + S·T²)
# 第一步：沿 S 轴（把 T 放进 batch，S 是序列长度）
x = x.reshape(T, S, D)
x = self_attention(x)   # 等价于对每个时间步 t 分别做 [S,D] attention，共 T 次 O(S²)

# 第二步：沿 T 轴（把 S 放进 batch，T 是序列长度）
x = x.permute(1, 0, 2)   # [S, T, D]
x = self_attention(x)     # 等价于对每个空间位置 s 分别做 [T,D] attention，共 S 次 O(T²)
x = x.permute(1, 0, 2)   # 还原 [T, S, D]
```

两步 attention 的组合使任意两个 token 之间可以通过最多两步互相影响，保留全局感受野。

**复杂度**：$O(T \cdot S^2 + S \cdot T^2) = O(TS(T+S))$，远小于全局 $O((TS)^2)$。

**使用**：Axial-DeepLab（图像，行×列）、TimeSformer（视频，时序×空间帧）、Wayformer（驾驶，时序×agent 邻居）。

详细说明见 [Attention 优化技术](./attention-optimization.md)。

---

### 8. MLA（Multi-head Latent Attention）

**K/V 先通过低秩压缩写入小 latent 向量，推理时从 latent 还原 K/V，大幅降低 KV Cache。**

```python
# 标准 MHA 的 KV Cache 大小：2 × L × H × D_head（每个 token 存 K 和 V）
# MLA：先把 K/V 压缩到低维 latent，Cache 只存 latent

d_c = D // 4   # latent 维度，约为原来的 1/4

# 下行投影（Down-projection）：把 x 压缩到低维 latent
c_KV = W_DKV(x)          # [L, d_c]，KV Cache 只存这个！

# 上行投影（Up-projection）：从 latent 还原 K/V（推理时实时计算）
K = W_UK(c_KV)            # [L, H, D_head]，从 latent 还原 K
V = W_UV(c_KV)            # [L, H, D_head]，从 latent 还原 V

# Q 也类似压缩，但 Q 不需要 Cache（只在当前 step 用）
c_Q = W_DQ(x)             # [L, d_c']
Q = W_UQ(c_Q)             # [L, H, D_head]
```

**KV Cache 节省**：标准 MHA 每个 token 存 `2 × H × D_head` 个值，MLA 每个 token 只存 `d_c`（约 D/4）个值，节省约 **8 倍**（以 DeepSeek-V2 的配置为例）。

**质量**：DeepSeek-V2/V3/R1 的实验结果显示，MLA 在 KV Cache 减少 8 倍的情况下，模型质量和标准 MHA 相当甚至略好。

**使用**：DeepSeek-V2、DeepSeek-V3、DeepSeek-R1。这是 2024 年的新变体，代表了"更极致地节省 KV Cache"的方向。

---

## 变体对比

| | Full Self | Causal Self | Cross | Local Self | **Deformable** | GQA | Sparse | Factorized | MLA |
|---|---|---|---|---|---|---|---|---|---|
| **核心变化** | 基础双向 | mask 未来 | Q/KV 不同源 | 只看近邻 | **可学习偏移采样** | K/V head 共享 | 局部窗口+少量全局 | 分维度 attention | K/V 先压缩 |
| **有 QK^T？** | ✓ | ✓ | ✓ | ✓ | **✗（权重由线性层预测）** | ✓ | ✓ | ✓ | ✓ |
| **典型模型** | BERT | GPT/LLaMA | Transformer Dec | MTR | **BEVFormer/Def.DETR** | LLaMA 3/Mistral | Longformer | Wayformer | DeepSeek-V2 |
| **解决什么** | — | 自回归 | 跨序列 | 大规模 token | **多尺度视觉+快收敛** | 推理内存 | 超长文档 | 多维结构 | 推理内存 |
| **复杂度** | $O(L^2D)$ | $O(L^2D)$ | $O(L_QL_{KV}D)$ | $O(NkD)$ | **$O(N_qMKD)$** | $O(L^2D)$ | $O(LwD)$ | $O(TS(T+S)D)$ | $O(L^2D)$ |

多头版本（MHA）可以和前四种变体任意组合，GQA/MLA 是 MHA 本身的变体。

**关键规律**：**输出的行数永远等于 Q 的行数**，不管 K/V 有多少行。

---

## 各场景中 Q/K/V 的直觉

### Wayformer Decoder（Cross-Attention）

```
Q = K 个 seed（意图向量）    ← 提问方："这个意图对应什么轨迹？"
K = scene_enc（场景 token） ← 被检索的信息库
V = scene_enc
输出: [K, D]，K 个 seed 各自从场景里聚合了相关信息
```

直觉：K 个"意图专家"各自向场景"询问"和自己意图相关的信息——直行专家关注前方道路，左转专家关注左侧车道。

### MTR Encoder（Local Self-Attention）

```
Q = 当前 polyline token       ← "我想了解周围情况"
K = 最近 16 个邻居 token
V = 最近 16 个邻居 token
输出: [N, D]，每个 token 融合了近邻的信息
```

直觉：每条道路段只听取周围道路的情况，不关心远处的道路。

### MTR Decoder（多重 Cross-Attention）

```
# Step 1：意图间的 self-attention
Q = K/V = [K 个意图查询]    ← 让不同意图感知彼此

# Step 2：意图向 agent 提问
Q = [K 个意图查询]
K = V = [N_a 个 agent]

# Step 3：意图向动态地图提问
Q = [K 个意图查询]
K = V = [局部 128 条 polyline]
```

直觉：每个意图专家依次看了场景里的车和自己轨迹附近的路况，做出更精确的预测。

---

## 常见混淆点

**Q 和 K/V 的行数不同时违反规则吗？**

不违反。唯一的约束是 Q 和 K **最后一维 D 必须相同**（才能做点积）。行数（序列长度）可以完全不同，Cross-Attention 天然就是 $L_Q \neq L_{KV}$ 的情况。

**Self-Attention 输入输出 shape 相同是必然的吗？**

不是，只是 Self-Attention 的特例。Q/K/V 来自同一序列所以行数相同，输出行数等于 Q 的行数，所以 shape 相同。Cross-Attention 的输出 shape 跟 Q 走，不等于输入 shape。

**为什么要 W_Q/W_K/W_V 三个矩阵，而不是直接用 x 做点积？**

三个独立线性变换给模型自由度：Q 学"如何查询"，K 学"如何被找到"，V 学"聚合什么信息"。直接用 x 做点积相当于三个矩阵都是单位矩阵，表达能力弱很多。

**Causal Attention 是所有 Encoder-Decoder 架构都需要的吗？**

不是。Causal Attention 解决的是**自回归生成**中的问题：训练时已知的过去和未知的未来被拼在同一个序列里，必须 mask 住未来防止"作弊"。Wayformer 完全不需要它，因为：
- Encoder 只处理历史数据（已知的过去 1 秒），全部已知，不存在"未来 token"，可以用 Full Self-Attention 双向看
- Decoder 一次性输出全部 80 步轨迹（`[K, 80, 4]`），不是自回归逐步生成的——历史和未来严格分离，历史进 Encoder，未来由 Decoder 直接输出

只有"已知和未知混在同一序列里，且逐步生成"的场景（LLM 的 next-token prediction）才需要 Causal Mask。

---

## Q/K/V 类比的历史与局限

Attention 机制来自 Bahdanau et al.（2014）的机器翻译工程问题：RNN seq2seq 把整个输入压缩成固定向量，长句翻译质量差；解法是让 decoder 每步都能"回去看"整个输入，对每个位置算权重再加权求和。Q/K/V 的形式化表述是 Vaswani et al.（2017）在 Transformer 论文里整理的，"图书馆检索"类比是事后总结，不是设计出发点。

类比是理解和沟通工具，不是验证工具——"听起来像图书馆检索"无法证明模型有效，最终靠实验。

---

## 和 wiki 内其他概念的关联

- [Wayformer](../30-papers/wayformer-2207.05844.md)：Cross-Attention 最详细的应用案例，seeds → scene_enc 的 QKV 流向
- [MTR: Motion Transformer](../30-papers/mtr-2209.13508.md)：Local Self-Attention + 多重 Cross-Attention 的组合使用
- [Attention 优化技术](./attention-optimization.md)：FlashAttention/GQA/RoPE 等在计算效率上的改进
- [Tensor 操作参考](./tensor-operations.md)：unsqueeze/reshape/permute 等用于调整 attention 计算所需的 tensor shape
- [位置编码（PE）](./positional-encoding.md)：RoPE/ALiBi 等 PE 与 attention 的结合方式
