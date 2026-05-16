# Transformer 架构

一句话总结：Transformer 用 self-attention 替代了 RNN 的时序递推，让序列建模可以完全并行化，同时捕获任意距离的依赖关系，成为现代 LLM 的基础骨架。

---

## 核心问题：为什么需要 Transformer

### RNN 的根本缺陷

在 Transformer 出现之前，序列建模的主流是 RNN（Recurrent Neural Network）及其变体 LSTM、GRU。RNN 的工作方式是逐 token 递推：

```
h_t = f(h_{t-1}, x_t)
```

这带来两个根本性的问题：

**1. 无法并行训练**

每一步的隐状态 `h_t` 依赖上一步的 `h_{t-1}`，必须按顺序计算。序列长度为 L 时，训练需要 L 步串行计算，无法充分利用 GPU 的并行能力。现代 GPU 有成千上万个 CUDA 核，但 RNN 每次只能用其中一小部分。

**2. 长距离依赖退化**

信息通过隐状态一步步传递，经过很多步后会被覆盖或稀释——这就是梯度消失问题的根源。LSTM 通过门控机制缓解了这个问题，但没有根本解决：序列越长，早期 token 的信息越难保留到末尾。

**Transformer 的解法**：抛弃递推，改用 self-attention——每个 token 直接和序列中所有其他 token 交互，距离不再是障碍，计算可以完全并行。

---

## Transformer Block 的结构

一个标准的 Transformer Block 由以下几个组件组成：

```
输入 x
  │
  ├─ LayerNorm (Pre-LN 风格)
  │
  ├─ Multi-Head Self-Attention
  │
  ├─ 残差连接 (x = x + attn_output)
  │
  ├─ LayerNorm
  │
  ├─ FFN（Feed-Forward Network）
  │
  └─ 残差连接 (x = x + ffn_output)
  │
输出 x
```

### Residual Connection（残差连接）

每个子层（attention 和 FFN）的输出都加回输入：

```
x = x + SubLayer(x)
```

**作用**：让梯度在反向传播时有一条"高速公路"直接流回早期层，解决深层网络的梯度消失问题。没有残差连接，超过几十层的网络几乎无法训练。直觉是：SubLayer 只需要学习"在原有表示上做什么修正"，而不是"从零开始表示这个 token"。

### LayerNorm

对每个 token 的特征维度做归一化（均值为 0，方差为 1），然后乘以可学习的缩放参数 γ 和偏移参数 β：

```
LayerNorm(x) = γ · (x - mean(x)) / std(x) + β
```

**注意**：LayerNorm 是对单个 token 的 D 维向量做归一化，而 BatchNorm 是对 batch 维度做归一化。LLM 用 LayerNorm 的原因是：batch 里每条序列长度不同，BatchNorm 不好处理变长序列。

### Pre-LN vs. Post-LN

原始 Transformer（Vaswani et al., 2017）用的是 **Post-LN**：

```
x = LayerNorm(x + SubLayer(x))
```

现代 LLM（GPT-2 之后）几乎全部改用 **Pre-LN**：

```
x = x + SubLayer(LayerNorm(x))
```

**为什么 Pre-LN 更好**：Post-LN 在深层网络里训练不稳定，需要精心设计的 warmup 学习率调度。Pre-LN 把 LayerNorm 放在 SubLayer 之前，梯度流更稳定，可以用更大的学习率，训练更容易收敛。代价是最后一层输出没有经过归一化，通常在整个网络末尾再加一个 LayerNorm。

**RMSNorm**：LLaMA 系列进一步简化，用 RMSNorm 替代 LayerNorm——去掉均值减法，只做方差归一化。计算更快，效果相当。

### FFN（Feed-Forward Network）

每个 Transformer Block 里有一个两层全连接网络，作用于每个 token 独立（不跨 token 交互）：

```
FFN(x) = W_2 · activation(W_1 · x + b_1) + b_2
```

- 第一层把维度从 D 扩展到 4D（原始 Transformer 的惯例，现代 LLM 通常是 8/3 × D）
- 激活函数：原始用 ReLU，现代 LLM 普遍用 SwiGLU（LLaMA）或 GELU（GPT 系列）
- 第二层把维度压回 D

**FFN 的作用**：attention 负责 token 之间的信息交换（"谁和谁相关"），FFN 负责对每个 token 做非线性变换（"这个 token 应该如何表示"）。研究表明 FFN 层存储了大量的事实性知识，可以类比为"记忆库"。

**参数量占比**：在标准 Transformer 里，FFN 的参数量约占整个模型的 2/3，attention 约占 1/3。

---

## Self-Attention 的 QKV 机制

Self-Attention 的详细计算见 [Attention 直觉](./attention-intuition.md)。这里只做概要。

每个 token 的表示 x 经过三个线性变换，得到 Query、Key、Value 三个向量：

```
Q = W_Q · x,  K = W_K · x,  V = W_V · x
```

然后用 Q 和所有位置的 K 做点积，得到注意力权重，再用权重聚合 V：

```
Attention(Q, K, V) = softmax(Q·Kᵀ / √D) · V
```

**核心直觉**：每个 token 用自己的 Q 去"询问"序列里所有 token 的 K，找到最相关的，再把对应的 V 加权聚合回来。这一步让每个 token 都能直接获取序列中任意位置的信息，复杂度 O(L²)，但完全可并行。

---

## Multi-Head 的作用直觉

单头 attention 只能从一个角度检索信息。Multi-Head Attention（MHA）并行做 H 次 attention，每次用不同的投影矩阵：

```
head_i = Attention(W_Qi · x, W_Ki · x, W_Vi · x)
output = W_O · concat(head_1, ..., head_H)
```

每个头的维度是 D/H（总参数量不变），但每个头学到不同的"关注角度"：

- 有的头可能专注于**语法依存**（动词找主语）
- 有的头可能专注于**语义相似**（近义词之间的关联）
- 有的头可能专注于**位置邻近**（相邻 token 之间的局部关系）
- 有的头可能专注于**指代消解**（代词找先行词）

这种分工是在训练中自动涌现的，不是人为设定的。多头机制让模型同时捕获多种类型的依赖关系，最后拼接整合。

**参数量**：H 个头的总参数量和单头完全一样——每头维度 D_head = D/H，H 个头合计仍是 D。多头是把同样的参数量用 H 种方式分配，而不是增加参数。

---

## 三类架构：Encoder-only / Decoder-only / Encoder-Decoder

Transformer 原始论文（Vaswani et al., 2017）是为机器翻译设计的，包含 Encoder 和 Decoder 两部分。后来的研究发现不同任务适合不同的架构变体。

### Encoder-only

**结构**：只有 Encoder，每个 token 的 self-attention 可以看到序列中所有位置（双向注意力）。

**典型模型**：BERT、RoBERTa、DeBERTa

**适用任务**：理解类任务——分类、NER、语义相似度、问答（抽取式）

**为什么双向**：Encoder 的目标是理解输入，不需要自回归生成。每个 token 可以同时看到左边和右边的上下文，表示更充分。

**训练目标**：Masked Language Modeling（MLM）——随机遮住 15% 的 token，让模型预测被遮住的词。

**局限**：不擅长生成任务，因为没有自回归机制。

### Decoder-only

**结构**：只有 Decoder，self-attention 用因果掩码（causal mask）——每个 token 只能看到自己和之前的 token（单向注意力）。

**典型模型**：GPT 系列、LLaMA、Mistral、Qwen、Claude

**适用任务**：生成类任务——文本生成、对话、代码生成、推理

**为什么单向**：自回归生成时，位置 t 的 token 在生成时还不知道位置 t+1 之后的内容，因果掩码保证了训练和推理的一致性。

**训练目标**：Next Token Prediction——预测序列中每个位置的下一个 token，损失是交叉熵。

**为什么 LLM 都用 Decoder-only**：
1. 训练目标简单统一，可以直接用所有文本做无监督预训练
2. 自回归生成天然支持 few-shot prompting（把示例和问题拼在一起作为 context）
3. 规模扩大后涌现出的 in-context learning 能力主要来自 Decoder-only 架构

### Encoder-Decoder

**结构**：Encoder 处理输入序列（双向注意力），Decoder 生成输出序列（单向注意力 + cross-attention 读取 Encoder 输出）。

**典型模型**：T5、BART、mT5、原始 Transformer（机器翻译）

**适用任务**：输入输出都是序列、且两者有明确对应关系的任务——机器翻译、摘要、问答（生成式）

**Cross-Attention 的作用**：Decoder 每一层都有一个 cross-attention 模块，Q 来自 Decoder 的当前状态，K/V 来自 Encoder 的输出。这让 Decoder 在生成每个 token 时都能"查阅"完整的输入序列。

**为什么逐渐式微**：T5 等模型在 few-shot 场景下表现不如同等参数量的 Decoder-only 模型；Encoder-Decoder 的参数利用率也不如 Decoder-only 高效（Encoder 和 Decoder 各用一半参数）。

### 三类架构对比

| | Encoder-only | Decoder-only | Encoder-Decoder |
|--|-------------|-------------|-----------------|
| **注意力方向** | 双向 | 单向（因果） | Encoder 双向 + Decoder 单向 |
| **训练目标** | MLM | Next Token Prediction | Seq2Seq |
| **代表模型** | BERT, RoBERTa | GPT, LLaMA, Qwen | T5, BART |
| **强项** | 理解、分类 | 生成、对话、推理 | 翻译、摘要 |
| **LLM 主流** | 否（已边缘化） | **是** | 否（逐渐减少） |

---

## 一个 Token 的完整前向传播路径（Decoder-only）

以 LLaMA 风格的 Decoder-only 模型为例，假设：
- 词表大小 V = 128,256
- 模型维度 D = 4,096
- 注意力头数 H = 32，每头维度 D_head = 128
- FFN 中间维度 = 14,336（约 3.5D）
- 层数 L = 32
- 输入序列长度 T = 512

**步骤 1：Token Embedding**

```
token_id: [T]  →  embedding lookup  →  x: [T, D]
                                          [512, 4096]
```

每个 token ID 查嵌入矩阵（形状 [V, D]），得到对应的 D 维向量。

**步骤 2：位置编码**

RoPE 不改变 x 的 shape，而是在每层 attention 计算 Q/K 时施加旋转变换，x 仍为 `[T, D]`。（详见 [位置编码](./positional-encoding.md)）

**步骤 3：循环经过 32 个 Transformer Block**

每个 Block 内部（以 Pre-LN 为例）：

```
x: [T, D]
  │
  ├─ RMSNorm → x_norm: [T, D]
  │
  ├─ Q = W_Q · x_norm  → [T, D]  → reshape → [T, H, D_head] = [512, 32, 128]
  ├─ K = W_K · x_norm  → [T, D]  → reshape → [T, H, D_head]
  ├─ V = W_V · x_norm  → [T, D]  → reshape → [T, H, D_head]
  │
  ├─ (RoPE 旋转 Q 和 K，shape 不变)
  │
  ├─ causal mask: 位置 t 只能看到 0..t，上三角置 -inf
  │
  ├─ attn_weights = softmax(Q·Kᵀ / √D_head)  → [T, H, T] = [512, 32, 512]
  │
  ├─ attn_output = attn_weights · V  → [T, H, D_head] = [512, 32, 128]
  │
  ├─ reshape + W_O  → [T, D] = [512, 4096]
  │
  ├─ 残差: x = x + attn_output  → [T, D]
  │
  ├─ RMSNorm → x_norm: [T, D]
  │
  ├─ FFN: [T, D] → [T, 14336] → SwiGLU → [T, 14336] → [T, D]
  │
  └─ 残差: x = x + ffn_output  → [T, D]
```

**步骤 4：最终 RMSNorm**

```
x: [T, D]  →  RMSNorm  →  x: [T, D]
```

**步骤 5：Language Model Head**

```
x: [T, D]  →  W_lm_head: [D, V]  →  logits: [T, V]
                                              [512, 128256]
```

最后对每个位置的 logits 做 softmax，得到词表上的概率分布。训练时用第 t 个位置的 logits 预测第 t+1 个 token（next token prediction）。

**Shape 变化总结**：

```
[T]  →  [T, D]  →  (×32 Block)  →  [T, D]  →  [T, V]
token ID   embedding                 hidden     logits
```

整个前向传播中，主 hidden state 的 shape 始终是 `[T, D]`，只在 attention 内部临时变成 `[T, H, D_head]`，在 FFN 内部临时变成 `[T, 4D]`，最后投影到词表维度 `[T, V]`。

---

## 和 wiki 内其他概念的关联

- [Attention 直觉](./attention-intuition.md)：QKV 机制、multi-head attention、self/cross/local attention 的详细计算
- [位置编码](./positional-encoding.md)：Transformer 如何感知 token 顺序——正弦 PE、可学习 PE、RoPE、ALiBi
- [Attention 优化技术](./attention-optimization.md)：FlashAttention（显存优化）、GQA/MQA（推理 KV cache 优化）、RoPE（长度外推）
- [分词与词表](./tokenization.md)：token 从文本到 ID 的过程，词表大小对 embedding 层参数量的影响
- [Instruction Tuning](./instruction-tuning.md)：预训练完成后，如何通过 SFT 让模型从"续写"变成"听指令"
- [RLHF](./rlhf.md)：SFT 之后的偏好对齐训练，让模型行为和人类意图对齐
- [Word Embedding](./word-embedding.md)：token embedding 的前身，以及神经网络语言模型的历史脉络
- [参数调度](./parameter-scheduling.md)：Transformer 训练时学习率 warmup 和 decay 的设计，和 Pre-LN 的稳定性直接相关

---

## 值得看的材料

**原始论文**

- Vaswani et al., 2017: *Attention Is All You Need*（Transformer 原始论文，arxiv: 1706.03762）——架构设计的出发点，机器翻译任务，Encoder-Decoder 结构

**直觉讲解**

- Andrej Karpathy: *Let's build GPT: from scratch, in code, spelled out*（YouTube）——从零实现 Decoder-only GPT，逐行讲解每个组件，是目前最好的动手入门材料
- Jay Alammar: *The Illustrated Transformer*（博客）——可视化 attention 的计算过程，适合理解 shape 变化

**架构演化**

- BERT（Devlin et al., 2018，arxiv: 1810.04805）：Encoder-only，MLM 预训练，理解任务的里程碑
- GPT-2（Radford et al., 2019）：Decoder-only，证明大规模无监督预训练的潜力
- T5（Raffel et al., 2019，arxiv: 1910.10683）：Encoder-Decoder，把所有 NLP 任务统一成 text-to-text
- LLaMA（Touvron et al., 2023，arxiv: 2302.13971）：现代 Decoder-only 的工程实践——Pre-LN + RMSNorm + RoPE + SwiGLU + GQA 的组合
