# 自回归生成（Autoregressive Generation）

一句话总结：LLM 推理时每次只生成一个 token，把它拼回序列后再做一次前向传播，如此循环直到遇到终止条件——这就是自回归生成，也是为什么推理比训练慢得多的根本原因。

---

## 核心问题：训练 vs. 推理的不对称

### 训练时：Teacher Forcing

训练时，模型看到的是完整的目标序列。以"今天天气很好"为例，模型在预测每个位置的 token 时，输入的都是**真实的前缀**（ground truth），而不是模型自己上一步的预测结果。

```
输入序列：[BOS] 今天 天气 很
目标 token：       今天 天气 很 好
```

这个做法叫 **teacher forcing**：不管模型上一步预测对没对，都强制"喂"正确答案作为下一步的输入。

**好处**：整个序列的所有位置可以**并行计算**。一次前向传播，就能得到序列中每个位置的 loss，梯度反传一次搞定。GPU 的并行能力被充分利用。

### 推理时：自回归循环

推理时没有"正确答案"可以喂——模型必须用自己的输出作为下一步的输入。

```
第 1 步：输入 [BOS]           → 预测 "今天"
第 2 步：输入 [BOS, 今天]     → 预测 "天气"
第 3 步：输入 [BOS, 今天, 天气] → 预测 "很"
第 4 步：输入 [BOS, 今天, 天气, 很] → 预测 "好"
第 5 步：输入 [BOS, 今天, 天气, 很, 好] → 预测 [EOS]
```

每一步都依赖上一步的输出，**无法并行**。生成一个长度为 L 的序列，需要 L 次串行的前向传播。

### 为什么推理比训练慢

这个不对称导致了推理效率的天然劣势：

- **训练**：一次前向传播处理整个序列（长度 L），计算量 O(L)
- **推理**：L 次前向传播，每次处理越来越长的序列，总计算量 O(L²)

实践中，生成速度通常以 **tokens/s** 衡量，对同等规模的模型，推理的 tokens/s 远低于训练时的 tokens/s——这是架构决定的，不是工程问题。

---

## 自回归生成的循环过程

每一步的核心操作：

1. **前向传播**：把当前序列（所有已生成的 token + 原始 prompt）送入模型，得到最后一个位置的 logits（一个大小为词表 V 的向量）
2. **采样**：对 logits 应用某种策略（见下节），从词表中选出下一个 token ID
3. **Append**：把新 token 拼到序列末尾
4. **检查终止条件**：如果生成了 EOS token 或达到 max_length，停止；否则回到第 1 步

```
初始序列: [prompt tokens]
         ↓
      前向传播
         ↓
   logits [V]  ←── 只看最后一个位置的输出
         ↓
      采样策略
         ↓
   新 token id
         ↓
  append 到序列
         ↓
  检查终止条件 → 结束
         ↓ 否
      前向传播（序列变长了 1）
         ↓
       ...
```

注意：每次前向传播，模型处理的序列长度都在增加。第 k 步时，序列长度是 `prompt_length + k`。

---

## Sampling 策略

前向传播输出的 logits 是原始分数，需要转换成概率分布（经过 softmax），再从中选 token。不同的选法对应不同的 sampling 策略，直接影响生成的质量和风格。

### Greedy Decoding（贪心解码）

每步选概率最高的 token：

```
next_token = argmax(softmax(logits))
```

**优点**：确定性，可复现，速度快。  
**问题**：容易陷入重复循环（"好的好的好的……"），缺乏多样性。对于开放式生成任务效果差。  
**适用场景**：代码补全、格式严格的结构化输出、需要可复现结果的场合。

### Beam Search（束搜索）

维护 B 条候选序列（beam），每步对每条候选序列扩展所有可能的 token，保留总分（累积 log 概率）最高的 B 条。

```
B=3 时：
步骤 1：保留概率最高的 3 个 token → 3 条候选
步骤 2：每条候选扩展 V 个 token → 3V 条，保留最好的 3 条
...
最终返回 B 条完整序列，取总分最高的
```

**优点**：比 greedy 更全局，能找到概率更高的序列。  
**问题**：计算量是 greedy 的 B 倍；生成结果仍然偏保守，多样性低；对于开放式生成，beam search 的输出往往比 sampling 更无聊。  
**适用场景**：机器翻译、摘要等有明确"最优答案"的任务；不适合对话和创意生成。

### Temperature Sampling

在 softmax 之前，用温度参数 T 缩放 logits：

```
p_i = softmax(logits / T)
```

- **T = 1.0**：原始分布，不改变
- **T < 1.0（低温）**：分布变尖，高概率 token 更突出，接近 greedy
- **T > 1.0（高温）**：分布变平，低概率 token 也有机会被选到，更随机

**直觉**：T 控制模型有多"大胆"。T=0 退化成 greedy；T→∞ 变成均匀随机采样。

**适用场景**：几乎所有对话和创意生成任务都会设置 T，通常在 0.7–1.2 之间。T 不单独使用，通常和 top-p 或 top-k 组合。

### Top-k Sampling

每步只从概率最高的 k 个 token 中采样，其余 token 概率清零：

```
保留 logits 中最大的 k 个值，其余设为 -inf
再做 softmax + 采样
```

**问题**：k 是固定的，但不同步骤的概率分布形状差异很大。有时候前 k 个 token 的概率已经覆盖了 99%，有时候第 k+1 个 token 也很合理——固定 k 无法适应这种变化。

**适用场景**：k=50 是常见默认值，但通常被 top-p 取代或组合使用。

### Top-p Sampling（Nucleus Sampling）

不固定候选数量，而是固定**累积概率阈值** p：把 token 按概率从高到低排列，累积概率达到 p 后截断，只从这个"核"（nucleus）中采样。

```
p=0.9 时：
token A: 0.5  → 累积 0.5
token B: 0.25 → 累积 0.75
token C: 0.1  → 累积 0.85
token D: 0.07 → 累积 0.92 ← 超过 0.9，截断
→ 只从 {A, B, C} 中采样
```

**优点**：候选集大小随概率分布自动调整。分布尖锐时（模型很确定），候选集小；分布平坦时（模型不确定），候选集大。比 top-k 更自适应。

**适用场景**：对话、写作、代码生成的主流选择。p=0.9 或 p=0.95 是常见默认值。

### 策略组合

实践中通常组合使用：

```
temperature=0.8, top_p=0.9
```

先用 temperature 调整分布形状，再用 top-p 截断长尾。这是大多数 LLM 推理框架（vLLM、llama.cpp、HuggingFace Transformers）的默认配置思路。

---

## 为什么自回归生成天然需要 KV Cache

每次前向传播时，Transformer 的 attention 层需要计算当前 token 对**所有历史 token** 的注意力。这意味着历史 token 的 Key 和 Value 矩阵每步都要重新计算——而它们其实没有变化。

第 k 步时，序列是 `[t_1, t_2, ..., t_{k-1}, t_k]`：
- `t_1` 到 `t_{k-1}` 的 K/V 在第 k-1 步就算过了
- 只有 `t_k` 的 K/V 是新的

**KV cache** 的思路：把每步计算出的 K/V 缓存起来，下一步直接复用，只计算新 token 的 K/V。这把每步的计算量从 O(k) 降到接近 O(1)（只算新 token），代价是显存里要存下所有历史 token 的 K/V。

KV cache 是自回归生成的标配优化，几乎所有推理框架都默认开启。随着序列变长，KV cache 的显存占用会线性增长，这是长序列推理的主要瓶颈之一。

详细原理见 [Attention 优化技术](./attention-optimization.md)。

---

## 生成终止条件

自回归循环需要明确的停止信号，否则会无限生成。

### EOS Token

词表中有一个特殊的 **EOS（End of Sequence）** token（不同模型 ID 不同，如 LLaMA 3 中是 `<|eot_id|>`）。训练时，所有样本的结尾都加了 EOS，模型学会了"什么时候该停"。推理时，一旦采样到 EOS，生成立即终止。

### max_length / max_new_tokens

硬性上限，防止模型陷入循环或生成过长输出。推理框架通常区分：
- `max_length`：包含 prompt 在内的总序列长度上限
- `max_new_tokens`：只限制新生成的 token 数量

### 自定义停止序列（Stop Sequences）

很多应用场景需要在特定字符串处停止，比如代码生成时遇到 `\n\n` 或 `</code>`。推理框架支持传入 stop sequences，匹配到即停止，不等 EOS。

---

## 和 wiki 内其他概念的关联

- [分词与词表（Tokenization）](./tokenization.md)：自回归生成的输入输出单元是 token，理解 token 的粒度有助于理解为什么"生成一个中文字"可能需要多步
- [Attention 机制](./attention-intuition.md)：每步前向传播的核心计算，自回归生成中 attention 的 Q 只来自新 token，K/V 来自全部历史
- [Attention 优化技术](./attention-optimization.md)：KV cache、GQA/MQA、FlashAttention——都是针对自回归推理瓶颈的优化
- [Instruction Tuning](./instruction-tuning.md)：训练阶段用 teacher forcing，但模型最终服务的是自回归推理——两者的分布差异（exposure bias）是 SFT 数据质量重要的原因之一
- [量化（Quantization）](./quantization.md)：推理速度和显存占用是自回归生成的主要约束，量化直接影响这两个指标
- [困惑度（Perplexity）](./perplexity.md)：衡量语言模型在 token 预测上的不确定性，和自回归生成的 logits 分布直接相关
