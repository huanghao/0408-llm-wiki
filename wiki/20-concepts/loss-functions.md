# Loss Functions：任务类型与 Loss 选择

## 核心问题

面对一个训练任务，用什么 loss？为什么？

判断依据只有一个：**训练信号的形态**——你有的是"正确答案的 token 序列"、"好/坏的相对排序"、还是"一个标量奖励分数"？

---

## NLL / Cross-Entropy Loss

**适用场景**：你有**每个位置的正确 token**。

$$\mathcal{L}_{\text{NLL}} = -\sum_t \log p_\theta(y_t \mid x, y_{<t})$$

每个 token 都有一个"标准答案"，loss 就是让模型对正确 token 的概率越高越好。

**直觉**：模型说"我觉得下一个词是'猫'的概率是 0.8"，正确答案就是"猫"，loss 就是 $-\log 0.8 = 0.22$。概率越高，loss 越小。

**用 NLL 的训练**：

| 任务 | 为什么能用 NLL |
|------|--------------|
| 语言模型预训练 | 下一个 token 就是正确答案 |
| SFT（监督微调） | 标注数据提供了完整的回答 token 序列 |
| 蒸馏（hard label） | 教师模型的输出 token 作为正确答案 |
| 翻译、摘要（seq2seq） | 目标语言序列就是正确答案 |

**不能直接用 NLL 的情况**：你没有逐 token 的正确答案——比如只知道"这段回答比那段好"，或者只有一个最终分数。

---

## KL 散度（KL Divergence）

**适用场景**：你有**两个概率分布**，想让它们靠近。

$$\mathcal{L}_{\text{KL}} = \sum_t p(t) \log \frac{p(t)}{q_\theta(t)}$$

**直觉**：NLL 是"对正确答案打分"，KL 是"对整个分布打分"——不只关心正确 token，而是关心模型的整体分布和目标分布的差距。

**和 NLL 的关系**：当目标分布是 one-hot（即有明确的正确 token）时，KL 和 NLL 等价。

**用 KL 的训练**：

| 任务 | 为什么用 KL |
|------|-----------|
| 蒸馏（soft label） | 让学生模型分布逼近教师模型的软概率分布 |
| RLHF 中的 KL 惩罚 | 防止 RL 训练后的模型偏离 SFT 基础模型太远 |
| VAE | 让隐变量分布逼近先验分布 |

---

## Ranking / Preference Loss（偏好损失）

**适用场景**：你有**相对排序**——"A 比 B 好"，但没有绝对正确答案。

**Bradley-Terry 模型**（奖励模型训练用）：

$$\mathcal{L} = -\log \sigma(r_w - r_l)$$

让"好答案"的奖励分 $r_w$ 高于"差答案"的奖励分 $r_l$。

**DPO Loss**（直接偏好优化）：

$$\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\right)$$

不需要奖励模型，直接从偏好对训练语言模型。见 [RLHF](./rlhf.md)。

**用 Ranking Loss 的训练**：

| 任务 | 为什么不用 NLL |
|------|--------------|
| 奖励模型训练 | 只有"哪个答案更好"，没有正确答案的 token 序列 |
| DPO 微调 | 同上，但跳过了奖励模型 |
| RLHF 的 RM 阶段 | 人工标注是排序，不是生成 |

---

## REINFORCE / 策略梯度

**适用场景**：你有**标量奖励**，但奖励在整段输出完成后才给出，且中间的采样步骤不可微。

$$\nabla_\theta \mathcal{L} = -\mathbb{E}\left[(R - b) \cdot \nabla_\theta \log p_\theta(\tau)\right]$$

其中 $b$ 是基线（通常是同批次的平均奖励），$\tau$ 是生成的 token 序列。

**直觉**：不知道每一步"对不对"，只知道最后结果"好不好"——好就把整条路径的概率提高，差就降低。见 [REINFORCE](./reinforce.md)。

**用策略梯度的训练**：

| 任务 | 奖励来源 |
|------|---------|
| GRPO（DeepSeek-R1） | 数学/代码答案对错 |
| Quiet-STaR | 思考后预测下一词的准确率提升 |
| PPO（InstructGPT） | 奖励模型打分 |
| STaR | 推理链是否导向正确答案 |

---

## 一张决策表

```
你有什么训练信号？
│
├── 每个位置的正确 token
│     └── NLL / Cross-Entropy
│
├── 软概率分布（教师模型输出）
│     └── KL 散度（蒸馏）
│
├── 相对排序（A 比 B 好）
│     ├── 有奖励模型 → Bradley-Terry（RM 训练）+ PPO
│     └── 无奖励模型 → DPO
│
└── 标量奖励（整段输出后打分）
      ├── 有价值函数估计 → PPO
      └── 无价值函数，用组内平均基线 → GRPO / REINFORCE
```

---

## NLL 的局限性

NLL 只能用于"有正确 token"的场景。它的问题是：

1. **只奖励正确答案，不惩罚坏答案的具体坏法**：两段都"不太对"的回答，NLL 看不出哪段更糟。
2. **对开放式生成不自然**：一道题可以有多种正确写法，NLL 把所有不在标注里的 token 都当错的。
3. **无法优化不可微的目标**：比如"回答是否有害"、"用户是否满意"——这些没有 token 级别的正确答案，必须用 RL 或 preference loss。

这就是为什么 RLHF 要在 SFT（NLL）之后再加 RM + PPO：SFT 教会模型"怎么说话"，RL 教会模型"什么值得说"。

---

## 相关概念

- **Perplexity**：NLL 的指数形式 $e^{\mathcal{L}_{\text{NLL}}}$，用于评估语言模型质量，见 [Perplexity](./perplexity.md)
- **[REINFORCE](./reinforce.md)**：策略梯度的基础算法
- **[RLHF](./rlhf.md)**：PPO、GRPO、DPO 的完整对比
