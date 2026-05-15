# Training Compute-Optimal Large Language Models (Chinchilla, 2203.15556)

**关于名字**：Chinchilla（龙猫）是论文作者给这个 70B 验证模型起的名字，与 DeepMind 此前的 280B 模型 Gopher（囊鼠）并列，都是小型哺乳动物。这是 DeepMind 给内部模型起动物名的传统，名字本身没有特别技术含义，只是用来和 Gopher 区分。

一句话总结：固定计算预算下，模型参数量和训练 token 数应等比例扩大（约 1:20），当时大多数大模型严重"过大欠训"；用同等 FLOPs 训出的 70B Chinchilla 全面超越 280B Gopher。

## 基本信息

- 论文：Training Compute-Optimal Large Language Models
- 作者：Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, Tom Hennigan, Eric Noland, 等（DeepMind）
- 机构：DeepMind
- arXiv：[2203.15556](https://arxiv.org/abs/2203.15556)
- 发表：NeurIPS 2022（Tech Report, March 2022）
- 本地原文：`raw/inbox/2203.15556.pdf`

## 核心问题

2022 年前，LLM 训练的主流策略是"增大参数量"：GPT-3 175B、Gopher 280B、MT-NLG 530B，但这些模型大多只训练了约 300B tokens。

Kaplan et al. (2020) 给出了一条 scaling law：计算预算增加 10× 时，模型应增大约 5.5×，数据应增加约 1.8×。这意味着更多计算应主要分给更大的模型。

本文的核心问题是：**给定固定 FLOPs 预算，如何最优地分配模型参数量 N 和训练 token 数 D？**

结论与 Kaplan et al. 截然相反：当时几乎所有大模型都"过大欠训"——应该用更小的模型训更多的数据。

## 方法 / 核心机制

### 三种独立估计方法

论文在 400+ 次训练运行（70M 到 16B 参数，5B 到 400B tokens）上用三种独立方法求解最优 $N_{opt}(C)$ 和 $D_{opt}(C)$：

**方法一：Fix model sizes，vary training tokens**
对每个模型大小训练不同长度，从训练曲线包络线中读出每个 FLOP 预算下的最低 loss，拟合幂律 $N_{opt} \propto C^a$，$D_{opt} \propto C^b$，得到 $a \approx 0.50$，$b \approx 0.50$。

**方法二：IsoFLOP profiles**
固定 9 个 FLOP 预算（$6 \times 10^{18}$ 到 $3 \times 10^{21}$），对每个预算训练不同大小的模型，取最终 loss 最低点。在 loss vs 参数量图上找到 U 形最低点，拟合得 $a \approx 0.49$，$b \approx 0.51$。

**方法三：Parametric loss function fitting**
对所有运行结果拟合参数化公式：

$$\hat{L}(N, D) = E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $(A, B, E, \alpha, \beta)$ 用 Huber loss + L-BFGS 拟合。从公式推导出最优分配：

$$N_{opt}(C) = G \left(\frac{C}{6}\right)^a, \quad D_{opt}(C) = G^{-1} \left(\frac{C}{6}\right)^b$$

$$\text{其中} \quad G = \left(\frac{\alpha A}{\beta B}\right)^{\frac{1}{\alpha+\beta}}, \quad a = \frac{\beta}{\alpha+\beta}, \quad b = \frac{\alpha}{\alpha+\beta}$$

方法三得到 $a \approx 0.46$，$b \approx 0.54$，仍接近 1:1。

### 核心结论

三种方法收敛到相同结论：**$N$ 和 $D$ 应该以近似相同的比例随计算增长。**

实用近似规则：$D_{opt} \approx 20 \times N_{opt}$（1B 参数模型最优 token 数约 20B）。

与 Kaplan et al. 的对比：Kaplan 建议计算增加 10× 时参数增大 5.5×、数据增加 1.8×（偏重参数）；本文建议两者等比例增大（参数和数据各增加约 3.2×）。差异来源：Kaplan 使用固定 learning rate schedule，导致对数据量较少时的 loss 系统性高估，从而错误地低估了数据的重要性。

### Chinchilla 验证

用 Gopher 的 FLOPs 预算（$5.76 \times 10^{23}$）训练 70B 参数、1.4T tokens 的 Chinchilla，与 Gopher（280B，300B tokens）直接比较。

## 关键结果 / 数据

### 主要对比（同等 FLOPs）

| 模型 | 参数 | 训练 tokens |
|---|---:|---:|
| GPT-3 | 175B | 300B |
| Gopher | 280B | 300B |
| MT-NLG | 530B | 270B |
| **Chinchilla** | **70B** | **1.4T** |

Chinchilla 用同等 FLOPs，参数量只有 Gopher 的 1/4，但训练数据是 4×。

### 下游任务表现

**MMLU（5-shot）**

| 模型 | 准确率 |
|---|---:|
| GPT-3 | 43.9% |
| Gopher | 60.0% |
| **Chinchilla** | **67.6%** |
| 专家预测 2023 年 SOTA | 63.4% |
| 人类专家平均 | 89.8% |

Chinchilla 不仅超越 Gopher（+7.6%），还超过了 2023 年的专家预测上限。

**BIG-bench**：Chinchilla 62 个任务中超越 Gopher 的有 58 个，平均提升 10.7%（65.1% vs 54.4%）。

**阅读理解（RACE）**：
- RACE-h：Chinchilla 82.3% vs Gopher 71.6%（提升 +10.7%）
- RACE-m：Chinchilla 86.8% vs Gopher 75.1%（提升 +11.7%）

**常识推理**：HellaSWAG 80.8%（Gopher 79.2%）；Winogrande 74.9%（Gopher 70.1%）；BoolQ 83.7%（Gopher 79.3%）。

**TruthfulQA（0-shot）**：Chinchilla 43.6% vs Gopher 29.5%，提升 +14.1%——说明更好的预训练建模本身可以大幅提升事实准确性。

### 推理成本优势

Chinchilla 参数量是 Gopher 的 1/4，内存占用和推理 FLOPs 也约为 1/4，对下游微调和部署更友好。

## 局限性

- **单 epoch 假设**：所有训练运行均在少于一个 epoch 的数据上（不重复），论文明确指出不覆盖重复数据场景。这一局限直接催生了 [Scaling Data-Constrained Language Models](./scaling-data-constrained-lms-2305.16264.md)。
- **规模外推不确定性**：最大对照实验只有 Chinchilla（70B）和 Gopher（280B），更大规模（数百 B 以上）的外推存在不确定性；作者在 Appendix E 中指出在极大 FLOP 预算下可能存在负曲率，意味着最优参数量可能被高估。
- **计算近似**：使用 FLOPs $\approx 6ND$ 的近似，忽略了 attention 等其他计算项。
- **架构固定**：所有实验基于相同的 Transformer 架构，不同架构（MoE、稀疏模型）是否适用相同 scaling law 未验证。
- **数据质量固定**：分析假设训练数据质量均一，不覆盖数据过滤、混合比例对 scaling 的影响（见 [Data Mixing Laws](./data-mixing-laws-2403.16952.md)）。
- **Safety 不覆盖**：Chinchilla 的 bias/toxicity 评估与 Gopher 相近，更好的预训练不能解决安全问题。

## 现状与影响

一句话定性：**Chinchilla scaling law 是 LLM 训练资源分配的奠基性框架，"20× token rule"成为标准参考；但随着数据约束、数据质量差异、多 epoch 训练等问题的出现，后续工作不断扩展和修正其适用边界。**

截至 2026，Chinchilla 的影响体现在：

- **直接改变训练实践**：Llama 系列（Llama 1 用 1T tokens、Llama 2 用 2T、Llama 3 用 15T）都是在 Chinchilla 框架下刻意延长训练 token 数的产物。Llama 3 将 8B 模型训到 15T tokens（相当于约 1900×N），远超 Chinchilla 的 20× 建议，因为推理成本驱动使人们愿意"过训"小模型。
- **"推理时代"的修正**：当推理成本重要时，最优分配点会偏向更小的模型+更多数据（相比纯训练最优），Chinchilla 没有显式建模推理成本，这是后续讨论的重要补充。
- **数据受限修正**：[Scaling Data-Constrained Language Models（2305.16264）](./scaling-data-constrained-lms-2305.16264.md) 扩展了 Chinchilla 到重复数据场景。
- **数据配比修正**：[Data Mixing Laws](./data-mixing-laws-2403.16952.md) 扩展了 Chinchilla 到多域数据配比问题。
- **Chinchilla 本身的模型**已被更高效的架构和训练方案超越，不再是实际部署的参考点，但 scaling law 框架仍是标准工具。

## 和 wiki 内其他概念的关联

- [Scaling Data-Constrained Language Models](./scaling-data-constrained-lms-2305.16264.md)：Chinchilla 假设数据无限；该论文将其扩展到数据受限（重复 epoch）场景，给出数据受限下的修正分配建议。
- [Data Mixing Laws](./data-mixing-laws-2403.16952.md)：在 Chinchilla 框架内进一步分析多来源数据配比的最优化问题。
- [Gopher](./gopher-2112.11446.md)：Chinchilla 直接使用 Gopher 的 FLOPs 预算做对比验证，280B Gopher 是 Chinchilla 提出"过大欠训"问题的典型案例。
- [Llama 3 Herd of Models](./llama-3-herd-of-models.md)：Llama 系列的训练决策（token 数远超 20×N）体现了在推理成本驱动下对 Chinchilla 的主动偏离。
- [Phi-1](./phi-1-2306.11644.md) / [Phi-2/Phi-3](./phi-2-phi-3.md)：Phi 系列走"高质量教科书数据"路线，对 Chinchilla 的挑战在于：数据质量不均一时，20× 规则可能不成立。
- [幂律与 Scaling](../20-concepts/power-law-and-scaling.md)：Chinchilla 的核心数学框架——loss 和 N/D 之间的幂律关系，等比例分配规则的推导来源。
- [MFU](../20-concepts/mfu.md)：MFU 和 FLOPs 是 Chinchilla 分析的基础度量单位。
- [Perplexity](../20-concepts/perplexity.md)：Chinchilla 用 held-out loss（cross-entropy）作为主要优化目标，perplexity 是其直接变换。

## 值得看的部分 / 相关资料

- **Abstract + Figure 1**：核心结论的最简洁表述；Figure 1 右图直观展示 Chinchilla 和 Gopher 在 efficient frontier 上的位置。
- **Section 3**：三种方法的推导，Table 2 汇总三种方法的指数估计，结论高度一致。
- **Equation (2) / (4)**：参数化 loss 公式和 efficient frontier 推导，被后续大量论文引用和扩展。
- **Table 3**：对不同参数量模型的最优 FLOPs 和 token 数估算，是实践中"要训多少 tokens"的参考表。
- **Section 4.2**：Chinchilla vs Gopher vs GPT-3 vs MT-NLG 530B 的全面下游任务对比。
- **Section 5 Discussion**：作者自己对局限性（单 epoch 假设、外推不确定性）的坦诚讨论。
- 前驱工作：Kaplan et al. 2020, *Scaling Laws for Neural Language Models*（arXiv:2001.08361）——Chinchilla 直接修正了其结论。
