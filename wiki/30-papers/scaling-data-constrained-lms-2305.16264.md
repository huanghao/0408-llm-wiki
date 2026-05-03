# Scaling Data-Constrained Language Models (2305.16264)

一句话总结：当可用数据有限时，重复训练数据最多 4 个 epoch 几乎无损，超过后收益递减；数据受限场景下的最优计算分配应多加 epoch、少加参数，与 Chinchilla 的建议相反。

## 基本信息

- 论文：Scaling Data-Constrained Language Models
- 作者：Niklas Muennighoff, Alexander M. Rush, Boaz Barak, Teven Le Scao, Aleksandra Piktus, Nouamane Tazi, Sampo Pyysalo, Thomas Wolf, Colin Raffel
- 机构：Hugging Face, Harvard University, University of Turku
- arXiv：[2305.16264](https://arxiv.org/abs/2305.16264)
- 会议：NeurIPS 2023
- 本地原文：`raw/inbox/2305.16264.pdf`
- 代码 / 模型：https://github.com/huggingface/datablations

## 核心问题

[Chinchilla scaling laws](./gopher-2112.11446.md)（Hoffmann et al., 2022）给出了固定计算预算下参数量和训练 token 数的最优比例——大致是 1:20（参数数量 × 20 ≈ token 数）。但 Chinchilla 的推导假设训练数据是无限的，每个 token 只用一次。

现实中，许多语言（低资源语言、专业领域）的高质量文本数据很快会被耗尽。如果 Chinchilla 要求的 token 数超过了可用数据量，该怎么办？

**核心问题**：在数据受限（data-constrained）场景下：
1. 重复数据（多 epoch）带来多少收益？（Return 问题）
2. 在重复数据和扩大参数之间，计算如何最优分配？（Allocation 问题）

## 方法 / 核心机制

### 实验规模

超过 400 次训练运行，模型参数量从 10M 到 9B，训练 token 数从 1M 到 900B，最多训练 1500 个 epoch，使用 GPT-2 架构和 C4 数据集。

### 数据受限 Scaling Law 公式

论文对 Chinchilla 的参数化公式做扩展，将重复数据的"有效 token 数"和"有效参数数"用指数衰减函数建模：

**有效 token 数 D'**：
$$D' = U_D + U_D R_D^* (1 - e^{-R_D / R_D^*})$$

其中 $U_D$ 是唯一 token 数，$R_D$ 是重复次数（epoch - 1），$R_D^*$ 是拟合常数（约 15，对应约 16 个 epoch 的"半衰期"）。直觉：每次重复，数据对模型的边际价值按指数递减。$R_D^* \approx 15$ 意味着在第 16 次重复时，额外 token 的价值已降至新 token 的 $1/e \approx 37\%$。

**"收益"的度量**：这里的收益用**验证 loss（validation loss）** 衡量，即模型在未见过的保留测试集上的交叉熵损失。loss 越低，模型的语言预测能力越强。重复数据能不能带来好处，就看每次额外训练是否能进一步降低 validation loss。论文同时也用 19 个下游 NLP 任务（如问答、常识推理等）的准确率来验证 loss 趋势是否在真实任务上一致。

类似地定义有效参数数 N'，用于描述"超量参数"（模型容量超出数据约束的最优值时）带来的收益递减。

将 $D'$ 和 $N'$ 代入 Chinchilla 公式 $L(N, D) = A/N^\alpha + B/D^\beta + E$，即可在重复数据场景下预测 loss。

### 三类实验设计

- **Fixed Unique Data**（§5）：固定 $D_C$，变化参数和 epoch——测 Allocation（如何分配计算）。
- **Fixed FLOPs**（§6）：固定总计算量，变化 $D_C$ 和重复次数——测 Return（重复数据值多少钱）。
- **Parametric Fit**：用 §3.1 的公式拟合所有实验，验证外推能力。

### 补充策略（§7）

除重复外，还研究了两类应对数据稀缺的策略：
- **代码数据增补**：用 Python 代码（The Stack）填补自然语言缺口，1:1 混合可使有效 token 数翻倍，自然语言任务性能无损。（The Stack 是 HuggingFace 发布的大规模开源代码数据集，包含多种编程语言的公开代码。论文把 Python 代码和 C4 英语网页文本 1:1 混合：假设原本只有 50B tokens 的自然语言数据，混入等量代码后总数据量变成 100B tokens，训练时等价于有了 2× 更多可用数据，不用重复，也能跑更多"新" token。）
- **过滤策略**：困惑度过滤对嘈杂数据有效，去重对干净数据（如 C4）帮助有限；在数据受限场景中，过度过滤反而减少可用数据，得不偿失。（举例：假设你有 100B tokens 的原始网页数据。过滤掉低质量文本后剩 50B tokens——数据更干净，但量少了一半，在数据受限场景下你只能重复更多 epoch 来弥补，而重复本身又有代价。论文发现：对 C4 这种本来就相对干净的数据，用困惑度过滤能提升质量，但去重（删除高度相似的文档）几乎没有帮助；而对更嘈杂的原始数据，困惑度过滤效果明显。结论：在数据受限场景下，不能无脑过滤，要权衡质量提升和数量损失。）

## 关键结果 / 数据

### Return：重复数据值多少钱？

**先澄清一个容易混淆的点**：这里比较的不是"重复 4 次比重复 1 次好不好"，而是：

> 给定固定的计算预算（FLOPs），用 **1 份数据重复 4 次** vs 用 **4 份独立新数据训 1 次**，哪个更好？

直觉上当然是 4 份新数据更好。论文的发现是：当数据实在不够、只能重复时，重复 ≤4 次的 loss 和理想情况（全是新数据）几乎一样好——差距在 0.5% 以内，可以忽略不计。这叫"几乎无损"。

换句话说：**"几乎无损"是好事**，意思是"重复少量次数可以接受，不会明显损害模型质量"。不是说"训 4 次才学明白"，而是说"被迫重复不超过 4 次，损失微乎其微，可以放心这么做"。

| 重复次数 | 验证 loss 影响 | 解读 |
|---:|---|---|
| ≤4 epoch | 几乎无损（loss 差异可忽略） | 和同等计算下用新数据效果相当，可接受 |
| ~16 epoch | 收益减半（$R_D^*$ 半衰期） | 明显不如新数据，但仍有帮助 |
| >16 epoch | 收益趋近于零 | 继续重复基本没有意义 |

具体数字：8.7B 参数模型训练 4 epoch（$D_C = 44$B unique tokens）的验证 loss 比 1 epoch（$D_C = 178$B unique tokens）仅高 0.5%。即：把 44B tokens 重复 4 次，效果几乎等同于有 178B 不重复新 tokens 的理想情形。

### Allocation：数据受限下如何分配计算？

Chinchilla 建议参数和 token 等比例扩大（1:20 比例）。在数据受限场景下，论文发现：

- 应将更多计算分配到 **更多 epoch** 而不是更大模型。
- 以同等 FLOPs（$9.3 \times 10^{21}$）、25B unique tokens 的数据约束为例，数据受限最优模型比 Chinchilla 建议的模型少 27% 参数，但在 loss 和下游任务上表现更好。
- 图 1 右图：数据受限的 efficient frontier 偏向更小的模型和更多的 epoch，与 Chinchilla frontier 明显不同。

### 代码增补效果

- 加入 Python 代码（占 50% token）：自然语言任务无性能下降，某些推理任务（WebNLG、bAbI）明显提升。（WebNLG 是"给定结构化知识三元组，生成描述句子"的任务；bAbI 是 Facebook 发布的系列推理问答任务，测试因果推断、路径追踪等逻辑能力。这两类任务在加入代码数据后提升，原因推测是代码训练增强了模型对结构化序列和逻辑推断的处理能力。）
- 有效计算等价于 2× 更多唯一自然语言数据。（这句话的意思是：加入 50% 代码数据后，模型在 19 个自然语言任务上的总体表现，相当于把自然语言数据量翻倍（但不加代码）的效果。换句话说，代码数据并没有"稀释"自然语言能力，反而相当于免费获得了一倍多的有效训练信号。这是一个"好出乎意料"的结论，说明代码和自然语言的表示学习在底层有相当程度的正迁移。）

### 下游任务验证

19 个 NLP 任务（0-shot 和 5-shot）的结果与 loss 趋势一致：≤4 epoch 重复的性能差异不显著，超过后开始下降。（为什么要单独做下游任务验证？因为 validation loss 降低不一定意味着实际任务性能提升——loss 只是语言建模的代理指标。论文在 19 个标准 NLP 任务（涵盖问答、推理、常识等）上验证：loss 的"≤4 epoch 无损"结论在实际任务准确率上同样成立——重复 1–4 次，各任务分数基本不变；重复过多后，分数开始下滑。这给了"≤4 epoch 可接受"更强的实践背书，不只是理论上的 loss 数字。）

## 局限性

- 所有实验使用 GPT-2 架构和 C4 数据集（英语网页文本），不同数据分布（代码、数学、多语言）的 $R_D^*$ 值可能不同。
- 模型最大 9B 参数，不直接覆盖 70B+ 规模的外推。
- 公式假设 loss 单调递减，但实际上极端重复（>44 epoch）会导致 loss 上升（过拟合），公式低估这种情形。
- 去重、过滤的策略实验仅用 19 个自然语言任务评估，代码和数学任务的结论可能不同。
- $R_D^* \approx 15$ 是在 C4 上拟合的，这一常数对其他数据集是否成立需另行验证。

## 现状与影响

一句话定性：**本文是数据受限 scaling 的奠基性定量工作，"4 epoch 几乎无损、超过后快速递减"的经验法则已成为预训练数据规划的基础参考；具体数字会随数据类型变化，但框架仍在沿用。**

截至 2026，本文的直接影响体现在：

- **预训练工程实践**：业界普遍接受"≤4 epoch 重复可接受"，超出后需评估数据增补或增加参数以外的策略。LLaMA-3 等模型的训练配方都考虑了 epoch 数和数据量的平衡。
- **被引证的具体结论**：多语言 / 低资源语言预训练项目（如 FinGPT）直接引用本文的数据约束分析。
- **与 Chinchilla 的修正关系**：本文并非推翻 Chinchilla，而是将其延伸到 Chinchilla 未覆盖的重复数据场景。两者结合提供了更完整的 allocation 指南。
- **局限**：$R_D^*$ 的具体值对不同数据分布有差异；后续论文（如 [Data Mixing Laws](./data-mixing-laws-2403.16952.md)）在此基础上进一步研究了数据配比问题。

## 和 wiki 内其他概念的关联

- [Data Mixing Laws](./data-mixing-laws-2403.16952.md)：本文聚焦"同一数据重复多少次"，Data Mixing Laws 聚焦"不同来源数据如何配比"；两者都是数据受限预训练的定量工具。
- [Gopher](./gopher-2112.11446.md)：Gopher 使用 Chinchilla scaling law 框架（Hoffmann et al. [42]），本文直接扩展并修正了该框架在 epoch > 1 时的预测。
- [DCLM](./dclm-2406.11794.md)：DCLM 研究数据过滤策略对模型质量的影响；本文研究重复策略，两者互补。
- [FineWeb](../20-concepts/fineweb.md)：大规模网页数据集的构建也要面对过滤后有效数据量减少的问题，本文提供了决定是否重复 vs 扩充代码数据的量化框架。
- [Phi-1](./phi-1-2306.11644.md) / [Phi-2/Phi-3](./phi-2-phi-3.md)：phi 系列走"高质量小数据"路线，也受数据约束；本文的 epoch 建议与 phi 系列在数据复用上的选择有联系。
- [Domain-Specific Pipeline](../20-concepts/domain-specific-pipeline-code-math.md)：代码数据增补策略（§7）与代码预训练 pipeline 直接相关。

## 值得看的部分 / 相关资料

- **Figure 1**：直观展示 return（左）和 allocation（右）的核心结论，是全文最重要的一张图。
- **Section 3.1**：数据受限 scaling law 的参数化公式推导，$D'$ 和 $N'$ 的指数衰减建模。
- **Figure 3**：100M unique tokens 下，IsoLoss contours 展示数据受限 efficient frontier 与 Chinchilla frontier 的偏差。
- **Figure 4**：Fixed FLOPs 实验，直观展示各个 epoch 数下 validation loss 曲线如何分叉。
- **Section 7 / Figure 6**：代码增补和过滤策略的下游任务比较，给出实操建议。
- **Appendix A**：$D'$ 和 $N'$ 公式推导细节。
- 相关工作：Hoffmann et al. 2022（Chinchilla，[42]），Hernandez et al. 2022（重复数据 scaling，[40]），Komatsuzaki 2019（"one epoch is all you need" 的早期论点，[51]）。

Sources:
- [Scaling Data-Constrained Language Models (arxiv.org)](https://arxiv.org/abs/2305.16264)
