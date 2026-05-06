# Self-Alignment with Instruction Backtranslation

一句话总结：用"反向翻译"思路把海量无标注网页文本自动配上指令，再让模型自己筛选高质量样本迭代训练，无需 GPT-4 蒸馏即在 Alpaca 排行榜上超越所有非蒸馏 LLaMA 模型。

## 基本信息

- 论文：Self-Alignment with Instruction Backtranslation
- 作者：Xian Li, Ping Yu, Chunting Zhou, Timo Schick, Omer Levy, Luke Zettlemoyer, Jason Weston, Mike Lewis
- 机构：Meta
- 发表：ICLR 2024
- arXiv：2308.06259（2023 年 8 月）
- 本地 PDF：`raw/inbox/2308.06259.pdf`

## 核心问题

**高质量 instruction tuning 数据从哪里来，且不依赖 GPT-4？**

此前路线的两端都有问题：
- **人工标注**（OpenAssistant、LIMA）：质量高但难以扩展，覆盖任务类型有限
- **GPT-4 蒸馏**（Alpaca-GPT4、WizardLLM、Vicuna）：受限于教师模型能力，且依赖强外部模型，不满足"自对齐"要求

本文的核心洞察：互联网上存在大量高质量人类写作文本（技术文章、教程、问答等），这些文本本身就是"优质的输出"——缺的只是与之对应的指令。如果能**自动给这些文本配上指令**，就能无限扩展指令数据，且输出质量有保证（人类写的）。这正是 MT 领域"回译（backtranslation）"思路的迁移：在 MT 中，用目标语言句子反向生成源语言句子来扩充双语数据。

## 方法：Self-Augmentation + Self-Curation

整体流程两步迭代（Figure 1）：

### 第 0 步：初始化

- **种子数据**：3,200 条来自 OpenAssistant 的人工标注 `(instruction, output)` 对（排名第 0 即最高质量的英文响应）
- **无标注数据**：Clueweb 语料库的英文部分，抽取 502k 段落作为候选输出 $\{y_i\}$

### 第 1 步：Self-Augmentation（生成指令）

用种子数据在**反向方向**微调一个 backward 模型 $M_{yx}$，即给定输出预测对应的指令：$p(x|y)$。

> **Backward 模型是怎么训练的？**
> 就是一个普通的 LLaMA（7B/33B/65B），用**种子数据**微调，但输入和输出方向反过来——种子数据里本来是 `(instruction → output)`，训练 backward 模型时改成 `(output → instruction)`。训练数据就是那 3,200 条 OpenAssistant 人工标注对，规模很小。微调后这个模型的能力是"看一段文本，猜它可能是在回答什么问题/指令"。
>
> 举例：给它一段网页文字"蒸鸡蛋的做法：先打散鸡蛋，加入 1.5 倍温水……"，它会生成指令 "如何蒸出嫩滑的鸡蛋羹？"。这就构成了一条训练对 `(指令, 网页文本)` 供后续用。

用 $M_{yx}$ 对每个网页段落 $y_i$ 生成候选指令 $\hat{x}_i$，构成候选增强数据集 $\mathcal{A} = \{(\hat{x}_i, y_i)\}$（502k 对）。

**为什么输出用人类写的网页文本而不是模型生成？** 人类写的文本在风格多样性、长尾任务覆盖上更丰富，且不受模型幻觉污染。

### 第 2 步：Self-Curation（质量筛选）

直接用全部 502k 增强数据训练效果差（Figure 2 实线最低），必须筛选。

- 从种子数据出发训练一个前向指令跟随模型 $M_0$
- 用 $M_0$ 对每个候选对 $(\hat{x}_i, y_i)$ 打 5 分制质量分 $a_i$（用专门的评分 prompt）
- 取 $a_i \geq k$（threshold $k=4$ 或 $k=5$）的子集 $\mathcal{A}_k^{(1)}$ 作为下一轮训练数据

**迭代**：用筛选后数据 + 种子数据联合微调得到 $M_1$，再用 $M_1$ 重新打分，得到 $\mathcal{A}_k^{(2)}$，最终模型 $M_2$ 即为 **Humpback**。共 2 次迭代。

**系统 prompt 标记**：种子数据用 $S_a$ = "Answer in the style of an AI Assistant."，增强数据用 $S_w$ = "Answer with knowledge from web search."，区分两类数据来源。

### 数据规模

| 数据集 | 样本数 |
|--------|--------|
| 种子数据 | 3,200 |
| 增强数据全集 $\mathcal{A}$ | 502,133 |
| 筛选后 $\mathcal{A}_5^{(2)}$（score ≥ 5）| 41,821 |
| 筛选后 $\mathcal{A}_4^{(2)}$（score ≥ 4）| 195,043 |

## 关键结果

### AlpacaEval（win rate vs. text-davinci-003，GPT-4 判断）

| 模型 | 类别 | 人工标注数 | 总样本数 | Win Rate |
|------|------|-----------|---------|----------|
| **Humpback 33B** | **非蒸馏** | **3k** | **45k** | **79.84%** |
| OASST RLHF 33B | 非蒸馏 | 161k | 161k | 66.52% |
| Guanaco 33B | 非蒸馏 | 9k | 9k | 65.96% |
| **Humpback 65B** | **非蒸馏** | **3k** | **45k** | **83.71%** |
| LIMA 65B | 非蒸馏 | 1k | 1k | 62.70% |
| Vicuna 33B | 蒸馏 | 140k | 140k | 88.99% |
| GPT-4 | 专有 | — | — | 95.28% |

Humpback 是**非蒸馏类别最高分**，用 3k 人工标注超越用 161k 标注的 OASST。

**用 LLaMA 2 70B 做基础模型时**（Humpback 70B）win rate 进一步提升到 87.94%，仅低于 Vicuna 33B（蒸馏模型）。

### 数据扩展效率（Scaling coefficient α，Table 2）

用 $w = \alpha \log N + C$ 拟合"训练样本数 vs 胜率"曲线，$\alpha$ 越大代表数据越高效：

| 数据来源 | α |
|----------|---|
| **Humpback（本文）** | **6.95** |
| WizardLLM（蒸馏） | 5.69 |
| Alpaca-GPT4（蒸馏） | 5.40 |
| LIMA（人工标注）| 2.86 |
| FLAN v2（NLP 任务格式化）| 0.22 |

Humpback 数据的扩展效率在所有方法中**最高**，且高于蒸馏数据。

### 关键消融（Figure 2 & 5）

- **不做 self-curation 的增强数据**：随数据量增加性能几乎不变甚至下降——说明单纯扩大数量无效，**质量筛选是核心**
- **只用增强数据（不加种子）**：效果差于联合训练——种子数据与增强数据互补，种子覆盖 assistant 风格，增强数据覆盖长尾任务
- **联合训练（种子 + 筛选增强）**：大幅优于单独使用任一来源

### 常识推理与 MMLU（Table 4）

Humpback 相比 LLaMA 基础模型在 Arc-C（+18.2pp）、OBQA（+5.4pp）、MMLU（+4.2pp）均有显著提升，说明自我增强的数据不仅提升对话能力，也改善了通用推理能力。

## 局限性

- **仅限文本生成任务**：指令由模型生成，指令类型受限于 backward 模型的覆盖范围；数学、代码等需要精确验证的任务无法通过这一方式可靠扩展（网页文本里相应内容也少）
- **Self-curation 依赖种子模型的判断力**：质量打分用的是同一个模型，如果种子模型本身对某类任务判断力弱，筛选结果也会偏差
- **输出是人类写的网页文本**：输出风格偏向文章/说明型，不一定符合 AI assistant 的对话风格；用系统 prompt 区分是工程补丁而非根本解决
- **实验基于 LLaMA 1（7B/33B/65B）**：2023 年 8 月的结果，基础模型较弱；用 LLaMA 2 70B 的实验仅补充验证，未系统展开
- **无标注数据质量**：Clueweb 语料质量参差不齐，自动筛选可能放入低质量或有害内容（论文未做安全性系统评估）

## 现状与影响

一句话定性：**Instruction Backtranslation 是"用无标注数据自举指令数据"路线的奠基性工作，证明了 self-alignment 在 instruction tuning 场景的可行性，但随着 GPT-4 蒸馏数据门槛降低和更强基础模型涌现，其"超越蒸馏"的优势已不再成立，核心思路（给无标注输出配指令）被后续合成数据流水线广泛采纳。**

> **"GPT-4 蒸馏门槛降低"是什么意思？**
> 2023 年初，用 GPT-4 蒸馏数据还有两道门槛：(1) API 费用高（生成 5 万条数据要花数百美元）；(2) OpenAI 使用条款禁止用 GPT-4 输出训练竞争模型，风险不明朗。到 2024 年，这两个门槛都大幅降低：API 价格下降超过 10 倍，模型能力更强（单次调用生成更多高质量内容），条款限制也更宽松；同时 Claude、Gemini 等替代来源出现。结果是：随便调几千次 API 就能生成高质量的指令数据集，蒸馏路线变得"人人都能做"，Instruction Backtranslation 靠避开蒸馏换来的竞争优势随之消失。

截至 2026 年的影响：

- **方法论影响**：给"人类写的输出"自动配指令的思路被多个后续工作借鉴，包括用无标注代码库或文档生成指令等变体。MagPie（2024）从另一角度解决同一问题：让 aligned LLM 自动生成指令而非给输出配指令
- **Self-curation 的启发**：用模型自评打分来筛选数据的做法（无需外部强模型判断）后来被 Deita、AlpaGasus 等工作系统化。本文是该思路的早期实证之一
- **实际限制已显现**：2024 年后 Llama 3/Qwen2 等更强基础模型配合简单的蒸馏数据已能达到远超 Humpback 的水平；Backtranslation 生成的指令质量受限于 backward 模型，在复杂推理任务上表现平庸
- **Humpback 模型本身**：未被广泛部署，影响主要在研究层面

## 和 wiki 内其他概念的关联

- [Self-Instruct](./self-instruct-2212.10560.md)：同样是"模型自生成数据"路线，但方向相反——Self-Instruct 让模型生成指令和输出，Instruction Backtranslation 让模型给现有输出生成指令；两者都证明了 self-alignment 的可行性
- [LIMA](./lima-2305.11206.md)：LIMA 展示了少量高质量人工标注（1k 条）就足以激发对齐能力；本文提出了不同结论：高质量数据可以靠自动筛选从海量无标注数据中挖掘，且持续增加高质量数据有增益
- [Deita](./deita-2312.15685.md)：同样用模型自评来筛选指令数据，但 Deita 在复杂度、质量、多样性三维打分上更系统；本文是用模型自评筛选的早期实证
- [MagPie](./magpie-2406.08464.md)：同为"不依赖人工标注的指令数据生成"，但路线不同：MagPie 利用 aligned LLM 的 chat template 直接触发生成用户指令，不需要 backward 模型
- [AlpaGasus](./alpagasus-2307.08701.md)：用 ChatGPT 评分筛选 Alpaca 数据的同期工作；本文用模型自评替代外部强模型评分，是更纯粹的 self-alignment
- [Instruction Tuning](../20-concepts/instruction-tuning.md)：本文是指令微调数据来源演化路线中"自监督/自对齐"分支的代表

## 值得看的部分 / 相关资料

- **Section 2（Method）**：Figure 1 完整展示了两步迭代流程，Section 2.2/2.3 分别详述 self-augmentation 和 self-curation
- **Section 3.3（Scaling Analysis）**：Figure 2/3 和 Table 2 是理解"数据质量 vs 数量"和"扩展系数 α"最直观的部分
- **Section 3.4（Model Quality）**：Table 3 的 AlpacaEval 对比和 Figure 4 的人类偏好评估
- **Section 3.5（Ablations）**：Figure 5 说明种子数据和增强数据的互补性，系统 prompt 的消融（Table 5）
- 相关工作：
  - Sennrich et al., 2015: *Improving Neural Machine Translation Models with Monolingual Data*——MT 回译的原始来源，本文思路的直接类比
  - Köksal et al., 2023: *Longform: Optimizing instruction tuning for long text generation*——并行工作，同样给人类文本配指令但用蒸馏模型

## 参考

- Li et al., 2023/2024: *Self-Alignment with Instruction Backtranslation*（arXiv:2308.06259，ICLR 2024）
- Sennrich et al., 2015: *Improving Neural Machine Translation Models with Monolingual Data*（MT backtranslation 原始论文）
- Zhou et al., 2023: *LIMA: Less Is More for Alignment*（LIMA，NeurIPS 2023）
