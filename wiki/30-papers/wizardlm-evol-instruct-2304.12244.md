# WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions（Xu et al., 2023）

一句话总结：Evol-Instruct 用 LLM 自动把简单指令迭代改写成更复杂的版本，用 250k 进化数据微调的 WizardLM-13B 在复杂指令跟随上超越 Vicuna，在代码/数学方面接近 ChatGPT，ICLR 2024。

## 基本信息

- 论文：WizardLM: Empowering Large Pre-Trained Language Models to Follow Complex Instructions
- 作者：Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, Qingwei Lin, Daxin Jiang
- 机构：Microsoft（主要），Peking University（Feng）
- 发表：ICLR 2024
- arXiv：2304.12244

## 核心问题

Instruction tuning 的两个主要数据瓶颈：

1. **规模问题**：人工标注昂贵，难以规模化
2. **复杂度分布问题**：人工标注的难度分布偏向简单/中等，高难度指令稀缺——而真实用户需求（代码、多步推理、复杂约束）恰恰需要复杂指令

Self-Instruct（Wang et al. 2022）解决了规模问题，但生成的指令复杂度仍然有限，难度分布与 ShareGPT 相比偏低（Figure 5a 对比可见）。

> **ShareGPT** 是用户自愿分享的 ChatGPT 对话记录，由第三方平台收集汇总。因为是真实用户与 ChatGPT 的交互记录，其指令多为多轮对话、较复杂的真实需求（如调试代码、分析文章、多步推理），难度分布明显高于 Alpaca/Self-Instruct 这类从种子指令批量生成的数据。Vicuna 就是用 ShareGPT 的 70k 多轮对话微调 LLaMA 得到的。WizardLM 论文里用 ShareGPT 的难度分布作为"自然人类需求复杂度"的参照基线，指出 Alpaca 生成的指令与之有明显差距。

本文的核心洞察：**LLM 不仅能生成指令，还能把简单指令改写成更复杂的版本**——通过迭代进化，可以系统性地提升整个数据集的复杂度分布，同时保持多样性。

## 方法：Evol-Instruct

### 整体框架

```
初始指令池 D(0)（Alpaca 52k）
    ↓ [重复 M=4 轮]
Instruction Evolver（ChatGPT）
    ├── In-Depth Evolving（深度进化，增加复杂度）
    └── In-Breadth Evolving（广度进化，增加多样性）
    ↓
Instruction Eliminator（过滤失败的进化）
    ↓
合并所有轮次 → 最终指令池（250k）
    ↓
随机采样 70k → 微调 LLaMA 13B → WizardLM
```

### In-Depth Evolving（深度进化）

目标：把一条指令改写成更复杂、更难的版本。五种操作，每次随机选其一：

| 操作 | 作用 | 示例（从"1+1=?"出发） |
|---|---|---|
| **Add Constraints** | 增加约束条件 | "在金德巴赫猜想的框架下，如何证明 1+1=2？" |
| **Deepening** | 加深概念深度 | "在什么情况下 1+1 不等于 2？" |
| **Concretizing** | 具体化/特殊化 | "如果 x=3，2x+3=7 成立吗？" |
| **Increase Reasoning Steps** | 增加推理步骤 | "x³+2x+3=7，x 的值是多少？" |
| **Complicate Input** | 复杂化输入格式（代码/公式/表格） | 把文字问题变成代码片段 |

每次进化只加 10–20 个词，避免复杂度跳变过大影响泛化。In-Depth Evolving 的核心 prompt 模板：
> "Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems (e.g., ChatGPT and GPT-4) a bit harder to handle. But the rewritten prompt must be reasonable, understood, and responded by humans."

### In-Breadth Evolving（广度进化）

目标：生成全新的、与原指令相关但话题更偏门的指令，以扩大 topic/skill 覆盖。

核心 prompt：要求 LLM 以给定指令为灵感，生成一条属于同一 domain 但更长尾（rare）的新指令，难度和长度保持相当。

### Elimination Evolving（淘汰机制）

过滤以下四类失败进化：
1. 进化后的指令与原指令相比没有信息增益（用 ChatGPT 判断）
2. 生成的响应含"sorry"且较短（<80 词）——说明模型无法回应进化后的指令
3. 响应只含标点和停用词
4. 进化后的指令明显包含 prompt 中的关键词（如"rewritten prompt"）——说明进化失败

### 微调细节

- **基础模型**：LLaMA 13B（Touvron et al. 2023）
- **数据**：从 250k 进化指令中随机采样 70k（与 Vicuna 训练量相同，保证对比公平）
- **对话格式**：Vicuna 的 chat prompt（"A chat between a curious user and an artificial intelligence assistant..."）
- **训练**：Adam optimizer，lr=2×10⁻⁵，batch size=4/GPU，8×V100 GPU，3 epochs，140 小时

## 关键结果 / 数据

### 自动评测（9 个 benchmark，Table 1）

| 模型 | Avg | MMLU | ARC | HellaSwag | TruthfulQA | HumanEval | GSM8k | AlpacaEval | MT-Bench | WizardEval |
|---|---|---|---|---|---|---|---|---|---|---|
| ChatGPT-3.5 | 76.15 | 70.0 | 85.2 | 85.5 | 47.0 | 48.1 | 80.8 | 89.37 | 7.94 | 100.0 |
| Alpaca-13B | 43.44 | 46.63 | 51.20 | 76.31 | 41.62 | 9.2 | 8.35 | 33.25 | 4.78 | 76.6 |
| Vicuna-13B | 54.60 | 50.84 | 58.53 | 79.94 | **52.68** | 11.5 | 24.34 | 70.43 | 6.21 | 86.9 |
| **WizardLM-13B** | **58.96** | 52.92 | **57.25** | **80.88** | 50.55 | **24.0** | **37.15** | **75.31** | **6.35** | **89.1** |

WizardLM-13B 在 HumanEval（代码）上 24.0 vs Vicuna 11.5；GSM8k（数学）37.15 vs Vicuna 24.34——代码和数学的提升最显著，正是复杂指令增多带来的效果。

### 人工评测（WizardEval，Figure 4b）

在由 218 条真实人类指令组成的 WizardEval 测试集上（29 个技能和领域），盲测比较：

- vs Alpaca-13B：Win 116 / Lose 50 / Tie 52
- vs Vicuna-13B：Win 90 / Lose 70 / Tie 58
- vs ChatGPT：Win 64 / Lose 88 / Tie 66

WizardLM 明显优于同级别开源模型；对 ChatGPT 在高难度场景也有竞争力（尤其代码类任务）。

### 进化轮次的消融（Figure 5b）

从 C0（原始 Alpaca）到 C4（4 轮进化后）：

| 进化轮次 | 平均难度 | 9 个 benchmark 平均分 |
|---|---|---|
| C0（Alpaca） | 3.0 | 41.25 |
| C1 | 5.40 | 48.39 |
| C2 | 6.35 | 53.83 |
| C3 | 6.84 | 55.75 |
| C4 | 7.08 | 57.61 |

每轮进化都带来明显提升，验证了"**指令复杂度 → 模型能力**"的单调正相关。

## 局限性

论文 Section 5 明确列出：

- **评测方法的扩展性问题**：依赖 GPT-4 和人工评测，成本高，难以大规模标准化
- **测试集覆盖问题**：WizardEval 可能不代表所有应用场景
- **进化质量依赖 ChatGPT**：Evol-Instruct 的进化和过滤都调用 ChatGPT API，引入了 ChatGPT 的偏差和成本；总计调用 API 约 52k×4×3=624k 次

另：
- **语言覆盖**：实验只涉及英语
- **进化可能引入噪声**：部分进化出的指令虽然通过了过滤，但从人类角度看仍然不自然
- **复杂度上限**：4 轮进化后难度增长开始放缓，进一步进化的收益递减

## 现状与影响

一句话定性：**Evol-Instruct 是指令数据"复杂度工程"的奠基方法——它证明了"让 LLM 自动提升指令难度"是可行且高效的路线，直接催生了 WizardCoder、WizardMath 等专用变体，并影响了后续大量合成指令数据的设计。**

- **WizardCoder / WizardMath**：Microsoft 团队将 Evol-Instruct 直接应用于代码（WizardCoder，2023）和数学（WizardMath，2023）领域，在同规模开源模型中分别刷新了 HumanEval 和 GSM8k/MATH 榜单
- **指令复杂度概念的扩散**：Evol-Instruct 的"逐步加深"思路被许多后续工作借鉴，包括 Deita（用复杂度得分筛选）、ShareGPT4 系列等
- **与 Self-Instruct / Alpaca 的定位区别**：Self-Instruct 解决"规模"，Evol-Instruct 解决"复杂度分布"，两者是互补关系而非替代——Evol-Instruct 以 Alpaca 52k 为种子，没有 Self-Instruct 就没有 Evol-Instruct
- **局限被后续工作修正**：Evol-Instruct 依赖 ChatGPT 进化，后续 WizardLM-2 等工作探索了用开源模型替代 ChatGPT 进行进化，降低了成本依赖

## 和 wiki 内其他概念的关联

- [Self-Instruct](./self-instruct-2212.10560.md)：Evol-Instruct 的直接前身，解决了生成指令的规模问题；Evol-Instruct 在此基础上进一步解决复杂度分布问题。WizardLM 以 Alpaca（Self-Instruct 生成）的 52k 数据为初始种子
- [Stanford Alpaca](./stanford-alpaca.md)：Alpaca 52k 是 Evol-Instruct 的种子数据集；WizardLM 将其进化为更复杂的 250k 数据集
- [Vicuna](./vicuna-open-source-chatbot.md)：WizardLM 的主要对比基准之一；Vicuna 从 ShareGPT 多轮对话出发，WizardLM 从 Evol-Instruct 单轮进化出发，代表两种数据路线
- [Deita](./deita-2312.15685.md)：用复杂度、质量、多样性三维自动筛选 instruction tuning 数据；复杂度维度的设计受 Evol-Instruct 启发，并进一步量化
- [Instruction Tuning](../20-concepts/instruction-tuning.md)：Evol-Instruct 是指令数据合成路线中"复杂度控制"方向的代表，属于该文档"合成数据演化"小节的核心案例
- [AlpaGasus](./alpagasus-2307.08701.md)：同期代表性工作，路线相反——AlpaGasus 做数据筛选（少而精），Evol-Instruct 做数据增强（复杂化）；两者共同验证了"数据质量 > 数据数量"这一命题

## 值得看的部分 / 相关资料

- **Section 3.2（Evol-Instruct 核心方法）**：五种 In-Depth Evolving 操作的 prompt 模板（Example 3.1–3.3）和 Elimination 四条过滤规则——最具工程参考价值。其中：
  - **Add Constraints** 的完整 prompt（Example 3.1）要求模型"在保持合理、人类可理解的前提下，在原指令中再加一条约束/要求，只允许增加 10–20 词"。Complicate Input 的 prompt（Example 3.2）需要 in-context demonstration，完整版见论文 Appendix D
  - **In-Breadth Evolving** 的 prompt（Example 3.3）要求"以 #Given Prompt# 为灵感，生成一条全新指令，需与原指令属于同一 domain 但更长尾（rare），难度和长度相当"
  - **Elimination Evolving** 四条过滤规则的核心逻辑：第 1 条（无信息增益）用 ChatGPT 对比前后指令判断，prompt 见 Appendix G；第 2 条（响应 <80 词且含 sorry）是最简单的启发式过滤，不依赖额外 API 调用，实践中最常触发
- **Figure 1**：从"1+1=?"出发的进化树，直观展示六种操作的效果
- **Figure 5**：难度 vs 进化轮次 + 性能 vs 进化轮次的消融——证明复杂度和性能的单调正相关
- **Table 1 & Table 2**：9 个 benchmark 的完整对比，以及不同种子数据/基模型/数据规模的消融
- 后续关键工作：
  - Luo et al. 2023, *WizardCoder*（arXiv:2306.08568）——将 Evol-Instruct 应用于代码，15B 超越所有开源代码模型
  - Luo et al. 2023, *WizardMath*（arXiv:2308.09583）——将 Evol-Instruct 应用于数学推理
  - Liu et al. 2023, *Deita*（arXiv:2312.15685）——在 Evol-Instruct 思路上加入量化的复杂度/质量打分筛选
