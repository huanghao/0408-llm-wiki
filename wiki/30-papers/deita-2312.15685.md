# Deita: What Makes Good Data for Alignment? (2312.15685)

一句话总结：Deita 把 instruction tuning 的数据选择问题拆成复杂度、质量、多样性三维，用自动打分和去冗余选择 6K/10K SFT 样本，证明少量高质量数据可以接近甚至超过大规模 SFT 数据。

## 基本信息

- 论文：What Makes Good Data for Alignment? A Comprehensive Study of Automatic Data Selection in Instruction Tuning
- 作者：Wei Liu, Weihao Zeng, Keqing He, Yong Jiang, Junxian He
- 单位：ShanghaiTech University, Beijing University of Posts and Telecommunications, Meituan, Alibaba Group, The Hong Kong University of Science and Technology
- 会议：ICLR 2024
- arXiv：[2312.15685](https://arxiv.org/abs/2312.15685)
- 代码与数据：[hkust-nlp/deita](https://github.com/hkust-nlp/deita)
- 本地原文：`raw/inbox/2312.15685.pdf`

注意：用户给出的“清华”与论文首页不一致；论文 PDF 首页列出的机构没有 Tsinghua。

## 核心问题

Instruction tuning 已经是对齐 LLM 的标准后训练步骤，但“什么样的 SFT 数据最值得训练”并不清楚。

这篇论文把问题形式化为：给定一个 instruction-response 数据池 `X` 和预算 `m`，选择子集 `S` 做 SFT，使训练后的 alignment performance `Q(S)` 最大。这里的 `m` 同时代表数据预算和训练计算预算。

论文研究的不是“如何生成更多数据”，而是：

- 如何衡量一条 instruction-response 样本是否值得保留？
- 如果只能选 6K 或 10K 条，应该按什么准则选？
- 数据质量、复杂度、多样性分别对 alignment 有多大影响？

## 方法 / 核心机制

Deita 的假设是：好的 instruction tuning 数据应该同时满足三点：

- **复杂度高**：指令不是简单模板题，而是有约束、有推理步骤、有具体上下文。
- **质量高**：回答准确、相关、详细、有帮助。
- **多样性高**：样本之间不要高度重复，能覆盖不同请求类型。

一个直观例子：如果原始 instruction 是“写一封道歉邮件”，复杂度增强后可能变成“以产品经理口吻给企业客户写一封道歉邮件，解释服务中断原因、给出补偿方案，并保持语气专业但不推卸责任”；质量增强则不是改题目，而是把回答从泛泛道歉改成结构清楚、有背景、有行动项、有补偿细节的邮件；多样性过滤则会避免最终数据集里充满几十条几乎相同的“写邮件”任务。

论文先分别做三组 controlled study，再把三维合成一个简单选择算法。

### Evol Complexity

复杂度打分不是直接问 ChatGPT“这条有多难”，因为直接打分容易把大部分样本都打成相似高分。

**Evol-Instruct 风格 prompt** 指的是让模型按固定操作逐步“进化”一条指令，而不是凭空生成新题。论文沿用 WizardLM / Evol-Instruct 的思路，要求 ChatGPT 对原始 instruction 做 adding constraints、deepening、concretizing、increasing reasoning steps 等变换。它的作用是制造一组同源但难度递增的样本，方便 judge 做相对排序。

论文采用 evolution-based scoring：

1. 从 Alpaca 随机取 2K seed 样本。
2. 对每条 instruction 用 Evol-Instruct 风格 prompt 做 5 次复杂度增强。
3. 得到原始 instruction + 5 个增强版本，共 6 个复杂度递增变体。
4. 让 ChatGPT 在同一 prompt 里对这 6 个变体排序和打分。
5. 用这些分数训练一个 LLaMA-1 7B scorer，之后对整个数据池自动预测复杂度分数。

关键点：同源样本的相对排序比单条绝对打分更细粒度，也更便宜，因为只需要标注小 seed 集。

### Evol Quality

质量打分类似复杂度打分，但增强对象从 instruction 换成 response。

例子：同一个问题“解释 Transformer 的 self-attention”，低质量回答可能只有一句“self-attention 让 token 关注其他 token”；高质量回答会说明 query/key/value、attention weight、上下文聚合、为什么能并行，以及它和 RNN 的差异。Deita 想学到的是后一类 response 的模式。

流程：

1. 对同一个 instruction-response 样本，要求 ChatGPT 逐步提升 response 质量。
2. 增强方向包括 helpfulness、relevance、depth、creativity、details。
3. 对原始 response + 5 个增强版本进行相对排序和打分。
4. 用打分结果训练 LLaMA-1 7B quality scorer。

论文发现质量维度对 `Xbase` 这种低质量、冗余数据池尤其重要。`Xbase` 是论文构造的“普通数据池”设定，由 Alpaca、Dolly、OAssist、FLAN 2022 组合而成，包含更多短回答、模板化回答和重复任务；它用来模拟实际收集 instruction 数据时常见的噪声环境。如果数据池本身质量已经较高，例如 `Xsota`，单独质量过滤的边际收益就会变小。

### Repr Filter / diversity-aware selection

多样性部分用 embedding-based filter 控制冗余。

例子：两条样本分别是“写一封请假邮件”和“帮我写一封病假申请邮件”，它们表面文字不同，但语义空间很近；如果数据预算只有 6K，就不应该让大量近邻样本占掉名额。Repr Filter 的目的就是优先留下高分样本，同时跳过这类近重复样本。

论文用 LLaMA-1 13B 编码样本，计算候选样本和已选集合中最近邻的 cosine distance，并设阈值 `τ = 0.9`。选择过程不是先随机去重，而是先按质量/复杂度排序，再逐条判断是否加入，从而保留高分样本同时避免集合过于重复。

## Deita 数据选择算法

最终算法叫 **score-first, diversity-aware data selection**：

1. 对每条样本预测复杂度分数 `c` 和质量分数 `q`。
2. 合成 evol score：`s = c * q`。
3. 按 `s` 从高到低排序数据池。
4. 从高分样本开始遍历，用 Repr Filter 过滤冗余样本。
5. 直到选满 `m = 6K` 或 `10K`。

这个算法可以理解成“先找最值得学的样本，再做语义去重”。如果一条样本复杂但回答差，`q` 低会把它压下去；如果回答好但任务太简单，`c` 低也会压下去；如果两条样本都高分但非常相似，Repr Filter 会只保留其中一条，给其他类型任务留预算。

论文用这个选择出的数据训练 Deita 模型，backbone 包括 LLaMA-1-13B、LLaMA-2-13B、Mistral-7B。

## 关键结果 / 数据

论文构造了两个数据池：

| 数据池 | 来源 | 样本数 | 设定 |
|---|---:|---:|---|
| `Xsota` | ShareGPT, UltraChat, WizardLM | 300K | 较复杂、较多样、质量较高 |
| `Xbase` | Alpaca, Dolly, OAssist, FLAN 2022 | 100K | 较低质量、冗余，更接近普通数据池 |

复杂度实验中，选 6K 样本训练 LLaMA-1-13B：

| 方法 | `Xsota` MT-Bench | `Xbase` MT-Bench |
|---|---:|---:|
| Random Selection | 5.84 | 4.93 |
| Instag Complexity | 6.18 | 4.98 |
| Evol Complexity | 6.27 | 5.57 |

质量实验中：

| 方法 | `Xsota` MT-Bench | `Xbase` MT-Bench |
|---|---:|---:|
| Random Selection | 5.84 | 4.93 |
| Response Length | 5.94 | 5.65 |
| Evol Quality | 6.19 | 5.67 |

多样性实验中：

| 方法 | `Xsota` MT-Bench | `Xbase` MT-Bench |
|---|---:|---:|
| Random Selection | 5.82 | 4.34 |
| Instag Diversity | 6.10 | 4.46 |
| Repr Filter | 6.17 | 4.68 |

这三张表的读法是：Table 2 说明“复杂度”不是 instruction length 或 perplexity 这种粗糙 proxy，Evol Complexity 更能选出对 MT-Bench 有帮助的样本；Table 3 说明“质量”在低质量数据池 `Xbase` 上收益更明显，因为它能过滤短、浅、错的回答；Table 4 说明“多样性”本身也有价值，只按高分选会被重复样本占预算，embedding-based Repr Filter 比 tag-based Instag Diversity 更稳。

最终模型结果中，Deita 用很少数据达到强 baseline：

| 模型 | 数据 | MT-Bench | AlpacaEval |
|---|---:|---:|---:|
| Vicuna-13B-v1.3 | 125K SFT | 6.39 | 82.11 |
| Deita-LLaMA1-13B | 6K SFT | 6.46 | 77.08 |
| LLaMA2-13B-Chat | >100K SFT + >1M RLHF | 6.65 | 81.09 |
| Deita-LLaMA2-13B | 10K SFT | 6.79 | 81.09 |
| Zephyr-beta | 200K SFT + 60K DPO | 7.34 | 90.60 |
| Deita-Mistral-7B | 10K SFT | 7.32 | 81.67 |
| Deita-Mistral-7B + DPO | 6K SFT + 10K DPO | 7.55 | 90.06 |

论文还报告了 data scaling：Deita 只用 3K 样本就能接近使用 `Xsota` 全量 300K 样本的 MT-Bench；继续增加数据时性能先上升后下降，说明即使在高质量数据池中，真正有益的 alignment 数据比例也有限。

这不是说“数据越少越好”，而是说在固定选择策略和固定训练 recipe 下，后面加入的样本会越来越像低边际收益甚至负收益样本：重复任务、低质量回答、和目标能力无关的样本会稀释训练信号。Deita 的结论更接近“先把最有用的数据选出来，再决定是否扩量”，而不是盲目把整个池子都训进去。

## 局限性

论文自身没有单独的 limitations section，但从实验设置可以看到几个边界：

- 主要贡献在 SFT 数据选择，不是完整 RLHF/RLAIF pipeline；DPO 只作为附加参考点。
- scorer 的训练依赖 ChatGPT 对 seed 样本的相对排序，因此仍继承 teacher/judge 的偏好和盲点。
- controlled study 主要用 MT-Bench、AlpacaEval、Open LLM Leaderboard 自动评测；虽然附录有人评估，但规模较小。
- 数据池是公开 instruction/chat 数据的组合，结论是否直接迁移到专业领域、工具调用、多轮 agent 数据，需要额外验证。
- 多样性 filter 基于 embedding 距离，能处理语义冗余，但不等于保证任务覆盖、风险覆盖或能力覆盖。

## 现状与影响

一句话定性：**Deita 是 instruction tuning 数据选择的代表性奠基工作之一；具体 scorer 和 Mistral/LLaMA-2 时代模型已过时，但“复杂度 + 质量 + 多样性”的选择框架仍活跃。**

截至 2026，Deita 不应被看作最佳 post-training recipe，而应看作数据选择范式的清晰基线：

- 仍在用的思想：小预算 SFT 下，自动数据选择比盲目扩数据更重要；复杂度、质量、多样性要一起考虑。
- 已被推进的方向：后续工作开始研究更大规模 instruction selection、targeted selection、多轮 dialogue selection、influence/gradient-based selection，以及 reasoning/code 任务中的 verifier-based filtering。
- 已被绕过的实现：直接训练 LLaMA-1 7B scorer、用早期 ChatGPT 做 seed ranking、用 MT-Bench/AlpacaEval 作为主要目标，已经不是 2026 的最佳实践。
- 仍值得读的原因：它把“数据质量 > 数据数量”从口号变成了可执行的 selection algorithm，并给出清楚 ablation。

这些后续方向大致是在补 Deita 没解决的部分：large-scale selection 关心选择算法能否扩到更大的候选池；targeted selection 不是选“通用好数据”，而是给 GSM8K、MMLU、客服、代码等目标任务选最有迁移价值的数据；multi-turn dialogue selection 把选择单位从单轮 instruction-response 扩到整段对话，并检查话题漂移、信息推进和问答格式一致性；influence/gradient-based selection 用训练梯度或影响函数估计“这条样本会不会降低目标验证集 loss”；verifier-based filtering 则在数学、代码、agent 任务里用执行器、单元测试或答案校验器替代 LLM judge。

相关后续读物：

- [Large-Scale Data Selection for Instruction Tuning](https://arxiv.org/abs/2503.01807)：研究 instruction-tuning 数据选择方法在更大规模候选池上的可扩展性。
- [Efficient Data Selection at Scale via Influence Distillation](https://arxiv.org/abs/2505.19051)：把样本对目标分布的 influence 蒸馏成可用于 fine-tuning 选择的权重。
- [Data Selection for Multi-turn Dialogue Instruction Tuning](https://arxiv.org/abs/2604.07892)：把选择对象扩展到多轮对话，关注全局覆盖和对话内部结构可靠性。
- [A Critical Look at Targeted Instruction Selection](https://arxiv.org/abs/2602.14696)：分析 targeted instruction selection 中真正起作用的是表示方式还是选择算法。

## 和 wiki 内其他概念的关联

- [Instruction Tuning](../20-concepts/instruction-tuning.md)：Deita 是 instruction tuning 数据选择的核心论文之一，补上了“已有数据池里怎么选”的问题。
- [AlpaGasus](./alpagasus-2307.08701.md)：AlpaGasus 主要用 ChatGPT 对 Alpaca 数据直接评分，Deita 更系统地拆成复杂度、质量、多样性。
- [LIMO](./limo-2502.03387.md)：两者都支持少量高质量数据的重要性；Deita 面向通用 alignment，LIMO 面向数学推理。
- [Human Feedback vs AI Feedback vs Verification](../40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md)：Deita 属于 AI feedback / model-based data selection，而不是 human preference 或外部 verifier 路线。
- [Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)：Deita 过滤的是 instruction-response 数据质量，verification 路线过滤的是可验证正确性，两者适用场景不同。

## 值得看的部分 / 相关资料

- Section 2：三组 controlled study，最值得看。
- Table 2 / 3 / 4：复杂度、质量、多样性分别带来的收益；读表时重点看同一数据预算 6K 下，相比 random selection 是否稳定提升，而不是只看单个绝对分数。
- Algorithm 1：最终选择算法很简单，适合作为自己的数据筛选 baseline；实现时只需要准备两个 scorer、一个 embedding 模型、一个相似度阈值，就能复刻“高分优先 + 去冗余”的骨架。
- Figure 2：数据越多不一定越好，增加到一定规模后 performance 下降。
- GitHub 项目：[hkust-nlp/deita](https://github.com/hkust-nlp/deita)

## 开放问题

- 对 2026 的 reasoning/code/agent 数据，复杂度分数应该由 LLM judge 给，还是由 verifier、执行器、环境回报给？
- `s = c * q` 是否过于简单？不同任务可能需要不同权重或非线性组合。
- 多轮对话是否应该按 turn 评分，还是按完整 conversation-level trajectory 评分？
- 如果目标模型、teacher 模型、judge 模型不同，数据选择是否需要 model-specific calibration？例如 GPT-4 认为“复杂且高质量”的样本，未必正好是一个 7B student 最能学会、最该优先学习的样本；校准问题就是要把 judge 的偏好映射到目标模型真实训练收益上。
