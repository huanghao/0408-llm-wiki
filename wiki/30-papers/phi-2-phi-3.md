# Phi-2 / Phi-3：小模型，教科书质量数据的持续演进

**Phi-2**：Microsoft Research 内部报告，2023 年 12 月发布（无 arxiv 论文，仅 HuggingFace model card）
**Phi-3**：*Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone*（Abdin et al., 2024）
**arxiv**：2404.14219（Phi-3 报告）
**机构**：Microsoft
**发表**：Phi-2 2023 年 12 月，Phi-3 2024 年 4 月（arXiv，更新至 2024 年 8 月）

> Phi-2 没有独立 arxiv 论文，内容来源于 HuggingFace model card 和 Microsoft 博客，部分细节据公开资料。Phi-3 内容来自 2404.14219 原文。

---

## 核心问题

**phi-2/phi-3 没有算法创新，但也不只是"更大的 phi-1"。核心看点有两个：**

### 1. 把优化目标从 Compute Optimal 换成 Data Optimal

此前领域共识是 Chinchilla 的 **Compute Optimal**：给定训练预算，怎么分配参数量和 token 数最划算。

phi-3 提出的问题是另一个轴：**给定部署约束（模型必须跑在手机上，参数量上限 ~4B），怎么用数据把这个参数量压榨到极限？** 优化目标从"省训练成本"变成"省推理成本"。这个视角的转变催生了 phi-3 的所有数据决策：主动过滤事实性内容（新闻、比赛结果），只保留对推理能力有正向贡献的内容，即使牺牲知识覆盖面。

Figure 3（MMLU 误差 vs 参数量对数图）是这个主张的核心实证：phi 系列整体在同等参数量下显著优于 Llama-2 系列，曲线的斜率差异直观展示了"数据最优"路线的优势。

### 2. 跨领域泛化的工程验证

phi-1 只在 Python 代码任务上验证了"教科书数据"的有效性。合理的怀疑是：代码结构清晰，合成教科书容易写，这个方法对通用语言理解不一定适用。phi-2 和 phi-3 逐步证明：这条路线在通用推理、数学、多语言上同样成立。这是工程验证而非理论贡献，但对确立整条路线的可信度至关重要。

**两个实际结果**：
- Phi-2（2.7B）：性能匹敌 25 倍参数量的普通训练模型
- Phi-3-mini（3.8B）：整体平均分接近 GPT-3.5，4-bit 量化后可在 iPhone 14 本地运行

---

## Phi-2（2.7B，2023 年 12 月）

### 核心变化

从 phi-1 的"代码专用"扩展为"通用推理"：

- **参数量**：1.3B → 2.7B（翻倍）
- **训练数据**：~1.4T tokens（包含 250B tokens 合成数据）
- **数据构成**：
  - Azure OpenAI GPT-3.5 生成的 NLP 合成教科书（常识推理、通用知识）
  - Falcon RefinedWeb + SlimPajama 筛选的高质量网页数据（用 GPT-4 质量过滤）
  - **没有 RLHF**：刻意不做对齐微调，保留为研究基座
- **上下文**：2048 tokens
- **训练算力**：96× A100-80GB，14 天

### 性能定位

论文声称在 <13B 参数模型里"接近 SOTA"，在推理类任务上匹敌参数量 25 倍的普通数据训练模型（Phi-3 报告的 Figure 3 用 MMLU error 展示了 Phi 系列整体的"数据最优"缩放曲线，明显优于 Llama-2 系列）。

具体 benchmark 数字未在 model card 中完整公开，Phi-3 报告的对比表中有部分 Phi-2 数据：

| 任务 | Phi-2 (2.7B) | Mistral-7B | Llama-3-Instruct-8B |
|------|-------------|------------|---------------------|
| MMLU | 56.3 | 61.7 | 66.5 |
| GSM-8K | 61.1 | 46.4 | 77.4 |
| HumanEval | 59.0 | 28.0 | 60.4 |
| Arc-C | 75.9 | 78.6 | 82.8 |

---

## Phi-3（3.8B–14B，2024 年 4 月）

**开放权重**：Phi-2 和 Phi-3 全系列均以 MIT 许可证发布，可免费商用，权重公开下载。这是 Microsoft 与 Meta（Llama）、Google（Gemma）共同推动的开放权重潮流的一部分。

### 模型家族

Phi-3 报告同时覆盖多个规模：

| 模型 | 参数量 | 训练 tokens | 上下文 | 架构特点 |
|------|--------|------------|--------|---------|
| phi-3-mini | 3.8B | 3.3T | 4K（128K via LongRope）| Llama-2 同款 block 结构，32 heads，32 layers，hidden 3072 |
| phi-3-small | 7B | 4.8T | 8K | tiktoken tokenizer，GQA，blocksparse attention，GEGLU |
| phi-3-medium | 14B | 4.8T | 4K | 40 heads，40 layers，hidden 5120 |
| phi-3.5-mini | 3.8B | — | 128K | phi-3-mini + 多语言 mid-training |
| phi-3.5-MoE | 16×3.8B（6.6B active）| — | 128K | MoE，top-2 routing，16 experts |
| phi-3.5-Vision | 4.2B | — | 128K | phi-3.5-mini + 视觉模态 |

### 训练方法

**数据**：延续 phi-1/phi-2 的路线，"教科书质量"数据 + "数据最优制度"（data optimal regime）：

- **两阶段预训练**：
  - Phase 1：以网页数据为主，建立通用知识和语言理解
  - Phase 2：加入更多高质量过滤网页 + 合成数据，强化逻辑推理和小众技能
- **合成数据**：LLM 生成，覆盖推理、数学、代码；多样性策略同 phi-1（随机词汇约束）
- **数据过滤哲学**："数据最优制度"——不追求 compute optimal，而是为给定参数量选择最优数据质量和数量。过滤掉"事实类"网页（如比赛结果），只保留能提升推理能力的内容

**后训练**：SFT + DPO（phi-1/phi-2 没有做 RLHF，phi-3 加入）：
- SFT：高质量多领域数据（数学、代码、推理、对话、安全）
- DPO：用拒绝样本引导，减少有害输出

### 关键结果

**phi-3-mini（3.8B）与同期模型对比**（来自 2404.14219 表格）：

| 任务 | phi-3-mini 3.8B | phi-3-small 7B | Mixtral-8x7B | GPT-3.5 v1106 | Llama-3-In-8B |
|------|-----------------|----------------|--------------|---------------|---------------|
| MMLU | 68.8 | 75.7 | 70.5 | 71.4 | 66.5 |
| GSM-8K | 82.5 | 89.6 | 64.7 | 78.1 | 77.4 |
| MATH | 41.3 | 34.6 | 11.1 | 45.3 | 28.2 |
| HumanEval | 58.5 | 61.0 | 37.8 | 62.2 | 60.4 |
| BigBench-Hard | 71.7 | 79.1 | 69.7 | 68.3 | 51.5 |
| MT-Bench | 8.38 | 8.70 | — | 8.35 | — |
| Average | 69.7 | 73.6 | 66.8 | 72.8 | 67.3 |

phi-3-mini（3.8B）在 MMLU、GSM-8K 上超过 Mistral-7B 和 Gemma-7B，整体平均分接近 GPT-3.5。

**部署亮点**：phi-3-mini 4-bit 量化后约 1.8GB，iPhone 14（A16 Bionic）本地运行 > 12 tokens/秒。

### phi-3.5 系列（2024 年后续更新）

- **phi-3.5-MoE**（16×3.8B，6.6B active）：在语言推理、数学、代码上超过 Llama-3.1 和 Mixtral 系列，接近 Gemini-1.5-Flash；RepoQA（代码库理解）上达到 GPT-4o 90% 以上
- **phi-3.5-mini**：128K 上下文 + 多语言能力，相比 phi-3-mini 有显著多语言提升
- **phi-3.5-Vision**：支持多图文交织输入

---

## Phi-4（14B，2024 年 12 月）

phi-4 是 phi 系列最自然的延伸，主要差异在于**把合成数据的重心从"通用推理"转向"数学"**，不引入新架构。

- **参数量**：14B（与 phi-3-medium 相同规模），MIT 许可证
- **arxiv**：2412.08905
- **核心改动**：合成数据生成策略大幅升级，专门为数学推理设计的"seed data → generate → verify → filter"流程；使用了更多由 GPT-4o 生成的数学习题和解题步骤
- **phi-4-mini**（3.8B）：同期推出，延续手机部署定位，数学能力相比 phi-3-mini 有显著提升

**结果（部分 benchmark）**：

| 任务 | phi-4 (14B) | phi-3-medium (14B) | GPT-4o |
|------|------------|-------------------|--------|
| MATH | **80.4** | 34.6 | 74.6 |
| AIME | 16.7 | — | 9.3 |
| HumanEval | 82.6 | 55.5 | 90.2 |
| MMLU | 84.8 | 78.0 | 85.7 |

MATH 上 phi-4 14B 超越 GPT-4o 是 2024 年末的一个标志性数字——14B 模型在数学专项上跑赢 frontier 大模型，再次验证"对准问题、高质量数据"的路线。

**这说明了什么**：phi-1 到 phi-4 的演进路径非常清晰——每一代把同一套"高质量合成数据 + 小模型"的方法对准一个新领域（代码 → 通用推理 → 数学），然后推高上限。框架本身没有变，变的是数据工程的目标域。

**建议单独文档？**：不需要，phi-4 的贡献是 phi 系列的量变而非质变，和本文档放在一起更能体现整条演进路线的逻辑。

---

## 局限性

1. **英语偏向**：phi-3-mini 多语言能力明显弱于英语（Figure 4 显示非英语 MMLU 大幅下降，phi-3.5-mini 通过 mid-training 改善）
2. **事实性知识不足**：主动过滤事实性网页数据导致知识覆盖面窄（论文自承）
3. **长上下文性能下降**：phi-3.5 在 RULER 128K 任务上性能显著低于 4K（phi-3 报告 Table 2 显示 phi-3.5-MoE 从 4K 的 94.8 降到 128K 的 64.2）
4. **合成数据偏差**：大量 LLM 生成数据可能引入系统性偏差，尤其在创意写作、主观判断类任务上
5. **架构无创新**：phi-3-mini 直接复用 Llama-2 block 结构，phi-3-small 引入 blocksparse attention 但也是已有技术
6. **phi-2 数据细节不透明**：没有独立 arxiv 论文，合成数据生成方法未完整公开

---

## 现状与影响

**定性：Phi 系列是"数据质量路线"最成功的连续性工程验证，phi-3-mini 是 2024 年小模型的标杆之一。**

- Phi-3-mini 发布时成为开源 <4B 参数模型的 SOTA，直接推动了 Gemma-2、Llama-3.2-3B 等竞品加速发布
- "数据最优制度"（data optimal regime）的提法被广泛引用，和 DeepSeek 的数据哲学形成呼应——在固定推理预算下，小模型 + 高质量数据比大模型 + 普通数据更经济
- phi-3.5-MoE 进一步验证了 MoE 在小规模上的可行性
- **当前（2026）视角**：phi-3 系列已被 phi-4（2025）超越，phi-4-mini 在数学推理上进一步推高了小模型上限；但 phi-3 确立的"教科书质量 + 两阶段预训练 + DPO"流程已成为业界小模型的标准范式

---

## 小模型的实际算力门槛

"小模型 + 高质量数据"这条路线的核心吸引力不只是便宜，而是**把强大能力推进到消费级设备可以运行的参数规模**。以下是 2024–2025 年代表性小模型的算力门槛（4-bit 量化是主流部署方式，性能损失约 3-5%）：

| 参数量 | 代表模型 | 最低设备 | VRAM / 内存 | 参考速度 |
|--------|---------|---------|------------|---------|
| 1–4B | phi-3-mini、Llama 3.2 3B、Gemma 3 4B | 手机（iPhone 14+）/ 任意笔记本 CPU | <2 GB | 10–30 tok/s on phone |
| 7–8B | Llama 3.1 8B、Qwen2.5 7B | M1/M2 MacBook（8GB 统一内存）/ RTX 3070 | 4–6 GB（4-bit） | 30–60 tok/s |
| 13–14B | phi-4 14B、Qwen2.5 14B | M2 Pro（16GB）/ RTX 4090 | 8–10 GB（4-bit） | 15–30 tok/s |
| 32B | QwQ-32B、Qwen2.5 32B | M2 Max（32GB）/ RTX 4090 | ~18 GB（4-bit） | 8–15 tok/s |
| 70B | Llama 3.3 70B、DeepSeek-R1 70B 蒸馏 | A100 40GB / 2×RTX 4090 | ~35 GB（4-bit） | 5–12 tok/s |

**最关键的三个认知**：

1. **7B 是个人工作站的甜蜜点（2025）**：M2/M3 MacBook Pro（16-36GB 统一内存）跑 7B FP16 / 14B 4-bit 很流畅，处理代码补全、文档总结、推理问答几乎没有瓶颈。
2. **能力边界不是参数量，是数据质量**：phi-3-mini（3.8B）在推理 benchmark 上超过 Llama 2（70B），DeepSeek-R1 14B 蒸馏版在数学上远超同参数量的普通模型——参数量只是"房间大小"，装进去的是什么决定了能力。
3. **推理模型 = 更慢但更强的小模型**：R1/QwQ 等推理模型的代价是生成更多 token（思考链），相同参数量下延迟是普通模型的 3-10 倍，但在数学/代码类任务上能力大幅领先。

详细全景见 → [开放权重模型全景（2023–2025）](./open-weight-models-landscape.md)

---

## 和 wiki 内其他概念的关联

- **[Phi-1](./phi-1-2306.11644.md)**：直接前驱，"教科书质量数据"命题的起点；Phi-2/Phi-3 是同一命题在通用推理的扩展
- **[随机森林](../20-concepts/random-forest.md)**：phi-1 用随机森林做代码质量过滤；Phi-2/Phi-3 延续相同的数据过滤思路，但规模更大、过滤器更复杂
- **[DCLM](./dclm-2406.11794.md)**：同期的数据过滤系统，方法互补——DCLM 用 fastText 分类器，Phi-3 用 LLM 质量标注 + 过滤
- **[Data Mixing Laws](./data-mixing-laws-2403.16952.md)**：Phi-3 的两阶段数据配比策略（Phase 1 网页为主，Phase 2 合成为主）和 Data Mixing Laws 的实证框架是同一问题的不同切入角度
- **[LIMO](./limo-2502.03387.md)**：同一逻辑在数学推理微调阶段的极端验证——小而精的数据，phi 系列是代码/通用版本
- **[Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)**：Phi-3 的合成数据生成是该概念的大规模工程实践

---

## 值得看的部分

- **Phi-3 报告 Figure 3**：Phi 系列 vs Llama-2 系列的 MMLU 误差 vs 参数量对数图——直观展示"数据最优"路线的优势
- **Phi-3 报告 Section 2**（Technical Specifications）：phi-3-mini/small/medium 的架构差异和设计权衡
- **Phi-3 报告 Section 3**（Academic Benchmarks）：完整对比表，含 phi-2 数字
- **Phi-3 报告 Section 4**（Multilingual and Long Context）：phi-3.5 的多语言和长上下文扩展方法
- **Phi-3 报告 Section 5**（Safety）：post-training 安全对齐流程，phi-3 与 phi-2 的重要区别
- **HuggingFace model card（microsoft/phi-2）**：Phi-2 技术细节的主要公开来源
