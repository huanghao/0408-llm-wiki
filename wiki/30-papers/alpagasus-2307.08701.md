# AlpaGasus: Training a Better Alpaca with Fewer Data

一句话总结：AlpaGasus 用 ChatGPT 对 Alpaca 的 52k 指令数据逐条评分，筛掉低质量样本，只用约 9k 条高质量数据微调出的模型在多项评测上反超原版 Alpaca——直接回答了"数据少但精是否胜过多而杂"。

**论文**：*AlpaGasus: Training a Better Alpaca with Fewer Data*  
**作者**：Lichang Chen, Shiyang Li, Jun Yan, Hai Wang, Kalpa Gunaratna, Vikas Yadav, Zheng Tang, Vijay Srinivasan, Tianyi Zhou, Heng Huang, Hongxia Jin  
**机构**：University of Maryland, Samsung Research America  
**arXiv**：2307.08701  
**发表**：ICLR 2024

## 核心问题

Stanford Alpaca 用 52k 条 self-instruct 生成的数据微调 LLaMA-7B，效果尚可但数据质量参差。直觉上这 52k 条数据里有相当一部分是低质量甚至有害的训练样本：

- 有些 instruction 含糊或无意义
- 有些 output 明显错误或答非所问
- 有些样本只是在重复，信息密度很低

核心问题：**能否用更少但质量更高的子集，训出比原版 Alpaca 更好的模型？**

## 方法 / 核心机制

### 用 ChatGPT 对数据质量打分

AlpaGasus 的核心创新是把数据过滤变成一个评分问题：

**打分 prompt 设计：**
```
We would like you to evaluate the quality of an AI assistant's response to the user's instruction.
Please rate on a scale of 1-5 considering:
- Accuracy
- Helpfulness
- Coherence

Instruction: <instruction>
Input: <input>
Response: <response>

Rating: [1-5]
Reason: <brief explanation>
```

用 ChatGPT（gpt-3.5-turbo）对全部 52k 条 Alpaca 数据逐条打分，要求同时给出评分理由。

**过滤策略：**
- 保留评分 ≥ 4.5 分（满分 5 分）的样本
- 过滤后约剩 **9k 条**（约 17% 的原始数据）

这个阈值的选择是通过实验确定的：论文比较了不同 threshold 下的训练效果，4.5 分是性价比最高的切点。

### 训练设置

- Base model：LLaMA-7B 和 LLaMA-13B
- 训练方式：与原版 Alpaca 完全相同（SFT，next-token prediction，只对 output 算 loss）
- 唯一变化：训练数据从 52k → 9k（过滤后高质量子集）

## 关键结果 / 数据

**主要评测结果（人工评测，pairwise comparison）：**

| 对比 | AlpaGasus-7B 胜率 |
|------|-----------------|
| AlpaGasus-7B vs Alpaca-7B | ~57% 胜 |
| AlpaGasus-7B vs text-davinci-003 | 接近持平 |

- AlpaGasus-7B（9k 数据）在人工评测中优于 Alpaca-7B（52k 数据）
- AlpaGasus-13B 表现进一步提升
- 用 GPT-4 作为 judge 的自动评测结论一致

**数据量 vs 质量的消融实验：**
- 随机从 52k 里抽 9k 训练：效果明显差于质量过滤后的 9k
- 说明性能提升来自**数据质量**而非仅仅数据量减少

**不同分数阈值的比较：**
- ≥3.0：几乎保留全部数据，效果接近原版 Alpaca
- ≥4.0：效果开始明显提升
- ≥4.5：最佳性价比
- ≥4.8：数据太少，效果反而下降

## 局限性

- 打分依赖 ChatGPT（gpt-3.5-turbo），打分本身有噪声，且评分标准随模型版本变化
- 只在 Alpaca 52k 数据上验证，对其他数据来源的泛化性未充分测试
- 过滤后的 9k 数据任务分布发生偏移（高分样本本身就偏向某类任务），可能丢失某些任务类型的覆盖
- 评测以人工 pairwise 为主，规模有限，存在评测偏差风险
- 没有深入分析什么类型的样本被过滤掉了，以及为什么

## 现状与影响

**还在用吗？** AlpaGasus 的具体数据集已被更强的数据和方法替代，但"用 LLM 打分过滤训练数据"这个思路已成 SFT 数据工程的标配。

**被什么取代？** 在数据质量控制方向，被更系统的方法取代：
- LIMA（"less is more"）：手工挑选 1k 条极高质量数据，进一步极端化了这个方向
- WizardLM 系列：在数据复杂度而非质量过滤上做文章
- Tulu 系列：系统比较多种数据来源和过滤策略
- 更新的方法用奖励模型（reward model）而非 ChatGPT 打分来做数据过滤

**贡献与实现路线是否分离？** 是。贡献（LLM-as-data-quality-judge 的实证可行性）已被广泛借鉴；实现路线（用 ChatGPT 打 1-5 分）已被更精细的 reward model 或多维评分取代。

**一句话定性**：AlpaGasus 是"数据质量 > 数据数量"这条路线的早期有力实证，奠定了 LLM 驱动的数据过滤作为 SFT 工程基础设施的地位。

## 和 wiki 内其他概念的关联

- [Stanford Alpaca](stanford-alpaca.md)：AlpaGasus 直接针对 Alpaca 数据集做过滤，是 Alpaca 路线的一个改进变体
- [Self-Instruct (2212.10560)](self-instruct-2212.10560.md)：Alpaca 的数据来自 Self-Instruct 方法，AlpaGasus 揭示了这类自动生成数据的质量问题
- [Unnatural Instructions (2212.09689)](unnatural-instructions-2212.09689.md)：同样研究合成数据的质量，但角度不同——Unnatural Instructions 关注多样性，AlpaGasus 关注过滤
- [Instruction Tuning with GPT-4 (2304.03277)](instruction-tuning-with-gpt-4-2304.03277.md)：类似结论，用更强的模型生成/评估数据效果更好
- [Instruction Tuning 概念](../20-concepts/instruction-tuning.md)：AlpaGasus 是"数据质量控制"子方向的代表工作

## 值得看的部分 / 相关资料

- Figure 2（打分 prompt 设计）：具体的评分 prompt 值得参考，可直接复用
- Table 1 / Figure 4（阈值消融）：不同分数阈值对训练效果的影响，直观展示质量过滤的边际效益
- §4.3（数据分布分析）：分析了哪些类型的数据被过滤掉
- 代码和数据：[github.com/gq-chen/alpagasus](https://github.com/gq-chen/alpagasus)
