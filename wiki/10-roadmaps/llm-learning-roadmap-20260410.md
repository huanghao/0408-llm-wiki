# LLM 学习线路图（截至 2026-04-10）

基于之前的基础知识梳理，下面这份清单聚焦“从 2024 年中后期到 2026-04-10，这段时间 LLM 领域最值得补的主线”。

## 0. 前置：核心概念的演化脉络

在进入前沿论文之前，先理清这些基础概念的血缘关系。**分代的标准是"解决了什么本质问题、打开了什么新的可能性"，不是时间顺序，也不是工程改进。**

---

### 第一代：统计语言模型（1990s–2000s）

**核心问题**：如何建模"下一个词出现的概率"？

**n-gram LM**：统计词序列频率，用历史 n-1 个词预测下一个词。

```
P("趣" | "机器 学习 很 有") = count("机器 学习 很 有 趣") / count("机器 学习 很 有")
```

**遇到的坎**：词之间完全独立——"猫"和"狗"是两个不相关的 key，关于"猫"的统计对"狗"毫无帮助。词表外的词（OOV）概率为 0。模型只能记忆，不能泛化。

**KenLM**（2011）是 n-gram LM 的高效工程实现，今天仍在数据管道里用（ccNet 用它做困惑度过滤），但它是工程优化，不是范式突破。

详见 [困惑度](../20-concepts/perplexity.md)。

---

### 第二代：词向量（2003–2014）

**核心突破**：用稠密向量表示词，语义相近的词向量相近，解决了 n-gram 的"词之间相互独立"问题。

**NNLM**（Bengio et al., 2003）：用神经网络做语言模型，目标和 n-gram 一样，但为了让网络泛化，它必须把语义相近的词映射到相近的向量——**embedding 是为了泛化而自然涌现的**。

**Word2Vec**（Mikolov et al., 2013）：把 embedding 从语言模型里剥离，专门优化向量质量，用负采样大幅加速训练。

**fastText**（Joulin et al., 2016）：Word2Vec 的工程变体，加入字符 n-gram（subword）处理 OOV 和拼写变体，并极大提速。**fastText 没有理论突破，但速度使它成为大规模数据管道的标准工具**（语言识别、质量过滤）。它属于这一代，不是独立的一代。

**GloVe**（Pennington et al., 2014）：换了训练目标（全局词共现矩阵分解），产物类似 Word2Vec。

**遇到的坎**：这些都是**静态向量**——每个词只有一个向量，"苹果"（水果）和"苹果"（公司）是同一个向量，无法处理一词多义。向量是离线算好的，无法根据上下文动态调整。

详见 [Word Embedding](../20-concepts/word-embedding.md)、[fastText](../20-concepts/fasttext.md)。

---

### 第三代：预训练 + 上下文动态向量（2017–2019）

**核心突破**：同一个词在不同上下文里得到不同的向量；大规模无监督预训练 + 下游任务 fine-tune 成为标准范式。

**为什么需要这一代**：静态向量的根本问题不是精度不够，而是**表示能力的天花板**——一个词只能有一个意思，这在语言里根本不成立。解决这个问题需要模型在处理每个词时能"看到"整个上下文，而不是预先算好一个固定向量。

**Transformer**（Vaswani et al., 2017）：提出注意力机制，每个词能直接看到序列里任意位置的其他词，彻底解决了 RNN/LSTM 的长距离依赖问题。这是现代 LLM 的结构基础。

**ELMo**（Peters et al., 2018）：第一次用双向 LSTM 生成上下文相关的动态向量，同一个词在不同句子里得到不同向量，一词多义问题得到缓解。

**BERT**（Devlin et al., 2018，Google）：用 Transformer encoder 替代 LSTM，双向理解上下文，在海量文本上预训练后 fine-tune 到下游任务。掀起了"预训练 + fine-tune"范式，几乎所有 NLP 任务都受益。

**RoBERTa / DistilRoBERTa**（2019）：BERT 的改进和蒸馏版，工程优化，今天仍被 Llama 3 用于质量过滤分类器。

**GPT**（Radford et al., 2018）、**GPT-2**（2019）：同期走生成方向——Transformer decoder，单向，预测下一个词。和 BERT 是同一代的两个分支：BERT 走理解，GPT 走生成。

**遇到的坎**：BERT 系擅长理解，不擅长生成；GPT 系能生成，但规模还不够大，能力有限。更根本的问题是：**模型只会"预测下一个词"，不会"按人的意图回答问题"**——它生成的是"统计上合理的续写"，不是"有用的回答"。

---

### 第四代：规模扩张与涌现（2020–2022）

**核心突破**：规模扩大到足够大时，模型涌现出之前没有的能力（few-shot learning、chain-of-thought 推理）。

**GPT-3**（Brown et al., 2020，OpenAI）：1750 亿参数，few-shot learning 涌现——不需要 fine-tune，只需要在 prompt 里给几个例子，模型就能完成新任务。这是一个质变，不只是量变。

**Scaling Law**（Kaplan et al., 2020；Chinchilla, 2022）：系统研究模型大小、数据量、计算量三者的关系，给出"给定算力预算怎么分配最优"的答案。成为所有认真做预训练的团队的基础参考。

**PaLM**（Chowdhery et al., 2022，Google）：540B 参数，多步推理能力进一步涌现，提出 MFU（Model FLOPs Utilization）作为训练效率的标准度量。详见 [MFU](../20-concepts/mfu.md)。

**数据工程同步成熟**：规模扩大的同时，数据质量成为瓶颈。ccNet（2019）、Gopher/MassiveText（2021）、DCLM（2024）是这条线上的里程碑。

**遇到的坎**：模型越来越大，但它只是在"预测合理的续写"，**不会按人的意图行事**。给它一个问题，它可能给出一个统计上合理但完全没用的回答。需要一种方法让模型"对齐"人类意图。

---

### 第五代：对齐与指令跟随（2022–2023）

**核心突破**：通过人类反馈的强化学习（RLHF）让模型从"预测续写"转变为"按指令回答"，ChatGPT 的出现标志着 LLM 进入大众视野。

**InstructGPT / RLHF**（Ouyang et al., 2022，OpenAI）：用人类对模型输出的偏好标注，训练 reward model，再用 PPO 强化学习优化语言模型。模型从"写出统计上合理的文字"变成"写出人类认为有用的文字"。

**ChatGPT**（2022）：InstructGPT 的产品化，第一次让普通用户感受到 LLM 的能力。

**Llama**（Touvron et al., 2023，Meta）：开源了高质量基础模型，让学术界和中小团队能够在此基础上做 fine-tune 和研究，大幅降低了入门门槛。

**遇到的坎**：RLHF 需要大量人工标注，成本高；reward model 本身可能被"博弈"（reward hacking）；模型在对话上表现好，但**复杂推理任务**（数学、逻辑、多步规划）仍然很弱。

---

### 第六代：推理能力与 Test-Time Compute（2024–2026）

**核心突破**：模型不再只是"一次性生成答案"，而是学会"先思考再回答"，推理能力出现质变；inference 时的计算量成为新的可调旋钮。

**Chain-of-Thought**（Wei et al., 2022）：让模型在给出答案前先写出推理步骤，复杂推理能力大幅提升。看似简单的 prompt 技巧，实际上揭示了一个深层现象：模型有推理能力，但需要"空间"来展开。

**o1 / DeepSeek-R1**（2024–2025）：用强化学习训练模型的推理过程本身，而不只是最终答案。模型学会了"想更久才回答"，在数学、代码、科学推理上出现质变。

**LIMO / s1**（2025）：发现少量高质量推理数据就能激发强大的推理能力，挑战了"推理需要海量数据"的直觉。

**当前状态**：这一代仍在快速演化中，核心问题从"模型能不能推理"转向"怎么控制 test-time compute、怎么让推理更可靠、怎么评测推理能力"。

---

### 演化总结

每一代的分界线不是时间，而是**解决了什么本质问题**：

```
n-gram LM（1990s）
  问题：词之间独立，无泛化能力
    ↓
词向量 / Word2Vec / fastText（2003–2016）
  突破：语义相近的词向量相近
  问题：静态向量，一词一义，无法根据上下文调整
    ↓
Transformer / BERT / GPT（2017–2019）
  突破：上下文动态向量，预训练+fine-tune 范式
  问题：只会"预测续写"，不会"按意图回答"
    ↓
规模扩张 / GPT-3 / Scaling Law（2020–2022）
  突破：涌现能力，few-shot learning
  问题：不对齐人类意图，推理能力弱
    ↓
RLHF / ChatGPT / 指令跟随（2022–2023）
  突破：模型学会"有用地回答"
  问题：复杂推理仍然弱，reward hacking
    ↓
Chain-of-Thought / o1 / R1（2024–2026）
  突破：推理能力质变，test-time compute 可调
  当前前沿：如何让推理更可靠、更高效、更可评测
```

**贯穿始终的不变量**：所有这些模型，本质上都是在估计同一个条件概率 P(下一词 | 上下文)，区别只是用什么结构参数化它、用什么方式训练它、用什么目标对齐它。

---

## 先说结论

过去这段时间，LLM 领域最重要的变化，不是又多了几个模型名，而是 4 件事变得清晰了：

1. 开放模型的工程配方成熟了：不再只看架构，而是看数据、post-training、RL、评测一整套怎么拼。
2. reasoning 变成主线：从“会不会推理”转向“怎么用 test-time compute、RL、少量高质量数据把推理能力激发出来”。
3. 长上下文和 agent 从 demo 变成独立研究方向：不是单纯 RAG，而是上下文利用率、computer use、真实任务执行。
4. 评测体系明显升级：旧 benchmark 已经太饱和，大家开始用更难、更真实、更抗污染的评测。

## 不建议按年份补，建议按问题链补

### 1. 现代开放模型到底是怎么做出来的

建议阅读顺序：

1. **The Llama 3 Herd of Models**  
   链接：https://arxiv.org/abs/2407.21783  
   看点：顶级系统报告是怎么组织 pretrain、post-train、safety、tool use 的。
2. **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model**  
   链接：https://arxiv.org/abs/2405.04434  
   看点：MLA + MoE + 低成本高性能的工程思路。
3. **DeepSeek-V3 Technical Report**  
   链接：https://arxiv.org/abs/2412.19437  
   看点：auxiliary-loss-free load balancing、multi-token prediction，以及现代大模型系统优化。
4. **Tulu 3: Pushing Frontiers in Open Language Model Post-Training**  
   链接：https://arxiv.org/abs/2411.15124  
   看点：更像一份现代 post-training cookbook，把 SFT、偏好优化、RL、评测串起来。

### 2. 推理能力为什么成了主线

建议阅读顺序：

1. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking**  
   链接：https://arxiv.org/abs/2403.09629  
   看点：它认真讨论了“模型能不能先想再说”。
2. **s1: Simple test-time scaling**  
   链接：https://arxiv.org/abs/2501.19393  
   看点：把很多人对 o1 类模型的直觉，变成一个相对简单、可复现的 recipe。
3. **LIMO: Less is More for Reasoning**  
   链接：https://arxiv.org/abs/2502.03387  
   看点：挑战“推理一定需要海量 reasoning data”这个直觉。
4. **DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning**  
   链接：https://arxiv.org/abs/2501.12948  
   看点：让“RL 直接激发 reasoning”这条路线真正出圈。

### 3. 数据工程：过滤只是起点

完整内容见 → [LLM 数据工程：过滤只是起点](data-engineering-llm.md)

核心图景：**过滤/去重**（ccNet/DCLM）→ **合成**（Phi）→ **质量提升**（AlpaGasus/Deita）→ **混合配方**（DoReMi）→ **课程安排**（Annealing/LIMA）。

必读论文：DoReMi、Phi-3、LIMA、AlpaGasus、Deita。

### 4. 自动驾驶数据工程

完整内容见 → [自动驾驶数据工程](data-engineering-av.md)

核心范式：**数据引擎飞轮**——部署车队 → 触发器自动挖掘 hard case → 人工精标 → 训练更好的模型 → 回到第 1 步。和 LLM 数据工程是同一套思维在不同模态的应用。

必看：Karpathy CVPR 2021 Tesla FSD 演讲（非论文）。

### 5. alignment / reward model 的关注点怎么变了

建议阅读顺序：

1. **Self-Rewarding Language Models**  
   链接：https://arxiv.org/abs/2401.10020  
   看点：模型既当选手又当裁判，这个设定为什么有吸引力，也有哪些风险。
2. **RM-R1: Reward Modeling as Reasoning**  
   链接：https://arxiv.org/abs/2505.02387  
   看点：reward model 不再只是“打分器”，而是在往“会推理的 judge”演化。

### 4. 长上下文、agent、真实世界任务

建议阅读顺序：

1. **Gemini 1.5: Unlocking multimodal understanding across millions of tokens of context**  
   链接：https://arxiv.org/abs/2403.05530  
   看点：长上下文正式进入研究主线，而不只是 marketing。
2. **RULER: What’s the Real Context Size of Your Long-Context Language Models?**  
   链接：https://arxiv.org/abs/2404.06654  
   看点：模型“支持 128K/1M context”不等于它真的会利用这么长的上下文。
3. **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments**  
   链接：https://arxiv.org/abs/2404.07972  
   看点：先把 agent 应该怎么测立住，而不是先看各种框架 demo。
4. **SWE-Lancer: Can Frontier LLMs Earn $1 Million from Real-World Freelance Software Engineering?**  
   链接：https://arxiv.org/abs/2502.12115  
   看点：用真实经济价值来评估 coding agent。
5. **Humanity’s Last Exam**  
   链接：https://arxiv.org/abs/2501.14249  
   看点：更新你对“当前前沿模型到底强到哪里了”的感知。

## 如果时间很少，只读 8 篇

1. DeepSeek-V3  
   https://arxiv.org/abs/2412.19437
2. Tulu 3  
   https://arxiv.org/abs/2411.15124
3. s1  
   https://arxiv.org/abs/2501.19393
4. DeepSeek-R1  
   https://arxiv.org/abs/2501.12948
5. Gemini 1.5  
   https://arxiv.org/abs/2403.05530
6. OSWorld  
   https://arxiv.org/abs/2404.07972
7. Phi-3（数据工程/合成数据）  
   https://arxiv.org/abs/2404.14219
8. DoReMi（数据混合配方）  
   https://arxiv.org/abs/2305.10429

### 5. 评测体系本身是怎么演化的

旧 benchmark 饱和之后，评测本身成了一个独立研究方向。这条线值得单独看，因为它决定了"我们现在用什么标准判断模型强不强"。

建议阅读顺序：

1. **MMLU: Measuring Massive Multitask Language Understanding**  
   链接：https://arxiv.org/abs/2009.03300  
   看点：理解"多任务知识问答"这类 benchmark 的基本设计，以及为什么它后来会饱和。

2. **HELM: Holistic Evaluation of Language Models**  
   链接：https://arxiv.org/abs/2211.09110  
   看点：从单指标评测转向多维度、多场景的系统性评测框架，代表了"评测体系应该怎么设计"的一种成熟思路。

3. **RULER: What's the Real Context Size of Your LLMs?**  
   链接：https://arxiv.org/abs/2404.06654  
   看点：一个具体案例——"支持 128K context"不等于"会用 128K context"，展示了 benchmark 设计如何暴露模型的虚报能力。

4. **Humanity's Last Exam**  
   链接：https://arxiv.org/abs/2501.14249  
   看点：旧 benchmark 饱和后，评测社区如何构造更难、更抗污染的题目；同时更新你对"前沿模型现在到底强到哪里"的感知。

5. **OSWorld: Benchmarking Multimodal Agents for Open-Ended Tasks in Real Computer Environments**  
   链接：https://arxiv.org/abs/2404.07972  
   看点：从静态题目转向真实环境执行，代表 agent 评测的新方向——不问"模型知不知道"，而问"模型能不能做到"。

**这条线的核心认知**：benchmark 不是中立的测量工具，它的设计决定了什么能力被看见、什么能力被忽略。读 benchmark 论文，要同时问"它在测什么"和"它测不到什么"。

### 6. Scaling Law：现在还值得看吗

**短答案**：值得，但要带着批判性去看，而不是当成"规律"来记。

Scaling law 的核心问题是：给定计算预算，怎么分配模型参数量和训练 token 数，才能得到最好的模型？这个问题在 2024–2026 年仍然是基础性的，所有认真做预训练的团队都在跑 scaling law 实验。

但"scaling law 论文"这个类别里，质量差异很大：早期工作的结论已经被更新的数据推翻，有些结论只在特定规模范围内成立。读的时候要问：这个结论在什么规模、什么数据、什么架构下得出的？

建议阅读顺序：

1. **Scaling Laws for Neural Language Models**（Kaplan et al., OpenAI，2020）  
   链接：https://arxiv.org/abs/2001.08361  
   看点：最早系统研究"模型大小、数据量、计算量"三者关系的论文，提出了幂律关系。结论后来被 Chinchilla 部分推翻，但作为理解 scaling law 概念的起点仍然必读。

2. **Training Compute-Optimal Large Language Models**（Hoffmann et al., DeepMind，2022，即 Chinchilla）  
   链接：https://arxiv.org/abs/2203.15556  
   看点：推翻了 Kaplan 的结论——给定计算预算，应该同时增大模型和数据量，而不是只增大模型。"Chinchilla-optimal"成为此后几年预训练的标准参考点。

3. **Scaling Laws for Data Filtering**（DCLM，2024）  
   链接：https://arxiv.org/abs/2406.11794  
   看点：把 scaling law 的思路扩展到数据过滤策略——不同过滤方式在不同数据规模下的效果曲线。是 scaling law 方法论在数据工程方向的延伸。

**2026 年的视角**：Kaplan 和 Chinchilla 的结论都是在"用完整的互联网数据训练"的假设下得出的。当训练数据开始包含大量合成数据、高质量精选数据时，这些结论的适用范围变得更模糊。Llama 3 的做法（用小模型实验 + scaling law 预测大模型性能）代表了当前工业界最实用的应用方式，而不是直接套用 Chinchilla 公式。

## 最该更新的认知

1. 模型变强，现在更多是 post-training + RL + inference-time compute 的故事，不只是 pretraining。
2. 推理已经从 prompt trick 变成独立训练范式。
3. 长上下文不能只看支持多长，要看模型会不会用。
4. agent 研究不要先迷信框架，先看评测和 failure mode。
5. benchmark 本身也在演化：旧的饱和了，新的在往更难、更真实、更抗污染的方向走。读模型报告时，要先看它用的是什么评测，再看数字。
6. 数据工程不只是过滤和去重：混合配方（DoReMi）、合成数据（Phi）、课程安排（LIMA/Annealing）同样关键，而且这套逻辑在自动驾驶等其他领域有直接的同构对应。
