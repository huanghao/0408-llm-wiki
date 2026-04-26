# LLM 学习线路图（截至 2026-04-10）

基于之前的基础知识梳理，下面这份清单聚焦“从 2024 年中后期到 2026-04-10，这段时间 LLM 领域最值得补的主线”。

## 0. 前置：核心概念的演化脉络

在进入前沿论文之前，先理清这些基础概念的血缘关系。它们不是并列的，而是一代代演化出来的——每一代都在解决上一代的核心局限。

### 问题主线：如何让机器理解语言

```
如何表示词的含义？
    ↓
如何建模词序列的概率？
    ↓
如何在大规模数据上快速处理？
    ↓
如何让模型真正理解语义、支持长上下文？
```

---

### 第一代：统计语言模型（1990s–2000s）

**n-gram 语言模型**

核心思想：统计词序列的频率，用历史 n-1 个词预测下一个词。

```
P("趣" | "机器 学习 很 有") = count("机器 学习 很 有 趣") / count("机器 学习 很 有")
```

**局限**：词之间相互独立——"猫"和"狗"在模型里是完全不同的两个 key，关于"猫"的统计无法帮助"狗"。没有泛化能力，只能记忆见过的词序列。

**KenLM**（2011）：n-gram LM 的高效实现，用 trie + 量化压缩，几十亿条 n-gram 塞进几百 MB，查询速度极快。ccNet 用它做困惑度过滤，就是因为在 TB 级数据上速度是决定性的。详见 [困惑度](../20-concepts/perplexity.md)。

---

### 第二代：词向量（2003–2014）

**NNLM → Word2Vec → GloVe**

**NNLM**（Bengio et al., 2003）：用神经网络做语言模型，目标和 n-gram 一样（预测下一个词），但换了实现方式。关键发现：为了让网络泛化，它必须把语义相近的词映射到相近的向量——**embedding 是为了解决 n-gram 的泛化问题而自然涌现出来的**，不是独立发明的。

**Word2Vec**（Mikolov et al., 2013）：把 embedding 从语言模型里剥离出来，目的变成"只要好的向量"。用负采样大幅加速训练，Skip-gram 模式（给定中心词预测上下文）成为标准方法。

**GloVe**（Pennington et al., 2014）：换了训练目标（全局词共现矩阵的分解），但产物和 Word2Vec 类似，都是静态词向量。

**共同局限**：静态向量——每个词只有一个向量，"苹果"（水果）和"苹果"（公司）是同一个向量，无法处理一词多义。

详见 [Word Embedding](../20-concepts/word-embedding.md)。

---

### 第三代：大规模快速分类（2016–2017）

**fastText**（Joulin et al., 2016–2017，Facebook）

在 Word2Vec 基础上做了两件事：
1. 引入**字符 n-gram**（subword）作为特征，解决拼写变体和 OOV 问题
2. 把分类器做得极快——CPU 上每秒处理数十万条文本

fastText 不追求语义深度，追求速度。这使它成为大规模数据管道的标准组件：语言识别（lid.176）、质量过滤（Llama 3 的 Wikipedia 引用分类器）都用 fastText。

**血缘**：Word2Vec 的直系后代，同样用 embedding + bag-of-words，只是加了 subword 特征和更快的训练。

详见 [fastText](../20-concepts/fasttext.md)。

---

### 第四代：上下文动态向量（2018）

**ELMo → BERT**

**ELMo**（Peters et al., 2018）：第一次用双向 LSTM 生成**上下文相关的动态向量**——同一个词在不同句子里得到不同的向量，解决了一词多义问题。

**BERT**（Devlin et al., 2018，Google）：用 Transformer encoder 替代 LSTM，双向理解上下文，在大量文本上预训练后 fine-tune 到下游任务。掀起了"预训练 + fine-tune"范式。

**RoBERTa**（Liu et al., 2019，Facebook）：BERT 的改进版，主要优化训练方式（更大 batch、更多数据、去掉 NSP 任务），效果更好。Llama 3 用 RoBERTa-based 分类器做质量过滤。

**DistilBERT / DistilRoBERTa**（Sanh et al., 2019）：知识蒸馏压缩版，参数量约 1/3，速度快 60%，精度损失约 3%。Llama 3 用它在 TB 级数据上打质量分，速度是决定性因素。

**共同局限**：BERT 系是 encoder-only，擅长理解，不擅长生成；上下文窗口仍然有限（512 token）。

---

### 第五代：Transformer 与生成式预训练（2017–2020）

**Transformer**（Vaswani et al., 2017，Google）：提出注意力机制（Attention），让每个词能直接看到整个序列里的任意其他词，彻底解决了 RNN/LSTM 的长距离依赖问题。这篇论文是现代 LLM 的基础。

**GPT**（Radford et al., 2018，OpenAI）：用 Transformer decoder（单向，只看左边），做自回归语言建模（预测下一个词）。和 BERT 同期，但方向相反——BERT 走理解，GPT 走生成。

**GPT-2**（2019）、**GPT-3**（2020）：规模不断扩大，涌现出 few-shot learning——不需要 fine-tune，只需要在 prompt 里给几个例子，模型就能完成新任务。

**核心认知**：GPT 系和 n-gram LM 的目标完全一样（预测下一个词的条件概率），只是用 Transformer 替代了计数表，上下文从 n-1 个词扩展到整个序列。**本质上都是在估计同一个条件概率**，区别只是用什么结构来参数化它。

详见 [困惑度 → LLM 和 perplexity 的关系](../20-concepts/perplexity.md)。

---

### 第六代：数据工程成为核心（2019–至今）

随着模型架构趋于成熟，数据质量成了决定性因素。

**ccNet**（Wenzek et al., 2019，Facebook）：第一个系统化的大规模数据清洗 pipeline，提出用 KenLM 困惑度做质量过滤、MinHash 去重、行级去重三件套，成为后续所有数据管道的参考。

**Gopher / MassiveText**（Rae et al., 2021，DeepMind）：系统化的启发式过滤规则，提出重复 n-gram 覆盖率过滤，被 Llama 3 直接引用。

**Llama 3**（Meta, 2024）：综合了 ccNet、Gopher、fastText、RoBERTa 的所有方法，代表了当前工业级数据管道的最佳实践。

**DCLM**（Li et al., 2024）：第一次系统对比不同数据策略，固定模型只改数据，416 个实验证明数据策略可造成 35–44% 的精度差异。

---

### 演化总结

```
n-gram LM（1990s）
  ↓ 解决泛化问题
Word Embedding / Word2Vec（2013）
  ↓ 加速 + subword 特征
fastText（2016）
  ↓ 解决一词多义
BERT / 上下文向量（2018）
  ↓ 换成生成式 + 扩大规模
GPT / Transformer（2017–2020）
  ↓ 数据质量成为瓶颈
ccNet / 数据工程（2019–至今）
  ↓ 系统化对比数据策略
DCLM / Llama 3（2024）
```

每一步都在解决上一步的核心局限，但**目标始终是同一个：估计 P(下一词 | 上下文)**。

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

### 3. alignment / reward model 的关注点怎么变了

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

## 如果时间很少，只读 6 篇

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
