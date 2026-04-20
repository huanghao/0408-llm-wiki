# LLM 学习线路图（截至 2026-04-10）

基于之前的基础知识梳理，下面这份清单聚焦“从 2024 年中后期到 2026-04-10，这段时间 LLM 领域最值得补的主线”。

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
