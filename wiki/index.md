# Wiki Index

This is the content-oriented entry point for the LLM wiki.

## Overview

- [Knowledge Base Overview](00-overview/knowledge-base-overview.md): repo purpose, structure, and operating model.
- [AV Data Pipeline Architecture](00-overview/av-data-pipeline-architecture.md): 自动驾驶数据处理全链路架构概览
- [PNC Model Architecture](00-overview/pnc-model-architecture.md): PNC 神经网络模型的架构、规模与训练数据
- [Tesla Data Engine](00-overview/tesla-data-engine.md): Karpathy 在 Tesla AI Day 演讲中描述的数据飞轮范式

## Roadmaps

- [LLM Learning Roadmap (2026-04-10)](10-roadmaps/llm-learning-roadmap-20260410.md): main post-2024 reading path focused on open models, reasoning, alignment, long context, and agent evaluation.
- [LLM 数据工程路线图](10-roadmaps/data-engineering-llm.md): 过滤、去重、数据混合配方、合成数据的学习路径
- [AV 数据工程路线图](10-roadmaps/data-engineering-av.md): 自动驾驶数据飞轮、标注体系、传感器融合的学习路径

## Concepts

**训练与优化**
- [Instruction Tuning](20-concepts/instruction-tuning.md): 指令微调的数据来源、演化路线与关键论文导读
- [Loss Functions](20-concepts/loss-functions.md): NLL、KL、DPO、REINFORCE 的选择决策树
- [RLHF](20-concepts/rlhf.md): InstructGPT、DPO、PPO、GRPO 的完整对比
- [REINFORCE](20-concepts/reinforce.md): 策略梯度基础算法，奖励/Return/Advantage 的区别
- [PPO 逐行讲解](20-concepts/ppo-explained.md): PPO 的梯度机制与监督学习的区别

**数据处理**
- [ccNet](20-concepts/ccnet.md): 困惑度过滤 + 去重 pipeline
- [MinHash](20-concepts/minhash.md): 近重复文档检测
- [fastText](20-concepts/fasttext.md): 语言识别与质量分类
- [Perplexity](20-concepts/perplexity.md): 语言模型困惑度及 KenLM
- [FineWeb](20-concepts/fineweb.md): HuggingFace 高质量网页数据集
- [DCLM](20-concepts/dclm.md): DataComp-LM 数据过滤框架
- [DolmIno](20-concepts/dolmino.md): 数据混合配方研究
- [Domain-Specific Pipeline](20-concepts/domain-specific-pipeline-code-math.md): 代码与数学数据处理
- [t-SNE and Embedding Visualization](20-concepts/tsne-dimensionality-reduction.md): 高维 embedding 可视化方法，适合看局部邻域和覆盖，不适合直接证明数据质量

**部署与推理**
- [ONNX](20-concepts/onnx.md): 跨框架模型中间格式，导出、运行时、图优化、量化全览
- [量化（Quantization）](20-concepts/quantization.md): 低精度部署为什么是开放权重生态的基础设施

**推理与搜索**
- [Monte Carlo Tree Search（MCTS）](20-concepts/mcts.md): UCB 平衡探索利用的树搜索算法，AlphaGo 核心组件，LLM 推理时用于结构化的 test-time compute 搜索

**Scaling 基础**
- [幂律与 Scaling（Power Law）](20-concepts/power-law-and-scaling.md): 幂律是什么、翻倍法则、在 LLM/语言/城市/地震等领域的普遍出现，以及为何指数小意味着收益递减

**基础概念**
- [Tokenization](20-concepts/tokenization.md): BPE、WordPiece、SentencePiece
- [Word Embedding](20-concepts/word-embedding.md): 词向量基础
- [MFU](20-concepts/mfu.md): 模型 FLOPs 利用率
- [Ablation Study](20-concepts/ablation-study.md): 消融实验方法论
- [Synthetic Data with Verification](20-concepts/synthetic-data-with-verification.md): 合成数据与验证
- [Minimax 博弈](20-concepts/minimax.md): 博弈论中对抗框架，GAN / RLHF reward hacking 的理论基础
- [随机森林（Random Forest）](20-concepts/random-forest.md): 集成学习方法，bagging + 特征随机性的决策树集成

## Papers

- [The Llama 3 Herd of Models](30-papers/llama-3-herd-of-models.md): a good first systems paper for building a modern LLM reading frame around data, scale, post-training, long context, and safety.
- [Data Mixing Laws](30-papers/data-mixing-laws-2403.16952.md): 用指数函数拟合数据配比与验证损失的定量关系，嵌套 Scaling Laws 预测 1B 模型最优配比，ICLR 2025
- [Phi-1](30-papers/phi-1-2306.11644.md): 1.3B 参数 + 7B 教科书质量数据，HumanEval 50.6%，超越 10 倍大的模型
- [Phi-2 / Phi-3](30-papers/phi-2-phi-3.md): 教科书质量路线扩展到通用推理，phi-3-mini 3.8B 匹敌 GPT-3.5，可本地运行于手机
- [DoReMi](30-papers/doremi-2305.10429.md): 用小代理模型自动优化预训练数据 domain 配比，280M→8B 加速 2.6×
- [LIMO](30-papers/limo-2502.03387.md): 817 条高质量 SFT 数据激发强数学推理能力
- [Quiet-STaR](30-papers/quiet-star-2403.09629.md): 让模型在每个 token 处静默思考，从普通文本中自发学习推理
- [nuScenes](30-papers/nuscenes-1903.11027.md): 自动驾驶多传感器数据集，360° 全向感知基准
- [nuPlan](30-papers/nuplan-2106.11810.md): 闭环 ML-based 自动驾驶规划基准，10,000+ 小时真实驾驶日志
- [DCLM](30-papers/dclm-2406.11794.md): 固定模型只改数据，系统对比数据过滤策略的影响
- [Gopher](30-papers/gopher-2112.11446.md): DeepMind 280B 模型，重复 n-gram 过滤方法被 Llama 3 引用
- [Instruction Tuning with GPT-4](30-papers/instruction-tuning-with-gpt-4-2304.03277.md): 首次系统验证用 GPT-4 生成指令数据和比较数据来蒸馏开源 assistant
- [Self-Instruct](30-papers/self-instruct-2212.10560.md): instruction tuning 合成数据路线起点，用模型自己生成 instruction / instance 再对齐自己
- [Unnatural Instructions](30-papers/unnatural-instructions-2212.09689.md): 15 个种子样本 → LLM 全自动生成 24 万条指令数据，合成数据媲美人工众包的实证
- [AlpaGasus](30-papers/alpagasus-2307.08701.md): 用 ChatGPT 对 Alpaca 52k 数据打分，只取 9k 高质量样本训练反超原版——"数据质量 > 数据数量"的早期实证
- [Stanford Alpaca](30-papers/stanford-alpaca.md): 把 Self-Instruct 工程化成低成本、可复现的开源 instruction-tuning recipe
- [Vicuna](30-papers/vicuna-open-source-chatbot.md): 用 ShareGPT 多轮对话把开源 assistant 从 instruction 模式推进到 chat 模式
- [LIMA](30-papers/lima-2305.11206.md): LLaMA 65B + 1,000 条精选 demonstrations，无 RLHF 也能激活强 assistant 行为，NeurIPS 2023
- [Deita](30-papers/deita-2312.15685.md): 用复杂度、质量、多样性三维自动筛选 6K/10K instruction tuning 数据，ICLR 2024
- [MagPie](30-papers/magpie-2406.08464.md): 只用 aligned LLM 的 chat template 触发自生成用户指令，百万级合成 alignment 数据，ICLR 2025
- [Scaling Laws for Neural Language Models (Kaplan et al.)](30-papers/scaling-laws-neural-lm-2001.08361.md): loss 与 N/D/C 各呈幂律；固定计算应优先扩大模型（Kaplan rule），被 Chinchilla 修正为等比例，2020
- [Chinchilla Scaling Laws](30-papers/chinchilla-2203.15556.md): 固定 FLOPs 下参数量与 token 数应等比例扩大（~20× rule），当时大模型普遍过大欠训；70B Chinchilla 以同等计算全面超越 280B Gopher，NeurIPS 2022
- [Scaling Data-Constrained Language Models](30-papers/scaling-data-constrained-lms-2305.16264.md): 数据受限场景下重复数据最多 4 epoch 几乎无损，超过后收益递减；最优分配应多加 epoch 少加参数，NeurIPS 2023
- [Are Emergent Abilities a Mirage?](30-papers/emergent-abilities-mirage-2304.15004.md): 涌现能力是评估指标非线性的人工产物，换用线性指标即变为平滑提升；数学证明 + 视觉模型实验，NeurIPS 2023 Oral
- [PaLM: Scaling Language Modeling with Pathways](30-papers/palm-2204.02311.md): Google 540B 密集 Transformer，Pathways 系统 6144 TPU v4 训练，提出 MFU 效率度量，CoT 推理和多语言 SOTA，2022
- [Dolma: an Open Corpus of Three Trillion Tokens](30-papers/dolma-2402.00159.md): AI2 发布的 3T token 开放英语预训练语料库，配套开源 Dolma Toolkit，OLMo 系列数据基础，数据策划过程最透明的同类语料库，ACL 2024
- [开放权重模型全景（2023–2026-05）](30-papers/open-weight-models-landscape.md): 从 Llama 2 到 Qwen3 / DeepSeek-R1 / OLMo 2 / Qwen3-Coder 的开放权重生态总览

## Comparisons

- [Human Feedback vs AI Feedback vs Verification](40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md): 后训练三类监督信号源的目标、优劣和适用任务对比
- [参数量 vs 能力：2023–2026 的时间压缩](40-comparisons/parameter-vs-capability-over-time-2023-2026.md): 公开参数量模型里，达到相近可用能力所需参数量如何随时间下降
- [Parameters vs Context vs Memory vs Skills](40-comparisons/parameters-context-memory-skills-agent-learning.md): agent 系统里参数、上下文、外部记忆和 skill 的学习边界

## Open Questions

- [Active Questions](50-questions/active-questions.md): current unresolved questions worth revisiting while reading.

## Meta

- [论文名称词源](90-meta/glossary-names.md): Gopher/Dolma/DoReMi/LIMA/Phi 等名字的含义、梗和来源，按字母序排列
- [研究者简介](90-meta/glossary-people.md): wiki 高频作者，按研究方向分组（Scaling/数据/对齐/推理/小模型/系统）
- [Benchmark 速查表](90-meta/glossary-benchmarks.md): wiki 里出现的评测集，按能力分类（综合/常识/QA/推理/NLI/数学/代码），含测什么、评测方式、污染风险
- [数据集速查表](90-meta/glossary-datasets.md): 预训练语料、指令微调数据集、评测数据集，含规模、来源机构、被哪些模型使用
- [机构与实验室速查](90-meta/glossary-orgs.md): Google DeepMind/OpenAI/Anthropic/Meta/AI2 等机构历史、分拆关系、代表工作，以及常见混淆点
- [Knowledge Base Conventions](90-meta/conventions.md): page rules and maintenance expectations.
- [Lint Report 2026-04-28](90-meta/lint-20260428.md): wiki health check findings and action items.
- [Lint Report 2026-05-03](90-meta/lint-20260503.md): wiki health check findings and action items.
- [Log](log.md): chronological record of ingests and updates.
