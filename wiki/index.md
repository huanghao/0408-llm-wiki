# Wiki Index

This is the content-oriented entry point for the LLM wiki.

## Overview

- [Knowledge Base Overview](00-overview/knowledge-base-overview.md): repo purpose, structure, and operating model.

## Roadmaps

- [LLM Learning Roadmap (2026-04-10)](10-roadmaps/llm-learning-roadmap-20260410.md): main post-2024 reading path focused on open models, reasoning, alignment, long context, and agent evaluation.

## Concepts

**训练与优化**
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

**基础概念**
- [Tokenization](20-concepts/tokenization.md): BPE、WordPiece、SentencePiece
- [Word Embedding](20-concepts/word-embedding.md): 词向量基础
- [MFU](20-concepts/mfu.md): 模型 FLOPs 利用率
- [Ablation Study](20-concepts/ablation-study.md): 消融实验方法论
- [Synthetic Data with Verification](20-concepts/synthetic-data-with-verification.md): 合成数据与验证

## Papers

- [The Llama 3 Herd of Models](30-papers/llama-3-herd-of-models.md): a good first systems paper for building a modern LLM reading frame around data, scale, post-training, long context, and safety.

## Comparisons

- Pending.

## Open Questions

- [Active Questions](50-questions/active-questions.md): current unresolved questions worth revisiting while reading.

## Meta

- [Knowledge Base Conventions](90-meta/conventions.md): page rules and maintenance expectations.
- [Lint Report 2026-04-28](90-meta/lint-20260428.md): wiki health check findings and action items.
- [Log](log.md): chronological record of ingests and updates.
