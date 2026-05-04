# Benchmark 速查表

wiki 里出现的评测集，按能力分类。"污染风险"指数据集测试集可能已出现在模型预训练语料中。

---

## 综合/语言理解

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **MMLU** | Massive Multitask Language Understanding | 57 个学科的多选题（高中/大学/专业级）| 5-shot 多选 | 中-高 | 高（题库公开）| 最常用的综合能力基准，Chinchilla/PaLM/Llama 3 都报告 |
| **BIG-bench** | Beyond the Imitation Game Benchmark | 150+ 多样任务，包括逻辑、翻译、数学、创意写作等 | few-shot，任务自定义 | 高（部分超人类）| 中 | 协作 benchmark，涵盖非常规任务，专门设计来挑战大模型 |
| **BIG-bench Hard (BBH)** | — | BIG-bench 中对 LLM 最难的 23 个子任务 | CoT few-shot | 高 | 低 | 常作为推理能力代理指标 |
| **SuperGLUE** | — | 8 项 NLU 任务：QA、推理、共指消解等 | fine-tune 或 few-shot | 中 | 高 | BERT 时代标准基准，现代大模型基本饱和 |
| **GLUE** | General Language Understanding Evaluation | 9 项 NLU 任务（更简单）| fine-tune | 低-中 | 高 | 已被 SuperGLUE 取代，现代模型轻松接近满分 |

---

## 常识推理

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **HellaSwag** | — | 给定情境选正确的后续句子（常识推理）| 0-shot 多选 | 中 | 高 | Dolma/Llama 消融实验最常用的代理指标，简单但对小模型区分度好 |
| **PIQA** | Physical Intuition QA | 物理常识问答（如"怎么切硬面包"）| 0-shot 多选 | 低-中 | 中 | 测物理世界直觉 |
| **OpenBookQA** | — | 小学科学问答 + 需要额外常识推理 | few-shot 多选 | 中 | 中 | |
| **ARC-Easy (ARC-E)** | AI2 Reasoning Challenge - Easy | 小学科学题（简单集）| 25-shot 多选 | 低 | 中 | |
| **ARC-Challenge (ARC-C)** | AI2 Reasoning Challenge - Challenge | 小学科学题（难集，模型和基于检索的系统都答错的题）| 25-shot 多选 | 中-高 | 中 | |
| **WinoGrande** | — | 填空式共指消解（Winograd schema 的大规模版本）| 0-shot | 中 | 中 | 测语言推理和常识 |
| **Winograd** | Winograd Schema Challenge | 原版共指消解问题 | 0-shot | 中 | 高（题目数量少）| WinoGrande 是其扩展版 |
| **LAMBADA** | — | 预测章节最后一个词（需理解长距离依赖）| 0-shot 完形填空 | 中 | 中 | |
| **StoryCloze** | — | 选择正确的故事结局 | 0-shot 多选 | 低-中 | 中 | |

---

## 阅读理解 / QA

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **SQuAD v1/v2** | Stanford Question Answering Dataset | 从维基百科段落里抽取式回答问题（v2 含无答案问题）| few-shot F1/EM | 中 | 高 | NLP 经典基准 |
| **DROP** | Discrete Reasoning Over Paragraphs | 段落理解 + 数值推理（加减计数等）| few-shot F1 | 中-高 | 中 | |
| **CoQA** | Conversational Question Answering | 对话式阅读理解 | few-shot F1 | 中 | 中 | |
| **QuAC** | Question Answering in Context | 对话式 QA，含弃权选项 | few-shot F1 | 高 | 低 | |
| **RACE-m/h** | ReAding Comprehension from Examinations | 中国初中（m）/高中（h）英语考试阅读理解 | few-shot | 中-高 | 低 | 注意和其他 QA 任务 setup 不完全兼容 |
| **TriviaQA** | — | 开放域问答（维基百科 + 网页）| closed-book EM | 中 | 高 | |
| **Natural Questions (NQ)** | — | 真实 Google 搜索问题 + 维基答案 | closed-book EM | 高 | 中 | |
| **Web Questions** | — | 基于 Freebase 的问答 | closed-book EM | 中 | 中 | |
| **BoolQ** | — | 段落 + 是/否问题 | 0-shot 二分类 | 中 | 中 | SuperGLUE 的子任务 |

---

## 自然语言推理（NLI）

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **ANLI (R1/R2/R3)** | Adversarial Natural Language Inference | 对抗性 NLI（专门让模型难以判断蕴含/矛盾/中立）| 0-shot 三分类 | 高 | 低 | 三个难度递增的 round |
| **RTE** | Recognizing Textual Entailment | 判断句子对是否存在蕴含关系 | few-shot | 中 | 高 | SuperGLUE 子任务 |
| **Copa** | Choice of Plausible Alternatives | 选择最合理的原因或结果 | 0-shot 多选 | 中 | 中 | SuperGLUE 子任务 |
| **WiC** | Word-in-Context | 判断同一个词在两个句子里是否同义 | few-shot | 中 | 中 | 一词多义理解 |
| **WSC** | Winograd Schema Challenge | 共指消解（SuperGLUE 版本）| few-shot | 中 | 高 | |
| **MultiRC** | Multi-Sentence Reading Comprehension | 从多段落里选择多个正确答案 | few-shot F1 | 中-高 | 中 | SuperGLUE 子任务 |
| **ReCoRD** | Reading Comprehension with Commonsense Reasoning | 新闻段落 + 完形填空 + 常识 | few-shot F1/EM | 中 | 中 | SuperGLUE 子任务 |
| **CB** | CommitmentBank | 短段落 + 三分类 NLI | few-shot | 中 | 高（题量少）| SuperGLUE 子任务 |

---

## 数学推理

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **GSM8K** | Grade School Math 8K | 小学数学应用题（需多步推理）| 8-shot CoT，EM | 中 | 高（2021 年发布，大量出现在网络）| 推理能力最常用代理，o1/R1/LIMO 必报 |
| **MATH** | — | 竞赛级数学题（AMC/AIME 等 5 个难度等级）| 4-shot CoT，EM | 很高 | 中 | 对顶级模型仍有区分度 |
| **AIME** | American Invitational Mathematics Examination | 高中数学竞赛（年度真题）| 0-shot，EM | 极高 | 低（年年更新）| o1/R1 的核心展示场景，LIMO 用 AIME24 |
| **NuminaMath** | — | 数学竞赛题合集（AMC/AIME/Olympiad）| — | 高 | 低 | 既是数据集也是评测集，LIMO 训练数据来源之一 |

---

## 代码生成

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **HumanEval** | — | 164 个 Python 函数补全（需通过单元测试）| pass@1 / pass@k | 中 | 高 | OpenAI 2021 发布，phi-1/Codex 的核心基准 |
| **MBPP** | Mostly Basic Python Problems | 374 个入门级 Python 编程题 | pass@1 | 中-低 | 高 | |

---

## 翻译 / 生成质量

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **ROUGE / ROUGE-L** | Recall-Oriented Understudy for Gisting Evaluation | 生成文本与参考文本的 n-gram 重叠 | 自动评测，非模型评测 | — | — | 摘要、翻译的自动评分指标，非 benchmark，属于指标 |
| **WMT** | Workshop on Machine Translation | 多语言机器翻译 | BLEU / COMET | 中-高 | 低（年年更新）| PaLM 多语言评测主要场景 |

---

## 领域多样性 / 困惑度

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **Paloma** | Perplexity Analysis for Language Model Assessment | 测模型对 120+ 个领域（新闻/论坛/学术/代码等）的困惑度分布 | perplexity（越低越好）| — | 低 | Dolma/DCLM 用来衡量数据多样性是否均衡，不是能力基准 |

---

## 偏见 / 安全

| 名称 | 全称 | 测什么 | 评测方式 | 难度 | 污染风险 | 备注 |
|---|---|---|---|---|---|---|
| **Winogender** | — | 职业性别刻板印象（代词共指）| 准确率 | 低 | 高 | PaLM 论文 Section 10 用来评测性别偏见 |
| **ToxiGen** | — | 模型生成的仇恨言论 | 分类器打分 | — | — | 安全评测，非能力基准 |
