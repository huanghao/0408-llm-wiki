# 开放权重模型全景（2023–2026-05）

一句话总结：到 2026-05-01 为止，开放权重模型已经从“能本地跑的替代品”演变成一个完整生态，主线分成通用模型、推理模型、代码模型、完全开放模型四条。

## 时间边界

- 本页最后核对时间：`2026-05-01`
- 2025 年内的重要更新已补齐到 Q3。
- 2026 年截至 2026-05-01，没有看到足以重写格局的新一代通用开放权重基础模型公开发布；当前格局仍主要由 2025 年的 Qwen3、Gemma 3、DeepSeek-R1、OLMo 2 32B、Qwen3-Coder 这些发布定义。

## 什么是“开放权重”

“开放权重”（open-weight）不等于“开源”（open-source）：

| 维度 | 开源 | 开放权重 | 闭源 |
|------|------|---------|------|
| 模型权重 | 公开 | 公开 | 不公开 |
| 训练代码 | 公开 | 通常不公开 | 不公开 |
| 训练数据 | 公开 | 通常不公开 | 不公开 |
| 商用限制 | 无或宽松 | 常有限制或附加条款 | 付费/API |

真正“全链路开放”的仍然稀少。到 2026-05-01，AI2 的 OLMo / Tülu 路线依然是最接近“可复现开放科学”的代表；Meta Llama、Google Gemma、Qwen、DeepSeek 这些主流家族更准确地说是开放权重。

## 演进脉络

### 2023：奠基期

| 模型 | 厂商 | 参数量 | 许可证 | 意义 |
|------|------|--------|--------|------|
| Llama 2 | Meta | 7B / 13B / 70B | Meta Research License | 学界和中小团队第一次有了真正可用的大底座 |
| Mistral-7B | Mistral AI | 7B | Apache 2.0 | 证明高质量 7B 能压过更大的 Llama 2 13B |
| Mixtral-8x7B | Mistral AI | 8x7B，12.9B active | Apache 2.0 | MoE 进入开放模型主流 |

### 2024：从“能用”到“有分工”

| 模型 | 厂商 | 参数量 | 许可证 | 关键意义 |
|------|------|--------|--------|---------|
| Phi-3-mini | Microsoft | 3.8B | MIT | 小模型进入手机/端侧可用区间 |
| DeepSeek-V2 | DeepSeek | 236B MoE，21B active | MIT | 高性价比 MoE 开始成为一条独立路线 |
| Gemma 2 | Google | 9B / 27B | Gemma Terms | Google 首次给出有竞争力的开放小模型 |
| Qwen2 / Qwen2.5 | Alibaba | 0.5B–72B | Apache 2.0 | 多语言、中文、代码、数学一起做强 |
| Llama 3.1 | Meta | 8B / 70B / 405B | Llama 3.1 License | 405B 把开放权重上限抬到“可对标闭源前沿”的级别 |
| OLMo 2 | AI2 | 7B / 13B | Apache 2.0 | 权重、数据、代码、recipe 一起公开 |
| DeepSeek-V3 | DeepSeek | 671B MoE，37B active | MIT | 大规模 MoE 在开放阵营里成熟 |
| Phi-4 | Microsoft | 14B | MIT | 小中模型继续向推理任务逼近 |

### 2025：格局真正定型

| 日期 | 模型 | 厂商 | 关键点 |
|------|------|------|-------|
| 2025-01-20 | DeepSeek-R1 | DeepSeek | 开放推理模型成为独立赛道；R1 + 蒸馏版把“reasoning”带到 1.5B–70B |
| 2025-03-12 | Gemma 3 | Google | 单机可跑的多模态开放模型，1B / 4B / 12B / 27B 全系列覆盖端侧到工作站 |
| 2025-03-13 | OLMo 2 32B | AI2 | 第一个在官方口径里超过 GPT-3.5-Turbo 和 GPT-4o mini 的 fully open 32B |
| 2025-04-29 | Qwen3 | Alibaba | “thinking / non-thinking”混合模式、MoE 下放、小模型能力跃迁 |
| 2025-07-22 | Qwen3-Coder | Alibaba | 代码模型进入 agentic coding 阶段，开始正面打 SWE-Bench / Browser-Use / Tool-Use |

## 2026-05 的主流版图

### 1. 通用开放权重底座

如果今天只看“最值得拿来做二次开发的通用基础家族”，核心还是这几条：

| 家族 | 当前代表 | 适合场景 | 主要优势 | 主要短板 |
|------|----------|---------|---------|---------|
| Qwen | Qwen3-8B / 32B / 30B-A3B / 235B-A22B | 通用助手、推理、Agent、多语言 | 覆盖尺寸全；Apache 2.0；中英文都强；thinking 模式实用 | 体系复杂，dense / MoE / thinking 切换带来选型成本 |
| Gemma | Gemma 3 4B / 12B / 27B | 单卡部署、轻量多模态、英语为主应用 | 单加速器友好；官方量化版齐全；视觉能力直接可用 | 生态和社区二改不如 Qwen / Llama 广 |
| Llama | Llama 3.1 / 3.3 70B，3.2 1B / 3B | 英语通用场景、产业生态兼容 | 生态最成熟之一；推理框架、蒸馏、微调工具最全 | 许可证不是 OSI 意义上的开源；小模型已不再明显领先 |
| DeepSeek | DeepSeek-V3 | 高性能服务端、MoE 成本优化 | active 参数效率高；通用能力强 | 更偏服务端；本地部署和精调门槛高 |
| OLMo | OLMo 2 7B / 13B / 32B | 研究复现、训练 pipeline 学习 | 真正 fully open；适合研究训练配方 | 产品生态和社区部署热度不如主流开放权重家族 |

### 2. 推理模型已经独立成层

2024 年还可以把“推理增强”当成通用模型的一种特性；到 2025 以后不行了，因为推理模型已经形成单独产品层：

| 模型 | 路线 | 今天的定位 |
|------|------|-----------|
| DeepSeek-R1 | 纯 RL 强化 reasoning，配蒸馏小模型 | 开放推理模型的分水岭，尤其适合数学、代码、复杂分析 |
| QwQ-32B | 面向 reasoning 的专门变体 | 说明“32B 推理专模”本身已能成为一类产品 |
| Qwen3 | 在同一模型里混合 thinking / non-thinking | 把“推理模型”和“通用助手”开始合并 |

最重要的变化不是 benchmark，而是交互范式变了：用户开始显式管理“是否思考、思考多久、是否允许工具调用”。

### 3. 编码模型已经独立成一条主线

编码模型不再只是“通用模型顺带会写代码”，而是单独的数据、训练、评测、产品链路。

| 日期 | 模型 | 参数与形态 | 关键点 |
|------|------|-----------|-------|
| 2024-09-19 | Qwen2.5-Coder | 1.5B / 7B / 32B 首发，后续扩成 0.5B / 1.5B / 3B / 7B / 14B / 32B | 代码预训练规模拉到 5.5T tokens，128K context，Apache 2.0 为主 |
| 2024-11 | Qwen2.5-Coder 全系列 | 0.5B–32B | 把“本地代码助手”从 7B 拉到 14B / 32B 可选层级 |
| 2025-07-22 | Qwen3-Coder-480B-A35B-Instruct | 480B MoE，35B active | 256K 原生上下文，可外推 1M；SWE-Bench Verified、Agentic Tool-Use、Browser-Use 直接对标闭源 coding agent |

这条线说明了一个结构性事实：代码模型的竞争不再只是 pass@1，而是三件事一起看：

1. repo 级长上下文
2. 可执行验证驱动的 RL
3. agentic coding 中的规划、工具调用、环境交互

换句话说，代码模型已经从“补全器”进化成“软件工程代理底座”。

### 4. “完全开放”仍然是小众但重要的分支

如果你的目标是：

- 研究训练 recipe
- 复现实验结论
- 合法清晰地重训 / 继续训练
- 分析数据混合、对齐、RLVR 这类机制

那么 OLMo / Tülu 线的重要性仍高于很多更强但不透明的开放权重模型。

这也是为什么“开放权重最强是谁”和“最值得学习的开放模型是谁”通常不是同一个答案。

## 本地部署速查

量化原理单独见 [量化](../20-concepts/quantization.md)。这页只保留“什么规模大致跑得动”的部署感知。

| 规模 | 常见量化 | 大致显存/统一内存 | 典型设备 | 适合做什么 |
|------|---------|------------------|---------|-----------|
| 1B–4B | 4-bit | 1–4 GB | 手机、轻薄本 | 分类、改写、简单助手、本地隐私任务 |
| 7B–8B | 4-bit / FP16 | 6–16 GB | MacBook Pro、消费级 GPU | 日常代码、RAG、小 Agent |
| 12B–14B | 4-bit / FP16 | 10–28 GB | 24GB 显卡、高配 Mac | 更稳的代码和推理 |
| 30B–32B | 4-bit | 18–24 GB | RTX 4090 / 统一内存大 Mac | 高质量单机助手 |
| 70B | 4-bit / 多卡 | 40GB+ | A100 / 多卡工作站 | 企业级高质量推理 |
| 200B+ MoE | 4-bit / 服务端 | 通常走多卡 | 服务器集群 | 前沿开源服务化部署 |

## 现在最该记住的结论

1. 小模型不再只是“缩水版”。
   2025 年之后，4B / 8B / 14B 已经能覆盖大量真实工作流，尤其是代码、RAG、工具调用和结构化输出。

2. active 参数比总参数更重要。
   Qwen3-30B-A3B、Qwen3-235B-A22B、DeepSeek-V3 这些模型都说明，MoE 时代只看总参数很容易误判成本和能力。

3. 通用模型、推理模型、代码模型正在分层。
   “一个模型包打天下”的叙事越来越弱；真正可用的系统通常会按任务选模型层。

4. fully open 和 open-weight 是两种不同价值。
   前者更适合研究与复现，后者更适合追前沿能力和产业落地。

5. 2026-05 的开放阵营里，Qwen 线最完整。
   不是说它在所有单项都绝对第一，而是它同时覆盖了通用、推理、代码、MoE、小模型、多语言和 Agent 入口。

## 现状与影响

- 一句话定性：开放权重模型已经从“闭源模型的便宜替代品”变成独立技术生态；其中 Qwen / DeepSeek / Gemma / Llama 负责产品竞争，OLMo / Tülu 负责研究透明度。
- 目前是否还在普遍使用：是，而且在本地部署、私有化部署、企业二次开发、学术复现里比 2024 年更重要。
- 哪些路线被证明成立：
  - 小模型高质量路线成立
  - MoE 的成本效率路线成立
  - reasoning 作为后训练层成立
  - 代码模型 agent 化成立
- 哪些旧判断已经过时：
  - “开源模型只能当 GPT-3.5 替代品”已经过时
  - “7B 只能做玩具 demo”已经过时
  - “开放模型缺少推理能力”也已经过时
- 今天仍然没解决的问题：
  - 真正 fully open 的前沿模型数量仍太少
  - 最强开放权重模型往往许可证和数据透明度不足
  - 多模态开放权重生态仍没有文本模型那样稳定成熟

## 相关页面

- [Phi-2 / Phi-3](./phi-2-phi-3.md)
- [The Llama 3 Herd of Models](./llama-3-herd-of-models.md)
- [域专属管道：Code 和 Math](../20-concepts/domain-specific-pipeline-code-math.md)
- [量化](../20-concepts/quantization.md)

## 来源

- DeepSeek, “DeepSeek-R1 Release”, 2025-01-20  
  https://api-docs.deepseek.com/news/news250120
- Google, “Introducing Gemma 3: The most capable model you can run on a single GPU or TPU”, 2025-03-12  
  https://blog.google/innovation-and-ai/technology/developers-tools/gemma-3/
- Ai2, “OLMo 2 32B: First fully open model to outperform GPT 3.5 and GPT 4o mini”, 2025-03-13  
  https://allenai.org/blog/olmo2-32b
- Qwen Team, “Qwen3: Think Deeper, Act Faster”, 2025-04-29  
  https://qwenlm.github.io/blog/qwen3/
- Qwen Team, “Qwen2.5-Coder: Code More, Learn More!”, 2024-09-19  
  https://qwenlm.github.io/blog/qwen2.5-coder/
- Qwen Team, “Qwen2.5-Coder Series: Powerful, Diverse, Practical.”, 2024-11  
  https://qwenlm.github.io/blog/qwen2.5-coder-family/
- Qwen Team, “Qwen3-Coder: Agentic Coding in the World”, 2025-07-22  
  https://qwenlm.github.io/blog/qwen3-coder/

## 开放问题 / 下一步

- 2026 年内如果 Meta、Mistral、Google、Qwen 再发布新一代开放权重大模型，需要把“2025 定型”这一判断重新检查一次。
- 可以单独补一页“开放权重代码模型谱系”，把 Qwen-Coder、DeepSeek-Coder、Code Llama、StarCoder、OpenHands-style agent stack 放在一起比较。
