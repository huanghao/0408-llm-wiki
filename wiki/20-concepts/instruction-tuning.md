# Instruction Tuning（指令微调）

一句话总结：Instruction tuning 是让预训练语言模型从"续写文本"变成"执行指令"的核心后训练步骤，它的核心问题始终是数据从哪来、质量怎么控制。

---

## 核心问题

预训练语言模型（LLM）的训练目标是预测下一个 token，这让模型学到了大量知识和语言结构——但它不会"听指令"。给它发送一条"帮我写一封邮件"，它可能只是把这句话接着续写下去，而不是真的去写邮件。

Instruction tuning 解决的就是这个问题：

**用带有 instruction 和对应 response 的数据对模型做监督微调（SFT），让它学会"收到指令该怎么回应"。**

---

## 数据从哪来：三条路线

这是 instruction tuning 最核心的问题。

### 路线 1：人工标注

最早的做法（如 InstructGPT / ChatGPT 底层）是雇人工标注者写 prompt 并写出高质量回答，再做 SFT。

- 质量最高
- 成本极高（每条几美元，大规模不现实）
- 覆盖面受人力限制

### 路线 2：Self-Instruct（模型自举）

**[Self-Instruct (2212.10560)](../30-papers/self-instruct-2212.10560.md)** 提出：让模型自己生成 instruction + input + output，再过滤掉无效、重复的数据，用来微调自己。

核心流程：
1. 175 条人工 seed instructions
2. 让模型不断生成新指令
3. 判断 classification vs. non-classification task
4. 生成对应 input-output instance
5. 过滤无效样本，加入训练集

结果：GPT-3 在 SUPER-NATURALINSTRUCTIONS 上提升 33%，与 InstructGPT-001 差距仅剩 5%。

关键认知：**过滤比生成更重要**——没有过滤，模型会反复生成模板化垃圾。

**[MagPie (2406.08464)](../30-papers/magpie-2406.08464.md)** 把“自举生成 instruction”推进了一步：不再需要人工 seed tasks，也不需要 few-shot prompt，而是只输入 aligned chat model 的 user pre-query template，让模型自回归地产生用户问题，再生成回答。它说明 instruction 数据可以从 aligned LLM 学到的 chat template 分布中直接采样出来。

### 路线 3：Teacher 蒸馏

Self-Instruct 的"source = target"有上限：teacher 的错误和盲点会直接进入训练数据。改进思路是用更强的模型当 teacher。

**[Stanford Alpaca](../30-papers/stanford-alpaca.md)**（2023-03）：
- 复用 Self-Instruct 的 175 个 seed
- 让 `text-davinci-003`（GPT-3.5 级别）生成 52K (instruction, output) 对
- 微调 LLaMA 7B
- 总成本 < $600，让"人人都能做 instruction tuning"变成现实

**[Instruction Tuning with GPT-4 (2304.03277)](../30-papers/instruction-tuning-with-gpt-4-2304.03277.md)**（2023-04）：
- 沿用 Alpaca 的 52K 指令集
- 只换 teacher：从 GPT-3.5 换成 GPT-4 重新生成答案
- 结论：LLaMA-GPT4 (7B) 对 ChatGPT 的相对分数约 91%，teacher 质量直接影响 student 泛化

关键认知：**teacher quality matters**——同样的题目，更强的老师写答案，学生会明显变强。

---

## 数据格式的演化：单轮 → 多轮

早期 instruction tuning（Self-Instruct、Alpaca）主要是**单轮**：一条 instruction，一条 response。

**[Vicuna](../30-papers/vicuna-open-source-chatbot.md)**（2023-03）把训练数据换成 ShareGPT 上真实用户分享的多轮 ChatGPT 对话（~70K），让模型学到了"聊天助手"的风格，而不只是"指令执行器"的风格。

这一步的影响不仅在于 Vicuna 模型本身，还在于它把两件事推上主流：
1. 多轮对话数据比单轮 instruction 数据更接近真实使用分布
2. GPT-4-as-a-judge 作为评测手段（后来发展成 MT-Bench / Chatbot Arena）

---

## 从 SFT 到 Preference Learning

纯 SFT 的问题是：它告诉模型"什么是好的回答"，但不告诉模型"好的回答比差的回答好在哪"。

为了让模型真正学到人类偏好，需要：

1. 收集 **comparison data**：同一个 prompt，多个回答，人类标注哪个更好
2. 用 comparison data 训练 **reward model**
3. 用 reward model 做 RL 训练（PPO 等），或者直接用 DPO 绕过显式 RM

InstructGPT 是这条路线的完整实现。  
Instruction Tuning with GPT-4 则开了一个重要的口：**用强模型自动生成 comparison data**（RLAIF 路线的前身）。

完整对比见 [Human Feedback vs AI Feedback vs Verification](../40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md) 和 [RLHF](./rlhf.md)。

---

## 数据质量 vs. 数据量

**[LIMA (2305.11206)](../30-papers/lima-2305.11206.md)** 是这条线的早期核心论文：LLaMA 65B 只用 1,000 条精心策划的 demonstrations 做 SFT，就能在 2023 年的人类偏好评估中接近 GPT-4 / Claude / Bard，并超过 Alpaca 65B 和 DaVinci003。它提出 Superficial Alignment Hypothesis：预训练已经学到大部分知识和能力，alignment 主要教模型用什么格式和风格与用户交互。

**[Deita (2312.15685)](../30-papers/deita-2312.15685.md)** 把 instruction tuning 数据选择拆成复杂度、质量、多样性三维：先用 evolution-based ranking 训练 complexity / quality scorer，再按 `complexity * quality` 排序并用 embedding filter 控制冗余。它说明在已有大数据池中，自动选择 6K/10K 高价值样本可以接近甚至超过使用十几万样本的 SFT baseline。

**[LIMO (2502.03387)](../30-papers/limo-2502.03387.md)** 的一个极端案例说明了质量的重要性：只用 817 条高质量 SFT 数据，可以激发出很强的数学推理能力。

**[Phi-1 (2306.11644)](../30-papers/phi-1-2306.11644.md)** 的教科书质量合成数据路线也是同一方向的证据：7B tokens 教科书质量数据训出的 1.3B 模型可以超越 10 倍大的对手。

一般规律是：
- 数据越多越好，**前提是质量不下降**
- 数据去重和过滤在 instruction tuning 阶段同样关键
- 专项数据（代码、数学、agent）比通用数据更能快速提升具体能力

---

## 现状与影响

**一句话定性**：Instruction tuning 是现代 LLM post-training stack 的必经步骤，具体 recipe 仍在快速演化，但"用 (instruction, response) 对做 SFT"这件事本身已经是行业默认。

2026 年的主流做法是：
- 多来源、多格式混合数据（单轮 + 多轮 + 代码 + 数学 + 工具调用）
- 用更强闭源模型生成 + 人工核查或 verifier 过滤
- SFT 之后接 preference optimization（DPO / RLHF / RLAIF）
- 专项后训练（reasoning、coding）已经成为重要的独立阶段

原始 Self-Instruct / Alpaca recipe 已经不是最佳实践，但它们揭示的核心规律仍然成立：
- synthetic instruction data 可以有效
- teacher 质量是数据质量的天花板
- 数据过滤不可省略

---

## 关键论文阅读路径

按时间线，适合这样读：

| 论文 | arXiv | 关键贡献 |
|------|-------|---------|
| Self-Instruct | [2212.10560](../30-papers/self-instruct-2212.10560.md) | 合成 instruction data 可行，bootstrapping + filtering 的基本骨架 |
| Stanford Alpaca | [博客](../30-papers/stanford-alpaca.md) | 把 Self-Instruct 工程化，< $600 做出 instruction-following LLaMA |
| Instruction Tuning with GPT-4 | [2304.03277](../30-papers/instruction-tuning-with-gpt-4-2304.03277.md) | teacher 质量直接影响 student；machine-generated comparison data 可行 |
| Vicuna | [博客](../30-papers/vicuna-open-source-chatbot.md) | 多轮 chat data；GPT-4-as-a-judge 走上主流 |
| LIMA | [2305.11206](../30-papers/lima-2305.11206.md) | 1,000 条高质量 demonstrations 足以激活强预训练模型的 assistant 行为 |
| Deita | [2312.15685](../30-papers/deita-2312.15685.md) | 在已有 instruction 数据池中按复杂度、质量、多样性自动选高价值 SFT 样本 |
| MagPie | [2406.08464](../30-papers/magpie-2406.08464.md) | 不用 seed 和 prompt engineering，利用 chat template 从 aligned LLM 自合成 instruction data |
| LIMO | [2502.03387](../30-papers/limo-2502.03387.md) | 817 条高质量数据可激发复杂推理——质量极端重要 |

如果还想看更底层的 pipeline，可以再读：
- [RLHF](./rlhf.md)：完整偏好优化链路
- [Synthetic Data with Verification](./synthetic-data-with-verification.md)：加了 verifier 的合成数据路线
- [Human Feedback vs AI Feedback vs Verification](../40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md)：三类监督信号的横向对比

---

## 常见误区

**"更多数据总是更好"**：不对。低质量数据不只没用，还可能引入噪声。LIMO 的 817 条胜过大量低质量数据。

**"SFT 之后就完成对齐了"**：不对。SFT 只教会了 behavior imitation，不能保证 preference alignment、safety 或 factuality。

**"自举生成的数据等于人类标注"**：不对。teacher 模型的错误和偏见会直接进入训练集，这也是为什么后来都加了过滤和更强 teacher。

**"instruction tuning 和 RLHF 是互相替代的"**：不对。现代 post-training pipeline 几乎都是先 SFT（instruction tuning），再做 preference optimization，两步不互斥。
