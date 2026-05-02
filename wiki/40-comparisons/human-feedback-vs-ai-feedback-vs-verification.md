# Human Feedback vs AI Feedback vs Verification

一句话总结：这三条路线都在回答“后训练监督信号从哪里来”，但它们优化的目标完全不同：human feedback 优化人类偏好，AI feedback 优化强模型风格，verification 优化可验证正确性。

## 为什么值得单独比较

2023 以后，后训练已经不只是“拿一批 SFT 数据微调一下”。

真正决定模型风格和能力边界的，常常是**监督信号来自哪里**：

1. 人类偏好
2. 更强模型的反馈或输出
3. 外部 verifier 的可验证正确性

如果不把这三类信号分开，很容易混淆几件本来不同的事：

- chat alignment
- teacher distillation
- reasoning / code correctness optimization

## 三条路线分别在做什么

| 路线 | 监督信号来源 | 优化目标 | 代表工作 |
|------|-------------|---------|---------|
| Human Feedback | 人类写答案 / 人类排序偏好 | 更符合人类主观偏好 | InstructGPT, 经典 RLHF |
| AI Feedback | 更强模型生成答案、评分、比较 | 更像强 teacher 的行为 | Self-Instruct, Alpaca, Instruction Tuning with GPT-4, Constitutional AI 的一部分 |
| Verification | 执行器、形式化验证、答案校验器 | 客观正确性和可检验成功率 | DeepSeek-R1, 代码执行验证, 数学 verifier |

## 1. Human Feedback

## 核心思想

让人类直接告诉模型：

- 什么是好回答
- 哪个回答更好
- 什么回答有害、误导、无用

这是最直接的对齐方式，因为目标本来就是让模型符合人的偏好。

## 优点

- 对“有用、礼貌、自然、无害”这类主观标准最直接
- 在产品助手场景里通常最贴近真实用户体验
- 是 chat assistant 早期成功的核心来源

## 缺点

- 贵
- 慢
- 标注一致性差
- 很难规模化覆盖长尾场景

## 最适合什么任务

- 通用聊天
- 拒答与安全风格
- 多轮对话行为
- 主观质量优化

## 2. AI Feedback

## 核心思想

让更强模型或同一模型本身，来生成后训练数据：

- 生成 instruction-output pairs
- 生成 comparison data
- 给回答打分
- 给出自我修正意见

这条路线的核心不是“人类喜欢什么”，而是：

**把更强 teacher 的行为模式廉价地转移给更弱或更开放的 student。**

## 内部分成两类

### 2.1 自举型

代表是 [Self-Instruct](../30-papers/self-instruct-2212.10560.md)。

特点：

- source 和 target 基本是同一个模型
- 重点是把模型已有能力组织成 instruction data
- [MagPie](../30-papers/magpie-2406.08464.md) 是这条线的后续强化版：它不再依赖人工 seed tasks，而是通过 aligned model 的 chat template 直接采样用户指令分布。

### 2.2 蒸馏型

代表是：

- [Stanford Alpaca](../30-papers/stanford-alpaca.md)
- [Instruction Tuning with GPT-4](../30-papers/instruction-tuning-with-gpt-4-2304.03277.md)
- [Vicuna](../30-papers/vicuna-open-source-chatbot.md)
- [Deita](../30-papers/deita-2312.15685.md)

特点：

- teacher 更强
- student 更弱或更开放
- 目标是风格和能力蒸馏
- 也可以用 teacher / judge 来筛选已有 SFT 数据，而不只是生成新数据

## 优点

- 成本远低于纯人工标注
- 扩数据速度快
- 很适合把一个 base model 快速做成 assistant

## 缺点

- 会继承 teacher 的偏见、盲点、口癖
- 可能只是在模仿 style，不是真的理解
- judge 和 teacher 如果是同一个模型，容易自洽但不一定客观

## 最适合什么任务

- instruction following
- chat style imitation
- preference data 冷启动
- 开源 assistant 蒸馏

## 3. Verification

## 核心思想

不是问“人或 teacher 喜不喜欢这个回答”，而是问：

**它对不对？能不能被外部系统验证？**

典型 verifier：

- 代码执行器
- 数学答案检查器
- 形式化证明器
- 单元测试
- 可验证环境回报

## 优点

- 信号最客观
- 很适合推理、数学、代码
- 可以支持 RL 规模化优化

## 缺点

- 只适用于“可验证”的任务
- 对开放式写作、复杂主观对话帮助有限
- answer-level verification 常常不能保证 process-level correctness

## 最适合什么任务

- 代码
- 数学
- 可验证推理
- 工具调用成功率

## 三者最关键的区别

| 问题 | Human Feedback | AI Feedback | Verification |
|------|----------------|------------|-------------|
| 谁给信号？ | 人 | 模型 | 外部规则 / 执行环境 |
| 优化什么？ | 主观偏好 | teacher 行为 | 客观正确性 |
| 最强场景 | 通用助手 | 蒸馏 assistant | reasoning / code |
| 最大问题 | 成本 | teacher 偏见 | 任务覆盖窄 |

## 什么时候该用哪一种

## 如果目标是“像个好助手”

优先：

1. human feedback
2. AI feedback

verification 通常不是主线。

## 如果目标是“像强模型一样说话”

优先：

1. AI feedback
2. 少量 human feedback 校正

## 如果目标是“答案必须做对”

优先：

1. verification
2. 再辅以 AI / human feedback 调整表达风格

## 2026 视角下的主流组合

今天真正有效的 post-training，通常不是三选一，而是组合：

1. **AI feedback 做大规模冷启动**
   快速生成 instruction / preference / style data

2. **Human feedback 做高价值校正**
   修正 safety、helpfulness、拒答边界、产品语气

3. **Verification 做专项强化**
   在数学、代码、agent 环境里把客观正确率继续往上推

所以现代 post-training 更像：

**AI feedback 负责规模，human feedback 负责偏好，verification 负责硬正确性。**

## 现状与影响

- 一句话定性：这不是三条互斥路线，而是现代后训练 stack 的三种监督信号源。
- 目前是否还在普遍使用：是，而且三者都在 2026 年的主流系统里同时存在。
- 哪条在扩张最快：
  - 2023 是 AI feedback 爆发
  - 2024–2025 是 verification 在 reasoning / code 上爆发
  - human feedback 没消失，但从“全量主力”变成“高价值稀缺信号”
- 最容易犯的错误：
  - 把 AI feedback 当成 human preference
  - 把 verification 当成通用对齐
  - 把某条路线在一个子域的成功，误认为它能统一解决所有 post-training 问题

## 相关页面

- [RLHF](../20-concepts/rlhf.md)
- [Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)
- [Self-Instruct](../30-papers/self-instruct-2212.10560.md)
- [MagPie](../30-papers/magpie-2406.08464.md)
- [Stanford Alpaca](../30-papers/stanford-alpaca.md)
- [Instruction Tuning with GPT-4](../30-papers/instruction-tuning-with-gpt-4-2304.03277.md)
- [Vicuna](../30-papers/vicuna-open-source-chatbot.md)
- [Deita](../30-papers/deita-2312.15685.md)

## 开放问题

- 哪些能力最依赖 human feedback，不能被 AI feedback 替代？
- verification 能否扩展到更开放的 agent 任务，而不只是在代码/数学里有效？
- 当 judge model 也是 teacher 时，如何避免评测和训练相互污染？
