# Parameters vs Context vs Memory vs Skills in Agent Learning

一句话总结：LLM 本体的参数没有变，不代表 agent 系统没有学习；只要外部状态会被未来读取并稳定改变行为，它就承担了类似“外挂参数”的功能。

## Decision Frame

讨论“模型有没有学习”时，先区分两个层级：

- **Model-level learning**：模型权重发生变化，例如 pretraining、SFT、DPO、RLHF、LoRA。
- **System-level learning**：模型权重不变，但 agent 的 prompt、memory、skills、工具、文件、检索库或代码发生持久变化，导致未来行为改变。

很多混乱来自把这两层混在一起。严格说，memory / skill / repo 文件不是神经网络参数；但从 agent 系统的输入输出行为看，它们可以像参数一样存储经验、偏好和操作策略。

## Four Storage Layers

| 层级 | 存在哪里 | 是否持久 | 是否改变权重 | 典型例子 |
|---|---|---:|---:|---|
| Parameters | 模型权重 | 是 | 是 | pretraining、SFT、DPO、LoRA |
| Context | 当前上下文窗口 | 否 | 否 | 当前 prompt、临时对话历史、一次性 few-shot examples |
| External memory | 模型外存储 | 是 | 否 | 用户偏好、项目笔记、RAG 文档、任务历史 |
| Skills / tools / code | 文件或可执行系统 | 是 | 否 | `SKILL.md`、shell scripts、API tools、repo conventions |

这个表的关键点是：**是否改变权重** 和 **是否改变未来行为** 不是同一个问题。

Context 不改变权重，但会改变本轮行为。Memory / skills 不改变权重，但会改变未来多轮行为。Parameters 改变权重，是最强、最不可逆、也最难审计的一类学习。

## Synthetic Data Has a Signal Ceiling

Self-Instruct、MagPie、Alpaca、Instruction Tuning with GPT-4 这些 synthetic instruction data 路线，都可以理解为把已有信号转成训练数据。

两类主要路线：

- **自举 / self-synthesis**：模型把自己已经隐含掌握的能力结构化成 instruction-response 数据，例如 [Self-Instruct](../30-papers/self-instruct-2212.10560.md)、[MagPie](../30-papers/magpie-2406.08464.md)。
- **teacher-student 蒸馏**：更强 teacher 生成答案、比较或评分，训练较弱 student，例如 [Stanford Alpaca](../30-papers/stanford-alpaca.md)、[Instruction Tuning with GPT-4](../30-papers/instruction-tuning-with-gpt-4-2304.03277.md)。

这两类方法都有信号上限：

- 纯自举不会凭空创造模型没学过的知识。
- 纯 teacher-student 蒸馏通常不会稳定超过 teacher 的真实能力边界。
- 生成数据会继承 source model / teacher 的偏见、盲点、风格和错误。

但这个“上限”不是一个简单的分数天花板。Student 可能在某些 benchmark 上超过 teacher 或官方 instruct model，因为：

- 数据筛选去掉了 teacher 输出里的噪声。
- student 的 base model 预训练分布不同，原本就有某些能力。
- benchmark 偏好特定回答格式、长度或风格。
- 训练目标更集中，student 在窄域里表现更强。
- pipeline 引入了外部 verifier、执行器、人工反馈、多 teacher、搜索或 RL，此时已经不是纯蒸馏。

更准确的说法是：**纯自举/纯蒸馏不能创造超出信号源的信息，但可以重新组织、压缩、筛选、放大已有能力；一旦加入外部验证或环境反馈，就可能突破原 teacher 的局部上限。**

## When External State Becomes Learning

如果一个 agent 完成任务后写入：

```text
这个 repo 的测试入口是 `uv run pytest`。
用户偏好短回答，不要展开背景。
遇到 mdv 批注时，先 `comments get --json`，处理后 `reply-batch`。
```

下次 agent 读取这些文件并改变行为，那么系统发生了学习。这个学习不是 gradient descent，但它满足三个条件：

- **持久化**：信息跨会话保存。
- **可调用**：未来任务能检索、读取或执行它。
- **行为相关**：它会改变未来输出、工具调用或决策路径。

因此，memory / skill 更适合被称为 **external state** 或 **non-parametric memory**，而不是“训练数据”。训练数据只有在被训练、检索或放入上下文时才会影响行为；memory / skill 的特殊之处在于它们被设计为未来会被 agent 主动调用。

## Skills as Externalized Parameters

Skill 和普通文档的区别在于：skill 通常包含触发条件、操作流程、工具约束和决策规则。

例如一个 `SKILL.md` 可能写：

```text
当用户要求处理 mdv 批注时：
1. 先打开文件。
2. 用 `mdv comments get --json` 读取批注。
3. 直接修改文档。
4. 用 `reply-batch` 回复。
```

这已经不只是“知识”，而是可执行的 policy fragment。它不在模型权重里，但会像参数一样影响 agent 的 action selection。

所以可以把 agent 看成：

```text
Agent behavior = base model parameters + current context + external memory + skills + tools + environment feedback
```

其中 `base model parameters` 是固定内核，后面几项是可变外部状态。

## Similarities

Parameters、memory、skills 的共同点：

- 都能存储过去经验。
- 都能影响未来行为。
- 都能把一次任务里的发现转化为后续能力。
- 都可能引入错误、偏见或过时信息。

这就是为什么“agent 修改 skill 文件”在系统层面很像学习。

## Differences

| 维度 | Parameters | External memory | Skills |
|---|---|---|---|
| 更新方式 | 梯度下降 / 权重合并 | 写入、检索、编辑 | 写入流程、规则、脚本 |
| 可解释性 | 低 | 高 | 高 |
| 泛化方式 | 隐式泛化 | 依赖检索和上下文 | 依赖触发条件和流程匹配 |
| 更新成本 | 高 | 低 | 低到中 |
| 风险 | 难回滚、难审计 | 检索错、过时 | 规则僵化、错误自动化 |
| 生效范围 | 全局 | 被检索时 | 被触发时 |

最大区别：参数学习会改变模型的默认行为；memory / skills 只有在被读入或触发时才生效。

## Practical Implications

对 agent 系统，应该把“学习”看成多层状态更新，而不只看权重是否变化。

几条实用判断：

- 如果信息只在当前 prompt 里，叫 in-context adaptation。
- 如果信息写入 memory / 文件，下次可检索，叫 external-memory learning。
- 如果信息写成 skill / workflow，叫 policy-level externalization。
- 如果模型权重更新，叫 parametric learning。
- 如果 agent 能自动修改 memory / skill / repo conventions，它就是一个会自我更新的系统，即使 base model 没有训练。

这也解释了为什么 agent 项目里的 `AGENTS.md`、`SKILL.md`、tools、repo notes 很重要：它们是系统行为的一部分，不是普通说明文档。

## Open Questions

- 什么时候应该把经验写入参数，什么时候应该写入 skill / memory？
- 外部 memory 越来越多时，如何避免“外挂参数污染”？
- skill 是不是应该有版本、测试、回滚和 lint，就像代码一样？
- 如果 agent 能自动改自己的 skills，谁来防止错误规则被固化？
- 对长期 agent，评价能力时应该评估 base model，还是评估 model + memory + tools 的完整系统？

## Related Pages

- [Instruction Tuning](../20-concepts/instruction-tuning.md)
- [Self-Instruct](../30-papers/self-instruct-2212.10560.md)
- [MagPie](../30-papers/magpie-2406.08464.md)
- [Deita](../30-papers/deita-2312.15685.md)
- [Human Feedback vs AI Feedback vs Verification](./human-feedback-vs-ai-feedback-vs-verification.md)
- [Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)
