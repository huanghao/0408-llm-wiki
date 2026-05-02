# Instruction Tuning with GPT-4

一句话总结：这篇 2023 年 Microsoft 论文证明了一个后来几乎成为行业默认做法的命题：用更强的闭源模型生成指令跟随数据，可以把较小的开放模型快速推到可用的 assistant 水平。

**论文**：*Instruction Tuning with GPT-4*  
**作者**：Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, Jianfeng Gao  
**机构**：Microsoft Research  
**arXiv**：2304.03277  
**时间**：2023-04-06  
**补充来源**：Microsoft Research 页面、项目页

## 背景概念

读这篇论文前，三个概念值得先确认一下：

**`text-davinci-003`** 是 OpenAI 2022 年底发布的 GPT-3.5 系列 API 模型，基于 175B 的 GPT-3 做了 instruction fine-tuning 和 RLHF，是同期 ChatGPT 底层的 API 版本。它能跟随指令、写代码、回答问题，已是当时最好的公开可用模型之一——但推理深度、长文一致性、拒绝有害请求方面明显弱于 GPT-4。

**Instruction-following data（指令跟随数据）** 是 (instruction, response) 对的集合，用来教模型"收到自然语言指令该怎么回答"。举两个例子：

```
instruction: 把下面这句话翻译成法语：The weather is nice today.
response:    Le temps est beau aujourd'hui.
```

```
instruction: 写一个 Python 函数，接受一个列表并返回其中的偶数。
response:    def get_evens(lst): return [x for x in lst if x % 2 == 0]
```

这类数据的价值在于：把一个"预测下一个 token"的预训练模型，变成能跟随用户指令的 assistant。

**GPT-3.5 到 GPT-4 的差距**：GPT-4 于 2023 年 3 月发布，在推理（数学、逻辑、多步问题）、代码质量、长文一致性上相对 GPT-3.5 有系统性提升，多数 benchmark 上得分高出一大截。这意味着 GPT-4 生成的"示范回答"在质量和准确度上普遍更好——这正是这篇论文的核心假设。

---

## 核心问题

当时 open-source instruction tuning 的主流做法是：

- 用人工标注数据做 SFT，成本高（人工标注每条通常数美元，52K 条约需几十万元人民币级别；用强模型 API 生成同规模数据成本低约 1–2 个数量级——论文没有给出精确对比数字，但这是定性共识）
- 或像 Alpaca 一样，用 `text-davinci-003` 生成 52K 指令数据，再微调 LLaMA

这篇论文问的是一个更直接的问题：

**如果把 teacher 从 GPT-3.5 级别换成 GPT-4，生成出来的 instruction-following data 是否足够好，好到能明显提升较小开放模型的泛化能力？**

它的意义不在算法创新，而在于把“更强 teacher 蒸馏出更强 student”这条路，在 instruction tuning 场景里第一次系统跑通。

## 方法 / 核心机制

## 1. 不重新造 instruction set，直接复用 Alpaca 的 52K 指令

**Alpaca 的 52K instructions** 来自 Stanford Alpaca 项目（2023 年 3 月）。他们用 Self-Instruct 方法——把少量人工写的示例喂给 `text-davinci-003`，让模型不断生成新的指令——最终产出约 52,000 条 (instruction, input, output) 三元组，涵盖翻译、写作、问答、代码等各类任务，整个生成过程花了大约 $500 美元的 API 费用。这 52K 条数据是当时最广泛使用的开源 instruction dataset。

这篇论文没有自己从头构造新指令，而是拿来 Alpaca 的 52K 条问题/指令，只把负责”写答案”的模型从 `text-davinci-003`（GPT-3.5 级别）换成 GPT-4——相当于用同一份题目，让更强的老师重新写了一遍答案。

这样做的好处是：

- instruction 集合基本相同
- prompt 模板基本相同
- 最大变化是 teacher model 质量

因此实验结论更接近”teacher 变强会不会直接提升 student”——变量被控制得很干净。

## 2. 生成四类 GPT-4 数据

论文实际构造了四类数据资产：

1. **English instruction-following data**  
   对 Alpaca 的 52K instruction，用 GPT-4 生成英文回答。

2. **Chinese instruction-following data**  
   先用 ChatGPT 把 52K instruction 翻成中文，再让 GPT-4 用中文回答。

3. **Comparison data**  
   让 GPT-4 给不同模型回答打分和比较，包括 GPT-4、GPT-3.5、OPT-IML 的输出，用来训练 reward model。

4. **Answers on Unnatural Instructions**  
   **Unnatural Instructions**（Honovich et al., 2022）是一个与 Alpaca 52K **完全独立**的数据集，用不同的方法（向模型提供"不自然"的奇特提示来激发多样化指令）生成了约 240K 条 (instruction, input, output) 对，覆盖的任务类型更广。这篇论文取其 core dataset（约 15K 条）上额外解码 GPT-4 回答，用来做大规模行为对照，**不用于训练主要 student 模型**。

这四类数据分别对应四个目标：

- 做 SFT
- 看跨语言泛化
- 做 reward modeling
- 看 student 是否在行为上逼近 teacher

## 3. 训练两个 instruction-tuned LLaMA 7B

论文训练了两个核心 student：

- `LLaMA-GPT4`：用 52K 英文 GPT-4 数据微调
- `LLaMA-GPT4-CN`：用 52K 中文 GPT-4 数据微调

这里的重点不是模型规模，而是对比：

- 同样是 `LLaMA 7B`
- 只比较喂 GPT-4 数据和喂 GPT-3 数据时，student 会差多少

## 4. 用 GPT-4 生成 comparison data，再训练 reward model

他们还用 GPT-4 生成偏好比较数据，并训练一个基于 `OPT 1.3B` 的 reward model。

这件事的研究意义大于结果本身：

- 之前开源 instruction-tuning 工作多数停在 SFT
- RLHF 很贵，主要贵在 comparison data
- 这篇论文把“用强模型生成 comparison data”明确提了出来

这可以看作后续 **RLAIF / AI feedback / machine-generated preference data** 路线的早期文本版原型之一。

## 关键结果 / 数据

## 1. GPT-4 数据明显优于 GPT-3 数据

在人类评测中，论文按 Anthropic 的 HHH 三条标准评估：

- Helpful
- Honest
- Harmless

对比 `LLaMA-GPT4` 和 Alpaca（可理解为 `LLaMA-GPT3`）时：

- 在 **Helpfulness** 上，`LLaMA-GPT4` 获得 **54.12%** 选票
- Alpaca 只有 **19.74%**
- 在 **Honesty** 和 **Harmlessness** 上，最大票数都落在 tie，但 Alpaca 略占优

这个结果说明：

- GPT-4 teacher 最大的提升首先体现在“有用性”
- 不是所有对齐维度都自动一起上升

## 2. 7B 的 LLaMA-GPT4 已经逼近 GPT-4 的回答风格

论文还直接拿 `LLaMA-GPT4` 和 GPT-4 做人类比较。结果不是 student 超过 teacher，而是：

- 在 HHH 三条标准上，`LLaMA-GPT4` 与 GPT-4 **表现相近**

这当然不能理解为“7B 等于 GPT-4”，更准确的理解是：

**对于这批 instruction-following evaluation tasks，student 已经能学到 teacher 的相当一部分行为风格。**

## 3. GPT-4 as judge 下，7B student 超过一些更大的开源对手

在 Vicuna 使用过的 80 个 unseen questions 上，论文让 GPT-4 做自动评测。

核心结果：

- `LLaMA-GPT4 (7B)` 对 ChatGPT 的相对分数约 **91%**
- `LLaMA-GPT4 (7B)` 对 GPT-4 的相对分数约 **83%**
- 用 reward model 做 best-of-5 排序后，最好一组可到：
  - 对 ChatGPT **94%**
  - 对 GPT-4 **87%**

同时：

- `LLaMA-GPT4 7B` 明显强于 `Alpaca 13B`
- 也强于未 instruction-tune 的 `LLaMA 13B`

这件事在 2023 年是很强的信号：**数据 teacher 的质量，足以压过一部分 student 参数量差距。**

## 4. 中文能力可以通过“翻译 instruction + GPT-4 中文回答”启动

论文还测了中文 instruction following。

主要结论不是“中文做得很好”，而是：

- 这条路径确实能让模型进入中文 instruction-following 场景
- 但 GPT-4 自身英文强于中文，因此中文结果整体弱于英文

也就是说，这篇论文更像是证明“跨语言可迁移”，不是证明“高质量中文对齐已经解决”。

## 局限性

## 1. 它没有解决 instruction set 的质量问题

论文复用 Alpaca 的 52K instructions，只替换 answer generator。  
这意味着它验证的是 “better answers help”，不是 “better instructions + better answers together help”。

作者自己也明确说了，未来工作之一是用 iterative self-instruct 继续扩 instruction set。

## 2. 规模很小，还是早期验证

论文自己标注为 **work in progress**，而且只用了：

- `52K` 数据
- `LLaMA 7B`

和后来的 ShareGPT、多轮对话、百万级合成数据、混合人类偏好数据相比，这只是一个非常早的起点。

## 3. reward model 只用于 decoding，不是真正的 RLHF

论文虽然训练了 reward model，但没有把它继续用于 RL 训练，只用它做 response ranking。

所以它更准确地说是：

- **SFT on GPT-4 data**
- 加上 **machine-generated comparison data**
- 但还不是完整的“GPT-4 feedback driven RLHF pipeline”

## 4. 自动评测高度依赖 GPT-4 as judge

它的大量结论来自 GPT-4 做裁判。

这在当时很自然，但也有明显问题：

- teacher 同时是 data generator 和 evaluator
- 容易偏向更像 GPT-4 风格的回答
- 不一定等价于真实用户偏好

## 5. ROUGE-L 结果揭示了“像 teacher”不等于“像 ground truth”

在 Unnatural Instructions 上，Alpaca 的平均 ROUGE-L 反而更高。  
论文解释是：

- GPT-4 和 LLaMA-GPT4 更倾向生成更 chat-like、更展开的回答
- 短标准答案任务里，这反而会拉低 lexical overlap

这也说明这篇论文测到的主要是 **assistant behavior imitation**，不是传统 benchmark 上的 exact-match 最优。

## 现状与影响

**一句话定性**：这是 GPT-4 蒸馏式 instruction tuning 的奠基性工作，不再是今天的最佳实践本身，但它开启的路线在 2026 年仍然是主流。

### 还在普遍使用吗？

严格说，**论文里的具体 recipe 已经不再是主流最佳实践**：

- 只用 52K SFT 数据太小
- 只做单轮指令跟随也太早期
- reward model 只做 reranking 也不够

但它的核心思想不仅还在用，而且已经扩散成行业默认：

- 用更强闭源模型生成 SFT 数据
- 用更强闭源模型生成 preference / comparison data
- 用 teacher 的回答风格蒸馏出可部署的开源 assistant

### 被什么取代了？

被取代的是**具体配方**，不是**核心思想**。

后来的主流做法通常是：

1. 更大的 synthetic SFT data
2. 混合 ShareGPT / 多轮对话数据
3. preference data + DPO / RLHF / RLAIF
4. 针对代码、数学、agent 等子域的专项后训练

### 核心思想贡献和具体实现是否分离？

是，而且分离得很明显。

今天大家仍然在用的，是这篇论文确认的两个判断：

1. **更强 teacher 生成的数据，能显著提升较小开源模型**
2. **机器生成 comparison data 是可行的**

但没有多少团队还会原样照搬：

- Alpaca 的 52K 指令集
- 单轮 instruction template
- 7B LLaMA + reward reranking 这套完整组合

### 2026 视角

- 仍成立的结论：
  - teacher quality matters
  - synthetic instruction tuning 可以极高性价比地提升开源模型
  - machine-generated feedback 是值得做的
- 已被超越的部分：
  - 数据规模
  - 多轮对话能力
  - 偏好学习方法
  - 专项 reasoning / coding post-training
- 当年可能被低估的部分：
  - comparison data 的价值。后来 RLAIF、judge-model、synthetic preference pipeline 基本都沿这条路继续扩展。

## 和 wiki 内其他概念的关联

- [RLHF](../20-concepts/rlhf.md)：这篇论文不是完整 RLHF，但它把 **machine-generated comparison data** 明确引入了 reward modeling 语境。
- [Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)：它属于“合成数据”大脉络，但这里没有 verifier，重点是 **teacher imitation**，不是 **correctness verification**。
- [Phi-2 / Phi-3](./phi-2-phi-3.md)：Phi 路线证明“小模型 + 高质量数据”可以非常强；这篇论文则是“高质量数据可以直接来自更强 teacher”。

## 值得看的部分 / 相关资料

- 论文第 2 节：四类 GPT-4 数据到底怎么构造
- 论文第 3.2 节：用 GPT-4 comparison data 训练 reward model
- Figure 3：HHH 人类评测，最能说明 GPT-4 数据相对 Alpaca 数据的价值
- Figure 4：GPT-4 自动评测，最能说明 7B student 对更大开源对手的优势
- 论文结论部分：作者自己对后续方向的判断很准，特别是“更多 GPT-4 data + 更大模型 + RLHF”
- Microsoft Research 页面：  
  https://www.microsoft.com/en-us/research/publication/instruction-tuning-with-gpt-4/
- 项目页：  
  https://instruction-tuning-with-gpt-4.github.io/

## 来源

- Peng et al., 2023, *Instruction Tuning with GPT-4*, arXiv:2304.03277  
- Microsoft Research publication page  
- Project page: `instruction-tuning-with-gpt-4.github.io`
