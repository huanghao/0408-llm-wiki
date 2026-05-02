# Self-Instruct: Aligning Language Models with Self-Generated Instructions

一句话总结：Self-Instruct 是 instruction tuning 合成数据路线的起点，它证明了模型可以用自己生成的指令数据来对齐自己，而不必一开始就依赖大规模人工标注。

**论文**：*Self-Instruct: Aligning Language Models with Self-Generated Instructions*  
**作者**：Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A. Smith, Daniel Khashabi, Hannaneh Hajishirzi  
**机构**：University of Washington, Tehran Polytechnic, Arizona State University, Johns Hopkins University, Allen Institute for AI  
**arXiv**：2212.10560  
**ACL Anthology**：ACL 2023 Long Paper

## 核心问题

2022 年 instruction-tuned 模型已经显示出很强的 zero-shot 泛化能力，但一个核心瓶颈很明显：

- 高质量 instruction data 很贵
- 人写 instruction 不仅费时，还要求任务设计能力和答案质量
- 公共 instruction datasets 数量、覆盖面、创造性都有限

这篇论文问的是：

**如果不依赖大规模人工 instruction data，能不能让模型自己生成 instruction / input / output，再反过来用这些数据把自己变成一个更好的 instruction-following model？**

## 方法 / 核心机制

## 1. 从很小的人工 seed set 开始

Self-Instruct 不是零样本凭空生成。

它从一个很小的人工任务池开始：

- `175` 个人工编写的 seed tasks

这些 seed instructions 只负责给模型一个“instruction 长什么样”的起点。

## 2. 自举生成 instruction

核心 pipeline 是一个 bootstrapping loop：

1. 从现有 task pool 里抽 instruction 作为 in-context examples
2. 让模型继续生成新的 instruction
3. 判断是 classification 还是 non-classification task
4. 为 instruction 生成 input-output instances
5. 过滤无效、重复、过于相似的数据
6. 把剩余样本加入训练集

**举例说明这个流程：**

假设 task pool 里已经有这些 seed instructions：

```
把下面这段文字翻译成西班牙语。
写一首关于秋天的俳句。
判断下面这句话是否含有负面情绪。（分类任务）
```

模型看到这三条后，被要求生成新的 instruction，可能产出：

```
把下面的句子改写成更正式的语气。
```

然后再为这条新 instruction 生成 input 和 output：

```
input:  “我觉得这个方案挺好的，咱们就这么干吧”
output: “本人认为该方案具有可行性，建议予以采纳。”
```

这个 (instruction, input, output) 三元组就是一条训练数据。循环 N 轮后，pool 里就从 175 条扩展到了 5 万条。

这个方法的关键不是”生成”，而是**生成后过滤**。  
如果没有过滤，模型很容易反复生成模板化、重复、无意义的指令。

**过滤具体怎么做：**

论文里用了几条规则：

- **ROUGE-L 相似度去重**：如果新生成的 instruction 和 pool 里已有的 instruction ROUGE-L > 0.7，直接丢弃——防止大量重复变体（比如”翻译成法语””翻译成德语””翻译成日语”占满数据集）
- **关键词黑名单**：含有”图片””图表””链接”等词的 instruction 丢弃，因为模型无法处理这类输入
- **长度过滤**：过短（<3 token）或过长的 instruction 丢弃
- **分类任务特殊处理**：对分类任务，要求 output 是有限类别中的一个，否则丢弃

最终 52K 条数据是从大约 200K+ 生成候选里过滤出来的。

## 3. source 和 target 是同一个模型

论文把它应用在 vanilla GPT-3 上：

- 用 GPT-3 生成 synthetic instruction data
- 再用这些数据 finetune GPT-3

这和后来的 teacher-student 蒸馏不一样。  
Self-Instruct 更像是：

**同一个模型通过自举方式，把潜在的 instruction-following 能力从预训练表示里”挖出来”。**

**蒸馏路线最出名的是什么：**

后来”用更强模型生成数据训练更弱模型”这条路线爆发出一批工作：

- **Stanford Alpaca**（2023-03）：用 `text-davinci-003`（GPT-3.5）生成 52K 数据训 LLaMA 7B，不到 $600
- **Instruction Tuning with GPT-4**（2023-04）：同样的题目，换 GPT-4 当 teacher，student 能力大幅提升
- **Vicuna**（2023-03）：用真实 ChatGPT 对话（ShareGPT）训 LLaMA，强调多轮 chat 风格

这三个是最有代表性的。它们都不是”模型自举自己”，而是”强模型当老师，弱模型当学生”——这和 Self-Instruct 的核心区别就在这里。

## 4. 最终生成的数据规模

经过过滤后，论文构造出：

- `52K` instructions
- `82K+` instruction instances

这组数据后来直接成为 Alpaca 等工作的底座。

## 关键结果 / 数据

## 1. 对 vanilla GPT-3 带来显著提升

论文报告：

- 在 `SUPER-NATURALINSTRUCTIONS` 上，相比原始 GPT-3，Self-Instruct finetuning 带来 **33% absolute improvement**

这是论文最核心的结果，因为它说明：

- synthetic instruction tuning 不是“看起来像真数据”
- 而是确实能显著改变模型的泛化行为

## 2. 人类评估上接近 InstructGPT-001

论文还专门构造了一组 **expert-written novel instructions**，避免只在已有 benchmark 分布里看效果。

在这组任务上：

- Self-Instruct tuned GPT-3 明显优于基于其他公开 instruction datasets 训练的版本
- 与 `InstructGPT-001` 只剩 **5% absolute gap**

这个结果的重要性在于，它说明 Self-Instruct 学到的不只是 benchmark 适配，而是更广义的 instruction-following ability。

## 3. 数据规模继续增大时仍有收益

论文还比较了不同 instruction 数量下的表现，显示：

- 从 seed tasks 到更大的 synthetic instruction pool，能力会持续提高

这也是后来大家愿意把 synthetic SFT data 一路扩到几十万、几百万的原因之一。

## 局限性

## 1. 强依赖基础模型本身已经够强

这篇论文自己也承认：

- Self-Instruct 依赖模型已有的 inductive bias
- 更大的、更强的基础模型更适合跑这套方法

也就是说，这不是“从弱模型制造强能力”的方法，而是“把已有能力结构化出来”的方法。

## 2. 容易偏向预训练里常见的任务分布

论文明确指出一个风险：

- gains 可能主要集中在预训练语料里本来就常见的任务/指令类型
- 对少见、创造性、长尾 instruction 可能更脆弱

换句话说，Self-Instruct 的“创造性”并不意味着它真的覆盖了长尾世界。

## 3. synthetic data 的质量上限受 source model 限制

因为 source 和 target 是同一个模型：

- teacher 的错误、偏见、盲点会直接进入生成数据
- 没有外部更强模型帮它纠偏

这也是后来 Alpaca、Instruction Tuning with GPT-4、WizardLM 等路线改用更强 teacher 的重要原因。

## 4. 还不是 chat assistant 时代的多轮对话训练

它主要处理的是：

- 单轮 instruction-following

还没进入：

- 多轮对话
- preference learning
- RLHF / DPO
- tool use / agent

**单轮 vs 多轮举例：**

单轮的训练数据长这样，就是一问一答：

```
instruction: 把下面这句话翻译成法语
input:       The weather is nice today.
output:      Le temps est beau aujourd'hui.
```

多轮的训练数据包含完整的对话历史：

```
user:      你好，帮我写一封请假邮件
assistant: 好的，请问请假几天，原因是什么？
user:      明天一天，发烧了
assistant: 好的，以下是一封请假邮件：\n\n尊敬的领导……
```

**区别在哪里：**

不只是"多了几轮"，训练方式也不同：

- 单轮：每条数据独立，直接 (instruction, output) 对做 SFT
- 多轮：需要把整段对话历史拼成一个序列输入模型，只对 assistant 的回复部分计算 loss，user 的话不算 loss

这个"只对 assistant 计算 loss"是关键——不然模型会同时学"怎么当 user"，反而乱。

多轮数据来源也不同：单轮数据可以靠模型自动生成，多轮真实对话更难凭空制造，所以 Vicuna 直接用了 ShareGPT 上用户分享的真实 ChatGPT 对话记录。

## 现状与影响

**一句话定性**：Self-Instruct 是合成 instruction data 路线的奠基性工作，具体配方已被超越，但“用模型生成 instruction data 再反过来训练模型”这条思路到 2026 仍然在用。**

### 还在普遍使用吗？

原版 recipe 本身已经不是主流最佳实践。

今天很少有人会：

- 只用同一个模型自举自己
- 只用 175 seeds 起步
- 只停在单轮 instruction 数据

### 被什么取代了？

被取代的是原始 recipe，不是核心思想。

后来的主流演化是：

1. **更强 teacher 生成数据**
   例如 Alpaca 用 `text-davinci-003`，Instruction Tuning with GPT-4 用 `GPT-4`

2. **加偏好信号**
   从纯 SFT 走向 comparison data、RLHF、DPO、RLAIF

3. **走向专项数据**
   代码、数学、tool use、agentic tasks 各自有不同 synthetic pipeline

### 核心思想贡献和具体实现是否分离？

是。

今天大家继承的是：

- synthetic instruction generation 可行
- instruction data 可以被系统化扩展
- “数据管道”本身可以成为能力瓶颈

但不太继承的是：

- source=target 的自举设定

### 2026 视角

- 仍成立：
  - synthetic instruction tuning 是后训练主线之一
  - 数据过滤和去重非常关键
- 已被超越：
  - teacher 质量
  - 数据规模
  - 多轮对话 / preference / reasoning 扩展
- 被低估的部分：
  - 它其实定义了后来 Alpaca 系谱的基本骨架

## 和 wiki 内其他概念的关联

- [Instruction Tuning with GPT-4](./instruction-tuning-with-gpt-4-2304.03277.md)：可以看作把 Self-Instruct 的 teacher 从“自己”升级到 GPT-4。
- [RLHF](../20-concepts/rlhf.md)：Self-Instruct 还没有 preference learning，属于 RLHF 之前的一步。
- [Synthetic Data with Verification](../20-concepts/synthetic-data-with-verification.md)：同属“合成数据”路线，但 Self-Instruct 几乎没有 verifier，重点在 bootstrapping 和 filtering。

## 值得看的部分 / 相关资料

- 论文摘要和结论：最浓缩地说明 33% 提升和 5% gap
- 方法部分 Figure 2：整个 bootstrapping pipeline
- 3.1 / 3.2：生成数据的统计和多样性分析
- Figure 7：数据规模和性能关系

## 来源

- Wang et al., 2023, *Self-Instruct: Aligning Language Models with Self-Generated Instructions*, arXiv:2212.10560 / ACL 2023
