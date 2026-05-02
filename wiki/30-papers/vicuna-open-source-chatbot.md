# Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality

⚠️ 来源说明：本页基于 LMSYS 项目博客与 FastChat 仓库，不是正式论文总结。

一句话总结：Vicuna 把开源 assistant 的重心从“单轮 instruction data”推向了“多轮真实对话数据”，并顺手把 GPT-4-as-a-judge 这件事推到了主舞台。

**项目**：*Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality*  
**团队**：LMSYS / Vicuna Team  
**时间**：2023-03-30  
**主要来源**：LMSYS 博客、`lm-sys/FastChat`

## 核心问题

Alpaca 证明了便宜的 synthetic SFT 可以做出像样的 instruction-following model，但它仍主要是：

- 单轮
- instruction 风格
- 更像 `text-davinci-003`

Vicuna 要解决的是另一个问题：

**如果把训练数据换成真实用户分享的多轮 ChatGPT 对话，开源模型能不能更像“聊天助手”而不是“指令执行器”？**

## 方法 / 核心机制

## 1. 数据从 synthetic instruction 换成 ShareGPT 对话

Vicuna 的核心变化不是 base model，而是数据：

- 在博客里写的是 **70K user-shared ChatGPT conversations**
- FastChat 仓库后续文档写的是 **approximately 125K user-shared conversations**

这两个数字反映的是项目不同阶段的公开口径。更稳妥的表述是：

- Vicuna 训练基于 **大规模 ShareGPT 多轮对话数据**

这比 Alpaca 的单轮 instruction-following demonstrations 更贴近真实 chat assistant 分布。

## 2. 仍然基于 LLaMA 微调，训练方法本质相同

Vicuna-13B：

- 基于 `LLaMA 13B`
- 用 ShareGPT 对话数据做 fine-tuning

FastChat 仓库明确说：

- 它的 fine-tuning code 基于 Stanford Alpaca
- 但新增了对 **multi-turn conversations** 的支持

所以可以把 Vicuna 理解成：

- **Alpaca training pipeline + ShareGPT data + multi-turn chat format**

### 训练方法：模型/Loss 基本不变，核心差异在数据格式

**Alpaca（单轮）的输入输出：**

```
[INST] Instruction: <指令>
Input: <输入>
[/INST]
Output: <答案>
```

每条训练样本是一问一答。Loss 只算 Output 部分（assistant 回答），忽略 instruction/input。

**Vicuna（多轮）的输入输出：**

```
USER: <第一轮用户消息>
ASSISTANT: <第一轮回答>
USER: <第二轮用户消息>
ASSISTANT: <第二轮回答>
...
```

整个对话拼成一个长序列，Loss 只算所有 ASSISTANT 回答部分，USER 消息不算梯度。

#### Loss mask 的实现机制

这里有两个子问题：**如何定位 assistant token 的范围**，以及**如何只对这些 token 算 loss**。

**第一步：tokenize 时记录边界**

tokenizer 把整段对话文本转成 token id 序列。关键是：分词是线性扫描全文的，所以知道每个字符在原文中的位置，也就知道每段 `ASSISTANT:...` 对应哪些 token 的下标范围。

```
原文:  USER: 你好  ASSISTANT: 你好呀  USER: 再见  ASSISTANT: 再见
token: [101][202][303][404][505][606][707][808][909][1010]...
        ^^^^^^^^^^^^^^^^^^^  ^^^^^^^^^^^  ^^^^^^^^^^^  ^^^^^^^^^^^^
        user_turn_1          asst_turn_1  user_turn_2  asst_turn_2
```

token 和原文字符并不是一一对应，但 tokenizer 返回的 `offset_mapping`（字符偏移）或 special token 分隔符（如 `<|im_start|>user` / `<|im_start|>assistant`）可以精确定位每段的 token 下标范围。实践上更常见的做法是：用 **special token** 做分隔符，tokenize 时直接插入，解析时找这些 token 的位置即可，不依赖字符对齐。

**第二步：构造 label 序列，-100 表示"不算 loss"**

PyTorch 的 `CrossEntropyLoss` 约定：label 为 `-100` 的位置会被 ignore。

训练时的 label 序列和 input 序列等长，但：

```
input:  [USER_token_1, ..., ASST_token_1, ..., USER_token_2, ..., ASST_token_2, ...]
label:  [  -100,       ...,  ASST_token_2,...,   -100,       ...,  ASST_token_3, ...]
                              ↑ shifted by 1 (next-token prediction)
```

- user 消息对应的位置：label 填 `-100`
- assistant 回答对应的位置：label 填**下一个 token 的 id**（next-token prediction 目标）

loss 计算时自动跳过 `-100` 的位置，梯度只从 assistant token 流回。这是 HuggingFace Trainer 的标准行为，FastChat/Alpaca 都用这套。

**第三步：forward pass 和 loss 计算完全不变**

模型本身不知道哪些是 user 哪些是 assistant，它只看到一串 token id。loss 函数收到的是模型输出 logits 和 label 序列，`-100` 的位置自动被 ignore。所以模型结构和训练循环完全不需要改动，改的只是数据预处理（如何构造 label）。

**为什么多轮更难？**

- 模型必须在 2~N 轮上下文中保持一致性
- 数据清洗要处理"对话被截断"的情况（过长对话要切片）
- 数据分布更接近真实 assistant 使用场景，但也更嘈杂

## 3. 数据清洗是关键工程点

FastChat 仓库提到，为了保证质量，他们会：

- 把 HTML 转回 markdown
- 过滤不合适或低质量样本
- 把过长对话切段以适应上下文窗口

这说明从 Alpaca 到 Vicuna，instruction tuning 的难点已经开始从“有没有数据”转向“对话数据怎么清洗和组织”。

## 4. GPT-4-as-a-judge 被公开推上前台

Vicuna 最出名的不是训练方法，而是评测叙事：

- 用 GPT-4 作为 judge
- 得出“90%* ChatGPT quality”的结论

博客自己也加了星号和免责声明：

- 这是 **fun and non-scientific evaluation**
- 更严谨的 evaluation 还需要后续工作

但即便如此，这一步仍然影响极大。  
后来的 MT-Bench / Chatbot Arena / LLM-as-a-judge 基本都是沿这条线 formalize。

## 关键结果 / 数据

## 1. 2023 年最强的开源 chat 叙事之一

LMSYS 博客原话是：

- `Vicuna-13B` achieves more than `90%*` quality of ChatGPT and Bard
- 在 90%* 的 case 里优于 LLaMA 和 Stanford Alpaca

这当然不是严格科学结论，但它在 2023 年足够震撼，因为它第一次把“开源聊天模型”带到了一个更接近真实产品体验的话语体系里。

## 2. 训练成本非常低

博客给出的训练成本约：

- **$300**

这个数字和 Alpaca 一样，起到的核心作用是重新设定预期：

- 不是说小团队能追平所有闭源能力
- 而是说“看起来像聊天助手”的体验，门槛已经降得很低

## 3. 对开源评测范式的影响比对模型本身更持久

从 2026 回看，Vicuna 的长期影响有两部分：

1. **多轮 chat data 比单轮 instruction data 更接近真实助手**
2. **LLM-as-a-judge 可以成为评测基础设施的一部分**

第二点的历史影响甚至比第一点更大。

## 局限性

## 1. “90% ChatGPT”结论本身很脆弱

博客自己已经提醒：

- 评测是 preliminary
- 评测不够科学

问题主要有：

- GPT-4 judge 可能偏向某种回答风格
- 问题集规模有限
- “90%”是相对分数，不等于真实使用体验的 90%

## 2. 数据来源有隐私和许可争议

ShareGPT 路线虽然效果好，但从一开始就伴随几个问题：

- 用户共享对话是否稳定可用
- 数据许可是否清晰
- 隐私和合规风险如何处理

FastChat 也明确说他们**不发布 ShareGPT dataset**。

## 3. 仍然主要是 imitation，不是真正的偏好优化

Vicuna 强在 chat style imitation，但并不意味着它解决了：

- preference learning
- safety alignment
- reasoning depth
- factuality

它本质上仍是高质量 chat SFT 的代表，而不是完整 post-training stack。

## 现状与影响

**一句话定性**：Vicuna 作为具体模型已经被后来一整代 chat / judge / arena 系统超越，但它是“多轮 chat data + GPT-4 judge”时代的开山项目之一。**

### 还在普遍使用吗？

原版 Vicuna 模型本身已经不是 2026 的主流使用对象。

### 被什么取代了？

被取代的是模型本身，但它引出的两条路线一直保留下来：

1. **多轮高质量 chat SFT**
2. **LLM-as-a-judge 评测**

后来的替代者包括：

- 更强的 open chat models
- MT-Bench / Chatbot Arena 体系
- 更完整的 preference + reasoning + tool-use post-training

### 核心思想贡献和具体实现是否分离？

是。

保留下来的思想：

- chat assistant 训练要看真实多轮对话分布
- judge model 可以大幅降低评测成本

被淘汰的具体实现：

- 早期 80 questions 风格的评测
- “90% ChatGPT quality”这种 headline 式表述
- 原始 Vicuna 模型本身

### 2026 视角

- 仍成立：
  - 多轮对话数据是 chat assistant 的关键
  - LLM-as-a-judge 有高实用价值
- 已被超越：
  - 评测 rigor
  - 模型能力
  - 对齐完整性
- 被低估的部分：
  - Vicuna 实际上是后续 Arena / MT-Bench / judge-based eval 的前奏

## 和 wiki 内其他概念的关联

- [Stanford Alpaca](./stanford-alpaca.md)：Vicuna 可以看作从 Alpaca 的单轮 instruction SFT 走向多轮 chat SFT。
- [Instruction Tuning with GPT-4](./instruction-tuning-with-gpt-4-2304.03277.md)：Vicuna 的博客几乎和这篇论文一起，把 GPT-4-as-a-judge 推到了主流视野。
- [RLHF](../20-concepts/rlhf.md)：Vicuna 不是 RLHF 路线，而是 chat-distribution imitation 路线。

## 值得看的部分 / 相关资料

- LMSYS 博客里“90%*”声明和脚注
- FastChat 仓库的 data cleaning / fine-tuning 部分
- 后续延伸论文：`Judging LLM-as-a-judge with MT-Bench and Chatbot Arena`

## 来源

- LMSYS Blog, *Vicuna: An Open-Source Chatbot Impressing GPT-4 with 90%* ChatGPT Quality*, 2023-03-30  
  https://www.lmsys.org/blog/2023-03-30-vicuna/
- GitHub: `lm-sys/FastChat`
