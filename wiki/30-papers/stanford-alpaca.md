# Stanford Alpaca

⚠️ 来源说明：本页基于 Stanford CRFM 博客和公开代码仓库，不是正式论文总结。

一句话总结：Alpaca 把 Self-Instruct 路线工程化到一个足够简单、足够便宜、足够可复制的配方，第一次让学术界和开源社区意识到“几百美元就能做出一个像样的 instruction-following model”。

**项目**：*Alpaca: A Strong, Replicable Instruction-Following Model*  
**机构**：Stanford CRFM / HAI / Stanford NLP  
**时间**：2023-03-13  
**主要来源**：Stanford CRFM 博客、`tatsu-lab/stanford_alpaca`

## 核心问题

Self-Instruct 已经证明 synthetic instruction data 可以有效，但它仍然更偏研究论文风格。

Alpaca 解决的是一个更工程化的问题：

**在学术预算下，能不能把一个开放权重基础模型快速变成接近 `text-davinci-003` 的 instruction-following assistant？**

这个问题之所以重要，是因为 2023 年初：

- ChatGPT / text-davinci-003 已经很强
- 但学术界几乎没有一个真正可复现、可本地训练、可对照实验的近似替代物

## 方法 / 核心机制

## 1. 基座换成 LLaMA 7B

Alpaca 不是在 GPT-3 上做，而是：

- 以 `LLaMA 7B` 为 base model

这意味着它把 synthetic instruction tuning 从“闭源模型内部可做”转成了“开放权重模型也能做”。

## 2. 数据沿用 Self-Instruct 风格，但 teacher 换成 text-davinci-003

Alpaca 的 52K demonstrations：

- 来自 `text-davinci-003`
- 数据生成过程是 **building upon the self-instruct method**
- 起点是 Self-Instruct 的 `175` 条 seed instruction-output pairs

也就是说，Alpaca 的真实创新不是“发明新方法”，而是：

- 沿用 Self-Instruct 的骨架
- 用更强的 teacher 来扩 instruction data
- 把整条 pipeline 压到低成本、可复现

## 3. 成本极低

Stanford 博客给出的数字非常关键：

- 生成 52K 数据：**<$500**
- 微调 7B LLaMA：**8 x A100 80GB，3 小时，<$100**

合起来：

- 整套配方 **< $600**

这正是 Alpaca 在 2023 年爆炸式传播的原因。它改变的不是 SOTA，而是大家对“门槛”的认知。

## 关键结果 / 数据

## 1. 在初步人工评测里接近 text-davinci-003

博客报告，在 Self-Instruct evaluation set 上做 blind pairwise comparison：

- Alpaca 胜 `90`
- text-davinci-003 胜 `89`

Stanford 自己也承认：

- 这个评估规模有限
- 标注者就是 5 位学生作者

所以这不能被理解为严格意义上的“Alpaca = text-davinci-003”，但它足以说明：

- 一个 7B 开放模型，已经可以在相当多单轮 instruction tasks 上表现得“像样”

## 2. 重点不是绝对性能，而是可复制性

Alpaca 真正的结果不是某个 benchmark 数字，而是：

- 一个小模型
- 少量合成数据
- 很低成本

就能做出接近闭源 instruction model 风格的东西。

从研究史角度看，Alpaca 更像是：

**“开源 instruction tuning 的复现模板”**  
而不是“严格评测下的能力冠军”。

## 局限性

## 1. 它继承了双重许可限制

Stanford 博客明确写了：

- LLaMA 是非商业许可
- instruction data 来自 `text-davinci-003`，OpenAI terms 也限制竞争性用途

所以 Alpaca 从一开始就不是一个真正可商用的开放生态底座。

## 2. 评测很初步

博客自己承认：

- 评测 limited in scale and diversity

因此“90 vs 89”更多是方向性信号，不是严谨结论。

## 3. 安全性明显不足

Stanford 也明确说：

- Alpaca 还没准备好 general use
- 存在 hallucination、toxicity、stereotypes
- 内容过滤和水印只是临时缓解措施

## 4. 仍然是单轮指令跟随

它没有真正进入：

- 多轮对话
- preference learning
- 高质量对齐
- reasoning-specific post-training

## 现状与影响

**一句话定性**：Alpaca 不是今天最值得直接复用的 recipe，但它是 2023 年开源 assistant 爆发的触发点之一。**

### 还在普遍使用吗？

原版 Alpaca recipe 已经不再普遍使用。

今天很少有人还会：

- 直接用 `text-davinci-003` 生成单轮 52K 数据
- 用“Alpaca 评测方式”来判断模型质量

### 被什么取代了？

它很快被几条路线替代：

1. **更强 teacher**
   GPT-4 比 text-davinci-003 更强

2. **更多真实对话数据**
   Vicuna / ShareGPT 路线把重点转向多轮 chat

3. **更完整后训练**
   preference data、DPO、RLHF、RLAIF

### 核心思想贡献和具体实现是否分离？

是。

被保留下来的是：

- “cheap synthetic SFT works”
- “开放权重小模型也能快速变成 assistant”
- “数据 recipe 本身是可复现研究对象”

被淘汰的是：

- teacher 选择
- 数据规模
- 评测方式
- 安全与部署假设

### 2026 视角

- 仍成立：
  - instruction-tuning 数据质量非常关键
  - 小模型通过高质量后训练可以跃迁
- 已被超越：
  - 数据来源
  - 多轮对话能力
  - 对齐与偏好学习
- 历史地位：
  - 开源 instruction-following 时代的标志性“点火项目”

## 和 wiki 内其他概念的关联

- [Self-Instruct](./self-instruct-2212.10560.md)：Alpaca 直接建立在 Self-Instruct 方法和 seed set 之上。
- [Instruction Tuning with GPT-4](./instruction-tuning-with-gpt-4-2304.03277.md)：可以看成 Alpaca 的 teacher 升级版路线。
- [RLHF](../20-concepts/rlhf.md)：Alpaca 主要停在 SFT，还没进入偏好优化阶段。

## 值得看的部分 / 相关资料

- Stanford CRFM 博客里的 Training recipe 段落
- 成本数字：`<$500` 数据、`<$100` 微调
- Known limitations：对 Alpaca 历史定位很关键
- GitHub 仓库：看生成数据和微调脚本比看“效果宣传”更有价值

## 来源

- Stanford CRFM, *Alpaca: A Strong, Replicable Instruction-Following Model*, 2023-03-13  
  https://crfm.stanford.edu/2023/03/13/alpaca
- GitHub: `tatsu-lab/stanford_alpaca`
