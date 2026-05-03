# Are Emergent Abilities of Large Language Models a Mirage? (Schaeffer et al., 2304.15004)

一句话总结：大模型的"涌现能力"（emergent abilities）并非模型本身的质变，而是研究者选用非线性/不连续评估指标所产生的人工产物——换用线性指标，"涌现"就消失，变成平滑提升。

## 基本信息

- 论文：Are Emergent Abilities of Large Language Models a Mirage?
- 作者：Rylan Schaeffer, Brando Miranda, Sanmi Koyejo
- 机构：Stanford University
- arXiv：[2304.15004](https://arxiv.org/abs/2304.15004)
- 发表：NeurIPS 2023 Oral（Outstanding Paper）
- 本地原文：`raw/inbox/2304.15004.pdf`

## 核心问题

Wei et al.（2022）[arXiv:2206.07682] 观察到：某些能力（如多步算术、词语联想、解码策略）在模型规模超过某个阈值时"突然出现"——在此之前模型几乎不能完成任务，超过阈值后性能急剧跃升。这被称为"涌现能力"（emergent abilities），并被广泛解读为 LLM scaling 存在不可预测的相变。

本文的核心问题：**涌现能力是模型内在的性质，还是评估方式选择的产物？**

## 方法 / 核心机制

### 核心论点

作者给出一个干净的数学框架：

1. **假设模型底层能力随规模平滑提升**（per-token error probability $\epsilon$ 单调递减）
2. **对不同指标做分析**：
   - 线性指标（如 token edit distance）：$\mathbb{E}[\text{score}] \approx L(1 - \epsilon)$，随 $\epsilon$ 线性变化 → **平滑曲线**
   - 非线性指标（如 accuracy = 全部 token 都对才得分）：$\mathbb{E}[\text{Accuracy}] \approx (1-\epsilon)^L$，指数关系 → **S 形曲线，看起来像涌现**

关键公式（Appendix A）：accuracy 是 L 个 token 全部正确的乘积：

$$\mathbb{E}[\text{Accuracy}] \approx (1 - \epsilon)^L$$

当 $\epsilon$ 从 0.2 缓慢降到 0.1（底层平滑改善），$(1-\epsilon)^L$ 的变化极为陡峭（比如 $L=10$ 时，从 $0.8^{10} \approx 0.11$ 跳到 $0.9^{10} \approx 0.35$）——这个"跳跃"完全由非线性放大造成，不是底层的质变。

**具体例子**：假设任务是"把一个 10 位电话号码翻译成另一种格式"，每个数字翻译正确的概率是 $(1-\epsilon)$。

| 每位正确率 $(1-\epsilon)$ | 全部 10 位都对（accuracy）| 平均翻对几位（edit distance 视角）|
|---|---|---|
| 0.80 | $0.80^{10} \approx 10.7\%$ | 平均翻对 8 位 |
| 0.85 | $0.85^{10} \approx 19.7\%$ | 平均翻对 8.5 位 |
| 0.90 | $0.90^{10} \approx 34.9\%$ | 平均翻对 9 位 |
| 0.95 | $0.95^{10} \approx 59.9\%$ | 平均翻对 9.5 位 |

- 用 accuracy 看：从 10.7% 到 59.9%，像是"中间某个阈值附近突然会了"
- 用 edit distance 看：8 位 → 8.5 位 → 9 位 → 9.5 位，完全是均匀的平滑爬坡

模型的底层能力（每位的正确率）从头到尾都在线性改善，"涌现"完全是 accuracy 这个"全对才算分"的评分规则放大出来的视觉效果。

类似地，ROUGE-L-Sum（基于最长公共子序列）也是非线性指标——论文在 Appendix A 证明它同样会在平滑变化的底层误差下产生急剧变化。

### 两类实验验证

**实验一：对"已涌现"的 LLM 任务换指标**

取 BIG-Bench 中被报告有涌现现象的 39 个任务，以及 GPT-4 的算术任务：
- 原始指标（accuracy / ROUGE-L-Sum）：复现出涌现——性能在某规模附近急剧跳升
- 换用线性指标（token edit distance / Brier score）：同样的模型、同样的任务 → **平滑提升，涌现消失**

这说明"涌现"依赖于指标选择，而非模型的真实能力变化。

**实验二：对已知平滑的视觉模型施加非线性指标**

取 LeNet 在 MNIST 上训练的 CNN family（test accuracy 已知随参数量平滑提升），施加 subset accuracy（K 张图片全部正确才得分）：
- 当 K=1：平滑 sigmoid 曲线（Figure 10B）
- 当 K=5：看起来像涌现——性能在某模型大小附近突然跳升（Figure 10C）

这直接用已知无涌现的系统"制造"出了涌现，证明 emergent ability 的视觉外观可以由指标决定。

### 推论：研究者可以控制"是否出现涌现"

论文指出：

> 通过选择合适的指标，研究者既可以让一个任务"表现出涌现"，也可以让它"表现为平滑"。

这不是说研究者会故意这么做，而是说"涌现"的观察结果对指标选择非常敏感，不能作为模型内在性质的证据。

## 关键结果 / 数据

| 验证维度 | 原始指标 | 线性指标 |
|---|---|---|
| BIG-Bench 39 任务（LLM）| 涌现（急剧跳升）| 平滑提升 |
| GPT-4 算术任务 | 涌现 | 平滑提升 |
| InstructGPT / LaMDA / Gopher | 涌现 | 平滑提升 |
| LeNet on MNIST（CV，已知平滑）| 平滑（K=1）| — |
| LeNet on MNIST（subset acc, K=5）| 人工制造出涌现 | 平滑（K=1）|

**指标对比直觉**：

- Accuracy（全对才得分）：L 个 token 都对的概率 $= (1-\epsilon)^L$，高度非线性，$\epsilon$ 小幅改善就导致得分跳升
- Token Edit Distance（Levenshtein 距离）：错误数线性累加，$\epsilon$ 小幅改善带来得分平滑提升
- ROUGE-L-Sum：最长公共子序列，也是非线性——论文从数学上证明它同样产生尖锐变化

## 局限性

- 论文只能证明"用线性指标看是平滑的"——不能完全排除模型内部确实发生了某种质变（只是在这个框架内无法区分）。作者自己也承认：**"我们的工作并不否定涌现能力的存在，而是指出目前的实证证据不足以支持这一结论。"**
- 覆盖的模型系列（GPT-4、InstructGPT、LaMDA、Gopher）均为闭源或旧模型，对更新的开源模型（如 Llama 3）是否成立未直接验证。
- 论文假设底层 per-token error probability 平滑递减，这是合理但未经直接测量的假设。
- 某些能力（如 chain-of-thought reasoning）可能依赖多个子能力同时到位，线性指标也可能在这里出现非线性聚合效应——这属于 power-law-and-scaling 中"能力合成说"的范畴，本文没有完全排除。

## 现状与影响

一句话定性：**本文是"涌现能力"研究的最重要反证，NeurIPS 2023 Oral 认可了其在方法论上的贡献；它并未终结涌现争论，但迫使后续研究在评估指标选择上更加谨慎。**

截至 2026，本文的影响体现在：

- **评估规范化**：NLP benchmark 设计开始更多使用连续/线性指标（如 perplexity、calibration score），而非仅报告 accuracy，以避免指标诱发的虚假涌现。
- **涌现争论的格局**：Wei et al. (2022) 的"涌现能力"结论没有被完全推翻——部分研究者认为确有真实涌现（如 chain-of-thought 能力在某些任务上），但评估方法必须区分"指标非线性"和"模型质变"两种解释。
- **对 scaling law 叙事的修正**：Chinchilla 等工作证明 loss 平滑下降；本文进一步说明"能力跃升"的叙事更多来自评估方式，使 scaling 的连续性叙事更具说服力。
- **被引用的场景**：每当有论文报告新的"涌现能力"时，同行评审通常要求验证该能力在线性指标下是否仍然突现。

## 和 wiki 内其他概念的关联

- [幂律与 Scaling（Power Law）](../20-concepts/power-law-and-scaling.md)：本文的核心论点和 power-law-and-scaling 中"测量假象说"完全对应；非线性指标把平滑的底层幂律放大成了外观上的突变
- [Chinchilla Scaling Laws](./chinchilla-2203.15556.md)：Chinchilla 证明 loss 随计算平滑下降；本文说明 accuracy 看起来的"跳跃"是指标问题，两者共同支持 LLM 能力平滑提升的叙事
- [Scaling Laws for Neural Language Models (Kaplan et al.)](./scaling-laws-neural-lm-2001.08361.md)：Kaplan 的幂律框架也预测平滑改善；本文提供了为什么实践中看起来不平滑的解释
- [LIMA](./lima-2305.11206.md)：LIMA 的 Superficial Alignment Hypothesis 也挑战了某种"突然对齐"的叙事，两篇论文都倾向于用"已有能力的激活"而非"新能力的涌现"来解释后训练效果

## 值得看的部分 / 相关资料

- **Figure 1**：一图总结全文——同一个 BIG-Bench 任务，accuracy 看到涌现，token edit distance 看到平滑提升
- **Figure 2**：用数学模拟展示"为什么非线性指标在平滑 $\epsilon$ 下产生 S 形曲线"
- **Figure 10**：MNIST/LeNet 实验——在已知平滑的视觉系统上"制造"涌现，是论文最有说服力的部分
- **Appendix A**：token edit distance、accuracy、ROUGE-L-Sum 的数学分析，证明各指标的线性/非线性性质
- 被本文直接质疑的工作：Wei et al. 2022, *Emergent Abilities of Large Language Models*（arXiv:2206.07682）
- 同期支持涌现存在的工作：Ganguli et al. 2022（BIG-Bench predictability），Steinhardt 2022（Future ML systems）
