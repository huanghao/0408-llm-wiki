# Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor

一句话总结：Unnatural Instructions 用 15 个人工样本作为种子，通过 LLM 自动生成 24 万条多样指令数据，证明了全自动生成的合成数据可以在多个 benchmark 上媲美甚至超越人工众包数据集。

### 为什么叫"Unnatural"？

名字来自对标数据集 **Super-Natural Instructions**（简称 SNI）——那是一个靠人工众包写的大规模指令数据集。

本文的核心观点是：LLM 生成的数据虽然"不自然"（不是真人写的），但训练效果一样好甚至更好。"Unnatural"既是对 SNI 的命名呼应，也在暗示：**不需要"自然"来源（真人），合成数据本身就够用**。

关键点不是省钱，而是一个更大的主张：**数据的多样性和覆盖面比来源是否"真实"更重要**。LLM 生成的指令反而因为没有众包标注者的惯性和偏见，任务类型更稀奇、覆盖更广（200 条样本里出现了 117 种不同任务类型）。

**论文**：*Unnatural Instructions: Tuning Language Models with (Almost) No Human Labor*  
**作者**：Or Honovich, Thomas Scialom, Omer Levy, Timo Schick  
**机构**：Tel Aviv University, Meta AI  
**arXiv**：2212.09689  
**发表**：ACL 2023

## 核心问题

Instruction tuning 的效果高度依赖有标注的指令数据，而有两种主要数据来源：

1. **人工众包数据集**（如 Super-Natural Instructions）：质量高，但贵、慢、覆盖面受限于标注者的创造力
2. **真实用户交互数据**（如 InstructGPT）：分布好，但需要上线的产品积累，成本高

能否用极少人力（约 1 小时）就生成一个既多样又有效的大规模指令数据集？

## 方法 / 核心机制

### 两阶段数据生成

**阶段一：核心数据集生成（Core Dataset）**

- 种子：只需 15 个来自 Super-Natural Instructions 的人工样本（分 5 组，每组 3 个）
- 用 text-davinci-002 作为生成模型 M，给 3 个示例 → 要求生成第 4 个
- 每条样本格式为四元组：`instruction + input argument + constraints + output`
  - instruction：任务描述，可含 `{INPUT}` 占位符
  - input argument：实例化指令的具体输入
  - constraints：输出空间限制（无则填 None）
  - output：greedy decoding 生成（优先准确性）
- 用 nucleus sampling（p=0.99）生成多样 input，greedy decoding 生成 output
- 三类自动过滤：格式不完整、与提示完全相同、instruction+input 完全重复
- 生成核心数据集 ~68,478 条

**阶段二：模板扩展（Template Expansion）**

- 用 LLM 对每条 instruction 生成两个 free-form 改写（paraphrase），保留 input/output
- 目的：打破核心数据集对结构化格式的依赖，增加格式多样性
- 最终数据集：**240,670** 条样本

### 数据质量分析

从 68,478 条核心数据集里随机抽取 200 条，由作者**人工逐条判断**是否正确。判断标准是三项全对：
1. instruction 本身逻辑上可执行（不是无意义的任务描述）
2. input argument 符合 instruction 要求的格式/内容
3. output 在给定 instruction + input 下是正确答案

三项全对才算"完全正确"，任意一项有问题就算错误并归类。结果：
- 56.5%（113/200）完全正确
- 4.5%（9/200）instruction 本身不合逻辑（任务描述自相矛盾或无法执行）
- 17.5%（35/200）instruction 合理，但 input 不符合要求（比如任务说"给一个列表"，input 只给了一个词）
- 21.5%（43/200）instruction 和 input 都合理，但 output 答错了

**"错误样本仍有训练信号"是什么意思？**

直觉上觉得"错误数据应该不能用"，但实际不一定。举论文里的一个真实例子：

- Instruction：给出一个国家列表及其首都，每条线索对应一个国家，根据线索填写国名
- Input：Clue 1: This capital city is on two different continents.（这个首都城市位于两个大洲）
- Output：Istanbul, Turkey

这条被标为"错误"——因为 input 格式不对（应该是国家+首都列表，但只给了一条线索）。**但 output 本身是对的**：伊斯坦布尔确实横跨欧亚两洲。

模型从这条数据里能学到的：当看到关于"跨两个大洲的首都"这类描述，应该回答 Istanbul, Turkey。这个 instruction-following 信号是有效的，尽管格式不符合原始定义。

更一般地说：错误数据里只要 instruction 和 output 之间的映射关系是合理的，模型就能从中学到有用的"当看到这类任务描述，应该产生这类输出"的模式，即使 input 格式有问题。这就是"错误样本仍有训练信号"的含义——噪声数据不等于零信息数据。

## 关键结果 / 数据

fine-tune T5-11B（T5-LM 变体）在四个 benchmark 上的表现（零样本设置）：

| 模型 | Super-Natural Instructions | T0: Zero-Shot | BIG-bench Hard | LMentry |
|------|---------------------------|---------------|----------------|---------|
| T0++ (12.5M 人工样本) | 40.3 | - | 20.2/13.9 | 38.3 |
| Tk-Instruct (75k 样本) | 45.6 | 41.4 | 5.8/11.8 | 35.7 |
| T5-LM on SNI (64k) | **54.0** | 44.0 | 10.2/**29.7** | 34.6 |
| T5-LM on Unnatural (64k) | 51.9 | 45.7 | 16.0/29.5 | 42.0 |
| + 模板扩展 (240k) | 49.3 | **49.0** | **28.1**/29.4 | **50.7** |

- 用 64k 自动数据训练，在 3/4 benchmark 上超过直接用 64k 人工数据的基线
- BIG-bench Hard 上比人工基线高 +18 点（原始格式）
- LMentry 上比人工基线高 +16 点
- 性能与样本数呈 log-linear 关系：扩大数据量可继续提升

## 局限性

- 依赖 text-davinci-002（GPT-3.5 级别闭源模型）生成数据，成本并非零
- 约 43% 样本 output 有错误；虽仍有训练信号，但在严格任务上会引入噪声
- 核心数据集格式（结构化四元组）与 T0: Zero-Shot 格式不匹配，导致不加模板扩展时零样本泛化差
- 在 Super-Natural Instructions 测试集上略弱于直接用 SNI 数据训练（因生成种子本身来自 SNI，导致格式过拟合）
- 仍被 FLAN-T5 超过（但 FLAN-T5 训练数据量大几个数量级）

## 现状与影响

**还在用吗？** 该数据集本身已被更强的合成数据方法（GPT-4 生成、Self-Instruct、Alpaca）超越，但其核心思路（LLM 生成 instruction 数据）已成行业标准实践。

**被什么取代？** 在数据规模和质量上，被 [Instruction Tuning with GPT-4](instruction-tuning-with-gpt-4-2304.03277.md)（GPT-4 生成高质量数据）和各类 RLHF/DPO 管线取代。数据生成思路上，与 [Self-Instruct](self-instruct-2212.10560.md) 并列为两条同期独立的合成数据路线。

**贡献与实现路线是否分离？** 是。贡献在于"极少种子即可自动生成有效大规模数据"的实证验证；实现路线（用 text-davinci-002 生成）已被开源模型替代。其"模板扩展→格式多样性→泛化能力提升"的洞察，在后续 instruction tuning 工作中被反复借鉴。

## 和 wiki 内其他概念的关联

- [Self-Instruct (2212.10560)](self-instruct-2212.10560.md)：同期独立工作，方法相近（少量种子 → LLM 扩增）；Self-Instruct 侧重 pipeline 设计和 175B GPT-3，本文侧重数据质量分析和 T5 微调实验
- [Stanford Alpaca](stanford-alpaca.md)：把类似思路工程化，用 text-davinci-003 生成 52k 数据微调 LLaMA
- [Instruction Tuning with GPT-4](instruction-tuning-with-gpt-4-2304.03277.md)：升级版，用 GPT-4 代替 GPT-3.5 生成更高质量数据
- [Instruction Tuning 概念](../20-concepts/instruction-tuning.md)：本文是该技术路线中"合成数据"分支的奠基工作之一

## 值得看的部分 / 相关资料

- §2.1 核心数据集生成流程（Figure 2、3）：理解 meta-prompt 设计和 few-shot 种子策略
- §3 数据分析（Table 1、3）：提供了对生成数据质量和任务多样性的定量分析
- §5 实验结果（Table 4）：清晰展示了合成数据 vs 人工数据的性能对比
- 数据集开源：[github.com/orhonovich/unnatural-instructions](https://github.com/orhonovich/unnatural-instructions)
