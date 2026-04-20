# The Llama 3 Herd of Models

这是一篇最适合拿来建立“现代 LLM 系统报告阅读框架”的论文。

## Citation

- Authors: Llama Team, AI @ Meta
- Year: 2024
- Link: https://arxiv.org/abs/2407.21783
- Local PDF: `raw/inbox/llama-3-herd-2407.21783.pdf`

## 为什么把它作为第一篇

如果现在直接跳进 `s1`、`DeepSeek-R1` 或 `OSWorld`，很容易只看到局部热点，而缺少对“现代基础模型到底是怎么做出来的”这件事的整体框架。

这篇论文最适合放在第一篇，不是因为它最前沿，而是因为它把下面这几个问题放在了一篇系统报告里一起回答：

1. pre-training 现在最重要的杠杆是什么
2. data mix 和 tokenizer 到底有多重要
3. post-training 现在的主流 recipe 长什么样
4. 长上下文、tool use、安全、评测是怎么并入同一个模型族的
5. 一个 405B 级别模型在工程和基础设施上意味着什么

## 先记住作者想表达的主线

这篇论文的主线非常明确，可以压缩成一句话：

**Llama 3 的提升，主要不是靠“惊艳的新架构”，而是靠更好的数据、更大的规模、以及尽量控制复杂度的系统设计。**

作者在引言里直接点了三个关键杠杆：

1. `data`
2. `scale`
3. `managing complexity`

这三个词就是你读这篇论文的主导航。

这里容易混淆的一点是：`data` 和 `scale` 不是一回事。

我建议这样区分：

1. `scale` 更偏总量和资源规模
   - 模型参数有多大
   - 训练 token 有多少
   - 总 FLOPs 有多少
2. `data` 更偏数据分布和数据质量
   - 数据从哪里来
   - 各类数据占比怎么配
   - 去重、清洗、过滤做得怎么样
   - code / math / multilingual 有没有专门管道

所以“data 更多”有时候确实属于 scale，但“data 更好”不是 scale，而是数据工程。

更直白一点：

1. `scale` 问的是：你喂了多少
2. `data` 问的是：你喂的是什么，以及这些东西被怎么加工过

这篇论文真正强调的是，Llama 3 同时提升了这两个维度。

## 读这篇时最该看的 6 个点

### 1. 它故意没有追求花哨架构

这是第一件最值得注意的事。Llama 3 继续用的是相对标准的 dense Transformer，而不是 MoE。

论文的态度很明确：

1. 架构上只做少量改动
2. 优先保证训练稳定性和可扩展性
3. 把主要收益放在数据质量、训练规模、post-training 上

这点很重要，因为它会帮你建立一个很现实的判断：

**现代顶级模型的进步，很多时候不是来自“新结构”，而是来自更成熟的 data + systems + post-training 配方。**

### 2. 数据工程是核心，不是配角

如果只看模型名字，很容易忽略这篇论文里最重的部分其实是数据。

你应该重点记住这些数字和选择：

1. pre-training 使用大约 `15T` multilingual tokens，远高于 Llama 2 的 `1.8T`
2. 数据混合比例大致是：
   - `50%` general knowledge
   - `25%` math/reasoning
   - `17%` code
   - `8%` multilingual
3. 做了多层去重：
   - URL-level
   - document-level
   - line-level
4. 做了大量质量过滤、PII 过滤、unsafe domain 过滤
5. 还专门做了 code / reasoning 的 domain-specific pipeline

这里最值得你更新的认知是：

**“训练一个模型”并不是把互联网文本全喂进去，而是构造一个高度加工过的数据分布。**

### 3. tokenizer 不再是小事

Llama 3 用了 `128K` vocabulary。

论文特别强调：

1. 它把 `100K` 的 `tiktoken` token 加上 `28K` 非英文语言相关 token
2. 相比 Llama 2 tokenizer，压缩率明显更好
3. 更好的压缩率意味着在相同训练算力下，模型能“读到更多文本”

这里的“压缩率”可以简单理解成：

**同样一段文本，需要多少 token 才能表示出来。**

如果压缩率更好，意思就是：

1. 同样的文本内容，需要的 token 更少
2. 或者反过来说，同样数量的 token，可以覆盖更多原始文本

因为 LLM 训练和推理很多成本都是按 token 计的，所以这会直接影响效率。

举个不严谨但够用的直觉例子：

1. 如果旧 tokenizer 处理一段文本要 100 个 token
2. 新 tokenizer 只要 80 个 token

那在固定训练预算下，模型就能多“读”25% 左右的原始文本内容。

所以 tokenizer 改进不是小修小补，它会影响：

1. 训练时的信息吞吐
2. 长上下文里的有效内容密度
3. 多语言场景下的表示效率

这是很多人读系统报告时会跳过去的地方，但其实非常关键。

你可以把它理解成：

**tokenizer 改进本质上是在提高“每单位训练算力对应的信息摄入效率”。**

### 4. 长上下文不是一开始就训练到 128K

这篇论文对长上下文的处理是非常工程化的，不是“一步到位”。

405B 的 pre-training recipe 分三段：

1. initial pre-training
2. long-context pre-training
3. annealing

其中长上下文阶段是：

1. 先在 `8K` 上做主训练
2. 最后再逐步扩到 `128K`
3. 用大约 `800B` tokens 做 long-context pre-training
4. 每次增大 context length 后，都要求：
   - 短上下文性能恢复
   - needle-in-a-haystack 能过

这点很重要，因为它说明：

**长上下文能力不是一个简单的配置项，而是一个单独的训练阶段。**

### 5. post-training recipe 很“保守”，但在今天仍然非常值得看

Llama 3 的 post-training 主流程不是 PPO，而是：

1. reward model
2. rejection sampling
3. SFT
4. DPO

作者明确说他们偏好这条路线，是因为它比更复杂的 RL 方法更稳定、更容易扩展。

这对你后面读 `Tulu 3`、`DeepSeek-R1` 很有帮助，因为你会看到两条越来越清晰的路线：

1. `SFT + preference optimization` 这一条工业上非常稳的路线
2. 更重 RL 的 reasoning 路线

所以这篇论文的价值之一，就是帮你先把“非 RL 主线”的现代配方立住。

从 `2026-04` 的角度看，这套 recipe 已经不是“唯一主流”，但它仍然是非常重要的基线。

更准确地说：

1. 在通用 assistant、chat、tool use、helpfulness 这些场景里，`SFT + preference optimization` 仍然是主干
2. 在高强度 reasoning 场景里，更重的 RL、test-time scaling、reasoning-specific data 已经明显抬头
3. 所以今天看 Llama 3，不应该把它理解成“过时做法”，而应该理解成 reasoning-era 之前最成熟的一版工业主线

在今天仍然最值得看的内容有：

1. 为什么工业系统偏爱稳定、可扩展、可迭代的 post-training recipe
2. rejection sampling、SFT、DPO 是怎么串起来形成流水线的
3. tool use、long context、safety 这些能力是如何并入同一个 post-training 框架的
4. 为什么很多能力提升其实来自数据策划，而不是换一个优化算法名字

### 6. 安全不是最后补个 appendix

Llama 3 这里有两个层面的安全：

1. 模型本身在 post-training 中做 safety SFT / safety DPO
2. 系统层再接 `Llama Guard 3`

这会让你看到一个更现实的工业观点：

**安全不是只靠 base model 学出来，而是模型层 + system layer 一起做。**

这也是为什么后面很多真实系统讨论“模型能力”时，不能把模型本体和部署时的 guardrail 混为一谈。

## 你可以先跳过什么

第一次读，不建议从头到尾线性硬读。

可以先跳过这些细节：

1. 大段基础设施和并行训练实现细节
2. 多模态扩展章节
3. 具体 benchmark 表格里的每个数字
4. 大量 appendix 风格的工程故障分析

这些不是不重要，而是放在第一遍会稀释主线。

## 第一遍建议阅读顺序

### 必读

1. 引言
2. `3.1 Pre-Training Data`
3. `3.2 Model Architecture`
4. `3.4` 里 pre-training recipe 的三段式
5. `4 Post-Training`
6. 安全章节里关于 `Llama Guard 3` 的部分
7. 结论

### 第二遍再补

1. scaling law 具体构造方法
2. 训练系统和基础设施
3. human eval 和 benchmark 表格
4. multimodal extension

## 读完这篇你应该建立的判断

读完第一遍，不要追求“记住所有数字”，而要能回答下面这些问题：

1. 为什么 Meta 选择 dense Transformer 而不是 MoE？
2. 为什么数据清洗、去重、数据 mix 会被放到这么高的优先级？
3. tokenizer 为什么会显著影响模型效果和训练效率？
4. 为什么长上下文是一个专门的 continued pre-training 阶段？
5. 为什么工业上的 post-training 常常优先选 SFT + RS + DPO，而不是直接上更重的 RL？
6. 为什么真实系统里的安全往往是 model-level 和 system-level 共同完成的？

如果这 6 个问题你能回答清楚，这篇第一遍就算读对了。

## 一页速记

如果只想记最核心的结论，可以先记这 5 句：

1. 现代模型提升的主因，很多时候是数据、规模和训练配方，不是花哨架构。
2. 数据工程已经是 foundation model 的核心能力，不是辅助环节。
3. tokenizer 直接影响训练效率和多语言支持，不是边角料。
4. 长上下文需要专门训练，而不是靠把配置改成 128K。
5. 工业上的 post-training 强调稳定、可扩展、易迭代，这解释了为什么 SFT + DPO 这么常见。

## 从 2026 年回看：Llama 做对了什么，错过了什么

这部分最好带着一点“历史定位感”去看，不要只用今天的强模型结果倒推说它不行。

### 做对了什么

1. 它非常早地把“数据工程是核心竞争力”这件事讲清楚了。
2. 它证明了 dense 模型在极强工程执行下，依然可以非常有竞争力。
3. 它把 long context、tool use、安全系统、multilingual 这些能力整合进了一个统一模型家族。
4. 它代表了 open-weight 大模型第一次非常系统地逼近闭源旗舰的那一波。
5. 它把现代工业 post-training 的稳定主线讲得很完整。

### 错过了什么

1. 它没有真正进入后来的 reasoning-first 路线。
2. 它对 RL 在 reasoning 能力激发上的潜力押注不够重。
3. 它的系统报告重点仍然是“通用旗舰 assistant”，而不是“会花更多 test-time compute 深度思考的模型”。
4. 它虽然支持长上下文和 tool use，但还不是后来 agent-native 那种研究中心。
5. 它代表的是 reasoning 爆发前夜的最强工业范式之一，而不是 reasoning 时代的最终答案。

所以你今天读它，最好的姿势不是问“它现在是不是掉队了”，而是问：

**reasoning 时代到来之前，工业界最成熟的 LLM recipe 长什么样？**

这个问题上，Llama 3 仍然很值。

## 读原文时可以带着的 8 个问题

这里的 8 个问题和前面的 6 个问题不是一回事：

1. 前面的 6 个问题是 `自测题`
   - 作用是检查你第一遍有没有读懂主线
   - 它们都有相对明确的参考答案
2. 这里的 8 个问题是 `延伸题`
   - 作用是带着历史视角和比较视角继续往下读
   - 它们很多没有单一标准答案，更像研究问题

所以正确用法是：

1. 先用前面的 6 个问题判断“我有没有读懂”
2. 再用这里的 8 个问题判断“我接下来还该往哪里想”

1. 15T tokens 到底意味着什么样的数据控制力？
2. 25% reasoning + 17% code 这个数据 mix 在今天看是不是已经成为常态？
3. 405B dense 模型和同时代 MoE 路线相比，取舍在哪里？
4. annealing 为什么对小模型帮助更大、对 405B 帮助有限？
5. Llama 3 的 long-context recipe 和后来的 Gemini / long-context 路线有什么共性？
6. 这里的 DPO recipe 和后来的 Tulu 3 有哪些延续？
7. Llama Guard 3 应该被视为模型能力的一部分，还是系统补丁？
8. 如果今天重做一遍，哪些地方会被 reasoning-era 的做法替换掉？

## 后续阅读连接

按现在这个顺序，下一篇最自然的是：

1. `DeepSeek-V3`
   因为它会把“工程配方”继续推进到更激进的系统优化和架构取舍上。
2. `Tulu 3`
   因为它会把 post-training 这条线讲得更清楚。
3. `DeepSeek-R1`
   因为它代表另一条更重 RL 的 reasoning 路线。

## 附录：第一遍自测题参考答案

下面这部分不是标准答案，而是“读完第一遍后，你至少应该能说到这个程度”。

### 1. 为什么 Meta 选择 dense Transformer 而不是 MoE？

参考答案：

因为这篇论文的优先级不是“单位训练成本极致最优”，而是“训练稳定、工程可控、容易大规模扩展”。  
MoE 在理论上更省推理或训练成本，但会引入更多系统复杂度、负载均衡问题和训练不稳定因素。  
Llama 3 这篇论文代表的取向是：先把 dense 路线做到极强，再用数据、规模、post-training 去拿收益。

### 2. 为什么数据清洗、去重、data mix 会被放到这么高的优先级？

参考答案：

因为当模型已经足够大时，原始互联网文本的噪声、重复、低质量分布会直接决定训练上限。  
Llama 3 的一个核心观点是：不是“有更多数据就够了”，而是“要把数据做成适合 foundation model 学习的分布”。  
所以 URL/document/line 级别去重、质量分类、code/reasoning 专项管道、data mix 调整，都不是辅助动作，而是主工作。

### 3. tokenizer 为什么会显著影响模型效果和训练效率？

参考答案：

因为训练和推理很多成本都是按 token 算的。  
同样一段文本，如果 tokenizer 更高效，就能用更少 token 表示；这意味着固定训练预算下，模型能看到更多原始文本。  
同时 tokenizer 也会影响多语言压缩率、长上下文里的信息密度，以及不同语言的表示公平性。

### 4. 为什么长上下文是一个专门的 continued pre-training 阶段？

参考答案：

因为 context length 提升会显著增加 self-attention 计算成本，也会改变模型需要适配的分布。  
所以不能简单把 context window 参数改成 128K 就结束，通常要经过一个专门阶段，让模型逐步适配更长序列。  
Llama 3 这里强调的是：每扩一次长度，都要检查短上下文性能有没有恢复，以及 needle-in-a-haystack 是否通过。

### 5. 为什么工业上的 post-training 常常优先选 SFT + RS + DPO，而不是直接上更重的 RL？

参考答案：

因为工业系统更看重稳定、迭代速度、可扩展性和调参成本。  
SFT + rejection sampling + DPO 这条链路虽然未必能把 reasoning 激发到最强，但它非常适合做通用 assistant、tool use、helpfulness、格式对齐这些能力。  
从 2026 年看，reasoning 场景里更重 RL 很重要，但这不意味着传统 post-training 主线失效了。

### 6. 为什么真实系统里的安全往往是 model-level 和 system-level 共同完成的？

参考答案：

因为单靠 base model 或单靠 post-training 很难覆盖所有真实部署场景。  
模型本身可以通过 safety SFT / safety DPO 学到更稳妥的行为，但系统层还需要输入输出分类器、拒答策略、policy engine 等补充防线。  
Llama 3 用 `Llama Guard 3` 说明了一个现实做法：安全通常是模型能力和系统护栏叠加出来的。

## 附录：3.1 Pre-Training Data 中的数据清洗方法

Llama 3 的 3.1 节描述了一套多层次的 Web 数据处理管道，从原始爬取数据到最终训练集要经过多个过滤和去重步骤。

### 整体管道

原文 3.1.1 节描述的顺序是：

```
原始 Common Crawl
    ↓ PII 和安全过滤（域名黑名单：PII 高风险站点、有害域名、成人内容）
    ↓ HTML 解析 + 文本提取（自定义解析器，保留数学/代码结构）
    ↓ 去重（URL-level → Document-level MinHash → Line-level ccNet 风格）
    ↓ 启发式过滤（n-gram 重复率、dirty word、token 分布 KL 散度）
    ↓ 模型质量过滤（fastText 快速分类器 + DistilRoBERTa 精细分类器）
最终训练数据
```

**注意**：原文描述的是各类过滤手段，并没有给出严格的线性执行顺序，也没有说明各环节过滤掉多少数据（漏斗比例）或各手段的 ROI。上图是根据原文叙述顺序整理的合理推断，实际工程实现中各步骤可能并行或交错执行。

下一节「各步骤简介」按功能分组介绍，和上图的顺序不完全对应，是刻意的——目的是把"做什么"说清楚，而不是还原执行顺序。

### 各步骤简介

**PII 和安全过滤**：最前端的粗粒度过滤。去掉已知含大量 PII 的网站、被 Meta 安全标准标记为有害的域名、以及已知成人内容域名。

**HTML 解析与文本提取**：从原始 HTML 提取正文，去掉 boilerplate 噪声结构。Llama 3 构建了自定义解析器，针对数学公式和代码内容做了特殊处理（保留 `alt` 属性中的数学文本），并发现移除 markdown 标记对以网页数据为主的模型有帮助。

**去重**：分三个层级：
- **URL-level**：跨全数据集对同一 URL 只保留最新版本
- **Document-level**：使用 [MinHash](../20-concepts/minhash.md)（Broder, 1997）做全局近重复去重
- **Line-level**：借鉴 [ccNet](../20-concepts/ccnet.md)（Wenzek et al., 2019）策略，每桶 30M 文档中出现超过 6 次的行视为 boilerplate 删除

**启发式过滤（Heuristic Filtering）**：处理去重无法覆盖的低质量模式：
- 用 **duplicated n-gram coverage ratio**（Rae et al., 2021）过滤由重复内容构成的行（如日志、错误信息）
- 用 **"dirty word" counting**（Raffel et al., 2020，即 C4 的方法）过滤域名黑名单未覆盖的成人网站
- 用 **token 分布 KL 散度**过滤与训练语料分布差异过大的文档（异常 token 比例过高）

**模型质量过滤（Model-based Quality Filtering）**：
- 用 [fastText](../20-concepts/fasttext.md)（Joulin et al., 2017）训练快速分类器，识别"会被 Wikipedia 引用"的文本（正例来自 Wikipedia 引用的网页）
- 用基于 **RoBERTa**（Liu et al., 2019a）的更重分类器，在 Llama 2 预测标注的数据上训练，给文档打质量分；为效率使用 **DistilRoBERTa**（Sanh et al., 2019）
- 实验评估了多种质量过滤配置的效果

**Code 和 Reasoning 专项管道**：参考 DeepSeek-AI et al.（2024）的做法，为代码和数学推理内容建立独立的域专属管道，使用在 Llama 2 标注数据上训练的 DistilRoBERTa 分类器，并针对代码/数学的 token 分布特点定制 HTML 提取和启发式过滤规则。

**多语言管道**：与英文管道并行，用 fastText 语言识别模型（176 种语言）分类，在各语言内部做 document-level 和 line-level 去重，用多语言 Llama 2 分类器做质量排序，通过 scaling law 实验确定多语言 token 的比例。

### 3.1 节引用的论文和工具

#### 去重

| 引用 | 用途 |
|------|------|
| Broder, 1997 — *On the resemblance and containment of documents* | Document-level MinHash 去重 → [MinHash](../20-concepts/minhash.md) |
| Wenzek et al., 2019 — *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data* | Line-level 去重策略 → [ccNet](../20-concepts/ccnet.md) |

#### 启发式过滤

| 引用 | 用途 |
|------|------|
| Rae et al., 2021 — *Scaling Language Models: Methods, Analysis & Insights from Training Gopher* | Duplicated n-gram coverage ratio，过滤重复行 |
| Raffel et al., 2020 — *Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer* | "Dirty word" counting（C4 方法），过滤成人内容 |

#### 模型质量过滤

| 引用 | 用途 |
|------|------|
| Joulin et al., 2017 — *Bag of Tricks for Efficient Text Classification* | fastText 质量分类器和语言识别 → [fastText](../20-concepts/fasttext.md) |
| Touvron et al., 2023a — *Llama 2: Open Foundation and Fine-Tuned Chat Models* | 提供质量分类器的标注预测（Llama 2 predictions） |
| Liu et al., 2019a — *RoBERTa: A Robustly Optimized BERT Pretraining Approach* | RoBERTa 架构，用于更重的质量分类器 |
| Sanh et al., 2019 — *DistilBERT, a distilled version of BERT* | DistilRoBERTa，质量打分的高效实现 |
| DeepSeek-AI et al., 2024 | Code/reasoning 专项管道的参考做法 |

#### Annealing 阶段（3.1.3）

| 引用 | 用途 |
|------|------|
| Li et al., 2024b | Annealing 时上采样高质量数据的策略参考 |
| OpenAI, 2023a | GSM8k/MATH annealing 效果评估的对比参考 |
| Cobbe et al., 2021 — *Training Verifiers to Solve Math Word Problems* | GSM8k benchmark |
| Hendrycks et al., 2021b — *Measuring Mathematical Problem Solving With the MATH Dataset* | MATH benchmark |
| Blakeney et al., 2024 | 用 annealing 评估小数据集价值的方法 |

### 为什么是这些动作，怎么知道做完之后数据变好了

这是读这一节最容易跳过去的问题，但也是最值得想清楚的。

#### 作者的答案：两个评估工具

**工具一：scaling law 小模型实验（3.1.2）**

具体做法分两步：

第一步，建立"训练损失 → benchmark 分数"的映射关系。在不同计算预算（$6 \times 10^{18}$ 到 $10^{22}$ FLOPs）下训练一批 40M 到 16B 的小模型，测量这些小模型在下游 benchmark 上的 negative log-likelihood（即语言模型损失），再把损失值和实际 benchmark 准确率做相关性拟合，得到一条"损失 → 准确率"的换算曲线。

第二步，用这条曲线预测大模型。对于一个候选数据 mix，先在小模型上跑，得到它的损失值，再通过第一步的换算曲线，预测出 405B 大模型在该数据 mix 上的 benchmark 表现。

所以"scaling law 预测大模型性能"的意思是：**不直接训练大模型，而是通过小模型的损失值，间接估算大模型的 benchmark 分数**。这样就可以在小模型上快速比较几十种数据 mix 候选，只把最优的那个拿去真正训大模型验证。

这个方法的核心假设是：小模型和大模型在同一数据 mix 上的相对排序是一致的。这个假设在实践中大体成立，但有边界——某些能力（比如 in-context learning、long-context）在小模型上几乎不显现，所以小模型实验对这些能力的数据配比判断力很弱。

**工具二：annealing 快速评估（3.1.3）**

这个方法解决的是另一个更具体的问题：**某个候选小数据集值不值得加进来**。对这类问题跑完整的 scaling law 实验太贵，Llama 3 用了一个更轻量的替代方案。

先解释 annealing 是什么。神经网络训练时有一个超参数叫学习率（learning rate），它控制每一步梯度更新的步长。学习率太大，模型参数更新剧烈，容易不稳定；学习率太小，更新太慢。通常的训练策略是：训练初期用较大的学习率快速学习，训练后期把学习率逐渐降低，让模型在当前数据分布上"稳定收敛"。这个"逐渐降低学习率"的过程就叫 annealing（退火，借用了物理上金属冷却的比喻）。

把学习率衰减到 0 意味着：训练完全停止，模型参数不再更新。这一步的作用是让模型把最后这批数据"吸收干净"，而不是在一个较高的学习率下半途而废。

具体做法：

1. 取一个已经正常训练到约 50% 进度的 8B 模型作为起点（不是从头训练，节省大量计算）
2. 在接下来的 40B token 训练中，把候选数据集的权重设为 30%，其余 70% 仍然是默认数据 mix
3. 在这 40B token 过程中，让学习率从当前值线性降低到 0——也就是说，这 40B token 既是在用候选数据继续训练，也是在做最后的收敛
4. 训练结束后，看 GSM8k 和 MATH 分数相比"没有加入候选数据集、同样做了 annealing"的基线有没有提升

如果提升了，说明候选数据集有价值；如果没有甚至下降，就排除。

为什么这个方法比 scaling law 实验便宜：scaling law 实验需要从头训练多个不同规模的模型；这里只是在一个已有模型上续训 40B token，相当于只跑了完整训练的一小段，计算量小得多。代价是结论没有 scaling law 那么可靠，但对于"要不要加这个数据集"这个二元判断来说已经够用。

这两个工具合在一起，构成了一个**实验驱动的数据决策流程**：不是靠直觉判断"这个数据应该好"，而是跑实验，让 benchmark 数字说话。

#### 但这里有一个根本性的循环问题

作者没有正面回答的一个问题是：**"质量更好"是相对于什么定义的？**

仔细看会发现，整个评估链条最终落在 benchmark 上——GSM8k、MATH、以及"key benchmarks"。这意味着：

1. 数据清洗的每一步，实际上都是在让数据分布更接近"能在这些 benchmark 上得高分"的分布
2. 所谓"质量过滤"，本质上是"过滤掉让 benchmark 分数下降的内容"
3. fastText 质量分类器用 Wikipedia 引用作为正例，隐含的假设是"被 Wikipedia 引用的文本 = 高质量"，而这个假设的验证依据同样是 benchmark

这不是在批评 Llama 3——这是整个领域在 2024 年的普遍做法，也是目前最可操作的方案。但它意味着：**"数据质量"这个词在这里不是一个独立的客观标准，而是"对特定 benchmark 有帮助"的代理指标**。

#### 我的看法

这件事有两层值得记住的含义。

第一层是方法论上的：**你永远无法独立于评估标准来定义数据质量**。Llama 3 选择 benchmark 作为评估标准，DCLM 也是如此，FineWeb 也是如此。不同的是评估标准的覆盖范围和可信度。这解释了为什么随着 benchmark 饱和，数据质量的定义也在跟着演化——不是因为"质量"的本质变了，而是评估工具变了。

第二层是实践上的：**这套"小模型实验 → 预测大模型"的框架是目前最务实的数据决策方法，但它有一个隐性成本**——你需要有足够多的小模型实验预算，以及一套可信的 benchmark 集合。对于资源有限的团队，这个循环根本跑不起来，只能依赖别人跑出来的结论（比如直接用 FineWeb 或 Dolmino 的数据配比建议）。

所以如果你将来要参与数据工程决策，最重要的不是记住"用 MinHash 去重"或"用 fastText 过滤"，而是记住这个更底层的问题：**你用什么 benchmark 来判断数据变好了？这个 benchmark 覆盖了你真正关心的能力吗？**

### 2026 年视角：哪些方法已经过时，哪些值得深入看

#### 可以不用深入的

**fastText 质量分类器（Wikipedia 引用正例）**：这个思路本身没问题，但它的具体做法——用"是否被 Wikipedia 引用"作为质量信号——已经被证明有明显偏差，会系统性地过滤掉口语化文本、对话、创意写作等在 post-training 里很有价值的内容。2024 年之后，主流做法转向用更强的模型（甚至 GPT-4 级别）直接对文档打质量分，或者用 reward model 的信号来反向筛选数据。fastText 本身作为语言识别工具仍然有用，但作为质量过滤器的地位已经明显下降。

**KenLM 困惑度过滤**：ccNet 的核心创新之一，但在今天基本被更强的模型质量过滤取代。KenLM 是 n-gram 模型，它的"困惑度低 = 质量高"这个假设在实践中会过度偏向正式文体，对多样性有损害。如果你不是在研究 2019–2022 年的数据管道历史，可以直接跳过。

**"Dirty word" counting（C4 方法）**：Raffel et al. 2020 年提出时是个实用 trick，但它本质上是一个关键词黑名单，误伤率很高（比如医学文本、新闻报道会被误过滤）。今天的安全过滤已经用专门训练的分类器替代，这个方法作为独立工具基本退出了主流管道。

#### 仍然是基础设施，不会消失的

- **URL-level 去重**：对同一 URL 只保留最新版本，用精确哈希实现，是管道最前端最便宜的一步
- **[MinHash（Document-level 去重）](../20-concepts/minhash.md)**：处理"内容相同但 URL 不同"的近重复，TB 级数据上不可替代；MinHash 文档里也包含了 line-level 去重（删 boilerplate 模板行）的说明
- **[域专属管道（Code/Math）](../20-concepts/domain-specific-pipeline-code-math.md)**：随着 reasoning 能力成为核心竞争力，这个思路反而越来越重要，2026 年已经延伸到合成数据生成 + execution-based 验证

#### 2026 年值得重点看的新方向

- **[DCLM](../20-concepts/dclm.md)**：把数据过滤策略变成可以系统比较的研究变量，结论是强模型过滤优于 fastText/KenLM
- **[FineWeb](../20-concepts/fineweb.md)**：开源高质量数据集 + 详细消融实验文档；FineWeb-Edu 证明"教育价值"比"Wikipedia 引用"更有效作为质量信号
- **[Dolmino](../20-concepts/dolmino.md)**：量化不同数据来源对不同能力的贡献，是数据配比决策的实证参考
- **[合成数据 + 验证](../20-concepts/synthetic-data-with-verification.md)**：2025–2026 年最大的范式变化，用强模型生成 + verifier 过滤，突破了真实数据的质量天花板，目前主要在代码和数学领域有效

**总结一句话**：Llama 3 的管道代表了 2024 年中期"工业级 Web 数据清洗"的成熟做法，其中去重部分仍然是标准，但质量过滤部分已经被更强的模型驱动方法超越，而整个领域的前沿已经从"如何清洗真实数据"转向"如何生成和验证合成数据"。

## Related Pages

- [LLM Learning Roadmap (2026-04-10)](../10-roadmaps/llm-learning-roadmap-20260410.md)
- [Active Questions](../50-questions/active-questions.md)
