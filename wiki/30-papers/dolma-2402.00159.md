# Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research (Soldaini et al., 2402.00159)

一句话总结：AI2 发布的 3 万亿 token 开放英语预训练语料库，配套开源数据管理工具包，为 OLMo 系列模型提供数据基础，也是同类语料库中文档最详尽的——设计原则、每个过滤决策的消融实验、数据 datasheet 全部公开。

## 基本信息

- 论文：Dolma: an Open Corpus of Three Trillion Tokens for Language Model Pretraining Research
- 作者：Luca Soldaini, Rodney Kinney, Akshita Bhagia, Dustin Schwenk 等（AI2 核心团队 + 合作机构）
- 机构：Allen Institute for AI（AI2），UC Berkeley，CMU，University of Washington，MIT 等
- arXiv：[2402.00159](https://arxiv.org/abs/2402.00159)（v2，2024 年 6 月）
- 发表：ACL 2024
- 数据集：[hf.co/datasets/allenai/dolma](https://huggingface.co/datasets/allenai/dolma)
- 工具包：[github.com/allenai/dolma](https://github.com/allenai/dolma)
- 本地原文：`raw/inbox/2402.00159.pdf`

## 核心问题

当前大多数高性能语言模型要么不公开训练数据，要么公开但不说明数据策划细节——这使得科学研究（数据如何影响模型能力？偏见从哪里来？）难以开展。能否构建一个**规模与主流模型训练数据相当、数据策划过程完全透明、所有工具全部开源**的预训练语料库？

## 方法 / 核心机制

### 数据来源与规模

Dolma v1.6 共 **3 万亿 token**（LLaMA tokenizer 计算），来自 6 类来源，从约 200 TB 原始文本清洗至 11 TB 最终数据集：

| 来源 | 类型 | UTF-8 字节（GB）| 文档数（百万）| Llama tokens（十亿）|
|---|---|---|---|---|
| Common Crawl | 网页 | 9,812 | 3,734 | 2,479 |
| GitHub | 代码 | 1,043 | 210 | 411 |
| Reddit | 社交媒体 | 339 | 377 | 89 |
| Semantic Scholar | 论文 | 268 | 38.8 | 70 |
| Project Gutenberg | 书籍 | 20.4 | 0.056 | 6.0 |
| Wikipedia/Wikibooks | 百科 | 16.2 | 6.2 | 4.3 |
| **总计** | | **11,519** | **4,367** | **3,059** |

注意：Common Crawl 占约 80% 的 token 量，Reddit 代码 等非网页来源提供多样性。

### 设计原则

论文明确了 4 条设计原则：

1. **与现有 LM 训练配方保持一致**：遵循 LLaMA、Gopher 等已验证的数据来源选择，降低设计风险
2. **基于证据做决策**：每个数据处理选择都通过消融实验（训练 1.2B 模型到 150B token）验证，而不仅凭直觉
3. **保持开放以便复现**：共享数据本身 + 策划过程的完整文档，避免 Books3 等版权存疑来源
4. **英语为主**：明确聚焦英语，以最大化对现有语言模型研究的通用性

### Dolma Toolkit

论文同时发布了高性能开源数据管理工具包，统一了两类核心操作：

- **filtering**：将语言过滤、质量过滤、内容过滤统一为一个接口，支持文档/段落/句子级别操作，可配置评分方法（语言模型困惑度、正则表达式、线性分类器）和处理动作（删除、替换、保留）。在内部测试中处理速率为 122 CPU 小时/TB，处理 200 TB 全量数据需约 5 天（c6a.48xlarge，192 vCPU）
- **mixing**：统一上下采样、去重、去污染操作，基于 Rust 实现，支持 Bloom filter 用于近线性时间重复检测

### 网页子集（Dolma-Web）处理流程

**最大的子集（2.28T token，来自 Common Crawl 2020-05 到 2023-06 共 25 个快照）**，处理管线分四步：

1. **语言过滤（CCNet）**：用 fastText 语言识别，保留英语得分 ≥ 0.5 的页面（过滤掉 61.7% 数据）；CCNet 同时通过段落级分组去除极常见段落（headers/navigation，约占 70%）；总计从 175.1 TB 过滤到 27.7 TB

2. **质量过滤（Gopher + C4 规则）**：结合 Gopher 全套规则（Gopher All）和 C4 的一条规则（C4 NoPunc，移除不以标点结尾的段落）。消融实验显示：
   - 单独用 C4 All：中等效果
   - 单独用 Gopher All：较好
   - **Gopher All + C4 NoPunc：最优**，在 HellaSwag 等评测上最高
   - KenLM 困惑度分桶（Wikipedia-like 高/中/低质量）与启发式规则正交，不互相替代

3. **内容过滤**：
   - **毒性内容**：训练专用 FastText 分类器（基于 Jigsaw Toxic Comments 数据集），分别识别"hate"和"NSFW"内容，在句子级别过滤。选择宽松阈值（τ=0.0004）以保证数据量；严格阈值（τ=0.4）移除内容更少（5.5–7.3%）但下游任务表现反而更差
   - **个人可识别信息（PII）**：用正则表达式（而非 Presidio 等重量级工具）检测邮件地址、IP 地址、电话号码，≤5 个 PII 的文档替换为占位符，高密度 PII 文档整体删除（影响 0.02% 文档）

4. **去重（三阶段）**：
   - **URL 精确去重**：过滤 53.2% 文档
   - **文档精确去重**：在 URL 去重后再过滤 14.9%
   - **段落精确去重**：用 Bloom filter 过滤 18.7% 的段落（含 boilerplate：同一作者署名行等）

### 代码子集（Dolma-Code）

来自 GitHub（经 the Stack 预处理），过滤 RedPajama v1 规则 + StarCoder 规则组合（消融显示比单独使用任一更好），411B token，24 种编程语言，去重使用 the Stack 的 MinHash + LSH 流程。

### 社交媒体子集（Dolma-Social）

来自 Reddit（Pushshift，2005–2023），80B token。关键处理：
- 保留 ≥3 票的评论（排除低质量/深层嵌套内容）
- 原子化内容（每条提交/评论独立文档）消融实验表现最好，优于整合线程
- 排除 26,123 个 banned 或 NSFW 子版块

## 关键结果 / 数据

### OLMo-1B 对比（Table 2）

论文用 Dolma 训练 1.2B 参数的 OLMo-1B 模型，在 8 个评测数据集上与同规模开源模型对比（零样本）：

| 模型 | 平均 |
|---|---|
| StableLM-2 (1.6B) | 66.5 |
| Pythia (1.1B) | 54.5 |
| TinyLlama (1.1B) | 59.4 |
| **OLMo-1B** | **60.3** |

OLMo-1B 好于除 StableLM-2 外的所有同规模基线（StableLM-2 用了 2T token 训练 2 epoch，数据构成未公开）。

### 数据多样性（Figure 5）

在 Paloma（多领域困惑度评测集）上对比 1.2B 模型：
- C4、mC4、RefinedWeb 等**单一来源语料**：困惑度高，对多样领域覆盖差
- Pile：多样性好（多来源），但规模较小
- **Dolma 和 RedPajama v1**：曲线接近 Pile，表明多来源组合即使包含大量网页也能保持良好的领域覆盖

### 消融实验的主要发现

- 质量/内容/去重三类过滤**叠加有正向效果**（Figure 3），组合使用比任一单独使用都好
- 模型和启发式质量过滤器**正交**（过滤相同内容的比例很少重叠），说明两者不互相替代
- 毒性内容宽松阈值（Low Threshold，保留更多数据）比严格阈值下游任务表现**更好**——过于激进的内容过滤反而损害模型性能
- Reddit 数据中，原子化内容（每条评论/提交独立）优于整合线程格式

## 局限性

- **仅英语**：fastText 语言识别可能存在漏网的非英语内容；明确不支持多语言场景
- **评测数据集不完整**：消融实验只用 8 个任务，且仅测了 1.2B 参数的小模型，可能不反映更大规模模型的行为
- **无法全面人工审查**：11TB 数据无法靠人工全面检查内容质量、偏见和潜在危害
- **来源代表性有限**：不包含版权受限数据（如 Books3），无法复现用了这类数据的模型
- **法律不确定性**：网页爬取数据的版权和公平使用仍是法律灰色地带，论文坦承这一点

## 现状与影响

一句话定性：**Dolma 是截至 2025 年文档最详尽的大规模开放英语预训练语料库，作为 OLMo 系列模型的官方训练数据仍在积极维护；其开源工具包（Dolma Toolkit）成为社区构建预训练数据集的重要基础设施。**

截至 2026 的影响：

- **OLMo 系列**：OLMo（1B/7B）、OLMo 2（7B/13B）都使用 Dolma 或其演进版本；Dolma v1.7 相比论文版本有显著下游任务提升，持续更新
- **数据透明度标准**：Dolma 附带的详尽 Datasheet（Appendix N）和消融实验报告成为同类工作的参照标准，推动了"发布数据集必须说明策划决策"的社区规范
- **工具包影响**：Dolma Toolkit 的 filtering + mixing 抽象被多个后续数据集项目借鉴
- **相对局限**：FineWeb（HuggingFace，2024）在网页数据过滤质量上做了更细致的研究，对于纯网页数据场景可参考性更强；但 Dolma 的多来源混合设计（代码+论文+社交+书籍）在领域覆盖上更全面

## 和 wiki 内其他概念的关联

- [ccNet](../20-concepts/ccnet.md)：Dolma-Web 的语言过滤和初步去重直接使用 CCNet 管线，是 CCNet 在大规模生产使用中的典型案例
- [fastText](../20-concepts/fasttext.md)：用于语言识别（保留英语文档）和毒性分类器基础
- [MinHash](../20-concepts/minhash.md)：代码子集去重使用 MinHash + LSH（来自 the Stack）
- [Perplexity](../20-concepts/perplexity.md)：KenLM 困惑度分桶作为质量信号（high/medium/low Wikipedia-like 内容），与启发式过滤正交
- [FineWeb](../20-concepts/fineweb.md)：同期网页数据集，专注更精细的质量过滤，两者侧重互补（Dolma 多来源混合 vs FineWeb 专注网页质量）
- [DCLM](../30-papers/dclm-2406.11794.md)：DataComp-LM，同样使用消融实验对比不同过滤策略，方法论与 Dolma 类似但更系统化
- [Gopher](../30-papers/gopher-2112.11446.md)：Dolma 的质量过滤规则直接来自 Gopher/MassiveText，Gopher All 规则是 Dolma 质量过滤的核心组成

## 值得看的部分 / 相关资料

- **Table 1（数据来源统计）**：一眼看出 3T token 的来源分布，Common Crawl 占主导
- **Section 4.2（Data Ablations）**：介绍消融实验方法论——1.2B 模型 + 150B token + 8 个评测集的框架，每个过滤决策都有消融支撑
- **Figure 1–3**：质量过滤、毒性过滤、三者叠加的消融曲线（HelaSwag 为代表），直观看到每类过滤的边际贡献
- **Section 5（Curating Dolma-Web）**：最详细的部分，完整管线：CCNet → 质量过滤 → 内容过滤 → 去重，每步都有消融数据
- **Section 9.2（Measuring Domain Fit）**：用 Paloma 评测数据多样性，图 5 清晰展示不同语料库对多领域覆盖的差异
- **Appendix N（Datasheet）**：完整数据集文档，包括来源、过滤规则细节、法律和伦理考量——是阅读数据集文档的参考范本
- 数据集主页：`hf.co/datasets/allenai/dolma`
- 工具包：`github.com/allenai/dolma`
- 后续：OLMo 2 技术报告（2024）介绍了在 Dolma 基础上的进一步改进

## 附录：消融实验计算成本粗估

论文没有给出总算力开销，但给出了足够的参数可以估算。

### 已知参数

| 参数 | 数值 | 来源 |
|---|---|---|
| 每次消融模型规模 | 1.2B 参数 | Section 4.2 |
| 每次消融训练量 | 150B tokens（early stop，非整 epoch）| Section 4.2 |
| 数据处理速率 | 122 CPU 小时/TB | Section 4.1 |
| 数据处理总量 | ~200 TB | Section 5 |
| 数据处理机器 | c6a.48xlarge（192 vCPU）| Section 4.1 |
| 数据处理总时间 | ~5 天 | Section 4.1 |

### 单次消融训练成本估算

150B tokens 是**固定的训练预算上限**，不是 epoch × 数据集大小。论文明确说"training models to completion is prohibitively expensive"，所以每次消融都在 150B tokens 处 early stop——这相当于在 Dolma 3T token 总量上走了约 5% 的一段，远不到一个完整 epoch。每次消融用的训练数据可以不同（不同过滤策略产生不同子集），150B token 是固定切断点，不是从某个固定数据集走若干 epoch 得来的。

训练 1.2B 模型到 150B tokens 的 FLOPs：

$$C \approx 6 \times N \times D = 6 \times 1.2 \times 10^9 \times 1.5 \times 10^{11} \approx 1.1 \times 10^{21} \text{ FLOPs}$$

换算成 A100（312 TFLOPS bfloat16，MFU 约 40%）：

$$t = \frac{1.1 \times 10^{21}}{312 \times 10^{12} \times 0.4} \approx 8,800 \text{ GPU 小时} \approx 37 \text{ 天（单卡）}$$

实际训练通常用 64–128 张 A100 并行，则每次消融约需 **1.5–3 天**。

### 消融实验总数估算

从论文的消融曲线来看：
- Figure 1（质量过滤）：约 5 组对比
- Figure 2（毒性过滤）：约 4 组对比
- Figure 3（三者叠加）：约 4 组
- Figure 4（Reddit 格式）：约 4 组
- Figure 5（跨语料对比）：约 6 组
- 代码子集、社交媒体子集各有独立消融

保守估计 **25–35 次独立训练运行**，含重复实验约 **30 次**。

### 总成本粗估

| 项目 | 估算 |
|---|---|
| 单次消融 GPU 小时（64 卡）| ~140 GPU 小时 |
| 总消融次数 | ~30 次 |
| 消融总 GPU 小时 | **~4,200 GPU 小时**（≈ 175 A100 天） |
| 数据处理（CPU）| 122 × 200 TB ≈ 24,400 CPU 小时（≈ 5 天 × 192 vCPU）|

**折合云计算成本（A100 约 $3/小时，c6a.48xlarge 约 $7/小时）：**
- 消融训练：~$12,600
- 数据处理：~$170,000（主要是 CPU 密集型，但时间短）

**数量级结论**：消融实验部分约花费 **数万美元 + 数百 A100 天**；数据处理管线本身主要是 CPU 密集型工作，GPU 占用很少。这个规模对于学术实验室来说不小，但远低于训练 7B+ 模型的成本——1.2B early stop 的设计选择本身就是在压缩消融成本。

> **注意**：以上为粗估，误差 2–3×。论文未报告实际 GPU 小时或美元成本，Appendix D.1 只描述了训练配置（batch size、learning rate schedule）而非总算力。
