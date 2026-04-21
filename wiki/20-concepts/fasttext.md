# fastText

## 是什么

fastText 是 Facebook AI Research（FAIR）于 2016–2017 年开源的文本分类和词向量工具库。它的核心贡献是两件事：

1. **词向量训练**：在 word2vec 基础上引入 subword（字符 n-gram）信息，能处理拼写变体和低频词
2. **文本分类器**：一个极简的浅层神经网络，速度极快，在很多任务上效果接近深度模型

在 LLM 数据处理管道里，fastText 主要以**分类器**的形式出现，而不是词向量工具。

## 为什么值得了解

fastText 分类器有一个很特殊的定位：**它快到可以在 TB 级数据上实时跑**。

这使它成为大规模预训练数据管道里的标准组件：

- 语言识别（language identification）：判断一段文本是哪种语言
- 质量过滤：判断一段文本是否"像高质量来源"
- 内容分类：判断文本是否属于某个类别（新闻、代码、成人内容等）

Llama 3、C4、RefinedWeb、Dolma 等主流预训练数据集都用了 fastText 做语言识别或质量过滤。

## 核心原理

### 模型结构

fastText 分类器的结构非常简单：

```
输入文本
  ↓ 分词 + n-gram 特征提取
  ↓ 词向量查表（embedding lookup）
  ↓ 对所有词向量取平均（bag-of-words 风格）
  ↓ 线性层 + softmax
输出类别概率
```

没有卷积、没有 attention、没有 RNN。就是"把词向量平均一下再做线性分类"。

### 为什么这么简单还有效

1. **Bag-of-words 在分类任务上出奇地强**：对于语言识别、主题分类这类任务，词的分布本身就包含足够信息，不需要建模词序
2. **subword 特征**：对词做字符 n-gram 分解，`running` 会被拆成 `<run`、`runn`、`unni`、`nning`、`ning>` 等片段，这使模型能处理未登录词和拼写变体
3. **负采样训练**：训练速度极快，一台普通机器几分钟就能训练出语言识别模型

**subword 的具体实现**：

词向量训练模式下，默认 n-gram 范围是 minn=3、maxn=6（监督分类模式默认关闭 subword，需手动开启）。n-gram 通过 FNV-1a 哈希映射到一个大小为 2,000,000 的 bucket，每个 n-gram 的 embedding ID = `nwords + (hash % bucket)`。哈希时正确处理 UTF-8 多字节字符边界。

负采样默认抽 5 个负例，负例按词频的 0.5 次方平滑后采样（高频词被适当压低）。

### 速度优势

fastText 在 CPU 上每秒可以处理数十万条文本。相比之下，即使是小型 BERT 模型也慢 2–3 个数量级。

这个速度差距在数据规模达到 TB 时是决定性的：

- Common Crawl 一个月的快照大约有 200–300 亿个文档
- 用 fastText 跑语言识别，几台机器几小时就能处理完
- 用 BERT 类模型，同样任务可能需要几千 GPU 小时

## 在预训练数据管道中的典型用法

### 语言识别

```
fastText 语言识别模型（lid.176.bin）
输入：一段文本
输出：语言标签 + 置信度（如 __label__zh 0.98）
```

这个模型支持 176 种语言，是目前最广泛使用的语言识别工具之一。

**lid.176 的训练细节**：训练数据来自 Wikipedia、Tatoeba（多语言例句库）和 SETimes（新闻语料），使用监督模式 + 字符 n-gram（minn=2, maxn=4）+ hierarchical softmax。未压缩版 126 MB，压缩版（product quantization + feature selection）仅 917 KB，精度损失约 0.5%。

### 质量过滤（Llama 3 的做法）

Llama 3 训练了两个 fastText 质量分类器：

1. **Wikipedia 引用分类器**：正例是被 Wikipedia 引用的网页文本，负例是随机爬取文本。分类器学到的是"像被权威来源引用的文本"的特征
2. **Books 相似度分类器**：正例是书籍文本，用于过滤出文风更正式、信息密度更高的内容

这两个分类器的输出分数会被用来对文档打分，低分文档被过滤掉。

## 局限性

1. **无法建模词序**：bag-of-words 假设，"猫吃鱼"和"鱼吃猫"对它来说是一样的
2. **对长文本不敏感**：平均池化会稀释局部特征，对需要理解段落结构的任务效果差
3. **分类边界粗糙**：质量过滤的"质量"定义取决于训练数据的正负例选取，存在偏差

但这些局限性在大规模数据清洗场景下往往不重要——你不需要完美判断每一条，只需要在 TB 级数据上快速过滤掉明显的低质量内容。

## 和其他工具的关系

| 工具 | 速度 | 精度 | 典型用途 |
|------|------|------|----------|
| fastText | 极快（CPU 秒级） | 中等 | 大规模初筛、语言识别 |
| BERT/RoBERTa | 慢（GPU） | 高 | 精细分类、小规模数据 |
| KenLM | 快（CPU） | 困惑度指标 | 语言模型质量过滤 |
| 规则过滤 | 最快 | 低 | URL 黑名单、字符统计 |

在实际管道里，这几种工具通常是叠加使用的，而不是互斥的。

## 参考

- Joulin et al., 2016: *Bag of Tricks for Efficient Text Classification*
- Bojanowski et al., 2017: *Enriching Word Vectors with Subword Information*
- fastText 官方：https://fasttext.cc
