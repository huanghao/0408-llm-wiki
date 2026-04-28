# ccNet

## 是什么

ccNet（Common Crawl Network）是 Facebook AI Research 于 2019 年开源的一套 **Common Crawl 数据处理管道**，目标是从原始网页爬取数据中提取高质量的单语言文本。

论文：*CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*（Wenzek et al., 2019）

它不是一个单一算法，而是一套完整的数据工程流程，包含语言识别、质量过滤和去重三个主要环节。

## 为什么值得了解

ccNet 是现代预训练数据管道的重要参考，影响了大量后续工作：

- Llama 3 的行级去重借鉴了 ccNet 的策略
- C4、The Pile、ROOTS、CulturaX 等数据集的处理管道都参考了 ccNet
- 它提出的"用语言模型困惑度做质量过滤"成为了一个标准方法

理解 ccNet 能帮你理解"现代预训练数据集是怎么从 Common Crawl 里做出来的"这个问题。

## 核心流程

```
Common Crawl WARC 文件
    ↓
1. 段落提取（paragraph extraction）
   从 HTML 中提取文本段落，过滤掉过短的段落

    ↓
2. 语言识别（language identification）
   用 fastText lid 模型识别语言，保留目标语言

    ↓
3. Document-level 去重（MinHash LSH）
   去掉近重复文档（见 MinHash 文档）
   目标：整篇文档和另一篇高度相似 → 删掉其中一篇

    ↓
4. 质量过滤（KenLM 困惑度过滤）
   用语言模型困惑度分段，保留"像正常文本"的部分
   目标：整篇文档质量差（乱码、低信息量）→ 删掉整篇

    ↓
5. 行级去重（Line-level Deduplication）
   删除跨文档反复出现的模板行（版权声明、导航菜单等）
   目标：某一行在全局出现太多次 → 从所有文档里删掉这一行，其余行保留

    ↓
高质量单语言文本
```

**三个去重/过滤步骤的目标完全不同**，容易混淆：

| 步骤 | 操作粒度 | 判断标准 | 结果 |
|------|----------|----------|------|
| Document-level 去重 | 整篇文档 | 和另一篇文档高度相似（MinHash Jaccard > 阈值） | 删掉重复的那篇 |
| 困惑度过滤 | 整篇文档 | 文档整体质量差（困惑度高） | 删掉整篇文档 |
| 行级去重 | 单行文本 | 这一行在全局出现次数过多（boilerplate） | 只删这一行，文档其余部分保留 |

Llama 3 说"借鉴 ccNet 的行级去重"，指的是第 5 步，不是困惑度过滤。

## 关键组件详解

### 1. 语言识别

使用 [fastText](./fasttext.md) 的语言识别模型（lid.176.bin），支持 176 种语言。

对每个文档预测语言标签，只保留置信度超过阈值的文档。这一步会过滤掉语言混杂的文档（比如一篇文章里夹杂多种语言）。

### 2. Document-level 去重（MinHash）

ccNet 使用 MinHash + LSH 做文档级去重，参见 [MinHash](./minhash.md) 文档。

ccNet 的特点是按语言分桶做去重，而不是在整个数据集上做。这样既减少了计算量，也避免了跨语言的误判。

### 3. KenLM 困惑度过滤（核心创新）

这是 ccNet 最重要的贡献之一。

**思路**：用在高质量语料（Wikipedia）上训练的 KenLM 语言模型，对每个文档计算**困惑度（perplexity）**——困惑度越低，文档越"像正常文本"。详见 [困惑度](./perplexity.md)。

**ccNet 的特有做法——按百分位分段**：不设固定阈值，而是对整个语言桶的文档按困惑度排序，取第 30 和第 60 百分位切成 head / middle / tail 三段。实践中通常只用 head 或 head+middle。这样阈值自适应，不同语言、不同时期的网页文本分布不同也能正确处理。

**为什么用 KenLM 而不是神经网络 LM**：KenLM 是一个 5-gram 语言模型，速度极快，CPU 上每秒处理数千文档。对于 TB 级数据，这个速度优势是决定性的。模型从 100 万篇 Wikipedia 文章训练而来，用 SentencePiece 分词（词表 65536），`lmplz -o 5` 统计 5-gram 频率，每种语言一个 `.arpa.bin` 文件。计算前文本会先做归一化：转小写、数字替换为 0、去掉标点。

可运行的 demo 见 `tools/kenlm_perplexity_demo.py`。

### 4. 行级去重（Line-level Deduplication）

这是 Llama 3 等后续工作借鉴最多的一个组件。**注意**：这一步和困惑度过滤目标完全不同——困惑度过滤删的是整篇低质量文档，行级去重删的是跨文档反复出现的模板行，文档的其余部分照样保留。

**问题**：很多网页文档的主体内容是独特的，但附带了大量重复的模板文字（boilerplate）：
- 版权声明："© 2023 All rights reserved"
- Cookie 提示："This website uses cookies to improve your experience"
- 导航菜单文字："Home | About | Contact | Privacy Policy"
- 广告语："Click here to subscribe to our newsletter"

这些内容在 document-level 去重时不会被去掉（因为每篇文章的主体是独特的），但它们对语言建模没有价值，反而会让模型学到这些模板短语。

**做法**：
1. 将整个数据集按语言分桶（每桶约 30M 文档）
2. 对每一行文本计算哈希值
3. 统计每个哈希值在桶内出现的次数
4. 出现次数超过阈值的行被标记为 boilerplate，从所有文档中删除

Llama 3 的阈值是 6 次（即一行文本在 3000 万文档里出现超过 6 次就删掉）。

**这个"巨大的 dict"到底是什么**：

哈希函数是 SHA-1 取前 8 字节，得到一个 `uint64` 整数。存储结构是一个 `{uint64 → uint8}` 的哈希表，key 是行的哈希值，value 是 0/1（是否已出现过）。

实现上用的是 C++ 的 `getpy.Dict`（见附录），每个 kv 对占 9 字节。每个分片约占 2GB RAM。这是一个精确的哈希表，不是布隆过滤器，不存在误判。

之所以"一个大 dict 就能解决"，是因为哈希碰撞极低（64-bit hash 碰撞概率约 1/2^64），且实现做了分片（每次只把一组分片加载进内存），而不是把整个数据集的所有行一次性装进去。

**行数量级估算**：Common Crawl 一个月快照约 200 亿文档，每篇文章平均 20 行，总行数约 **4000 亿行**。去掉空行和极短行，有效行约 1000 亿量级。按每行哈希 9 字节，全量存储需要约 900 GB——这就是为什么必须分片：把文档按语言分桶（每桶约 3000 万文档），每桶行数约 6 亿，占内存约 5 GB，勉强可以在一台机器上处理。

**分片加载的读写模式**：想象一个图书管理员要统计全馆所有书里出现超过 6 次的句子。他的做法是：

1. 第一遍从头到尾翻所有书，每遇到一行就在本子上记一笔（哈希 → 计数）。翻完一批书，本子记满了，把本子存起来，换一本新本子继续翻下一批。
2. 第二遍再从头到尾翻所有书，每遇到一行就查本子，计数超过 6 的行删掉。

关键是：翻书的顺序是固定的（文件 1、文件 2、文件 3……），不会跳来跳去。这就是"顺序 I/O"——磁盘最擅长按顺序读，跳跃读（随机 I/O）才慢。分片只是把"本子"分成几本，每次只带一本本子翻完对应那批书，而不是试图把所有书的所有行一次性都记进一本本子（内存装不下）。

## ccNet 的数据质量

ccNet 用这套管道从 Common Crawl 里提取了覆盖 100+ 语言的高质量文本数据集，并公开发布。

对比实验显示，在 ccNet 数据上训练的语言模型，在多语言 benchmark 上明显优于直接用原始 Common Crawl 训练的模型。这验证了"数据质量比数据量更重要"这个直觉。

## 影响和后续工作

| 数据集/项目 | 借鉴 ccNet 的部分 |
|-------------|-------------------|
| C4 | 质量过滤思路 |
| The Pile | 多语言处理管道 |
| ROOTS | 完整借鉴 ccNet 管道 |
| CulturaX | 基于 ccNet 输出数据做进一步清洗 |
| Llama 3 | 行级去重策略 |
| RefinedWeb | 更激进的 ccNet 风格过滤 |

## 局限性

1. **KenLM 困惑度有偏差**：Wikipedia 训练的 LM 会偏好百科风格文本，可能过度过滤掉口语化、对话式文本
2. **行级去重会误伤**：一些有价值的固定短语（比如法律条款、标准定义）也会被过滤掉
3. **语言识别在短文本上不可靠**：很短的段落语言识别置信度低，容易误判

这些局限性在后续工作中都有针对性的改进，但 ccNet 的基本框架仍然是标准参考。

## 附录

### 用困惑度过滤 PDF 转 MD 的乱码

PDF 转 Markdown（如 markitdown）后，图表和公式往往变成乱码行，比如：

```
|--|--|--|--|
$\mathbf{W}_{q}^{(i)} \in \mathbb{R}^{d \times d_{k}}$
Figure 3: ▪ ▪ ▪ ▪ ▪▪ ▪▪▪▪ ▪ ▪ ▪ ▪ ▪▪
```

**可以用困惑度过滤，但有几个注意点**：

1. **行级过滤比文档级更合适**：PDF 转 MD 的文档主体往往是正常文本，只有少数行是乱码，应该按行计算困惑度，删掉高困惑度的行，而不是扔掉整篇文档。

2. **需要用对应语言的 KenLM 模型**：ccNet 提供了 100+ 语言的预训练模型（https://dl.fbaipublicfiles.com/cc_net/lm/）。下载对应语言的 `.arpa.bin`，用 `kenlm` Python 包调用。

3. **阈值需要手动调**：不同类型文档的困惑度分布差异大（学术论文 vs 网页文章）。建议先在一批样本上看分布，再定阈值。

4. **局限**：LaTeX 公式虽然"乱"，但有固定格式，困惑度不一定最高；真正的乱码（乱字符、乱序）才会得到极高困惑度。如果目标是精确过滤公式，正则匹配（`\$...\$`、`\begin{equation}`）比困惑度更准。

**结论**：困惑度过滤适合去掉真正的乱码行（乱字符、随机序列），对结构化但格式特殊的内容（公式、表格）效果有限，两者结合效果最好。

---

### SentencePiece

SentencePiece 是 Google 开源的**无监督分词工具**，不依赖语言规则，直接从文本数据学习分词方式。

**为什么需要它**：中文没有空格，阿拉伯语形态复杂，不同语言的分词逻辑差异极大。SentencePiece 用统一算法处理所有语言——把文本当成字符序列，学习哪些字符组合应该合并成一个 token。

**两种算法**：
- **BPE（Byte Pair Encoding）**：反复合并出现最频繁的相邻字符对。`"机器学习"` 可能被切成 `["机器", "学习"]` 或 `["机器学", "习"]`，取决于训练语料里哪种组合更常见。
- **Unigram**：从大词表开始，反复删除对整体 loss 影响最小的词，直到词表达到目标大小。

ccNet 用 SentencePiece 把文本切成 subword token，再送给 KenLM 统计 n-gram。词表大小 65536 是一个在覆盖率和模型大小之间的权衡。

---

### Kneser-Ney 平滑

n-gram 模型的核心问题：训练语料再大，也不可能覆盖所有可能的 n-gram 组合。遇到没见过的 n-gram，概率估计为 0，整段文本的概率就变成 0，perplexity 变成无穷大。

**Kneser-Ney 的做法**：

1. **折扣（discounting）**：把每个见过的 n-gram 的计数减去一个固定值 D（通常 0.75），把省下来的概率质量分给没见过的 n-gram。
2. **回退（backoff）**：如果 5-gram 没见过，退化到 4-gram；4-gram 没见过，退化到 3-gram，以此类推，直到 unigram。
3. **延续概率（continuation probability）**：unigram 的概率不用词频，而用"这个词出现在多少种不同上下文之后"来估计——出现在越多不同上下文后的词，越可能出现在新上下文里。

实际效果：Kneser-Ney 让没见过的 n-gram 也能得到合理（非零）的概率，避免了 perplexity 爆炸，是 n-gram LM 的标准平滑方法。

---

### getpy

getpy（https://github.com/atom-moyer/getpy）是一个 C++ 实现的 Python 哈希表库，专为大规模数值型 key-value 存储设计。

**和 Python dict 的区别**：Python dict 的每个 entry 都是 Python 对象，带有引用计数、类型指针等元数据，一个 `{int: int}` 的 entry 实际占约 200 字节。getpy 直接按固定类型存储（如 `uint64 → uint8`），每个 entry 只占 9 字节，内存节省约 20 倍。

**代价**：类型固定，不能混存不同类型；API 是批量操作（传 numpy 数组），不支持单条 `dict[key]` 语法；比 Python dict 略慢（但内存省了 20 倍，对大规模场景是值得的）。

ccNet 用它存行哈希表：key 是 `uint64`（SHA-1 前 8 字节），value 是 `uint8`（见过 0/1）。

## 参考

- Wenzek et al., 2019: *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*
- 代码：https://github.com/facebookresearch/cc_net
- KenLM：https://github.com/kpu/kenlm
