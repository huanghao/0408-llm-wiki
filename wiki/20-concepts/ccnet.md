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

    ↓
4. 质量过滤（KenLM 困惑度过滤）
   用语言模型困惑度分段，保留"像正常文本"的部分

    ↓
高质量单语言文本
```

## 关键组件详解

### 1. 语言识别

使用 fastText 的语言识别模型（lid.176.bin），支持 176 种语言。

对每个文档预测语言标签，只保留置信度超过阈值的文档。这一步会过滤掉语言混杂的文档（比如一篇文章里夹杂多种语言）。

### 2. Document-level 去重（MinHash）

ccNet 使用 MinHash + LSH 做文档级去重，参见 [MinHash](./minhash.md) 文档。

ccNet 的特点是按语言分桶做去重，而不是在整个数据集上做。这样既减少了计算量，也避免了跨语言的误判。

### 3. KenLM 困惑度过滤（核心创新）

这是 ccNet 最重要的贡献之一。

**思路**：用在高质量语料（Wikipedia）上训练的 n-gram 语言模型，对每个文档计算**困惑度（perplexity）**。困惑度越低，说明文档越"像正常文本"。

**困惑度的直觉**：如果你用一个在正常文本上训练的语言模型去预测一段文字，预测越准（困惑度越低），说明这段文字越符合正常文本的分布。

- 低困惑度：流畅的自然语言，结构清晰
- 高困惑度：乱码、随机字符串、高度重复的内容、代码混杂文本

**分段策略**：ccNet 把数据集按困惑度分成三段（head/middle/tail），分别对应高/中/低质量。实践中通常只用 head 或 head+middle。

**为什么用 KenLM 而不是神经网络 LM**：KenLM 是一个 n-gram 语言模型，速度极快，可以在 CPU 上每秒处理数千文档。对于 TB 级数据，这个速度优势是决定性的。

### 4. 行级去重（Line-level Deduplication）

这是 Llama 3 等后续工作借鉴最多的一个组件。

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

## 参考

- Wenzek et al., 2019: *CCNet: Extracting High Quality Monolingual Datasets from Web Crawl Data*
- 代码：https://github.com/facebookresearch/cc_net
- KenLM：https://github.com/kpu/kenlm
