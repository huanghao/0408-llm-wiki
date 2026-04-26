# 分词与词表（Tokenization）

## 问题背景：开放词表

语言模型需要把文字转成数字 ID，再查嵌入矩阵。最简单的做法是建一个词表，每个词对应一个 ID。

**问题**：词表永远不可能穷举所有词。

- 新词、专有名词、拼写错误：`ChatGPT`、`huanghao03`、`helllo`
- 低频词：训练语料里只出现一两次，统计不可靠
- 多语言：50 种语言的词表加在一起会有几千万个词

这就是**开放词表问题（OOV, Out-of-Vocabulary）**。

---

## 三种解决思路

### 思路一：字符级（Character-level）

把每个字符作为一个 token。词表极小（几百个字符），完全没有 OOV。

**问题**：序列太长。"机器学习" 变成 4 个 token，一篇文章可能有几万个 token，模型需要建模极长的依赖关系，计算量爆炸。

### 思路二：词级（Word-level）

把每个词作为一个 token。序列短，语义单元清晰。

**问题**：词表巨大，OOV 问题严重，无法处理拼写变体。

### 思路三：Subword（子词）

在字符和词之间找平衡：**常见的词保持完整，罕见的词拆成更小的片段**。

```
"tokenization" → ["token", "ization"]
"huanghao03"   → ["hu", "ang", "hao", "03"]
"的"           → ["的"]   ← 高频，保持完整
"机器学习"     → ["机器", "学习"]  或  ["机器学习"]  ← 取决于词表
```

这是目前所有主流 LLM 使用的方案。

---

## Subword 的三种主要算法

### BPE（Byte Pair Encoding）

**原理**：从字符开始，反复合并出现最频繁的相邻字节对。

**训练过程**：

```
初始词表：所有单个字符
语料：["low low low lower newer"]

第1步：统计相邻字节对频率
  l-o: 4次, o-w: 4次, e-r: 2次, n-e: 1次...
  最频繁：l-o → 合并成 "lo"

第2步：更新语料，继续统计
  ["lo-w lo-w lo-w lo-w-e-r n-e-w-e-r"]
  最频繁：lo-w → 合并成 "low"

重复直到词表达到目标大小（比如 50000 个 token）
```

最终词表里既有单个字符（兜底），也有常见词片段，还有完整的高频词。

**GPT 系列用的是 BPE 的变体**：在字节级别操作（Byte-level BPE），把所有文本先转成 UTF-8 字节序列再做 BPE，彻底消除 OOV——任何文本都能被表示，最坏情况退化成逐字节编码。

### WordPiece

**原理**：和 BPE 类似，但合并标准不同——不选频率最高的对，而是选**合并后能最大化训练数据似然**的对。

$$\text{score}(A, B) = \frac{\text{count}(AB)}{\text{count}(A) \times \text{count}(B)}$$

分子是合并后的频率，分母是两个片段各自的频率。这个比值高说明 A 和 B 经常一起出现，合并是有意义的。

**BERT 用 WordPiece**。子词片段用 `##` 前缀标记（`"playing"` → `["play", "##ing"]`）。

### Unigram Language Model

**原理**：从大词表开始，反复删除"去掉后对训练数据 log likelihood 影响最小"的词，直到词表缩小到目标大小。

和 BPE/WordPiece 的方向相反：BPE 是从小到大合并，Unigram 是从大到小裁剪。

**优点**：可以给每种分词方式计算概率，支持采样（训练时随机选择不同的分词方式，增加鲁棒性）。

**SentencePiece 同时支持 BPE 和 Unigram**，ccNet 用的是 Unigram 模式。

---

## LLM 的 token 是怎么做的

以 GPT-4 / LLaMA 为例，流程如下：

**1. 收集训练语料**（几百 GB 到几 TB 的文本）

**2. 在语料上训练 BPE 分词器**

```python
# 用 tokenizers 库（HuggingFace）训练
from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=["corpus.txt"], vocab_size=50000, min_frequency=2)
tokenizer.save_model(".")
```

**3. 得到两个文件**
- `vocab.json`：token → ID 的映射表（词表）
- `merges.txt`：合并规则列表（按优先级排序）

**4. 推理时用合并规则编码**

```
输入："机器学习很有趣"
→ 先转成字节序列
→ 按 merges.txt 里的规则从高优先级到低优先级依次合并
→ 输出 token ID 序列：[23456, 789, 1234]
```

**实际数字**：

| 模型 | 词表大小 | 算法 |
|------|----------|------|
| GPT-2 | 50,257 | Byte-level BPE |
| GPT-4 | ~100,000 | Byte-level BPE (cl100k) |
| LLaMA 3 | 128,256 | Byte-level BPE |
| BERT | 30,522 | WordPiece |
| T5 | 32,000 | SentencePiece Unigram |

词表越大，每个 token 平均包含的字符越多，序列越短，但嵌入矩阵越大。

---

## 分词对模型的影响

**序列长度**：同样一段文字，词表大的分词器切出的 token 更少，模型能处理的"有效文本量"更多。

**跨语言不均衡**：BPE 在英文语料上训练，英文词往往是完整 token，中文、阿拉伯文可能每个字都是单独 token，导致同样的信息量需要更多 token——这是多语言 LLM 的已知问题。

**数字和代码**：`"1234"` 可能被切成 `["12", "34"]` 或 `["1", "2", "3", "4"]`，取决于训练语料里数字的分布。这是 LLM 做数学计算出错的原因之一。

---

## 和 n-gram / fastText 的关系

- **n-gram 语言模型（KenLM）**：通常用空格分词（或 SentencePiece），词表固定，OOV 用平滑处理
- **fastText**：用字符 n-gram 作为 subword 特征，不需要显式词表，直接哈希——这是一种更轻量的 subword 思路，不需要训练分词器
- **LLM**：用 BPE/WordPiece/Unigram 训练专用分词器，词表是模型的一部分，和权重一起发布

三者都在解决同一个问题（OOV + 词表大小），只是在速度、灵活性、表达能力之间做了不同的权衡。

## 参考

- Sennrich et al., 2016: *Neural Machine Translation of Rare Words with Subword Units*（BPE 原始论文）
- Schuster & Nakamura, 2012: *Japanese and Korean Voice Search*（WordPiece）
- Kudo, 2018: *Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates*（Unigram LM）
- HuggingFace tokenizers 文档：https://huggingface.co/docs/tokenizers
- tiktokenizer（可视化 GPT token 切分）：https://tiktokenizer.vercel.app
