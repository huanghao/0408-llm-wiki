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

### 推理：给一段文本打分

推理是训练完成后使用模型的过程，以分类 `"机器学习 很 有趣"` 为例：

```
输入文本："机器学习 很 有趣"

步骤1：分词（按空格切词）
  词列表：["机器学习", "很", "有趣"]

步骤2：每个词展开成字符 n-gram（minn=2, maxn=4），加上词本身
  "机器学习" → 词本身 + ["机器", "器学", "学习", "机器学", "器学习", "机器学习"]
  "很"       → 词本身（太短，无 n-gram）
  "有趣"     → 词本身 + ["有趣"]

步骤3：每个特征通过哈希映射到 embedding ID，查嵌入矩阵取对应行向量
  "机器学习" → hash → ID 42 → [0.2, -0.1, 0.5, 0.3]
  "机器"     → hash → ID 61 → [0.1, -0.2, 0.3, 0.2]
  "很"       → hash → ID  7 → [0.1,  0.4, 0.1, 0.0]
  ...（其余特征同理）

步骤4：对所有向量取平均（bag-of-words）
  avg = 所有向量之和 / 特征数  →  [0.15, 0.12, 0.32, 0.18]

步骤5：线性层 + softmax
  avg × 输出矩阵 B → 各类别得分 → softmax → 概率
  {"正面": 0.82, "负面": 0.18}

输出：正面（置信度 82%）
```

没有卷积、没有 attention、没有 RNN。就是"把词向量平均一下再做线性分类"。

**哈希分桶是什么**：

嵌入矩阵的每一行对应一个特征，需要给每个特征分配一个行号（ID）。fastText 用两段空间：

```
嵌入矩阵行号：
  [  0  ~  V-1  ]  ← 词的 ID，V = 词表大小（训练语料里出现过的不重复词的数量）
  [  V  ~ V+B-1 ]  ← n-gram 的 ID，B = bucket 大小（固定值，默认 200 万）
```

**词表大小 V** 是训练语料里出现过的不重复词的数量，和数据集直接相关（词越多，V 越大）。用 `minCount` 参数过滤低频词可以控制 V 的大小。

**n-gram ID 为什么要加 V**：词和 n-gram 共用同一张嵌入矩阵，为了不冲突，n-gram 的 ID 从第 V 行开始。`n-gram ID = V + (hash(n-gram) % B)` 的意思是：先用哈希把 n-gram 压缩到 0~B-1 范围内，再偏移 V，确保不和词的 ID 重叠。

**B 是固定的，不随数据变化**：不管语料有多大，n-gram 的种类有多少，bucket 始终是 200 万个槽。不同 n-gram 可能映射到同一个槽（哈希碰撞），但 200 万足够大，实践中影响不大。这样嵌入矩阵总大小始终是 `(V + 200万) × DIM`，可以预先分配内存。

**"词向量查表"的意思**：嵌入矩阵是一张大表，每行对应一个词或 n-gram 的向量。给定 ID，直接取对应行——不做乘法，只是内存寻址，速度极快。查表等价于 one-hot 乘矩阵，原理详见 [Word Embedding → Embedding Lookup 的本质](./word-embedding.md)。

### 训练：让模型学会分类

训练阶段用带标签的数据（如 `"这部电影很好看" → 正面`）更新模型参数。

**PyTorch 伪代码**：

```python
import torch
import torch.nn as nn

class FastTextClassifier(nn.Module):
    def __init__(self, vocab_size, bucket_size, dim, n_classes):
        super().__init__()
        # Embedding 层：本质是一个可学习的查找表，shape (V+B, DIM)
        # 训练时等价于 Linear(V+B, DIM, bias=False)，但推理时只做数组下标操作
        self.embedding = nn.Embedding(vocab_size + bucket_size, dim)
        # 分类头：线性层，无激活函数（CrossEntropyLoss 内部已包含 softmax）
        self.classifier = nn.Linear(dim, n_classes)

    def forward(self, ids):          # ids: (N,)  N = 词数 + 字符n-gram数
        x = self.embedding(ids)      # (N, DIM)  查表
        x = x.mean(dim=0)            # (DIM,)    取平均
        logits = self.classifier(x)  # (C,)      线性变换，无激活
        return logits

# 训练
model = FastTextClassifier(V, B=2_000_000, dim=16, n_classes=176)
loss_fn = nn.CrossEntropyLoss()      # 内部做 softmax + log + NLL
optimizer = torch.optim.Adam(model.parameters())

logits = model(ids)                  # 前向
loss = loss_fn(logits.unsqueeze(0), label)  # 计算损失
loss.backward()                      # 反向传播
optimizer.step()                     # 更新参数
```

**Embedding 层的本质**：

- **训练时**：逻辑上等价于 `Linear(V+B, DIM, bias=False)`，即一个矩阵乘法——one-hot 向量乘以权重矩阵，梯度可以反向传播更新权重。
- **推理时**：实现上只做数组下标操作 `weight[id]`，直接取矩阵对应行，不做任何乘法，极快。
- **两者数学等价**，但推理时的实现更高效。PyTorch 的 `nn.Embedding` 就是这样实现的。

**分类头要不要激活函数**：

分类头 `Linear(DIM, C)` **不加激活函数**，直接输出 logits（未归一化的分数）。激活函数（softmax）放在 loss 函数里：`CrossEntropyLoss = softmax + log + NLLLoss`。这样做数值更稳定（避免 softmax 上溢/下溢），是 PyTorch 的标准做法。推理时如果需要概率，用 `F.softmax(logits)` 手动加。

**DIM 多大合适**：

DIM=100 对于一般分类任务完全够用，fastText 的官方 benchmark 多用 100–300。设置原则：

| 场景 | 推荐 DIM | 理由 |
|------|----------|------|
| 语言识别（lid.176） | 16 | 任务简单，语言间差异大，低维足够 |
| 通用文本分类 | 100–200 | 平衡精度和速度 |
| 质量过滤分类器 | 100 | DCLM/Llama 3 的实践 |
| 词向量（无监督） | 300 | 需要捕捉更细腻的语义 |

DIM 越大，表达能力越强，但参数量和推理时间线性增加。fastText 的优势在于 DIM 可以很小（16–100）仍然有效，因为任务本身不复杂。

**OH-2.5 和 ELI5 分类器是什么**：

DCLM 里提到的 OH-2.5 和 ELI5 是两个 fastText 分类器的**训练数据来源**，不是两种不同的模型架构。它们都是"embedding + 取平均 + 线性分类头"这个结构，参数规模也相近（DIM≈100，V+B≈200万），区别只在于：

| | OH-2.5 分类器 | ELI5 分类器 |
|--|--------------|-------------|
| 正例来源 | OpenHermes 2.5（高质量指令微调数据，问答/对话） | Reddit r/explainlikeimfive（用简单语言解释复杂概念） |
| 负例 | 随机网页文本 | 随机网页文本 |
| 学到的"质量"定义 | 像精心整理的问答文本 | 像通俗易懂的解释性文本 |
| 训练样本量 | 约数十万条 | 约数十万条 |

两个分类器学到的"高质量"定义略有不同，混合使用可以覆盖更广的质量维度。DCLM 的实验发现 OH-2.5+ELI5 混合的效果比单独使用任一个都好。

之所以都叫"fastText 分类器"，是因为模型结构完全相同，只是训练数据不同——这正是 fastText 的一个优点：换一批训练数据就能得到针对不同目标的分类器，成本极低。

lid.176 实际参数量：V ≈ 数十万词，B = 200万，DIM = 16，C = 176。
嵌入矩阵约 `(V+200万) × 16`，分类层 `16 × 176`——参数几乎全在嵌入矩阵里。

训练循环：`loss.backward()` 自动反向传播，`optimizer.step()` 更新参数。

**稀疏更新**：每次 forward 只用到几十个特征 ID，反向传播梯度也只流到这几十行，其余行不更新——更新代价和文本长度成正比，而不是词表大小。这是 fastText 在大词表上高速训练的核心原因。原理详见 [Word Embedding → Embedding Lookup 的本质](./word-embedding.md)。

**嵌入矩阵从哪来**：训练开始时全是随机数。通过大量样本的反复更新，正面词的向量逐渐靠近，负面词的向量靠近，两组向量互相远离。详见 [Word Embedding](./word-embedding.md)。

### 两级特征

fastText 同时用两种粒度的特征：

- **词级（word n-gram）**：把相邻词组合成特征，捕捉局部词序，如 `["机器", "学习"]` → bigram `"机器_学习"`。用 `-wordNgrams` 参数控制，默认=1（只用单词）。
- **字符级（character n-gram / subword）**：在每个词内部切字符片段，`running` → `<run`、`runn`、`ning>`…，目的是处理未登录词和拼写变体。词表外的词（OOV）也能通过共享的字符 n-gram 得到合理的向量。

**为什么叫 subword**：sub = 子，subword = 词的子单元，比词更细粒度但不是单个字符。

### 速度优势

fastText 在 CPU 上每秒可以处理数十万条文本。相比之下，即使是小型 BERT 模型也慢 2–3 个数量级。

这个速度差距在数据规模达到 TB 时是决定性的：

- Common Crawl 一个月的快照大约有 200–300 亿个文档
- 用 fastText 跑语言识别，几台机器几小时就能处理完
- 用 BERT 类模型，同样任务可能需要几千 GPU 小时

**实测数据**（见 `tools/fasttext_demo.py`）：在 M 系列 Mac CPU 上，用 900 条数据训练一个三分类（英/中/法）语言识别模型只需 0.15 秒，推理吞吐达 **81 万条/秒**，单条延迟 0.001 毫秒。BERT-base 在 GPU 上约 50–200 条/秒，差距约 4000–16000 倍。

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

**标签从哪来，需要多少训练数据**：

标签是自动构造的，不需要人工标注：
- Wikipedia 引用分类器：Wikipedia 的参考文献列表里有大量外链 URL。把这些 URL 对应的网页文本作为正例，从 Common Crawl 随机采样同等数量的文本作为负例，标签就有了。
- Books 分类器：直接用书籍数据集（如 Books3、Project Gutenberg）作为正例，随机爬取文本作为负例。

数据量方面，fastText 分类器对训练数据量要求不高，通常几万到几十万条就够用（Llama 3 技术报告没有公开具体数字，但 fastText 的典型用法在 10 万级样本上就能达到很好的效果）。这也是 fastText 相比深度模型的优势之一：数据效率高。

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

## 附录：fasttext Python 库使用

安装：`pip install fasttext-wheel`

### 训练分类器

训练文件每行格式：`__label__类名 文本内容`

```python
import fasttext

# 训练
model = fasttext.train_supervised(
    input="train.txt",   # 每行 "__label__xx 文本"
    epoch=10,
    lr=0.5,
    dim=16,
    wordNgrams=2,        # 加 word bigram 特征
)

# 推理（单条）
label, prob = model.predict("this is english text")
# label = ('__label__en',), prob = array([0.99])

# 推理（批量，更快）
labels, probs = model.predict(["text one", "text two"])

# 保存 / 加载
model.save_model("model.bin")
model = fasttext.load_model("model.bin")
```

### 加载预训练模型

fastText 官方提供两类预训练模型（https://fasttext.cc/docs/en/english-vectors.html）：

| 模型 | 大小 | 用途 |
|------|------|------|
| `lid.176.bin` | 126 MB | 语言识别，支持 176 种语言 |
| `lid.176.ftz` | 917 KB | 语言识别压缩版，精度损失 <0.5% |
| `cc.en.300.bin` | ~4 GB | 英文词向量（Common Crawl + Wikipedia，300维） |
| `cc.zh.300.bin` | ~4 GB | 中文词向量 |
| `wiki.en.bin` | ~9 GB | 英文词向量（Wikipedia，300维） |

实际使用最多的是 `lid.176.ftz`（不到 1 MB，够用）。词向量模型体积大，通常只在需要迁移学习时才下载。

```python
model = fasttext.load_model("lid.176.ftz")
label, prob = model.predict("今天天气很好")
# label = ('__label__zh',), prob = array([0.999])
```

### 训练词向量

词向量训练是无监督的，不需要标签，只需要大量文本。流程：

1. 准备语料文件（每行一句话，纯文本）
2. 选择模型：`skipgram`（给定中心词预测上下文）或 `cbow`（给定上下文预测中心词）
3. 训练后可以查询任意词的向量，包括 OOV 词（通过字符 n-gram 合成）

```python
# 训练（skipgram 通常比 cbow 效果好，但慢一些）
model = fasttext.train_unsupervised(
    "corpus.txt",
    model="skipgram",   # 或 "cbow"
    dim=100,
    epoch=5,
    minn=3, maxn=6,     # 字符 n-gram 范围
)

# 查词向量
vec = model.get_word_vector("running")     # shape (100,)，词表内
vec = model.get_word_vector("runing")      # 拼写错误也能得到向量（通过 n-gram）

# 查最近邻
model.get_nearest_neighbors("猫", k=5)    # 返回最相似的 5 个词
```

词向量训练的原理详见 [Word Embedding](./word-embedding.md)。

常用参数一览：

| 参数 | 默认 | 说明 |
|------|------|------|
| `epoch` | 5 | 训练轮数 |
| `lr` | 0.1 | 学习率 |
| `dim` | 100 | 向量维度 |
| `wordNgrams` | 1 | word n-gram 最大阶数 |
| `minn`/`maxn` | 0/0 | 字符 n-gram 范围（分类默认关闭） |
| `minCount` | 1 | 词频阈值 |

完整可运行示例见 `tools/fasttext_demo.py` Demo 2。

## 参考

- Joulin et al., 2016: *Bag of Tricks for Efficient Text Classification*
- Bojanowski et al., 2017: *Enriching Word Vectors with Subword Information*
- fastText 官方：https://fasttext.cc
