# MinHash

## 是什么

MinHash（Minimum Hash）是 Andrei Broder 于 1997 年提出的一种近似集合相似度算法，核心目标是：**在不两两比较的情况下，快速找出大量集合中的近重复对**。

在 LLM 预训练数据处理中用来做 document-level 去重：判断两篇文档是否"基本相同"（复制粘贴、轻微改写）。

**为什么不直接用精确哈希**：精确哈希只能找到完全相同的文档，改一个词就产生完全不同的哈希值。MinHash 对轻微改写有鲁棒性。

**为什么不直接两两比较**：1 亿篇文档两两比较需要 $10^{16}$ 次操作，完全不可行。MinHash + LSH 可以把复杂度降到接近线性。

---

## 一、MinHash 与 Jaccard 相似度的关系

### Jaccard 相似度

MinHash 估计的目标是 Jaccard 相似度：

$$J(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

直觉：两个集合有多少重叠，范围 0（完全不同）到 1（完全相同）。

### 这里用的哈希函数是什么

MinHash 里的哈希函数和密码学哈希（MD5、SHA）的目的不同，但底层要求有一个共同点：**把输入映射到一个均匀分布的输出空间**。

具体要求：
- **均匀性**：集合里的每个元素，被哈希到任意位置的概率相等。这样"最小值"才不会系统性地偏向某类元素。
- **独立性**：$k$ 个哈希函数之间要相互独立，这样 $k$ 次估计才是独立的随机变量，才能用大数定律收敛。
- **不需要抗碰撞**：MinHash 不在乎两个不同元素哈希到同一个值（碰撞），因为它只关心"最小值是否相等"这个事件，不需要从哈希值反推原始元素。

实践中通常用 **MurmurHash** 或 **xxHash** 这类非密码学哈希函数——它们均匀性好、速度极快（比 SHA256 快 10–100 倍），不需要密码学安全性。用 $k$ 个不同的种子（seed）初始化同一个哈希函数，就能得到 $k$ 个"独立"的哈希函数（严格说是近似独立，实践中足够）。

### 单个哈希函数的情况

对于一个随机哈希函数 $h$，有如下性质：

$$P\bigl(\min_{x \in A} h(x) = \min_{x \in B} h(x)\bigr) = J(A, B)$$

**证明**

把 $A \cup B$ 里的所有元素按哈希值从小到大排列。因为哈希均匀，这个顺序等价于对 $A \cup B$ 做随机排列，每个元素排第一的概率相等。

关键等价关系：**"排第一的元素属于 $A \cap B$"↔"$A$ 和 $B$ 的最小哈希值相等"**

为什么等价？分两个方向：

→ 方向：设排第一的元素 $x \in A \cap B$。$x$ 同时在 $A$ 里，所以 $x$ 是 $A$ 里哈希值最小的元素（因为 $x$ 在整个 $A \cup B$ 里都排第一）；同理 $x$ 也是 $B$ 里哈希值最小的元素。于是 $\min_A h = h(x) = \min_B h$，两集合最小哈希值相等。

← 方向：设 $\min_A h = \min_B h = v$。设 $A$ 里哈希值为 $v$ 的元素是 $x$，$B$ 里哈希值为 $v$ 的元素是 $y$。因为哈希函数对不同输入产生相同值的概率极低（近似忽略），$x = y$，即同一个元素同时在 $A$ 和 $B$ 里，$x \in A \cap B$。这个元素就是排第一的那个。

所以：

$$P(\text{最小哈希相等}) = P(\text{排第一的元素} \in A \cap B) = \frac{|A \cap B|}{|A \cup B|} = J(A, B)$$

**用 max 或 median 可以吗？**

可以，任何固定的排名位置（第一、最后、第 $k$ 小）都满足同样的等价关系，证明结构完全相同。实践中用 min 有两个工程优势：哈希值通常是无符号整数，min 天然有界且实现简单；此外 min 在流式处理（一次遍历更新）中比 median 更高效。

所以单个哈希函数给出的是一个 0/1 的随机变量：相等（1）或不相等（0），期望等于 $J(A,B)$，但方差很大——单次估计要么完全对要么完全错。

### k 个哈希函数：收敛速度

用 $k$ 个独立的随机哈希函数，对每个集合取 $k$ 个最小哈希值，估计量是这 $k$ 次 0/1 结果的平均值，是 $J$ 的无偏估计，标准误差为：

$$\text{SE} = \sqrt{\frac{J(1-J)}{k}} \leq \frac{0.5}{\sqrt{k}}$$

上界在 $J = 0.5$ 时取到（最难估准的情况）。

| k | 最坏情况标准误差 |
|---|---|
| 64 | ±0.063 |
| 128 | ±0.044 |
| 256 | ±0.031 |
| 512 | ±0.022 |

标准误差是估计值的**波动幅度**，不是错误概率。$k=128$、真实 $J=0.8$ 时，95% 情况下估计值落在 $[0.730, 0.870]$ 以内。统计原理的完整推导见：[统计估计基础](./statistical-estimation-basics.md)。

**k = 128 的时间开销**：$n$ 词的文档生成 $n - 4$ 个 5-gram（每个窗口滑动一步），所以 200 词 ≈ 196 个 5-gram，1000 词 ≈ 996 个 5-gram。1000 词大约是 2–3 页 A4，对应约 1000 个 5-gram，计算 128 个哈希值约需 **1–6ms**（MurmurHash，单核，128,000 次哈希操作）。k 翻倍时间线性翻倍，但标准误差只减少 $1/\sqrt{2} \approx 30\%$，边际收益递减——这是选 128–256 而不是 512 的主要原因。

---

## 二、特征化：把文档变成集合

MinHash 需要把文档表示成一个**集合**才能计算 Jaccard。特征化的选择直接影响"相似"的定义。

### n-gram（最常用）

把文本切成连续的 n 个 token 的片段，取所有片段构成集合。

**字符级 n-gram**：
```
文本: "hello"
3-gram: {"hel", "ell", "llo"}
```

**词级 n-gram（shingle）**：
```
文本: "the quick brown fox"
2-gram: {"the quick", "quick brown", "brown fox"}
```

**n 的选择**：
- n 太小（1-gram）：集合元素太常见，不同文档的 Jaccard 虚高，去重精度差
- n 太大（10-gram）：集合元素太稀疏，轻微改写就会导致相似度骤降，漏报多
- 文本去重通常用 **5-gram 词级**：对词序变化有一定鲁棒性，又不会过于稀疏

### 不同领域的典型特征化方式

| 领域 | 特征化方式 | 原因 |
|------|-----------|------|
| 文本去重（NLP） | 词级 5-gram | 对改写鲁棒，粒度合适 |
| 网页去重（搜索引擎） | 字符级 3–5-gram | 对语言无关，处理多语言更方便 |
| 代码去重 | token 级 n-gram（AST token） | 代码 token 比自然语言 token 更有语义 |
| 基因组学 | k-mer（DNA 子序列） | DNA 序列天然是字符集合，k=21 是常用值 |
| 推荐系统 | 用户行为集合（点击/购买的 item ID） | 直接用 item ID 作为集合元素，不需要 n-gram |
| 图像去重 | 感知哈希（pHash）的 bit 集合 | 图像没有自然的"词"，用像素特征替代 |

**核心原则**：特征化方式决定了"相似"的含义。用词级 n-gram，相似 = 词序相近；用 item ID 集合，相似 = 行为模式相近。选特征化方式之前要先想清楚"我想找的是什么意义上的相似"。

---

## 三、大规模化：LSH（Locality Sensitive Hashing）

### 问题

有了 $n$ 篇文档的签名向量之后，如果要两两比较，仍然是 $O(n^2)$ 的操作。1 亿篇文档 = $5 \times 10^{15}$ 次比较，不可行。

### LSH 的核心思路

LSH 的目标是：**不比较所有对，只比较"可能相似"的对**。

方法是把签名向量切成 $b$ 个 band，每个 band 有 $r$ 行（$b \times r = k$）：

```
签名向量（k=12，b=3，r=4）：

band 1: [h1, h2, h3, h4]
band 2: [h5, h6, h7, h8]
band 3: [h9, h10, h11, h12]
```

对每个 band，把该 band 的 $r$ 个值拼在一起做哈希，得到一个桶 ID。**两篇文档只要在任意一个 band 里落入同一个桶，就成为候选对**，再做精确比较。

### 为什么这样有效：S 曲线

两篇文档被 LSH 选为候选对的概率，是 Jaccard 相似度 $s$ 的函数：

$$P(\text{成为候选对}) = 1 - (1 - s^r)^b$$

这个函数的形状是一条 **S 曲线**（sigmoid 形状）：
- 当 $s$ 很低时，概率接近 0（低相似度的文档对很少被选中）
- 当 $s$ 很高时，概率接近 1（高相似度的文档对几乎都被选中）
- 曲线的"转折点"（概率从低跳到高的位置）由 $b$ 和 $r$ 控制

**转折点近似**：$s^* \approx (1/b)^{1/r}$

通过调整 $b$ 和 $r$，可以把转折点对准目标阈值（比如 0.8），使得：
- 相似度 > 0.8 的文档对：几乎都被选中（高召回）
- 相似度 < 0.8 的文档对：几乎都被过滤（低误报）

### b 和 r 的权衡

固定 $k$，$b \times r = k$：

| 参数变化 | 效果 |
|---------|------|
| 增大 $b$（减小 $r$） | S 曲线更陡，转折点更低，召回率更高，但误报增多，计算量增大 |
| 减小 $b$（增大 $r$） | S 曲线更平缓，转折点更高，精确率更高，但漏报增多 |

**典型配置**（$k=128$，目标阈值 0.8）：$b=16$，$r=8$，转折点 $\approx (1/16)^{1/8} \approx 0.78$。

### LSH 之后还需要精确验证

LSH 只是候选筛选，不是最终判断。进入同一个桶的文档对，还需要：
1. 计算真实的签名相似度（比较签名向量中相等位置的比例）
2. 或者直接计算真实 Jaccard（更慢但更准确）

这一步过滤掉 LSH 的误报（碰巧落入同一个桶但实际不相似的对）。

---

## 四、其他需要知道的概念

### 实现变体：one-permutation MinHash

标准 MinHash 需要 $k$ 个独立哈希函数，计算 $k$ 次。**One-permutation MinHash**（Li & König, 2011）只需要一次随机排列，把签名向量分成 $k$ 个区间，每个区间取最小值。速度是标准 MinHash 的 $k$ 倍，精度略低但在实践中差距很小。大规模系统（如 Google、Meta 的数据管道）通常用这个变体。

### 误差来源

MinHash 的误差有两个来源：
1. **估计误差**：用 $k$ 个哈希函数估计 Jaccard，误差 $O(1/\sqrt{k})$（前面已讨论）
2. **特征化误差**：n-gram 集合不完全等价于"文档相似"，特征化方式会引入系统性偏差（比如两篇讨论完全不同话题的文章，如果都有大量常见词，n-gram Jaccard 可能虚高）

第二类误差通常用**预处理**缓解：去停用词、lowercase、去标点，减少无意义的高频 n-gram。

### 去重策略：保留哪一篇

MinHash 找到近重复对之后，需要决定保留哪一篇。常见策略：
- **保留最新版本**（URL-level 去重后已处理）
- **保留最长文档**（内容更完整）
- **保留来源质量更高的文档**（配合域名质量分）

Llama 3 等大规模管道通常用 connected components 算法：先把所有近重复对连成图，每个连通分量只保留一篇（通常是质量最高的）。

### 和 SimHash 的关系

SimHash（Charikar, 2002）是另一种常用的近似相似度哈希，用于估计**余弦相似度**而不是 Jaccard。Google 用它做网页去重。

- MinHash：适合集合相似度（Jaccard），文本去重的标准选择
- SimHash：适合向量相似度（余弦），对词频有加权，对长文档更鲁棒

两者不是竞争关系，适用场景不同。

---

## 工作流程总结

```
原始文档集合
    ↓
1. 特征化：文档 → n-gram 集合
   "the quick brown fox" → {"the quick", "quick brown", "brown fox"}

    ↓
2. MinHash 签名：k 个哈希函数，各取最小值
   doc_A → [h1_min, h2_min, ..., hk_min]（k 维向量）

    ↓
3. LSH 分桶：把签名切成 b 个 band，每个 band 哈希到一个桶
   → 同桶的文档对成为"候选近重复对"

    ↓
4. 精确验证：对候选对计算真实相似度，过滤误报

    ↓
5. 去重：每组近重复文档保留一篇
```

---

## 参数选择速查

| 参数 | 典型值 | 说明 |
|------|--------|------|
| n-gram 大小 | 5（词级） | 对改写鲁棒，不过于稀疏 |
| 签名长度 k | 128–256 | 标准误差约 ±0.03–0.04 |
| 相似度阈值 | 0.7–0.8 | 根据去重激进程度调整 |
| band 数 b | k/8 左右 | 使转折点对准目标阈值 |
| band 行数 r | 8 左右 | 同上，b × r = k |

---

## 和 Line-level 去重的关系

MinHash 解决"整篇文档近重复"。但有一类重复它处理不了：文档主体独特，但附带了大量在整个数据集里反复出现的模板行（版权声明、Cookie 提示、导航菜单）。

**Line-level 去重**（借鉴 [ccNet](./ccnet.md)，Wenzek et al., 2019）专门处理这个问题：将数据按语言分桶（每桶约 30M 文档），统计每行出现次数，超过阈值（Llama 3 用 6 次）的行从所有文档中删除。复杂度 $O(n)$，比 MinHash 更便宜。两者互补，不是替代。

---

## 业界实现：用库还是自己写？

**短答案**：小规模用库，大规模（亿级文档）自己写或深度定制。

### 主要现成库

**datasketch**（Python，最常用）

```
pip install datasketch
```

提供 MinHash、MinHashLSH、MinHashLSHForest 等完整实现。接口简洁，适合原型开发和中等规模（千万级文档以内）。内部用 MurmurHash，支持自定义参数。

**text-dedup**（Google Research，2022）

专门为 NLP 预训练数据去重设计，支持 MinHash、SimHash、suffix array 等多种方法，有完整的命令行接口，可以直接处理 HuggingFace datasets 格式的数据。是目前最接近"开箱即用的工业级工具"的开源实现。

**Spark + 自定义 MinHash**

大规模（百亿 token 级别）通常在 Spark 上实现分布式 MinHash，Spark MLlib 有内置的 MinHashLSH，但通常需要针对文本去重场景做定制（调整特征化方式、桶策略等）。

### 为什么大规模要自己写

1. **内存布局**：库的通用实现通常用 Python 对象存储签名，亿级文档下内存开销不可接受。自己写可以用 numpy 数组或 C++ 连续内存布局，内存减少 10–100 倍。

2. **哈希函数选择**：datasketch 默认用 hashlib（较慢），大规模管道通常换成 MurmurHash3 的 C 扩展，速度快 5–10 倍。

3. **分布式策略**：LSH 分桶后的候选对匹配，需要跨机器的 join 操作，库通常不处理这部分，需要自己设计分布式流程。

4. **管道集成**：真实管道里 MinHash 只是一个步骤，需要和前后的过滤、质量打分等步骤无缝衔接，库的接口通常不够灵活。

**结论**：学习和实验用 datasketch，生产环境用 text-dedup 或自定义实现。理解原理比会用库更重要——大规模场景下你总会遇到需要改库的地方。

---

## Demo

两个 demo 文件在 `tools/` 目录下，用相同的 5 篇文档对比两种实现。

### Demo 1：从零实现（无第三方依赖）

```bash
python tools/minhash_from_scratch.py
```

代码：[`tools/minhash_from_scratch.py`](../../tools/minhash_from_scratch.py)

演示完整流程：n-gram 特征化 → MinHash 签名（k=128）→ LSH 分桶（b=16, r=8）→ 候选对验证。用 hashlib.md5 + seed 模拟 k 个独立哈希函数（演示用，速度慢，生产环境换 mmh3）。

**典型输出**：

```
── 真实 Jaccard 相似度 ──
  doc_A vs doc_B: 0.714   ← 轻微改写，高相似
  doc_C vs doc_D: 0.091   ← 语义相似但词重叠低
  doc_A vs doc_C: 0.000
  doc_A vs doc_E: 0.000

── MinHash 估计的 Jaccard ──
  doc_A vs doc_B: 0.703   ← 误差 0.011，在理论上界 ±0.044 以内
  doc_C vs doc_D: 0.086
  doc_A vs doc_C: 0.000
  doc_A vs doc_E: 0.000

── LSH 候选对（b=16, r=8，转折点阈值≈0.78）──
  doc_A vs doc_B  估计 Jaccard=0.703
```

### Demo 2：使用 datasketch 库

```bash
pip install datasketch
python tools/minhash_datasketch.py
```

代码：[`tools/minhash_datasketch.py`](../../tools/minhash_datasketch.py)

接口更简洁，内部用 MurmurHash，适合原型开发。结果和 Demo 1 基本一致（因为同样是 k=128 的无偏估计）。

**典型输出**：

```
── datasketch 估计的 Jaccard ──
  doc_A vs doc_B: 0.711
  doc_C vs doc_D: 0.086
  doc_A vs doc_C: 0.000
  doc_A vs doc_E: 0.000

── LSH 查询（threshold=0.5）──
  doc_A 的近邻: ['doc_B']
```

### 两个 demo 的关键观察

- **MinHash 测词重叠，不测语义**：doc_C 和 doc_D 语义相似（都在说"模型需要大量训练数据"），但词级 5-gram 几乎没有重叠，Jaccard 只有 0.09。这是 MinHash 的根本局限，不是 bug。
- **LSH 阈值影响召回**：Demo 1 的转折点约 0.78，doc_A/doc_B 的相似度 0.71 略低于转折点，LSH 可能漏掉它。调低阈值（比如 b=20, r=6）可以提高召回，但误报也会增多。

---

## 参考

- Broder, 1997: *On the resemblance and containment of documents*（MinHash 原始论文）
- Indyk & Motwani, 1998: *Approximate Nearest Neighbors: Towards Removing the Curse of Dimensionality*（LSH 理论基础）
- Li & König, 2011: *Theory and Applications of b-Bit Minwise Hashing*（one-permutation 变体）
- Charikar, 2002: *Similarity Estimation Techniques from Rounding Algorithms*（SimHash）
- Wenzek et al., 2019: *CCNet* → [ccNet](./ccnet.md)
- datasketch：`pip install datasketch`，Python 实现，适合原型开发
- text-dedup（Google Research）：工业级去重工具，支持多种方法
- mmh3：`pip install mmh3`，MurmurHash3 的 Python 绑定，生产环境哈希首选
