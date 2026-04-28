# 困惑度（Perplexity）

## 是什么

困惑度（perplexity）是衡量语言模型对一段文本"感到困惑的程度"的指标。

直觉：**perplexity = 模型平均每步面临多少种等可能的选择**。

- perplexity = 10：模型每预测一个词，平均只有 10 种合理选项 → 文本可预测，结构清晰
- perplexity = 10000：模型每步面临 10000 种选项 → 文本高度混乱，像乱码

## 为什么有用

给定一个在高质量语料（如 Wikipedia）上训练的语言模型，困惑度可以用来衡量一段文本"像不像正常文本"：

- **低困惑度**：流畅的自然语言，符合训练语料的分布
- **高困惑度**：乱码、随机字符串、高度重复内容、代码混杂文本

这是 ccNet 做数据质量过滤的核心思路：用 Wikipedia 训练的 KenLM 模型给网页文本打分，困惑度高的文档质量差，过滤掉。

## P(W) 是什么，怎么算出来的

$P(W)$ 是语言模型给这段文本打出的概率，表示"这段文字在训练语料的分布下有多可能出现"。

语言模型把整段文本的概率拆成每个词的条件概率之积：

$$P(W) = P(w_1) \cdot P(w_2|w_1) \cdot P(w_3|w_1,w_2) \cdots P(w_N|w_1,\ldots,w_{N-1})$$

n-gram 模型（如 KenLM）只看前 n-1 个词，比如 5-gram 只看前 4 个词：

$$P(w_i | w_1,\ldots,w_{i-1}) \approx P(w_i | w_{i-4},\ldots,w_{i-1})$$

这些条件概率从训练语料的**词频统计**中直接算出来：

$$P(w_i | w_{i-4},\ldots,w_{i-1}) = \frac{\text{count}(w_{i-4},\ldots,w_{i-1},w_i)}{\text{count}(w_{i-4},\ldots,w_{i-1})}$$

就是"这个 5-gram 出现了多少次"除以"前 4 个词（context）出现了多少次"。举例：

```
训练语料中：
  "机器 学习 很 有 趣"  出现 120 次
  "机器 学习 很 有 用"  出现  80 次
  "机器 学习 很 有 ___" 共出现 200 次（context 总数）

→ P("趣" | "机器 学习 很 有") = 120 / 200 = 0.6
→ P("用" | "机器 学习 很 有") =  80 / 200 = 0.4
→ P("价" | "机器 学习 很 有") =   0 / 200 = 0   ← 从未见过
```

最后一种情况（概率为 0）会让整段文本的概率变成 0，perplexity 变成无穷大。这就是为什么需要平滑算法（Kneser-Ney）——给没见过的 n-gram 一个小的非零概率，避免一个陌生词把整篇文章的得分打崩。

**连乘展开成连加（对数技巧）**：

P(W) 是所有词条件概率的连乘，取 log 后变成连加：

$$\log P(W) = \sum_{i=1}^{N} \log P(w_i \mid w_{i-n+1},\ldots,w_{i-1})$$

其中每一项 $\log P(w_i \mid \text{context}_i)$ 就是查一次词频表算出来的：

$$\log P(w_i \mid \text{context}_i) = \log \frac{\text{count}(\text{context}_i, w_i)}{\text{count}(\text{context}_i)}$$

**`model.score(line)` 做了什么**：把一行文本按上面的公式逐词查表，把每个 $\log P(w_i \mid \text{context}_i)$ 加起来，返回整句的 $\log_{10} P(\text{line})$。这是一个负数，越接近 0 说明这行文本在模型看来越"正常"。

**perplexity 和条件概率的关系**：

PP(W) 不是概率，但它和条件概率有直接关系。把公式展开：

$$\text{PP}(W) = P(W)^{-1/N} = \left(\prod_{i=1}^{N} P(w_i|\text{ctx}_i)\right)^{-1/N}$$

这等价于所有词条件概率的**几何平均数的倒数**。如果每个词的条件概率都是 $p$，那么 $\text{PP} = p^{-1}$。

直觉：如果模型在每一步都有 100 种等可能的选择（每种概率 = 1/100），PP = 100。所以 PP 就是"平均每步面临多少种等可能选择"——它描述的是条件概率的倒数，但以几何平均的方式把整段文本归一化成一个数。

**LLM 和 perplexity 的关系**：

LLM（大语言模型）本质上也是在做同一件事：预测下一个 token 的概率分布 $P(w_i \mid w_1,\ldots,w_{i-1})$。perplexity 就是衡量这个预测能力的标准指标：

- 模型预测越准 → 每步的 $P(w_i \mid \text{ctx})$ 越高 → PP 越低
- LLM 训练的目标就是最小化 perplexity（等价于最大化训练集的 log likelihood）

区别只是：KenLM 用词频查表估计概率，LLM 用神经网络估计概率。两者都输出 $P(w_i \mid \text{ctx})$，perplexity 的计算公式完全一样。所以 LLM 的 benchmark 里经常用 perplexity 来比较不同模型——在同一个测试集上，perplexity 越低说明模型对文本的预测能力越强。

## 计算方式

语言模型对一段文本 $W = w_1, w_2, \ldots, w_N$ 的困惑度定义为：

$$\text{PP}(W) = P(W)^{-1/N}$$

即整段文本概率的 $-1/N$ 次方。N 是 token 数，用来归一化长度（否则长文本概率天然更低，困惑度会虚高）。

**等价形式**（用对数概率计算，避免浮点下溢，见附录）：

$$\text{PP}(W) = 10^{-\frac{\log_{10} P(W)}{N}}$$

KenLM 返回 log₁₀ 概率，所以 ccNet 的实现是：

```python
perplexity = 10.0 ** (-doc_log_score / doc_length)
```

其中 `doc_log_score` 是所有行的 `model.score(line)` 之和，`doc_length` 是所有行的词数之和（每行 `len(line.split()) + 1`）。

**为什么用 log₁₀ 而不是 log₂**：两者在数学上完全等价，只是底数不同，最终算出的 perplexity 数值一样（因为 $P(W)^{-1/N}$ 与对数底数无关）。KenLM 选择 log₁₀ 是工程惯例——ARPA 语言模型格式（`.arpa` 文件）规范里规定用 log₁₀，便于人工阅读和调试。信息论里常用 log₂（单位是 bit），自然语言处理里两种都有，只要统一就行。

## 具体例子

假设一篇文章有 3 行，KenLM 打分如下：

```
行1："机器学习是人工智能的分支"    score = -15.2，length = 6
行2："深度学习使用多层神经网络"    score = -18.4，length = 9
行3："asdf qwer 1234 zxcv"        score = -48.0，length = 5

doc_log_score = -15.2 + (-18.4) + (-48.0) = -81.6
doc_length    = 6 + 9 + 5 = 20

perplexity = 10^(81.6 / 20) = 10^4.08 ≈ 12000
```

去掉乱码行，只算前两行：

```
doc_log_score = -15.2 + (-18.4) = -33.6
doc_length    = 15

perplexity = 10^(33.6 / 15) = 10^2.24 ≈ 174
```

一行乱码把整篇文章的困惑度从 174 拉到 12000。

## log probability 是什么

`model.score(line)` 返回的是**以 10 为底的对数概率**（log₁₀ P），是一个负数。

- 一句通顺的中文，P ≈ 10⁻²⁰，log₁₀ P ≈ -20（接近 0，说明概率高）
- 一段乱码，P ≈ 10⁻¹⁰⁰，log₁₀ P ≈ -100（非常负，说明概率极低）

用对数的原因：一段文本的概率是每个词条件概率的乘积，词数一多就会下溢到 0。对数把乘法变成加法，数值稳定。

## 局限性

1. **依赖训练语料**：Wikipedia 训练的 LM 偏好百科风格，会给口语化、对话式文本打出偏高的困惑度，可能误伤
2. **对高度重复文本失效**：全是"的的的的的"这样的文本，困惑度反而很低。原因是 n-gram 模型只看局部窗口——bigram 只看前一个词，"的"后面跟"的"在中文里条件概率本来就不低，模型完全"不困惑"。5-gram 稍好，但"的的的的的的的的"这个 5-gram 一旦在训练语料里出现过，照样得低分。这是 n-gram 模型的根本局限：它不理解语义，只统计词序列频率。ccNet 用行级去重来补充处理这类重复内容，两者配合使用
3. **语言依赖**：必须用对应语言的模型，跨语言打分没有意义

## 在 ccNet 中的具体用法

ccNet 的特有做法：不设固定阈值，而是**按百分位分段**：

1. 对整个语言桶里的所有文档计算困惑度
2. 取第 30 和第 60 百分位作为切分点
3. 分成 head（低困惑度，高质量）/ middle / tail（高困惑度，低质量）三段
4. 实践中通常只用 head 或 head+middle

这样做的好处是阈值自适应——不同语言、不同时期的网页文本分布不同，百分位切分比固定阈值更鲁棒。

## 附录：KenLM 工具链

KenLM 的训练流程（`lmplz`）、`.arpa.bin` 文件格式、推理用法等工具细节，见 [ccNet](./ccnet.md) 文档的「KenLM 困惑度过滤」一节——ccNet 是 KenLM 在数据清洗中最典型的使用场景。

---

## 附录：add-k 平滑的分母为什么要乘以词表大小

代码里的平滑：

```python
count = ctx_counts.get(word, 0) + k          # 分子：实际计数 + k
total = sum(ctx_counts.values()) + k * V      # 分母：context 总计数 + k×V
```

**为什么分母加的是 `k * V` 而不是直接加 `k`**？

目标是让所有词的概率加起来等于 1（概率归一化）。对于一个 context，词表里有 V 个词，每个词的平滑后概率是：

$$P_{\text{smooth}}(w \mid \text{ctx}) = \frac{\text{count}(\text{ctx}, w) + k}{\text{分母}}$$

把所有 V 个词加起来，要求总和 = 1：

$$\sum_{w \in V} P_{\text{smooth}}(w \mid \text{ctx}) = \frac{\sum_w \text{count}(\text{ctx}, w) + k \cdot V}{\text{分母}} = 1$$

所以分母必须等于 $\sum_w \text{count} + k \cdot V$，也就是 `sum(ctx_counts.values()) + k * V`。

**直觉**：分子给每个词都加了 k，一共 V 个词，分母就要加 k×V，才能保证概率之和不变（还是 1）。如果分母只加 k，概率之和就会超过 1，不合法。

**k×V 会不会太大**？不会，因为 k 很小。代码里 k=0.01，V=几百个词，k×V ≈ 几。而 `sum(ctx_counts.values())` 是这个 context 在语料里出现的总次数，可能是几百到几万。所以 k×V 只是在分母上加了一个小量，效果是把每个词的概率从 0 稍微抬高一点点，不会让分母膨胀到不合理的程度。

**这是 add-k 平滑，不是 Kneser-Ney**：

- **add-k 平滑**（k=1 时也叫 Laplace 平滑，Laplace 是 add-k 的特例）：给每个 n-gram 计数加一个常数 k，分母加 k×V。k=1 是最原始的版本（Laplace 1812 年提出），实践中 k 通常取更小的值（0.01～0.5）效果更好，所以 add-k 是更通用的说法。简单，但效果一般——它对高频词和低频词一视同仁，分配给没见过词的概率往往过多或过少。
- **Kneser-Ney**：更精细的方法，核心思想不同：用"这个词出现在多少种不同 context 之后"（延续计数）来估计低阶 n-gram 的概率，而不是用词频。直觉是：一个词如果总是跟在同一个 context 后面（如"旧金山"后面的"山"），在新 context 里出现的可能性就低；反之如果出现在各种 context 后面，就更可能出现在新 context 里。

demo 代码（`perplexity_explained.py` 和 `kenlm_perplexity_demo.py`）用的是 add-k，因为它实现简单易理解。ccNet 实际用的是 Kneser-Ney（由 lmplz 自动应用）。

## 附录：浮点下溢与对数技巧

**什么是浮点下溢**：计算机用有限位数表示浮点数，能表示的最小正数大约是 `5e-324`（float64）。一段 100 个词的文本，每个词的概率假设是 0.01，整段概率就是 $0.01^{100} = 10^{-200}$，远小于 `5e-324`，计算机会直接把它存成 `0.0`——这叫下溢（underflow）。一旦变成 0，后续所有运算都失效。

**对数技巧**：取对数把乘法变成加法，把极小数变成普通负数：

```
原始：P(W) = 0.01 × 0.02 × 0.005 × ...    → 很快下溢到 0.0

取 log：
  log P(W) = log(0.01) + log(0.02) + log(0.005) + ...
           = -2 + (-1.7) + (-2.3) + ...
           = -6.0（普通负数，不会下溢）
```

**从 log P 还原 perplexity**：

$$\text{PP}(W) = P(W)^{-1/N} = \left(10^{\log_{10} P(W)}\right)^{-1/N} = 10^{-\log_{10} P(W) / N}$$

所以只需要 `log P(W)`（一个普通负数），就能算出 perplexity，全程不需要接触那个极小的 `P(W)` 本身。

**Python 里的体现**：

```python
import math

# 直接算概率：100 个词每个概率 0.01
prob = 0.01 ** 100
print(prob)  # 0.0  ← 下溢了！

# 用对数：
log_prob = 100 * math.log10(0.01)  # = 100 * (-2) = -200
perplexity = 10 ** (-log_prob / 100)  # = 10^2 = 100  ← 正确
```

这就是为什么语言模型 API（包括 KenLM）都返回 log probability 而不是 probability 本身。

## 附录：Kneser-Ney 平滑

Kneser-Ney 是 n-gram 语言模型中效果最好的平滑算法，lmplz 默认使用它。

**add-k 的问题**：add-k 给所有没见过的 n-gram 分配相同的概率质量，但这不合理。比如"旧金山"这个词，在训练语料里几乎只出现在"旧金山"这个固定搭配里（context 极少变化）；而"的"这个词出现在各种各样的 context 之后。两者同样没见过某个 context，直觉上"的"更可能出现在新 context 里。

**Kneser-Ney 的核心思想**：用"这个词出现在多少种不同 context 之后"（延续计数）来估计低阶概率，而不是用词频。

定义延续计数：

$$N_{1+}(\bullet\ w) = |\{v : \text{count}(v, w) > 0\}|$$

即"有多少种不同的词 $v$ 出现在 $w$ 之前"。"的"的延续计数极高，"山"（在"旧金山"中）的延续计数很低。

**公式**：

$$P_{\text{KN}}(w \mid \text{ctx}) = \frac{\max(\text{count}(\text{ctx}, w) - D,\ 0)}{\text{count}(\text{ctx})} + \lambda(\text{ctx}) \cdot P_{\text{KN}}(w \mid \text{shorter ctx})$$

- 第一项：折扣后的直接概率（见过的 n-gram 用这项，没见过时为 0）
- 第二项：回退概率（λ 是权重，递归用更短 context 的概率补充）

**λ 是什么**：λ(ctx) 是"归一化补偿系数"，不是可调参数，而是由 D 和训练数据自动决定的：

$$\lambda(\text{ctx}) = \frac{D}{\text{count}(\text{ctx})} \times |\{w : \text{count}(\text{ctx}, w) > 0\}|$$

直觉：context 后面见过 $U$ 个不同的词，每个词的计数都被减去了 D，总共"省出"了 $U \times D$ 的概率质量。把这些质量除以 count(ctx) 就得到 λ——它代表"要分配给没见过词的总概率份额"。λ 越大，说明这个 context 后面接过的词种类越多，给回退项的预算也越多。

**回退到更短 context 的逻辑**：

为什么不用其他方法，而是用"更短的 context"？因为这是最自然的近似：如果没见过 "A B C → w"，那 "B C → w" 是目前能利用的最相关的统计信息。更短 context 见过的词更多，统计更可靠。递归一直到 unigram（最底层用延续计数），最终所有词都有非零概率。

**归一化：怎么保证加起来等于 1**：

对一个固定的 ctx，把所有词 w 的概率加起来：

$$\sum_w P_{\text{KN}}(w \mid \text{ctx}) = \sum_w \frac{\max(\text{count}-D, 0)}{\text{count}(\text{ctx})} + \lambda(\text{ctx}) \cdot \sum_w P_{\text{KN}}(w \mid \text{shorter ctx})$$

- 第一项之和 = $\frac{\sum_w \text{count}(\text{ctx},w) - U \times D}{\text{count}(\text{ctx})} = 1 - \lambda(\text{ctx})$（U 个见过的词各减 D）
- 第二项之和 = $\lambda(\text{ctx}) \times 1 = \lambda(\text{ctx})$（递归假设低阶已归一化）

两项相加 = $1 - \lambda + \lambda = 1$。λ 的设计正是为了让这个等式成立——它是"第一项缺了多少，第二项就补多少"。

**举例**（bigram，D=0.75，词表共 1000 个延续计数）：

```
训练语料：
  "旧金山" 后面跟过：["的"×50, "市"×30]
  → count("旧金山") = 80，U = 2 种词
  → λ = (0.75/80) × 2 = 0.01875

  "的" 延续计数 = 500（出现在 500 种不同词之后）
  "山" 延续计数 = 3
  P_continuation("的") = 500/1000 = 0.5
  P_continuation("山") = 3/1000  = 0.003

P("的" | "旧金山") = (50-0.75)/80  + 0.01875 × 0.5   = 0.615 + 0.009 = 0.624
P("市" | "旧金山") = (30-0.75)/80  + 0.01875 × 0.003 = 0.366 + 0.00006 ≈ 0.366
P("山" | "旧金山") = 0/80          + 0.01875 × 0.003 = 0     + 0.00006 ≈ 0.00006
P("的" | "旧金山") + P("市" | ...) + P("山" | ...) + ... = 1  ✓
```

"省出"的 $2 \times 0.75 = 1.5$ 个概率单位（分母是 80），就是 $\lambda = 1.5/80 = 0.01875$，全部通过 $\lambda \times P_{\text{continuation}}$ 分配给了"没见过的词"（以及见过的词的回退部分）。

**为什么比 add-k 好**：add-k 对所有没见过的词分配相同的概率，Kneser-Ney 通过延续计数让"出现在更多 context 里的词"获得更高的回退概率，更符合语言的实际分布。

---

## 附录：KenLM 工具包

KenLM 分两个独立部分，经常被混淆：

### C++ 工具（需要从源码编译）

提供训练和格式转换命令：

| 工具 | 作用 |
|------|------|
| `lmplz` | 从文本训练 n-gram 模型，输出 `.arpa` 文件 |
| `build_binary` | 把 `.arpa` 压缩成 `.bin`，加载更快、更省内存 |
| `query` | 命令行打分工具 |

编译方法：
```bash
git clone https://github.com/kpu/kenlm
cd kenlm && mkdir build && cd build
cmake .. && make -j4
# 生成的二进制在 build/bin/lmplz 和 build/bin/build_binary
```

本项目编译路径：`/Users/huanghao/workspace/sources/kenlm/build/bin/`

### Python 包（pip 安装）

只提供**推理**接口，不包含 `lmplz`：

```bash
pip install kenlm
```

```python
import kenlm

model = kenlm.Model("zh.arpa.bin")   # 加载模型（.arpa 或 .arpa.bin 都可以）
print(model.order)                    # n-gram 阶数，比如 5

# 整句打分（返回 log₁₀ P，负数）
score = model.score("机器学习 很 有趣", bos=True, eos=True)

# 逐词打分（返回迭代器，每项是 (log₁₀P, ngram_length, is_oov)）
for logp, ngram_len, oov in model.full_scores("机器学习 很 有趣", bos=True, eos=True):
    print(f"  logP={logp:.3f}  用了{ngram_len}-gram  OOV={oov}")

# 检查词是否在词表里
print("机器学习" in model)   # True / False

# 有状态打分（逐词流式处理，不需要完整句子）
state = kenlm.State()
model.BeginSentenceWrite(state)
out_state = kenlm.State()
logp = model.BaseScore(state, "机器学习", out_state)
```

### 可下载的预训练模型

ccNet 提供了 50 种语言的预训练 5-gram KenLM 模型（Wikipedia 训练）：

```
https://dl.fbaipublicfiles.com/cc_net/lm/{lang}.arpa.bin    # 模型文件
https://dl.fbaipublicfiles.com/cc_net/lm/{lang}.sp.model    # 配套 SentencePiece 分词模型
```

支持的语言（ISO 639-1 代码）：

```
af ar az be bg bn ca cs da de el en es et fa fi fr gu he hi
hr hu hy id is it ja ka kk km kn ko lt lv mk ml mn mr my ne
nl no pl pt ro ru uk zh
```

中文模型约 3.2 GB，英文约 3.4 GB。使用时需要先用对应的 `.sp.model` 做 SentencePiece 分词，再送给 KenLM 打分（ccNet 的 `perplexity.py` 封装了这个流程）。

## 参考

- 逐步演示计算过程：`tools/perplexity_explained.py`（训练→逐词打分→困惑度，每步有数字）
- ccNet 过滤完整流程：`tools/kenlm_perplexity_demo.py`
- KenLM：https://github.com/kpu/kenlm
- ccNet 预训练 KenLM 模型：https://dl.fbaipublicfiles.com/cc_net/lm/
