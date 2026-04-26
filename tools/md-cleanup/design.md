# MD Cleanup: Layout Artifact Detection

## 问题背景

PDF/论文 → markitdown → Markdown 的转换过程中，布局信息丢失，产生大量无法阅读的"排版乱码"。
正文段落本身可读，但图、表、页码、图注等由于格式丢失变得无意义。

---

## 噪声类型分类

### 1. 字符级乱码
PDF 页面角落的元数据块（arXiv ID、版本号）被按字符逐行提取。
```
5↵2↵0↵2↵r↵p↵A
```

### 2. 图坐标/图例粘连
图的坐标轴数值和图例文字被拼成一行。
```
10152025303540Core evals score at scale s3032343638404244...
```

### 3. 表格数字列堆叠
表格列变成垂直堆叠的短行。
```
400M-1x↵1B-1x↵3B-1x
412M↵1.4B↵2.8B
```

### 4. 表格文字列堆叠（难点）
词汇正常但不成句，规则无法识别。
```
Llama2↵DeepSeek↵Mistral-0.3↵QWEN-2
Dataset↵C4↵Dolma-V1↵RefinedWeb
```

### 5. 孤立小段落（最难）
单行短文本，本身看起来无害，只有放在上下文里才能识别。
```
Model
Open dataset?
Models we trained
57.1
```

### 6. 页码
单行纯数字。

---

## 四层检测流程（按执行顺序）

### 层 1 — TOC + 页码（最强信号，最先执行）

TOC 是 PDF 自带的结构化信息，最可靠。优先处理，建立全文章节骨架。

- 从 PDF 书签提取 TOC（level + title + page）
- 根据 level 推算编号（`"Introduction"` → `"1 Introduction"`），拼出完整候选文本
- 在对应页码行附近的窗口内做精确匹配（sim ≥ 0.78），双重验证
- 匹配到的标题段落：转换为对应层级的 `#` Markdown 标题
- 页码行（1-4 位纯数字）：静默删除

**效果（dclm）**：17 heading 全部 sim=1.00，10 页码行删除，零误判。

### 层 2 — 规则（44 处：8 char_noise + 2 figure + 34 table）

不依赖模型，针对"长得像噪声"的段落：

| 规则 | 目标 | 典型案例 |
|---|---|---|
| 孤立纯标点 | arXiv 元数据残留 | `]` |
| 单字符行比例 > 60% | 字符级乱码 | `r↵p↵A↵1↵2`、`✗↵✗↵✗` |
| 无空格长行（≥40字符）| 图内容粘连 | 图内坐标文字粘连 |
| 连续数字组开头的长行 | 图坐标轴粘连 | `10201021...Core evals score...` |
| 短行 + 高数字比例 | 表格数字列 | `400M-1x↵1B-1x↵3B-1x` |
| 低字母比例（<30%）| 表格数字列 | `412M↵1.4B↵2.8B` |

**注**：字符乱码和纯数字列的 perplexity 反而低（OOV 词、数字全归一化），KenLM 对这类无效，必须靠规则。

### 层 3 — KenLM perplexity（16 处）

用在 arXiv 摘要上训练的 trigram 模型，对每个段落打困惑度分。
词汇正常但不成句的段落（表格文字列、表头碎片）perplexity 高。

- 训练语料：`ccdv/arxiv-summarization` 5000 条摘要
- 模型：trigram KenLM，约 10MB binary
- 默认阈值：perplexity > 50000

| 典型案例 | perplexity |
|---|---|
| `Llama2↵DeepSeek↵Mistral-0.3↵QWEN-2` | 224,234 |
| `Dataset↵C4↵Dolma-V1↵RefinedWeb` | 98,346 |
| `Params Tokens` | 90,165 |
| `Train tokens Train FLOPs` | 82,878 |

### 层 4 — Sandwich（13 处）

前三层建立了足够多的锚点，夹在锚点之间的单行短段落几乎必然是噪声。

**判断条件（同时满足）：**
1. 未被前三层标记
2. 单行，字数 ≤ 20，不以句号/问号结尾
3. 向前、向后都能找到锚点（任意已标记的 artifact 或 heading）

**注**：多行段落（如论文标题）不参与 sandwich 检测，避免误判。

| 典型案例 | 说明 |
|---|---|
| `CORE`、`EXTENDED` | 表头列 |
| `Open dataset?`、`Pool size` | 表头列 |
| `57.1`、`63.7` | 孤立指标值 |
| `Models we trained` | 表格分组标签 |

---

## 检测结果（dclm 论文，184 段）

| 层 | 命中 | 核心信号 | 典型案例 |
|---|---|---|---|
| 1 TOC + 页码 | 17 heading + 10 page | PDF 结构元数据 | `# 1 Introduction`，删页码 |
| 2 规则 | 44 | 数字/字符比例 | `✗↵✗↵✗`，`412M↵1.4B` |
| 3 KenLM | 16 | 语言模型困惑度 | `Llama2↵DeepSeek↵...` |
| 4 Sandwich | 13 | 上下文位置 | `CORE`、`57.1` |
| **总计 artifact** | **83** | | |
| **可读保留** | **84** | | |

---

## 日志格式（JSONL）

每次运行可通过 `--log` 输出决策日志，便于排错。

**每行一条段落记录：**
```json
{"lineno": 191, "label": "rule:table", "pp": 186883, "preview": "400M-1x↵1B-1x↵3B-1x↵7B-1x↵7B-2x"}
```

**最后一行是汇总 stats：**
```json
{"stats": {"rule:char_noise": 8, "readable": 84, "kenlm": 16, "toc:heading:1": 6, ...}}
```

**label 前缀约定：**

| 前缀 | 含义 |
|---|---|
| `readable` | 正常段落，保留 |
| `toc:heading:N` | TOC 匹配到的 N 级标题 |
| `toc:page_number` | 页码行，删除 |
| `rule:<type>` | 规则命中，type 为 char_noise/figure/table |
| `kenlm` | KenLM 高 perplexity |
| `sandwich` | 夹在锚点间的孤立短段落 |

前缀设计保证向后兼容：新增层次用新前缀，现有记录格式不变。

---

## 文件说明

| 文件 | 说明 |
|---|---|
| `toc_anchored_cleanup.py` | **主入口**：四层完整流程 |
| `toc_match.py` | TOC 标题匹配模块（编号推算 + 页码锚点） |
| `extract_toc.py` | 从 PDF 提取 TOC（PyMuPDF）|
| `cleanup.py` | 层 2：规则分类器 |
| `kenlm_cleanup.py` | 层 3：KenLM perplexity 分类器 |
| `train_lm.py` | 训练 arXiv KenLM 模型 |
| `compare.py` | 规则 vs KenLM 对比工具（开发用）|

**运行：**
```bash
# 完整流程
python tools/md-cleanup/toc_anchored_cleanup.py input.md \
  --pdf source.pdf -o output.md --log output.log.jsonl -v

# 训练 KenLM 模型（首次需要）
python tools/md-cleanup/train_lm.py
# → data/kenlm_academic/en_academic.binary
```

---

## 训练语料：arXiv Summarization 数据集

**来源**：Hugging Face `ccdv/arxiv-summarization`  
**内容**：arXiv 论文摘要，约 20 万条，每条约 150-300 词  
**为什么选它**：语言风格与目标文档（学术论文 PDF 转 Markdown）高度匹配，只用摘要字段体积小，5000 条即可训练有效的 trigram 模型（约 30 秒）
