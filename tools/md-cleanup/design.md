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

## 检测流程（按执行顺序）

### 层 1 — TOC + 页码（精确匹配，不参与打分）

TOC 是 PDF 自带的结构化信息，最可靠，直接确定，不走打分流程。

- 从 PDF 书签提取 TOC（level + title + page）
- 根据 level 推算编号（`"Introduction"` → `"1 Introduction"`），拼出完整候选文本
- 在对应页码行附近的窗口内精确匹配（sim ≥ 0.78 + 页码双重验证）
- 匹配到的标题段落：转换为对应层级的 `#` Markdown 标题
- 页码行（1-4 位纯数字）：静默删除

**效果（dclm）**：17 heading 全部 sim=1.00，10 页码行删除，零误判。

### 层 2-4 — 联合打分（保守删除）

层 1 之后，剩余段落通过三个信号联合打分，**总分超过阈值才删除**，避免单一信号误伤。

#### 信号 A：规则置信度（0-1）

规则层不再直接删除，改为输出置信度：命中规则越多、越强，分数越高。

| 规则 | 权重 | 目标 |
|---|---|---|
| 孤立纯标点 | 1.0 | arXiv 元数据残留 |
| 单字符行比例 > 60% | 1.0 | 字符级乱码 |
| 无空格长行（≥40字符）| 0.8 | 图内容粘连 |
| 连续数字组开头的长行 | 0.8 | 图坐标轴粘连 |
| 短行 + 高数字比例 | 0.6 | 表格数字列 |
| 低字母比例（<30%）| 0.6 | 表格数字列 |

最终取所有命中规则的最高权重作为规则置信度。

#### 信号 B：KenLM 分位数置信度（0-1，自适应）

不再使用固定阈值（50000），改为文档内自适应：

```
pp_score = rank(para.pp) / total_paragraphs   # 段落在文档内的 perplexity 分位数
```

perplexity 越高（分位越靠后），置信度越高。阈值随文档整体分布自动变化，数学密集的论文不会被误伤。

#### 信号 C：Sandwich 置信度（0 或 0.5）

单行、字数 ≤ 20、不像完整句子，且前后都有锚点 → 置信度 0.5（作为辅助信号）。
多行段落不参与（避免论文标题被误删）。

#### 联合决策

```
score = max(rule_score, pp_score) * 0.7 + sandwich_score * 0.3
if score >= 0.5: artifact
```

逻辑：主信号（规则或 KenLM）至少有一个较强，sandwich 作为辅助加权。
单靠 sandwich（0.5 × 0.3 = 0.15）不足以删除，必须有主信号支撑。

---

## 检测结果（dclm 论文，184 段）

| 版本 | artifact | readable | 说明 |
|---|---|---|---|
| 旧版（OR 逻辑）| 83 | 84 | 任一层命中即删 |
| 新版（联合打分）| 54 | 103 | 多信号联合，自适应阈值 |

新版更保守：artifact 减少 29 处，readable 增加 19 处。
代价是少量表头碎片（`CORE`、`EXTENDED`、`Dataset`）因联合分数不足（~0.49）未被删除，
但同时也避免了正文段落被误伤。

**边界分析**（score 最高的 readable，最接近误删）：
- `Table 4: Quality filtering comparison...` score=0.498 → 正确保留（表格标题是正文）
- `CORE`、`EXTENDED` score=0.49 → 漏检（表头碎片，联合信号不够强）
- `authors have changed organization since then.` score=0.495 → 正确保留（正文脚注）

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
