# PDF → Markdown 处理方案设计

## 当前进展

`20250421-llama3-exp/` 目录下的文件状态：

| 文件 | 说明 | 状态 |
|------|------|------|
| `llama3-raw.md` | markitdown 原始转换输出，保留副本 | ✅ |
| `main-body.md` | 按 TOC 插入章节标题后的正文（5988 行） | ✅ |
| `main-body-clean.md` | 去噪后的正文（4053 行） | ✅ |
| `main-body-with-images.md` | 嵌入已确认位图的版本 | ✅ |
| `main-body-clean.translation.json` | 双语 sidecar（段落 ID → 译文） | 🔄 部分（15 段） |
| `main-body-bilingual.md` | 上下对照双语版 | 🔄 部分 |
| `references.md` | 参考文献部分 | ✅ |
| `structured.md` | 正文 + 参考文献合并版 | ✅ |
| `assets/figures/` | 从 PDF 提取的嵌入位图（7 张有效图） | ✅ |

## 处理步骤

### Step 1：PDF → 原始 Markdown ✅
工具：`markitdown`。正文段落质量好；表格、图片内嵌文字混乱；无章节标题。

### Step 2：章节结构化 ✅
读取 PDF outline（`.toc.json`），对照正文行内容插入 `#`/`##`/`###` 标题（98 个标题）。

### Step 3：正文清理 ✅
- 删除 Contributors/Acknowledgements
- 图表噪音块替换为 `> **[Figure N]** caption` 占位符
- 删除孤立页码行（PDF 分页残留）
- 独立公式行用 `$$...$$` 包裹

### Step 4：图片还原 🔄 部分完成
- 嵌入位图（7 张）：已提取并嵌入 → `main-body-with-images.md`
- 矢量图（绝大多数 Figure）：**跳过**，效果难保证，占位符保留

### Step 5：引用关联 ⏭️ 跳过（非关键路径）
自研 regex 准确率约 82%，暂不处理。

### Step 6：双语对照 🔄 原型验证完成
- Sidecar 方案：`main-body-clean.translation.json`，key = 段落 ID（`{章节路径}-p{序号}`）
- 本地翻译：mdv 内置 `opus-mt-en-zh` ONNX 模型，零 token
- 展示：`build_bilingual.py` 生成 `main-body-bilingual.md`，上下交替排版
- 当前状态：Introduction 章节 15 段已翻译，全文翻译待跑

---

## 图片方案（待实现）

### PDF 图片的存储方式

PDF 内部图片有两种存储方式，决定了提取策略：

**1. 原始位图流（Raw bitmap stream）**，`enc=image`
- 图片在 PDF 内部是未压缩的原始像素数据
- `pdfimages` 提取时读取原始像素后**重新编码**成 PNG 输出
- 不是"原来就是 PNG"，是工具转换的，但质量无损
- 特征：`ratio` 很低（如 0.6%），说明原始数据远大于提取后的 PNG

**2. JPEG 原生嵌入**，`enc=jpeg`
- 图片在 PDF 内部**直接存的就是 JPEG 文件**
- `pdfimages -all` 会原样输出为 `.jpg`，内容和原始 JPEG 完全一致，无转换损失
- llama3 论文中只有 Figure 12（img-006）是这种格式

**3. 矢量图**（不在 `pdfimages` 范围内）
- PDF 原生绘图指令（折线图、散点图、热图等），不是像素图
- 需要用 `pdftoppm` 渲染整页后按坐标裁剪

### llama3 论文图片分析

`pdfimages -list` 输出共 13 条，实际有效图片 **7 张**，其余 6 张是 alpha mask（smask）：

| img 文件 | 页码 | 编码 | 尺寸 | 对应 Figure（已确认） |
|----------|------|------|------|----------------------|
| img-000 | p.1  | raw→PNG | 23666×6483 | **Meta logo**（封面 logo，不是 Figure） |
| img-001 | p.1  | smask | 23666×6483 | img-000 的透明度通道 |
| img-002 | p.21 | raw→PNG | 1882×812 | **Figure 8**：Code translation example（Python→PHP 截图） |
| img-003 | p.21 | raw→PNG | 1556×700 | **Figure 9**：Improving generated code quality with system prompts |
| img-004 | p.27 | raw→PNG | 9240×6968 | **Figure 11**：Processing file uploads（工具使用截图） |
| img-005 | p.27 | smask | 9240×6968 | img-004 的透明度通道 |
| img-006 | p.40 | JPEG原生 | 1606×726 | **Figure 16**：Human evaluation results on code execution tasks |
| img-007 | p.54 | raw→PNG | 1200×800 | **Figure 27 左图**：Throughput-latency trade-off in FP8 inference |
| img-008 | p.54 | smask | 1200×800 | img-007 的透明度通道 |
| img-009 | p.54 | raw→PNG | 1200×800 | **Figure 27 右图**：Throughput-latency trade-off in FP8 inference |
| img-010 | p.54 | smask | 1200×800 | img-009 的透明度通道 |
| img-011 | p.63 | raw→PNG | 12150×2958 | **Figure 29**：Architecture of speech interface for Llama 3 |
| img-012 | p.63 | smask | 12150×2958 | img-011 的透明度通道 |

**确认方法**：使用 `pymupdf` 提取每页图片的 xref（PDF 对象 ID）和位置坐标，再从页面文本中提取 Figure caption，按 y 轴坐标（从上到下）和 xref 顺序匹配。

**关键发现**：
- img-000 是 Meta logo，不是论文 Figure 1（Figure 1 是矢量图）
- Figure 11 在 p.27（不是 Figure 10），Figure 16 在 p.40（不是 Figure 12）
- p.54 的两张图都属于 Figure 27，是同一 Figure 的左右两个子图
- 绝大多数 Figure（折线图、散点图等）是矢量图，不在嵌入位图范围内

**绝大多数 Figure（Figure 2-7, 13-27 等）是矢量图**，不在嵌入位图范围内。

### 提取命令

```bash
# 提取所有嵌入位图（JPEG原样输出，其余编码为PNG）
pdfimages -all input.pdf output_prefix

# 仅提取为PNG（统一格式，JPEG也转PNG）
pdfimages -png input.pdf output_prefix

# 矢量图：按页渲染为图片
pdftoppm -r 150 -png input.pdf prefix
# 然后用 pdfplumber 检测 Figure 坐标后裁剪
```

目标：把提取出的图片放到 `assets/figures/` 目录，替换正文中的占位符为 `![Figure N](../assets/figures/figN.png)`。

---

## 引用关联方案（待实现，非关键路径）

### 问题
正文引用格式为作者-年份式（`Vaswani et al., 2017`），参考文献无编号。
PDF 转换后 references.md 条目存在换行断裂，解析有难度。

### 自研 regex 方案（已验证）
解析 references.md 重建条目，提取 lastname+year 生成 anchor，正文 regex 替换为 Markdown 链接。当前准确率约 82%。

### 可选开源库

| 库 | 语言 | 特点 |
|----|------|------|
| **[grobid](https://github.com/kermitt2/grobid)** | Java（REST API） | 直接处理 PDF，输出 TEI XML，引用关系已关联，工业级准确率，需跑服务 |
| **[refextract](https://github.com/inspirehep/refextract)** | Python | CERN 出品，pip 可装（需 libmagic），只做提取不做关联 |
| **[anystyle](https://anystyle.io)** | Ruby CLI | 专做参考文献格式解析，准确率高，需 Ruby 环境 |
| **[pybtex](https://pybtex.org)** | Python | 解析 BibTeX，仅适用于有 .bib 文件的场景 |
| **[scholarly](https://github.com/scholarly-python-package/scholarly)** | Python | Google Scholar 查询补全元数据，联网，不做本地解析 |

---

## 整理设计思路

### 为了方便处理，把pdf的信息分成两个部分
- 给agent看的：可以拆分成多个文件，json/md格式哪个合适用那个。用一个manifest组合他们。原则是渐进式披露
- 给人看的：一篇md即可

### 怎么评价还原度？有什么合理的指标吗？


### 转换过程

- 通过markitdown把pdf转成raw md
- 提取pdf outline作为toc
- 把raw md变成structed md（需要llm）

- 提取pdf里的图片，但怎么知道图片的编号，还是学术论文就是从1开始编号即可

- 把structed md按照学术论文的结构拆分成一些部分（需要llm）
Abstract、目录、正文、附录、 Contributors/Acknowledgements、authors、References、图、表、公式、其他
拆分的主要目标是：为后续步骤省token，和结构化查询以及索引

- 生成一个人阅读的md（你想个合适的文件名）
在正文里清理图、表、公式位置的噪音（占位符、坐标轴乱码等），替换成占位符，但要保留图表的caption（需要llm）
摘要、正文、参考引用合并成人阅读版本
把正文里的参考应用关联对markdown anchor
把图片放回到对应的占位符位置

### 后续使用
- 中英文对照翻译
用单独的服务逐段翻译
可以用一个json存储就好，能够表达和原文的关联（需要非常稳定）。方便展示成一段原文，一段中文的形式
可以针对单独的段落用更好的模型再翻译，替换进去

- 评注：mdv已经有这个功能了，只在人阅读的版本上做批注

### 待评估

- 评估哪些环节可以写成脚本，直接用就行，评估哪些要持续使用llm能力
- 评估这些工作需要什么能力的模型，尽可能便宜好用就行
- 由于过程中好多环节都需要llm，所以需要一个稳定的llm sdk
  关键是需要带audit能力，知道每次调用是在干什么，以及token用量和花费，能够溯源和分析
- 一旦pdf2md转换过程做完，这些文件就定死了。需要一个方式告诉后续使用场景下，这些文件不要修改

---

## 讨论记录

### 问题1：流程是否需要严格串行？

当前流程链：`raw md → structured md → 按章节拆分 → 人阅读版本`，多个 LLM 步骤串联，风险是错误累积。
哪些步骤可以并行，而不是严格串行？

### 问题2：还原度评估指标

候选方向：
- **结构完整性**：章节数/图表数/公式数是否和 PDF outline 对得上
- **文本覆盖率**：关键词/摘要句子的保留率（可用 ROUGE 类指标）
- **人工抽样**：随机抽几段对比原文，成本低但不系统

### 问题3：给 agent 看的格式选择

- JSON：适合结构化查询（如"找第3节的所有公式"）
- MD：适合上下文注入，更省 token
- 取决于下游 agent 的主要使用方式

### 问题4：LLM SDK 的 audit 能力

"需要一个稳定的 LLM SDK，带 audit 能力"——自己实现还是用现成方案（LangSmith、Helicone 等）？
这个选择会影响整体架构。
