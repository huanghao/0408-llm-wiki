# AGENTS

This repository is a personal LLM learning wiki. Treat it as a persistent knowledge base, not a scratchpad.

## Core model

There are three layers:

1. `raw/`: immutable source material. Read from here. Do not rewrite source contents.
2. `wiki/`: LLM-maintained markdown knowledge base. This is the main artifact to update.
3. `AGENTS.md`: the schema and operating rules for maintaining the wiki.

## Primary goals

When working in this repo, optimize for accumulation:

1. Convert raw sources into durable wiki pages.
2. Update existing pages instead of duplicating the same idea in many places.
3. Maintain cross-links between pages.
4. Keep a compact content index in `wiki/index.md`.
5. Append a chronological record in `wiki/log.md`.

## Directory conventions

- `raw/inbox/`: newly added sources waiting for ingestion
- `raw/processed/`: sources already ingested
- `raw/assets/`: local images and attachments
- `wiki/00-overview/`: high-level entry pages and maps
- `wiki/10-roadmaps/`: reading plans and staged study routes
- `wiki/20-concepts/`: concept pages
- `wiki/30-papers/`: individual paper pages
- `wiki/40-comparisons/`: compare/contrast pages
- `wiki/50-questions/`: open questions, hypotheses, TODOs
- `wiki/90-meta/`: repo maintenance notes, lint reports, conventions
- `templates/`: page templates

## Concept page rules（wiki/20-concepts/）

创建概念文档前，先问三个问题：

**1. 信息量够不够独立成文？**
一个概念是否值得单独一个文件，取决于它有没有足够展开的内容：机制、参数、对比、局限、应用场景。如果只有两三段话，没有可展开的细节，就不要单独成文——把它内联到使用它的文档里，或者作为相关概念文档的一节。

**2. 它归属于哪个更大的主题？**
如果一个概念在逻辑上是另一个概念的子问题或补充（比如 line-level 去重之于 MinHash），放在那个主题文档里作为一节，而不是平铺成独立文件。归属关系比独立性更重要。

**3. 这是概念本身，还是概念之间的关系？**
概念文档只说"这个概念是什么"。关系、对比、综述、"为什么选这个而不是那个"——这些属于使用这些概念的上下文文档（论文页、roadmap、comparison 页），不要单独建一个"XXX 对比"或"XXX 汇总"文件。

## Page rules

Prefer short, structured markdown pages.

Each durable wiki page should usually have:

1. A title
2. A one-sentence summary near the top
3. Source links or source references
4. Internal links to related pages when relevant
5. A final section for open questions, tensions, or next reads when useful

## Ingest workflow

When the user asks to ingest a source:

1. Read the source from `raw/inbox/` or a provided URL/file.
2. Decide which existing wiki pages should be updated.
3. Create or update:
   - one source-specific page if needed
   - affected concept pages
   - affected comparison or roadmap pages
   - `wiki/index.md`
   - `wiki/log.md`
4. Move the raw file from `raw/inbox/` to `raw/processed/` only if the user wants file organization handled automatically.

## Query workflow

When the user asks a question:

1. Read `wiki/index.md` first.
2. Read the most relevant wiki pages.
3. Synthesize an answer from the wiki, not directly from memory when possible.
4. If the answer creates durable value, offer to file it back into the wiki as a new or updated page.

## Lint workflow

When asked to lint or health-check the wiki, look for:

1. orphan pages
2. stale claims
3. duplicated notes that should be merged
4. missing cross-links
5. empty sections or TODO-heavy pages
6. important concepts referenced repeatedly but lacking their own page

Write lint findings to `wiki/90-meta/`.

## Paper page writing principles

When writing or updating a page in `wiki/30-papers/`, follow these rules:

1. **原文优先，不猜测**：所有关于论文内容的陈述必须能在原文中找到依据。不确定的内容不写，或明确标注"推断"。不用记忆或常识填充原文没说的细节。

2. **引用必须指明**：原文中引用了其他论文的地方，wiki 页面里也要指明对应引用（作者、年份、论文名）。不能把引用来源的内容当作论文本身的贡献来写。

3. **总结原文，不改写原文**：wiki 页面的目标是把原文信息压缩成可快速回顾的结构，而不是换一种说法重新把原文写一遍。优先提炼"这篇论文在这个问题上的立场是什么"，而不是"这篇论文说了什么"。

4. **必须包含「现状与影响」小节**：每篇论文文档在局限性之后、值得看的部分之前，必须有独立的「现状与影响」一节，回答以下问题：
   - 该方法目前是否还在普遍使用？
   - 如果不再使用，被什么方法/方向取代了？
   - 核心思想贡献和具体实现路线是否分离（思想被引用，但实现被绕过）？
   - 一句话定性：最佳实践 / 奠基性工作 / 已被超越 / 仍活跃
   - 当时的贡献 vs. 今天（2026）的视角：哪些结论仍成立，哪些被超越，哪些被低估或高估

   这个视角对读者判断"要不要深入学这篇"至关重要，不可省略。

5. **神经网络论文必须覆盖的六个部分**：对于提出或改进神经网络模型的论文，除标准章节外，必须覆盖以下内容（可作为独立节或在方法节中展开）：

   | 部分 | 覆盖内容 | 重要性 |
   |------|---------|--------|
   | **模型架构** | 输入 shape → 各层变换 → 输出 shape；附伪代码（含维度注释）| 核心，必须有 |
   | **Loss 函数** | 分类/回归/辅助 loss 各是什么，为什么这样设计，WTA/NLL 等机制 | 核心，必须有 |
   | **消融实验** | 哪个模块贡献最大，去掉某模块指标变化多少；这是理解设计动机的最直接证据 | 核心，必须有 |
   | **训练细节** | 优化器、学习率调度、batch size、正则化、训练步数 | 复现时必须 |
   | **数据** | 数据集规模、划分方式、预处理；和 Chinchilla/scaling 规律的对照 | 复现时必须 |
   | **评测指标** | 每个指标的含义和局限，为什么这个指标能（或不能）衡量目标能力 | 评估时必须 |

   **写作优先级**：架构 + Loss + 消融实验 > 训练细节 + 数据 > 评测指标说明。消融实验往往比结果表格更有信息量——Table 3 里去掉某模块后 mAP 跌多少，比 Table 1 的 SOTA 结果更能解释"为什么这样设计"。

   **伪代码规范**：架构伪代码必须标注每步输入/输出 shape（如 `[B, L, D]`），说明每个维度的含义（B=batch, L=序列长度, D=特征维度），以及关键的 reshape/permute 操作。如果多模态输入有多个 shape，分别列出。

## LaTeX 写作规范

写完公式后自我检查，常见错误：

| 错误类型 | 错误写法 | 正确写法 |
|---------|---------|---------|
| 数学块内下划线多余转义 | `n\_{\text{true}}` | `n_{\text{true}}` |
| 花括号不配对 | `\frac{a}{b` | `\frac{a}{b}` |
| `\overline` 作用范围不明 | `\overline\log p` | `\overline{\log p}` |
| 下标直接跟命令 | `_\text{foo}` | `_{\text{foo}}` |

通用原则：`$$...$$` 块内 `_` 和 `^` 不需要转义；多字符上下标和 `\overline` 作用域都要加 `{}`；写完检查括号是否配对。

## Style

Prefer substance over polish.

Do not turn the wiki into a diary of chat transcripts. Convert chat output into durable notes.

Avoid redundant pages when an update to an existing page is better.

Prefer markdown links and relative paths inside the repo.
