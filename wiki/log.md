# Wiki Log

## [2026-04-10] bootstrap | Initialize personal LLM learning wiki

- Created `raw/`, `wiki/`, and `templates/` skeleton.
- Added `AGENTS.md` to define wiki maintenance rules for Codex.
- Added initial roadmap page.
- Added `wiki/index.md` and this log.

## [2026-04-10] ingest | Download first paper batch into raw/inbox

- Downloaded the first LLM paper batch from arXiv into `raw/inbox/`.
- Added `raw/inbox/manifest-2026-04-10.md` to map filenames to paper titles and arXiv URLs.
- Covered open models, reasoning, alignment, long context, agents, and evaluation.

## [2026-04-16] paper | Add first reading note for Llama 3

- Added a first paper page for `The Llama 3 Herd of Models`.
- Framed it as a reading guide rather than a full paper summary.
- Focused on data, scale, tokenizer, long-context training, post-training recipe, and system-level safety.

## [2026-04-21] experiment | pdf2md pipeline prototype with Llama 3 paper

Working directory: `20250421-llama3-exp/`

**已完成的步骤：**

1. **PDF → raw markdown**：用 `markitdown` 转换，保存为 `llama3-raw.md`（7240 行）。
2. **章节结构化**：读取 `llama3-herd-2407.21783.pdf.toc.json`，对照正文插入 `#`/`##`/`###` 标题，生成 `main-body.md`（5988 行，98 个标题）。
3. **正文清理**：删除 Contributors/Acknowledgements、图表噪音块替换为占位符、删除孤立页码行、公式用 `$$` 包裹，生成 `main-body-clean.md`（4053 行）。
4. **PDF 图片分析**：用 `pdfimages` 和 `pymupdf` 确认嵌入位图 7 张（其余 6 张为 alpha mask），并通过页面坐标 + caption 文本匹配到具体 Figure 编号（Figure 8/9/11/16/27/29；img-000 是 Meta logo 非 Figure）。
5. **图片嵌入**：生成 `main-body-with-images.md`，将 6 个已确认 Figure 的占位符替换为 Markdown 图片链接。
6. **双语对照原型**：
   - 设计 sidecar 方案：原文 md 不动，译文存 `main-body-clean.translation.json`（key = 段落 ID）。
   - 用 mdv 内置 `opus-mt-en-zh` ONNX 模型（本地，零 token）翻译 Introduction 章节 15 个段落作为验证。
   - `build_bilingual.py` 合并生成 `main-body-bilingual.md`，上下交替排版（原文 + `> 🌐 译文`），已在 mdv 中打开。

**工具脚本（`tools/`）：**
- `translate_sidecar.py`：提取段落 → 调用 mdv 翻译 API → 增量写入 sidecar JSON
- `build_bilingual.py`：原文 md + sidecar JSON → 双语对照 md

**设计文档：** `docs/0421-pdf2md-design.md`

**跳过的步骤（标记为待做）：**
- 矢量图提取（需 `pdftoppm` + 坐标裁剪，效果待评估）
- 表格还原（PDF 表格提取质量差，需专项处理）
- 参考引用关联（自研 regex 准确率约 82%，非关键路径）
