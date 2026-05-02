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

## [2026-05-01] update | Refresh open-weight landscape page from comments

- Rewrote `wiki/30-papers/open-weight-models-landscape.md` to extend the timeline through `2026-05-01`.
- Added a dedicated section for coding-model evolution, centered on `Qwen2.5-Coder` and `Qwen3-Coder`.
- Split quantization into a standalone concept page: `wiki/20-concepts/quantization.md`.
- Updated `wiki/index.md` to include both the new concept page and the refreshed open-weight landscape page.

## [2026-05-01] ingest | Add note for Instruction Tuning with GPT-4

- Downloaded `arXiv:2304.03277` into `raw/inbox/2304.03277.pdf`.
- Added `wiki/30-papers/instruction-tuning-with-gpt-4-2304.03277.md`.
- Updated `wiki/20-concepts/rlhf.md` to connect RLHF reward modeling with machine-generated comparison data.
- Updated `wiki/index.md` with the new paper entry.

## [2026-05-01] ingest | Expand instruction-tuning lineage and add feedback comparison

- Downloaded `2212.10560` into `raw/inbox/2212.10560.pdf`.
- Added `wiki/30-papers/self-instruct-2212.10560.md` for the original Self-Instruct paper.
- Added `wiki/30-papers/stanford-alpaca.md` as a project-note page based on the Stanford CRFM blog and repo.
- Added `wiki/30-papers/vicuna-open-source-chatbot.md` as a project-note page based on the LMSYS blog and FastChat repo.
- Added `wiki/40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md`.
- Updated `wiki/index.md` to include the new lineage pages and the comparison page.

## [2026-05-01] write | Add parameter-vs-capability timeline comparison

- Added `wiki/40-comparisons/parameter-vs-capability-over-time-2023-2026.md`.
- Framed the page around publicly disclosed parameter counts plus public capability signals, rather than pretending a full industry-wide parameter curve exists.
- Updated `wiki/index.md` with the new comparison entry.

## [2026-05-01] refactor | Split parameter-vs-capability timeline into chat, reasoning, and code

- Refactored `wiki/40-comparisons/parameter-vs-capability-over-time-2023-2026.md` into a hub page.
- Added `wiki/40-comparisons/parameter-vs-capability-chat-over-time-2023-2026.md`.
- Added `wiki/40-comparisons/parameter-vs-capability-reasoning-over-time-2023-2026.md`.
- Added `wiki/40-comparisons/parameter-vs-capability-code-over-time-2023-2026.md` with extra emphasis on the shift from code completion to repo-level and agentic coding.
- Updated `wiki/index.md` with the three split timeline pages.

## [2026-05-02] revise | Merge split parameter timeline back into one file

- Reworked `wiki/40-comparisons/parameter-vs-capability-over-time-2023-2026.md` back into a single document with internal sections for chat, reasoning, and code.
- Removed the three temporary split pages under `wiki/40-comparisons/`.
- Updated `wiki/index.md` to remove the split-page entries.

## [2026-05-02] ingest | Add Deita paper note

- Downloaded `arXiv:2312.15685` into `raw/inbox/2312.15685.pdf`.
- Added `wiki/30-papers/deita-2312.15685.md`.
- Updated `wiki/20-concepts/instruction-tuning.md` to include Deita in the data-quality/data-selection lineage.
- Updated `wiki/40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md` to classify Deita as AI-feedback/model-based data selection.
- Updated `wiki/index.md` with the new paper entry.

## [2026-05-03] ingest | Add MagPie paper note

- Downloaded `arXiv:2406.08464` into `raw/inbox/2406.08464.pdf`.
- Added `wiki/30-papers/magpie-2406.08464.md`.
- Updated `wiki/20-concepts/instruction-tuning.md` to connect MagPie to the self-synthesis instruction data lineage.
- Updated `wiki/40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md` to classify MagPie under AI feedback / self-synthesis.
- Updated `wiki/index.md` with the new paper entry.

## [2026-05-03] note | Clarify agent learning boundaries

- Added `wiki/40-comparisons/parameters-context-memory-skills-agent-learning.md`.
- Framed synthetic data limits, teacher/student signal ceilings, and the distinction between model-level learning and system-level learning.
- Updated `wiki/index.md` with the new comparison entry.

## [2026-05-03] ingest | Add LIMA paper note

- Downloaded `arXiv:2305.11206` into `raw/inbox/2305.11206.pdf`.
- Added `wiki/30-papers/lima-2305.11206.md`.
- Updated `wiki/20-concepts/instruction-tuning.md` to include LIMA in the data-quality/data-quantity lineage.
- Updated `wiki/index.md` with the new paper entry.

## [2026-05-03] note | Add t-SNE concept note from MagPie comments

- Added `wiki/20-concepts/tsne-dimensionality-reduction.md`.
- Linked the MagPie data-analysis section to the new t-SNE concept note.
- Updated `wiki/index.md` with the new concept entry.
