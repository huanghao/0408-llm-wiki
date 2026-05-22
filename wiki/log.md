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

## [2026-05-22] ingest | PersFormer + OpenLane: 3D Lane Detection and Benchmark (arXiv 2203.11089)

- 下载 `raw/inbox/2203.11089.pdf`。
- 新增 `wiki/30-papers/persformer-openlane-2203.11089.md`：以数据集为主——OpenLane 200K 帧/14 类/最多24条/帧/Waymo数据/7步LiDAR+SLAM标注流水线/F-Score评测协议；PersFormer 为配套基线模型（IPM+Deformable Attn，F-Score 50.5）。
- 更新 `wiki/index.md`，`wiki/90-meta/glossary-datasets.md`（修正 OpenLane 帧数和描述）。

## [2026-05-22] ingest | BEVFormer: BEV Perception from Multi-Camera Images (arXiv 2203.17270)

- 下载 `raw/inbox/2203.17270.pdf`。
- 新增 `wiki/30-papers/bevformer-2203.17270.md`：200×200 BEV 查询 + SCA（Pillar投影+多相机deformable采样）+ TSA（ego-motion对齐历史BEV），PyTorch 风格伪代码含 VRM/SCA/TSA/BEVFormer 完整类，nuScenes val NDS 0.517 / test 0.569，上海 AI Lab，ECCV 2022。
- 更新 `wiki/index.md`，`wiki/90-meta/glossary-names.md`（BEVFormer），`wiki/90-meta/glossary-people.md`（Jifeng Dai, Wenhai Wang, Hongyang Li）。
- 更新 `.claude/commands/ingest.md`：强调伪代码必须用 PyTorch 风格（nn.Module 类定义，附正反例）。

## [2026-05-22] ingest | BEV-LaneDet: 3D Lane Detection Baseline (arXiv 2210.06006)

- 下载 `raw/inbox/2210.06006.pdf`。
- 新增 `wiki/30-papers/bev-lanedet-2210.06006.md`：Virtual Camera（同质化相机参数）+ KPR（BEV 网格逐格 4 路预测）+ STP（双尺度 MLP 特征投影），OpenLane F-Score 58.4 vs PersFormer 47.8，185 FPS TensorRT，HAOMO.AI 2022。
- 更新 `wiki/index.md`，`wiki/90-meta/glossary-datasets.md`（OpenLane、Apollo 3D Lane Synthetic），`wiki/90-meta/glossary-people.md`（Jian Qin）。

## [2026-05-21] ingest | TNT: Target-driveN Trajectory Prediction (arXiv 2008.08294)

- 下载 `raw/inbox/2008.08294.pdf`。
- 新增 `wiki/30-papers/tnt-2008.08294.md`：三阶段流水线（目标点预测→目标条件轨迹估计→NMS 打分选 K 条），VectorNet 场景编码，Argoverse minFDE₆=1.29/MR=0.09，奠定"意图分解"范式，CoRL 2020。
- 更新 `wiki/index.md`，`wiki/90-meta/glossary-names.md`（TNT），`wiki/90-meta/glossary-people.md`（更新 Hang Zhao 和 Jiyang Gao），`wiki/90-meta/glossary-datasets.md`（Argoverse Forecasting、INTERACTION、SDD）。

## [2026-05-20] ingest | MapTR: Online Vectorized HD Map Construction (arXiv 2208.14437)

- 下载 `raw/inbox/2208.14437.pdf`。
- 新增 `wiki/30-papers/maptr-2208.14437.md`：等价置换建模（permutation-equivalent modeling）+ 层次化 query decoder + 三项 loss（cls/p2p/dir）；nuScenes 45.9 mAP @25.1 FPS（nano）/ 58.7 mAP（tiny），ICLR 2023 奠基之作。
- 更新 `wiki/index.md`，`wiki/90-meta/glossary-names.md`（MapTR/HDMapNet/VectorMapNet/GKT/VAD），`wiki/90-meta/glossary-people.md`（Xinggang Wang, Bencheng Liao）。

## [2026-05-20] update | 无图规划（HD Map Free）概览 + nuScenes 场景结构补充

- 新增 `wiki/00-overview/av-mapless-planning.md`：在线建图替代 HD Map 的技术路线，nuScenes/NAVSIM/Bench2Drive benchmark 对比，MapTR→VAD→SparseDrive 模型谱系，nuScenes 开环指标失真问题，工业落地现状，推荐阅读路线。
- 更新 `wiki/00-overview/av-open-ecosystem.md`：规划 leaderboard 表中加入 Bench2Drive、nuScenes E2E 规划条目，并注明 BEV-Planner 指标失真警告。
- 更新 `wiki/30-papers/nuscenes-1903.11027.md`：新增「场景的组织方式与区分维度」一节，覆盖四层数据结构、4 个地图区域、scene.description 自由文本机制、agent 级运动模式分类、和 nuPlan 场景体系的对比。
- 更新 `wiki/index.md`。

## [2026-05-19] ingest | PLUTO: Pushing the Limit of Imitation Learning-based Planning (arXiv 2404.14327)

- Added `wiki/30-papers/pluto-2404.14327.md`：横纵解耦 Transformer + CIL + 可微辅助 loss，首个在 nuPlan Val14 超越 PDM-Closed 的学习方法，HKUST 2024。
- Updated `wiki/index.md`，`wiki/90-meta/glossary-names.md`（PLUTO 词源），`wiki/90-meta/glossary-people.md`（Jie Cheng, Qifeng Chen）。

## [2026-05-19] ingest | 3D Occupancy Prediction 四篇核心论文

- 下载并阅读 4 篇论文 PDF（MonoScene/TPVFormer/VoxFormer/Occ3D）。
- 注意：arxiv 2302.11655 实际是 K-12 cybersecurity 论文，正确 Occ3D ID 为 2304.14365，重新下载。
- 新增 `wiki/30-papers/monoscene-2112.00726.md`：首个单目 RGB SSC，FLoSP+3D CRP+新 loss，Inria，CVPR 2022。
- 新增 `wiki/30-papers/tpvformer-2302.07817.md`：TPV 三视图表示，camera-only occupancy 经典 baseline，Tsinghua，CVPR 2023。
- 新增 `wiki/30-papers/voxformer-2302.12251.md`：两阶段稀疏 voxel query + MAE-like completion，LiDAR-assisted SSC 代表，NYU/NVIDIA，CVPR 2023。
- 新增 `wiki/30-papers/occ3d-2304.14365.md`：建立 Occ3D-nuScenes/Waymo benchmark + 自动标注 pipeline，NeurIPS 2023，现行最常用 occupancy 评测标准。
- 更新 `wiki/index.md`，`wiki/90-meta/glossary-names.md`（TPVFormer/VoxFormer/MonoScene/Occ3D/CTF-Occ/BEV 词源），`wiki/90-meta/glossary-datasets.md`（SemanticKITTI/Occ3D-nuScenes/Occ3D-Waymo）。

## [2026-05-03] note | Add t-SNE concept note from MagPie comments

- Added `wiki/20-concepts/tsne-dimensionality-reduction.md`.
- Linked the MagPie data-analysis section to the new t-SNE concept note.
- Updated `wiki/index.md` with the new concept entry.

## [2026-05-22] ingest | Gen-LaneNet: A Generalized and Scalable Approach for 3D Lane Detection (arXiv 2003.10656)

- 下载 `raw/inbox/gen-lanenet-2003.10656.pdf`（34MB）。
- 新增 `wiki/30-papers/gen-lanenet-2003.10656.md`：虚拟 top-view 坐标系 anchor + 两阶段解耦框架，发布 Apollo 3D Lane Synthetic 数据集（含 Balanced/Rarely Observed/Visual Variants 三划分），Baidu Apollo，ECCV 2020。
- 更新 `wiki/index.md`（Gen-LaneNet 条目）。
- 更新 `wiki/90-meta/glossary-names.md`（Gen-LaneNet / 3D-LaneNet 词源）。
- 更新 `wiki/90-meta/glossary-people.md`（Yuliang Guo，Baidu Apollo）。
- `wiki/90-meta/glossary-datasets.md` 中 Apollo 3D Lane Synthetic 条目已存在，无需更新。
