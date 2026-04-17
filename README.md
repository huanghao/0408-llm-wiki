# LLM Wiki

这是一个个人学习 LLM 的知识库，不是单次问答的临时目录。

设计目标：

1. `raw/` 只放原始材料，作为不可变输入。
2. `wiki/` 放 LLM 维护的结构化知识页、综述、问题清单和比较页面。
3. `AGENTS.md` 约束 Codex 在这个目录里的工作方式，让知识库可以持续积累，而不是每次重新从原始材料里“临时检索”。

## 目录结构

- `raw/inbox/`
  新放进来的论文、网页转存、截图、草稿笔记，等待处理。
- `raw/processed/`
  已处理过的原始材料，可保留原始命名。
- `raw/assets/`
  图片、图表、论文配图、本地附件。
- `wiki/index.md`
  内容索引，进入知识库时优先读它。
- `wiki/log.md`
  时间线日志，记录 ingest / query / lint。
- `wiki/00-overview/`
  总览页、入口页、阅读地图。
- `wiki/10-roadmaps/`
  学习路线、阶段性读书单。
- `wiki/20-concepts/`
  概念页，如 scaling、reasoning、alignment、MoE、agent。
- `wiki/30-papers/`
  论文页，一篇一页，记录摘要、贡献、争议、关联页面。
- `wiki/40-comparisons/`
  横向比较页，如 DPO vs RLHF、RAG vs long context。
- `wiki/50-questions/`
  研究问题、待验证假设、阅读中的疑问。
- `wiki/90-meta/`
  维护规则、页面模板、知识库健康检查结果。
- `templates/`
  论文页、概念页、比较页模板。

## 推荐工作流

1. 把新材料放进 `raw/inbox/`
2. 让 Codex ingest 单个来源
3. Codex 更新相关 wiki 页面、`wiki/index.md`、`wiki/log.md`
4. 周期性执行 lint，查孤儿页、过时结论、缺失交叉引用

## 起步内容

当前已初始化：

1. 目录骨架
2. 面向 Codex 的 `AGENTS.md`
3. 一份 LLM 学习线路图
4. 索引和日志文件

后续可以继续补：

1. 论文摘要页
2. 概念页之间的交叉链接
3. 个人问题清单和阶段性结论
