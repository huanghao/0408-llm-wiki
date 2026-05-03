摄入一份资料并写 wiki 文档。支持 arxiv 论文、演讲、技术博客、技术报告等多种来源。

用法：`/ingest <论文标题 / arxiv ID / URL / 资料描述>`

---

## 第一步：判断来源类型

根据 $ARGUMENTS 判断走哪条路径：

| 特征 | 来源类型 | 路径 |
|------|---------|------|
| 包含 arxiv ID（如 2305.10429）或论文标题 | arxiv 论文 | → 路径 A |
| 包含 URL（http/https） | 网页/博客/技术报告 | → 路径 B |
| 描述是演讲/talk/presentation，无 URL | 演讲（无直接原文） | → 路径 C |

---

## 路径 A：arxiv 论文

1. **确认 arxiv ID**：若 $ARGUMENTS 是标题，搜索确认 ID
2. **下载 PDF**：
   ```
   curl -L https://arxiv.org/pdf/<arxiv-id> -o raw/inbox/<arxiv-id>.pdf
   ```
   失败则尝试 `https://ar5iv.org/html/<arxiv-id>`
3. **读取内容**：提取标题、作者、机构、核心贡献、实验结果、局限性
4. **写 wiki 文档**：路径 `wiki/30-papers/<简短名>-<arxiv-id>.md`

---

## 路径 B：网页 / 博客 / 技术报告

1. **抓取内容**：用 WebFetch 读取 URL
2. **若有 PDF 链接**：curl 下载到 `raw/inbox/<名称>.pdf`
3. **写 wiki 文档**：路径 `wiki/30-papers/<简短名>.md` 或 `wiki/00-overview/<简短名>.md`（技术概述类放 overview）

---

## 路径 C：演讲 / Talk（无直接原文）

1. **搜索资料**：用 WebSearch 搜索演讲摘要、笔记、slides、相关报道
2. **明确标注来源性质**：
   - 文档开头注明"来源：演讲，非论文，无法直接引用原文"
   - 内容基于公开资料综合，有不确定的地方注明"据报道"或"据演讲描述"
3. **写 wiki 文档**：路径视内容性质选择
   - 介绍某个系统/方法 → `wiki/30-papers/<名称>.md`
   - 介绍某个概念/架构 → `wiki/00-overview/<名称>.md`

---

## wiki 文档必须包含的 sections

（按 AGENTS.md Paper page writing principles）

- 核心问题
- 方法 / 核心机制
- 关键结果 / 数据（有则写，无则说明）
- 局限性
- **现状与影响**（必须有：还在用吗？被什么取代？一句话定性）
- 和 wiki 内其他概念的关联
- 值得看的部分 / 相关资料

## 最后

更新 `wiki/index.md`，在对应区加入新条目。

---

## 注意

- 所有内容必须有来源依据，不猜测，不确定的地方明确标注
- 路径 C 的内容质量低于 A/B，写完后在文档开头注明"⚠️ 内容基于公开资料，非原文"
- 现状与影响一节必须明确：还在用吗？被什么取代？一句话定性
