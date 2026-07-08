# The Energy Footprint of Humans and Large Language Models

**来源**：CACM Blog（Communications of the ACM），2024年6月  
**作者**：Sasha Luccioni、Yacine Jernite（HuggingFace）、Emma Strubell（CMU）  
**URL**：https://cacm.acm.org/blogcacm/the-energy-footprint-of-humans-and-large-language-models/  
**配套论文**：Power Hungry Processing: Watts Driving the Cost of AI Deployment? — FAccT 2024，arxiv 2311.16863

> ⚠️ CACM 原文为 blog 文章，部分全文受 403 限制无法直接获取。以下数据来自多个二次报道和配套论文，相互交叉核实，关键数字标注来源。

---

## 核心问题

"LLM 比人类更费电"是一个流行叙事，但它混淆了两个完全不同的比较框架：

1. **训练 vs 人类成长**：LLM 训练确实耗能巨大（GPT-4 训练约 10,000 MWh）
2. **单次推理 vs 人类完成同等任务**：这个比较的结论截然相反

本文聚焦第二个框架——以"完成同一写作任务"为基准，量化 LLM 推理能耗与人类代谢能耗的差距。

---

## 方法 / 核心机制

### 对比基准设定

选取**写一篇 250 词文章**为任务单元：

- 250 词 ≈ 333 tokens（英文平均 ~0.75 词/token）
- 人类参考时长：1 小时完成 250 词写作

### LLM 推理能耗测量

| 模型 | 硬件 | 能耗 | 备注 |
|------|------|------|------|
| Llama 65B | 服务器 GPU | ~4 J/token → 333 tokens = 1,332 J ≈ **0.00037 kWh** | 实测数据 |
| Llama 3 8B | Apple M3 本地 | **< 200 J**，约 20 秒完成 | 本地小模型 |

### 人类代谢能耗

人体基础代谢约 80 W（静息），认知工作时额外增加有限（仅 ~5%），但整个身体 1 小时运转消耗**数 kWh** 的食物能量当量（消化、体温维持、心跳等全部计入）。

**关键比值**：人类完成同等写作任务消耗的生理能量 > LLM 推理能耗的 **300 倍**（3 个数量级）。

---

## 关键结果 / 数据

| 维度 | 人脑 | LLM 推理（Llama 65B） |
|------|------|-----------------------|
| 任务能耗 | ~kWh 量级（整体代谢）| **0.00037 kWh**（333 tokens）|
| 功耗 | 全身 ~80 W，脑 ~20 W | 服务器 GPU 700 W+，但计算极快 |
| 时间尺度 | 1 小时写 250 词 | 数秒到数十秒 |
| 效率倍数 | — | 推理能耗约为人类代谢的 **1/300** |

### 配套论文（Power Hungry Processing, FAccT 2024）的关键数据

实验规模更大：88 个模型，30 个数据集，10 类任务。

- **生成任务 vs 判别任务**：多模态生成（图像/文本）比纯文本分类贵 2–4 个数量级
- **训练 vs 推理**：BLOOM 系 LLM 训练消耗 ≈ 2–5 亿次推理的能量
- **图像生成**：Stable Diffusion XL 单次生成约消耗 1 次手机充电量的能量
- **结论**：为完成同等任务，多用途生成模型比专用判别模型贵数个量级

---

## 局限性

1. **边界划定问题**：人类"任务能耗"只算了代谢，没算食物生产的上游碳排放；LLM 只算了推理，没算数据中心冷却、网络传输。两者不对称。

2. **任务复杂度无法归一化**：人写 250 词 ≠ LLM 生成 250 词，质量和认知深度差异显著，简单用 token 数做分母有失公平。

3. **人类比较不适用于训练阶段**：当考察 LLM 训练（10,000 MWh for GPT-4）vs 人类 18 年学习（估算 ~3,155 kWh 体能），结论完全反转，LLM 训练比人类成长贵约 **1390 万倍**。

4. **大规模部署时能耗线性叠加**：ChatGPT 服务规模下，1M 用户/天 × 2.9 Wh/query ≈ 29,000 kWh/天，相当于 2,700 户美国家庭一天的用电量。

5. **仅考虑英文、特定任务**：不同语言、不同任务类型（代码生成 vs 摘要 vs 分类）的差距差异极大。

---

## 现状与影响

**当前状态**：该分析框架在 2024–2025 年被广泛引用，尤其在 AI 可持续性政策讨论中。

**影响**：
- 纠正了"LLM 一定比人类费电"的简单叙事，推动更精确的任务级能耗对比
- 配套论文 Power Hungry Processing（FAccT 2024）是该团队的正式学术发表，方法论更完整，被 ACL 2025 能耗评估工作引用
- Sasha Luccioni 在 HuggingFace 持续推动模型能耗标准化评测（CodeCarbon 工具）

**被哪些工作延伸**：
- ACL 2025：[Energy Considerations of Large Language Model](https://aclanthology.org/2025.acl-long.1563.pdf)——更系统的 LLM 能耗评估框架
- Frontiers in Communication 2025（Dauner & Socher）：14 个模型 × 1000 道 MMLU 题，reasoning 模型排放量是标准模型 4–6 倍

**一句话定性**：奠基性分析视角，从"训练总能耗"转向"推理任务级能耗"，是 AI 可持续性讨论的关键参照，但数据本身较简单，配套 FAccT 论文的方法论更可靠。

**2026 年视角**：结论仍然成立——单次推理确实比人类完成同等任务的代谢开销低，但这个比较在大规模部署时失去意义；随着推理规模和 reasoning 模型复杂度增加，能耗差距在快速收窄。

---

## 和 wiki 内其他概念的关联

- [[mfu]]：MFU（模型 FLOPs 利用率）是度量 GPU 计算效率的指标，与推理能耗直接相关——MFU 越高，单位能量完成的计算越多
- [[llm-inference-frameworks]]：vLLM/SGLang 等推理框架通过 PagedAttention 和 continuous batching 提升 GPU 利用率，是降低每 token 能耗的工程路径
- [[kv-cache]]：KV Cache 减少重复计算，直接影响推理阶段的实际 FLOPs 和能耗
- [[quantization]]：量化（INT8/INT4）降低推理功耗和显存占用，是绿色部署的关键工具

---

## 值得看的部分 / 相关资料

**必看**：
- [Power Hungry Processing（FAccT 2024，arxiv 2311.16863）](https://arxiv.org/abs/2311.16863)——本 blog 的正式学术版，88 模型 × 30 数据集，方法更严格
- [CACM 原文](https://cacm.acm.org/blogcacm/the-energy-footprint-of-humans-and-large-language-models/)——需要 ACM 订阅

**延伸阅读**：
- [ACL 2025 能耗评估](https://aclanthology.org/2025.acl-long.1563.pdf)：更新的系统评估框架
- [Reconciling contrasting narratives on LLM environmental impact（Nature Scientific Reports 2024）](https://www.nature.com/articles/s41598-024-76682-6)：对"LLM vs 人类"各种对比叙事的综合梳理，结论：human-to-LLM 效率比 40–4400 倍不等，取决于模型大小和任务类型
- [CodeCarbon](https://codecarbon.io/)：Luccioni 团队开发的 Python 工具，自动追踪 ML 实验碳排放
