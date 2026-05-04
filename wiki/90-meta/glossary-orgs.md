# 机构与实验室速查

wiki 里频繁出现的 AI 研究机构，包括它们的历史渊源、分拆关系，以及各自的代表模型/工作。

---

## 主要机构关系图

```
Google（2001 创立）
  ├── Google Brain（2011，Mountain View）────────────┐
  └── DeepMind（2010 创立，2014 被 Google 收购）───┤
            └── 2023 年两者合并为 Google DeepMind ──┘

OpenAI（2015 创立，Musk/Altman/Brockman 等）
  └── 2021 年部分团队出走 → Anthropic（2021，Dario/Daniela Amodei）

Meta AI（Facebook AI Research，FAIR，2013）
  └── 独立研究部门，产出 LLaMA/OPT/Segment Anything 等

Allen Institute for AI（AI2，2014，Paul Allen 创立）
  └── 学术属性，产出 OLMo/Dolma/DCLM/Semantic Scholar 等

EleutherAI（2020，草根开源社区）
  └── GPT-Neo/GPT-NeoX/Pythia/The Pile，HuggingFace 有深度合作

HuggingFace（2016 创立，纽约）
  └── 开源生态平台，自研模型包括 BLOOM（与 BigScience 合作）/SmolLM/FineWeb 等
```

---

## 机构详情

### Google DeepMind

| 属性 | 内容 |
|---|---|
| **前身** | Google Brain（2011）+ DeepMind（2010，伦敦），2023 年合并 |
| **属性** | 企业研究部门（Alphabet 子公司）|
| **规模** | 合并后约 3,000+ 研究员 |
| **LLM 代表工作** | PaLM（2022），PaLM 2（2023），Gemini（2023–），Gopher（2021），Chinchilla（2022）|
| **数据工作** | MassiveText（Gopher 数据，未开源）|
| **特点** | 资源极其充裕，但数据/模型多数闭源；Gemini 是当前 Google 主力模型系列 |

### OpenAI

| 属性 | 内容 |
|---|---|
| **创立** | 2015 年，Elon Musk / Sam Altman / Greg Brockman 等创立，非营利转 capped-profit |
| **属性** | 企业研究 + 产品公司 |
| **LLM 代表工作** | GPT 系列（GPT-3/3.5/4/4o/o1），InstructGPT，CLIP，Whisper，Codex |
| **对齐工作** | InstructGPT（RLHF 范式），Constitutional AI 的竞争对手 |
| **特点** | 产品（ChatGPT）驱动，近年研究发表明显减少；o1 系列标志推理时代 |

### Anthropic

| 属性 | 内容 |
|---|---|
| **创立** | 2021 年，Dario Amodei / Daniela Amodei + 多名 OpenAI 前员工出走创立 |
| **属性** | AI safety 导向的企业研究 |
| **LLM 代表工作** | Claude 系列（Claude 1/2/3/3.5/4）|
| **独特贡献** | Constitutional AI（CAI），Scaling Laws 论文（Kaplan et al. 2020 的核心作者）|
| **特点** | 资金雄厚（Amazon 战略投资），AI safety 定位，论文发表比 OpenAI 多 |

### Meta AI (FAIR)

| 属性 | 内容 |
|---|---|
| **全称** | Meta Fundamental AI Research（FAIR）|
| **创立** | 2013 年，Yann LeCun 主导 |
| **属性** | 企业研究部门，但开源文化强 |
| **LLM 代表工作** | LLaMA 系列（1/2/3/3.1/3.2），OPT，LIMA，MagPie |
| **特点** | 开源大模型生态的核心推动者，LLaMA 3 是目前最强开源基础模型之一；FAIR 和 GenAI 两个团队有时各自发表 |

### Allen Institute for AI (AI2)

| 属性 | 内容 |
|---|---|
| **创立** | 2014 年，微软联创 Paul Allen 出资，西雅图 |
| **属性** | 学术+公益研究机构 |
| **LLM 代表工作** | OLMo 系列（1B/7B/OLMo 2），Dolma，DCLM |
| **数据工作** | Dolma（3T 开放预训练语料），peS2o/S2ORC（学术论文），DCLM（过滤框架）|
| **评测工作** | Paloma，MMLU 相关，ARC（AI2 Reasoning Challenge）|
| **特点** | 注重数据透明度和可复现性，论文配套数据/代码/模型全开放；"做 AI 研究的 AI2"定位 |

### EleutherAI

| 属性 | 内容 |
|---|---|
| **创立** | 2020 年，Discord 上的草根开源社区，后注册为非营利组织 |
| **属性** | 开源社区，研究员分散在全球 |
| **LLM 代表工作** | GPT-Neo，GPT-NeoX-20B，Pythia（scaling 研究用），The Pile |
| **评测工作** | LM Evaluation Harness（lm-eval，事实上的开源模型评测标准框架）|
| **特点** | 资源有限但社区活跃，奠定了"开源 LLM 可复现研究"的基础设施；lm-eval 被几乎所有开源模型论文使用 |

### HuggingFace

| 属性 | 内容 |
|---|---|
| **创立** | 2016 年，原来是聊天机器人公司，后转型为 ML 开源平台 |
| **属性** | 开源平台 + 部分研究团队 |
| **平台产品** | Model Hub（模型托管），Datasets Hub，Spaces（demo），Transformers 库 |
| **自研工作** | BLOOM（与 BigScience 合作），FineWeb，SmolLM，MTEB（embedding 评测）|
| **特点** | ML 开源生态的"GitHub"，几乎所有开源模型/数据集都在这里托管；研究团队（Leandro von Werra 等）产出 FineWeb/FLAN 等数据工作 |

### BigScience / BigCode

| 属性 | 内容 |
|---|---|
| **性质** | HuggingFace 发起的国际学术协作项目 |
| **LLM 代表工作** | BigScience → BLOOM（176B 多语言模型，2022）；BigCode → StarCoder / The Stack |
| **特点** | 开放治理，注重数据版权和 opt-out 机制；The Stack 支持开发者申请移除自己的代码 |

### 高校研究组（LLM 相关）

| 机构 | 代表团队/PI | 代表工作 |
|---|---|---|
| **Stanford** | Percy Liang（HELM/CRFM），Eric Zelikman（STaR/Quiet-STaR），Sang Michael Xie（DoReMi）| Constitutional AI 批评，HELM 评测框架，STaR 推理，DoReMi 数据配比 |
| **UW（华盛顿大学）** | Luke Zettlemoyer，Hannaneh Hajishirzi，Yejin Choi，Noah A. Smith | LLaMA/OLMo/Self-Instruct/MagPie，与 AI2 高度重叠 |
| **CMU** | Graham Neubig，Yiming Yang | 多语言，指令跟随；Unnatural Instructions（Honovich et al. 来自 CMU/Google）|
| **上交 GAIR** | Pengfei Liu | LIMO，推理数据，评测 |
| **MIT** | Yoon Kim，部分合作 | Scaling 理论，数学推理 |

---

## 常见混淆点

**Google Brain vs Google DeepMind**：Google Brain 是 Mountain View 的 Google 内部团队（产出 Transformer、T5、BERT），DeepMind 是伦敦的独立实验室（产出 AlphaGo、Gopher、Chinchilla）。2023 年两者合并成 Google DeepMind，现在新论文统一署名 Google DeepMind。

**OpenAI 和 Anthropic 的关系**：Anthropic 的核心团队（Dario/Daniela Amodei 等）2021 年从 OpenAI 离职创立，两家是直接竞争关系。Dario 曾是 OpenAI 研究 VP，Scaling Laws for Neural LMs（2020）的 Kaplan/McCandlish 等人后来都在 Anthropic。

**Meta FAIR vs Meta GenAI**：Meta 内部有两个 AI 团队。FAIR（Fundamental AI Research，Yann LeCun 主导）更偏学术/基础研究，LLaMA 系列来自 FAIR。GenAI 团队更偏产品/应用（Meta AI 助手）。两者有时联合发表，有时独立发表。

**AI2 和 UW 的关系**：AI2 位于西雅图，和华盛顿大学（UW）地理上相邻，很多研究员双栖（如 Luke Zettlemoyer、Hannaneh Hajishirzi 都是 UW 教授 + AI2 研究员），论文经常联合署名。
