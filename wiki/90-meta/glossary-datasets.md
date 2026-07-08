# 数据集速查表

wiki 里出现的数据集，按用途分类。"规模"以 token 数计（B=十亿，T=万亿），tokenizer 不统一时为近似值。

---

## 预训练语料（网页/通用）

| 名称 | 来源机构 | 规模 | 语言 | 核心特点 | 被哪些模型用 |
|---|---|---|---|---|---|
| **Common Crawl** | 非营利组织 | 原始 PB 级，清洗后约 1–10T | 多语言 | 互联网爬取的全量快照（2007 年至今，每月一次），是所有网页语料的上游来源 | 几乎所有大模型的数据基础 |
| **C4** | Google | ~175B | 英语 | Common Crawl 经 T5 论文过滤（去重+质量规则），Llama/RedPajama 的网页子集来源之一 | T5、LLaMA、Dolma（补充）|
| **mC4** | Google | ~6.4T | 101 语言 | C4 的多语言版，质量参差不齐 | mT5，部分多语言模型 |
| **RefinedWeb** | Falcon（TII UAE）| ~5T | 英语为主 | 基于 Common Crawl，MacroFilter + dedup，Falcon 模型专属语料，部分公开 | Falcon-40B |
| **RedPajama v1** | Together AI | ~1.2T | 英语 | LLaMA 训练数据的开源复现（7 类来源：CC+C4+GitHub+Arxiv+Books+Wikipedia+StackExchange）| 开源复现基准 |
| **RedPajama v2** | Together AI | ~30T（未过滤）| 英语 | 原始 CC 爬取 + 轻量过滤，主要作为"原料"供外部过滤实验使用 | DCLM 实验基础 |
| **The Pile** | EleutherAI | ~825B | 英语 | 22 类来源的多样组合（含 Books3/ArXiv/GitHub/Wikipedia 等），开源且文档详尽 | GPT-NeoX，Pythia，多个学术模型 |
| **ROOTS** | BigScience | ~1.6T | 46 语言 | 多语言，包含大量非英语资源，英语子集仅约 30% | BLOOM |
| **Dolma** | AI2 | ~3T | 英语 | 6 类来源（CC+GitHub+Reddit+S2ORC+Gutenberg+Wikipedia），完整文档 + 开源工具包 | OLMo 系列 |
| **FineWeb** | HuggingFace | ~15T（原始），精选版 ~1.3T | 英语 | 精细 CC 过滤（教育质量分类器 + 严格 dedup），专注网页质量 | 多个开源实验 |
| **OpenWebText** | EleutherAI | ~38B | 英语 | Reddit 高赞链接内容的复现（模仿 OpenAI WebText），GPT-2 数据近似 | GPT-Neo，多个小模型 |
| **WebText** | OpenAI | 未公开规模 | 英语 | 来自 Reddit 高赞链接的网页文本，GPT-2 训练数据，**未开源** | GPT-2 |
| **MassiveText** | DeepMind | ~10.5TB | 英语 | Gopher 的训练数据，包含 MassiveWeb+Books+Wikipedia+News+GitHub+C4，**未开源** | Gopher，Chinchilla |
| **MassiveWeb** | DeepMind | — | 英语 | MassiveText 的网页子集，用 Gopher 规则过滤 CC，**未开源** | Gopher |

---

## 预训练语料（代码）

| 名称 | 来源机构 | 规模 | 语言 | 核心特点 | 被哪些模型用 |
|---|---|---|---|---|---|
| **The Stack** | BigCode | ~6.4T（v1），~67B（去重后）| 350+ 编程语言 | GitHub 开源代码，permissive license 过滤，MinHash dedup，支持 opt-out | StarCoder，Dolma 代码子集 |
| **GitHub（原始）** | — | 1TB+ | 多语言 | 直接爬取 GitHub 公开仓库，质量参差不齐，需额外过滤 | The Pile，Dolma-Code 的上游 |

---

## 预训练语料（学术/书籍/百科）

| 名称 | 来源机构 | 规模 | 语言 | 核心特点 | 被哪些模型用 |
|---|---|---|---|---|---|
| **peS2o (S2ORC)** | AI2 | ~40M 篇论文，~57B tokens | 英语 | 开放获取学术论文（Semantic Scholar Open Research Corpus），Dolma 学术子集 | Dolma，OLMo |
| **ArXiv** | — | ~1.5M 篇，~70B tokens | 英语 | 预印本平台，The Pile/RedPajama/LLaMA 均包含 | LLaMA，GPT-NeoX，The Pile |
| **Wikipedia** | Wikimedia | ~20B tokens（英语）| 多语言 | 最常见的高质量百科语料，几乎所有模型都包含 | 所有主流模型 |
| **Project Gutenberg** | — | ~6B tokens | 英语为主 | 70,000+ 公版书籍（版权过期），Dolma/LLaMA 书籍子集来源之一 | Dolma，LLaMA，The Pile |
| **Books3** | — | ~100B tokens | 英语 | 包含大量版权图书的盗版数据集（Library Genesis 来源），**版权存疑**，The Pile 包含但 Dolma 回避 | The Pile，LLaMA v1（GPT-3 类似来源）|

---

## 预训练语料（社交媒体/论坛）

| 名称 | 来源机构 | 规模 | 语言 | 核心特点 | 被哪些模型用 |
|---|---|---|---|---|---|
| **Pushshift Reddit** | Reddit/学术 | ~800B tokens（原始）| 英语为主 | Reddit 2005–2023 全量提交和评论，Dolma 社交子集来源。2023 年后 Reddit 限制 API 访问，大规模获取困难 | Dolma，OpenWebText，The Pile |
| **StackExchange / StackOverflow** | — | ~30B tokens | 英语 | 技术问答平台，The Pile/RedPajama 均包含 | The Pile，LLaMA，RedPajama |

---

## 计算机视觉检测数据集

| 名称 | 来源机构 | 规模 | 特点 | 被哪些工作用 |
|---|---|---|---|---|
| **COCO（MS-COCO）** | Microsoft（Lin et al., ECCV 2014）| train2017: 118K 图片，val2017: 5K，test-dev: 20K | 目标检测/分割/关键点标准 benchmark；80 类物体；AP/AP₅₀/AP₇₅/AP_S/AP_M/AP_L 评测体系；COCO test-dev 排行榜是检测器的事实标准 | DETR, Deformable DETR, DINO, Faster R-CNN, YOLO 系列等几乎所有检测方法 |
| **Objects365** | 旷视（Shao et al., ICCV 2019）| 1.7M 标注图片，365 类 | 大规模检测预训练数据集，类别覆盖日常物体；常用于 DETR-like 模型预训练后再在 COCO 上微调 | DINO-SwinL（预训练），Soft Teacher 等 |
| **ImageNet** | Stanford/Princeton（Deng et al., CVPR 2009）| 1K 类 1.2M 图片（IN-1K）/ 22K 类 14M 图片（IN-22K）| 图像分类预训练数据集；IN-1K 用于 ResNet 预训练，IN-22K 用于 SwinL 预训练 | 几乎所有视觉 backbone 的预训练基础 |

---

## 自动驾驶数据集

| 名称 | 来源机构 | 规模 | 特点 | 被哪些工作用 |
|---|---|---|---|---|
| **WOMD（Waymo Open Motion Dataset）** | Waymo | ~100K 场景，每个 20 秒 | 大规模运动预测数据集，包含 HD map + 所有参与者完整状态序列（位置/速度/朝向/类型），聚焦于交通流运动预测，非感知或规划 | TrafficGen（训练数据），ScenarioNet（场景来源）|
| **Argoverse Forecasting（Argoverse 1 Motion）** | Argo AI（Chang et al., CVPR 2019）| 333K 场景，每个 5s，10Hz | 运动预测数据集，2s 历史→3s 预测；含车道中心线 HD map；minADE/minFDE/MR 三指标体系；后被 Argoverse 2 升级 | TNT, Wayformer, MTR/MTR++, VectorNet |
| **INTERACTION Dataset** | Zhan et al.（arXiv:1910.03088, 2019）| ~40K 场景（4 类场景合计）| 专注复杂交互的驾驶数据集，含环岛/无信号路口/有信号路口/合流并道四类；高密度交互场景；含 HD map | TNT, MultiPath 等预测基线 |
| **Stanford Drone Dataset（SDD）** | Robicquet et al.（arXiv:1601.00998, 2016）| 大学校园俯拍视频 | 无地图的行人轨迹数据集，无人机俯视图，2.5 Hz，2s 历史→4.8s 预测；常用于行人预测 benchmark，以像素为单位评测 | TNT, Social GAN, DESIRE, PECNet, SoPhie |
| **OpenLane** | 上海 AI Lab + SenseTime（Chen et al., ECCV 2022）| 200K 标注帧（train 798段+val 202段），test 150段不公开 GT；基于 Waymo Open | 首个真实世界大规模 3D 车道线 benchmark；GT 由 LiDAR+SLAM 半自动生成；14 类车道线；最多 24 条/帧；F-Score + X-error + Z-error 评测体系；wiki 页见 PersFormer+OpenLane 文档 | BEV-LaneDet, PersFormer, Gen-LaneNet, 3D-LaneNet |
| **Apollo 3D Lane Synthetic** | Baidu Apollo（Guo et al., ECCV 2018）| 10,500 帧合成图像，3 场景 | 合成数据集，含 Balanced/Rarely Observed/Visual Variants 三个子集，每个独立 train/test；camera height + pitch 已知，无精确外参 | BEV-LaneDet, PersFormer, Gen-LaneNet, 3D-LaneNet |
| **SemanticKITTI** | KIT（Behley et al. ICCV 2019）| 22 outdoor driving sequences，LiDAR 256×256×32 (0.2m voxel) | 自动驾驶室外 LiDAR 语义场景数据集，21 类（19 语义+1 free+1 unknown），常用于 SSC（Semantic Scene Completion）任务的训练和评测 | MonoScene, VoxFormer, TPVFormer |
| **Occ3D-nuScenes** | Tsinghua MARS Lab（NeurIPS 2023）| 1000 场景，40K 帧，6 路环视相机 | 基于 nuScenes 建立的 3D occupancy 预测 benchmark；三步自动标注 pipeline；16 类 + GO；voxel size 0.4m；带 LiDAR 和 camera visibility mask | TPVFormer, BEVFormer, CTF-Occ 等；现行最常用的 occupancy 评测 benchmark |
| **Occ3D-Waymo** | Tsinghua MARS Lab（NeurIPS 2023）| 1000 sequences，200K 帧，5 路相机 | 基于 Waymo Open Dataset 建立；14 类 + GO；voxel size 0.05m（迄今分辨率最高）；范围 [-80m, 80m] | CTF-Occ, BEVFormer-Fusion 等 |

---

## 指令微调数据集

| 名称 | 来源机构 | 规模 | 特点 | 被哪些工作用 |
|---|---|---|---|---|
| **Self-Instruct（原始数据）** | UW | 52K 条 | GPT-3 自动生成的 instruction/instance 对，种子 175 条 | Self-Instruct 论文 |
| **Alpaca data** | Stanford | 52K 条 | text-davinci-003 生成的指令数据（基于 Self-Instruct），成本约 $500 | Stanford Alpaca |
| **ShareGPT** | 社区 | ~90K 对话 | ChatGPT 对话的真实用户分享，多轮，质量较高但版权不明 | Vicuna，后续众多模型 |
| **Evol-Instruct** | WizardLM | 250K 条 | 用 ChatGPT 迭代"进化"指令，逐步增加复杂度和多样性 | WizardLM 系列 |
| **FLAN** | Google | 1800+ 任务，数千万条 | 将大量 NLP 数据集转换为指令格式，FLAN-T5/PaLM 的指令微调基础 | Flan-T5，Flan-PaLM |
| **LIMA 1000** | Meta FAIR | 1,000 条 | 手工精选的高质量 prompt-response，LIMA 论文的核心数据 | LIMA |
| **MagPie data** | UW/AI2 | 300K–1M 条 | 用 aligned LLM 的 chat template 自动生成，无人工介入 | MagPie 论文 |
| **Deita 6K/10K** | HKUST | 6K / 10K 条 | 从 Evol-Instruct 等来源自动筛选，平衡复杂度/质量/多样性 | Deita |
| **LIMO-Pool** | SJTU GAIR | ~100K 条（筛选前），817 条（最终）| 来自 NuminaMath/AIME 等数学推理语料，LIMO 的训练数据 | LIMO |

---

## 评测框架 / 质量评测数据集

| 名称 | 来源机构 | 用途 | 特点 |
|---|---|---|---|
| **Jigsaw Toxic Comments** | Google/Jigsaw | 训练毒性分类器 | 维基百科评论 + 人工标注毒性标签，Dolma 用它训练 FastText 过滤器 |
| **MT-Bench** | LMSYS | 评测 chat 模型的多轮对话质量 | 80 个多轮对话题目，用 GPT-4 打分，Vicuna 论文用 |
| **Paloma** | AI2 | 评测模型对多领域文本的困惑度分布 | 120+ 领域，用来诊断数据多样性，非能力基准 |
