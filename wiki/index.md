# Wiki Index

This is the content-oriented entry point for the LLM wiki.

## Overview

- [Knowledge Base Overview](00-overview/knowledge-base-overview.md): repo purpose, structure, and operating model.
- [Neural Combinatorial Optimization for VRP](00-overview/neural-vrp.md): 用 attention encoder-decoder + REINFORCE 端到端学习 VRP 启发式求解器；AM→POMO→混合 OR 的演化路线及 2026 年前沿
- [AV Data Pipeline Architecture](00-overview/av-data-pipeline-architecture.md): 自动驾驶数据处理全链路架构概览
- [自动驾驶模型评测全栈概览](00-overview/av-model-evaluation.md): 感知/预测/规划/端到端各层评测指标、主流 benchmark 及其局限，开环 vs 闭环的根本区别
- [自动驾驶开放生态](00-overview/av-open-ecosystem.md): 数据集/模型权重/Leaderboard 全景，WOMD/Argoverse/nuScenes/nuPlan 开放程度和许可证，OpenDriveLab 生态
- [PNC Model Architecture](00-overview/pnc-model-architecture.md): PNC 神经网络模型的架构、规模与训练数据
- [Tesla Data Engine](00-overview/tesla-data-engine.md): Karpathy 在 Tesla AI Day 演讲中描述的数据飞轮范式

## Roadmaps

- [LLM Learning Roadmap (2026-04-10)](10-roadmaps/llm-learning-roadmap-20260410.md): main post-2024 reading path focused on open models, reasoning, alignment, long context, and agent evaluation.
- [LLM 数据工程路线图](10-roadmaps/data-engineering-llm.md): 过滤、去重、数据混合配方、合成数据的学习路径
- [AV 数据工程路线图](10-roadmaps/data-engineering-av.md): 自动驾驶数据飞轮、标注体系、传感器融合的学习路径

## Concepts

**训练与优化**
- [Instruction Tuning](20-concepts/instruction-tuning.md): 指令微调的数据来源、演化路线与关键论文导读
- [Loss Functions](20-concepts/loss-functions.md): NLL、KL、DPO、REINFORCE 的选择决策树
- [Softmax 与交叉熵](20-concepts/softmax-and-cross-entropy.md): softmax 求导过程，交叉熵损失与 softmax 的链式法则，与 Actor 梯度更新的对比
- [强化学习基础](20-concepts/rl-fundamentals.md): 从监督学习的边界出发，覆盖 agent/环境/策略/V值/Q值/Advantage/REINFORCE/Actor-Critic，配老虎机和格子世界代码
- [Bellman 方程](20-concepts/bellman-equation.md): V 值的自引用递推关系，策略迭代收敛的数学基础，TD 误差的来源
- [参数调度：衰减、Warmup 与 Clip](20-concepts/parameter-scheduling.md): 指数/线性/余弦/阶梯/Warmup+余弦五种调度形式对比，以及 PPO clip 为什么不是衰减
- [RLHF](20-concepts/rlhf.md): InstructGPT、DPO、PPO、GRPO 的完整对比
- [REINFORCE](20-concepts/reinforce.md): 策略梯度基础算法，奖励/Return/Advantage 的区别
- [PPO 逐行讲解](20-concepts/ppo-explained.md): PPO 的梯度机制与监督学习的区别

**数据处理**
- [ccNet](20-concepts/ccnet.md): 困惑度过滤 + 去重 pipeline
- [MinHash](20-concepts/minhash.md): 近重复文档检测
- [fastText](20-concepts/fasttext.md): 语言识别与质量分类
- [Perplexity](20-concepts/perplexity.md): 语言模型困惑度及 KenLM
- [FineWeb](20-concepts/fineweb.md): HuggingFace 高质量网页数据集
- [DCLM](20-concepts/dclm.md): DataComp-LM 数据过滤框架
- [DolmIno](20-concepts/dolmino.md): 数据混合配方研究
- [Domain-Specific Pipeline](20-concepts/domain-specific-pipeline-code-math.md): 代码与数学数据处理
- [t-SNE and Embedding Visualization](20-concepts/tsne-dimensionality-reduction.md): 高维 embedding 可视化方法，适合看局部邻域和覆盖，不适合直接证明数据质量

**部署与推理**
- [ONNX](20-concepts/onnx.md): 跨框架模型中间格式，导出、运行时、图优化、量化全览
- [量化（Quantization）](20-concepts/quantization.md): 低精度部署为什么是开放权重生态的基础设施

**推理与搜索**
- [Monte Carlo Tree Search（MCTS）](20-concepts/mcts.md): UCB 平衡探索利用的树搜索算法，AlphaGo 核心组件，LLM 推理时用于结构化的 test-time compute 搜索
- [UCB1 与多臂老虎机问题](20-concepts/ucb-bandit.md): Auer et al. 2002 的经典 bandit 算法，UCB1 公式的来源与遗憾界理论，MCTS 和在线决策的理论基础
- [UCT: Bandit Based Monte-Carlo Planning](30-papers/uct-kocsis-szepesvari-2006.md): 把 UCB1 应用到树搜索每个节点，提出 UCT 算法，现代 MCTS 的奠基之作，AlphaGo 的直接前身，ECML 2006
- [A Survey of Monte Carlo Tree Search Methods（Browne et al.）](30-papers/mcts-survey-browne-2012.md): MCTS 第一个五年（2006–2011）的系统综述，覆盖 UCT 推导、约 40 种增强（RAVE/UCB1-Tuned/FPU 等）、非游戏应用，IEEE TCIAIG 2012

**Scaling 基础**
- [幂律与 Scaling（Power Law）](20-concepts/power-law-and-scaling.md): 幂律是什么、翻倍法则、在 LLM/语言/城市/地震等领域的普遍出现，以及为何指数小意味着收益递减

**图结构学习**
- [图神经网络（GNN）](20-concepts/gnn.md): 消息传递框架、GCN/GAT/GIN 变体对比、节点/边/图分类任务，附 Karate Club 纯 PyTorch demo（src/gnn_demo.py）

**概率与统计**
- [概率分布速查](20-concepts/probability-distributions.md): 幂律、指数、泊松、对数正态、GMM、Pareto、Weibull、Beta、Dirichlet、负二项、学生 t——密度函数/均值方差/应用/图形判断/统计检验

**基础概念**
- [Tokenization](20-concepts/tokenization.md): BPE、WordPiece、SentencePiece
- [Word Embedding](20-concepts/word-embedding.md): 词向量基础
- [MFU](20-concepts/mfu.md): 模型 FLOPs 利用率
- [Ablation Study](20-concepts/ablation-study.md): 消融实验方法论
- [Synthetic Data with Verification](20-concepts/synthetic-data-with-verification.md): 合成数据与验证
- [Minimax 博弈](20-concepts/minimax.md): 博弈论中对抗框架，GAN / RLHF reward hacking 的理论基础
- [随机森林（Random Forest）](20-concepts/random-forest.md): 集成学习方法，bagging + 特征随机性的决策树集成
- [高斯混合模型（GMM）](20-concepts/gaussian-mixture-model.md): K 个高斯加权叠加建模多峰分布，运动预测轨迹输出的主流方式，Winner-Takes-All loss 驱动多模态分化
- [分布式训练](20-concepts/distributed-training.md): 数据并行 vs 模型并行，AllReduce 梯度同步原理，DistributedDataParallel，数据切分提升 I/O 效率
- [Attention 优化技术](20-concepts/attention-optimization.md): Factorized/FlashAttention/MQA-GQA/Latent Queries/Linear Attention/RoPE——各类优化手段的原理、适用场景和工业采用现状
- [Tensor 操作参考](20-concepts/tensor-operations.md): reshape/permute/expand/einsum 详解，内存布局原理，为什么 reshape 和 for 循环等价

## Papers

- [Perceiver](30-papers/perceiver-2103.03206.md): cross-attention bottleneck 把超高维输入（50k 像素/音频/点云）压入小 latent 数组，无领域专用结构处理任意模态，ImageNet/AudioSet/ModelNet40 全覆盖，ICML 2021
- [The Llama 3 Herd of Models](30-papers/llama-3-herd-of-models.md): a good first systems paper for building a modern LLM reading frame around data, scale, post-training, long context, and safety.
- [Data Mixing Laws](30-papers/data-mixing-laws-2403.16952.md): 用指数函数拟合数据配比与验证损失的定量关系，嵌套 Scaling Laws 预测 1B 模型最优配比，ICLR 2025
- [Phi-1](30-papers/phi-1-2306.11644.md): 1.3B 参数 + 7B 教科书质量数据，HumanEval 50.6%，超越 10 倍大的模型
- [Phi-2 / Phi-3](30-papers/phi-2-phi-3.md): 教科书质量路线扩展到通用推理，phi-3-mini 3.8B 匹敌 GPT-3.5，可本地运行于手机
- [DoReMi](30-papers/doremi-2305.10429.md): 用小代理模型自动优化预训练数据 domain 配比，280M→8B 加速 2.6×
- [LIMO](30-papers/limo-2502.03387.md): 817 条高质量 SFT 数据激发强数学推理能力
- [rStar-Math](30-papers/rstar-math-2501.04519.md): MCTS 驱动的四轮自演化，7B SLM 数学推理达 o1-preview 水平；PPM 偏好训练替代精确 Q 值标注
- [Quiet-STaR](30-papers/quiet-star-2403.09629.md): 让模型在每个 token 处静默思考，从普通文本中自发学习推理
- [nuScenes](30-papers/nuscenes-1903.11027.md): 自动驾驶多传感器数据集，360° 全向感知基准
- [nuPlan](30-papers/nuplan-2106.11810.md): 闭环 ML-based 自动驾驶规划基准，10,000+ 小时真实驾驶日志
- [Axial-DeepLab](30-papers/axial-deeplab-2003.07853.md): 2D self-attention 分解为两个 1D axial-attention，position-sensitive 相对位置编码，全景分割 COCO +2.8% PQ，参数量少 3.8×，ECCV 2020
- [NAVSIM](30-papers/navsim-2406.15349.md): 非反应式仿真评测框架，用真实数据替代仿真器，PDM-Score 综合指标，CVPR 2024 竞赛 143 支队伍，NeurIPS 2024
- [UniAD](30-papers/uniad-2212.10156.md): 规划导向端到端 AD，五模块 query 接口串联（跟踪/建图/运动预测/占据预测/规划），nuScenes 全面 SOTA，CVPR 2023 Best Paper
- [MetaDrive](30-papers/metadrive-2109.12674.md): 可组合自动驾驶 RL 模拟平台，BIG 算法程序化生成 + Waymo/Argoverse 真实数据导入，300 FPS 轻量运行，ScenarioNet 直接前身，TPAMI 2022
- [ScenarioNet](30-papers/scenarionet-2306.12241.md): 统一场景描述格式整合 Waymo/nuScenes/nuPlan/L5/Argoverse，MetaDrive 模拟器支持闭环 RL/IL 和 AD stack 测试，NeurIPS 2023
- [TrafficGen](30-papers/trafficgen-2210.06609.md): 数据驱动交通场景生成，encoder-decoder + 自回归从 WOMD 学习车辆放置和长轨迹，生成数据改善 RL 安全性，ScenarioNet 场景嵌入工具，ICRA 2023
- [Waymo Open Motion Dataset（WOMD）](30-papers/waymo-open-motion-dataset.md): Waymo 运动预测数据集，103K 场景×20s，HD map + agent 状态序列，运动预测 benchmark 标准，无原始传感器数据
- [Argoverse Motion Forecasting](30-papers/argoverse-motion-forecasting.md): Argo AI 运动预测 benchmark，Brier-minFDE 为主指标（距离+置信度综合），速度自适应 MR 阈值，与 WOMD 互补
- [Wayformer](30-papers/wayformer-2207.05844.md): 同质化 attention 架构家族，Early/Late/Hierarchical 三种融合策略系统对比，Early Fusion 最优，WOMD+Argoverse 双榜 SOTA，Waymo，ICRA 2023
- [MTR: Motion Transformer](30-papers/mtr-2209.13508.md): Motion Query Pair（静态意图锚点+动态搜索查询）驱动迭代轨迹精化，WOMD 边际/联合预测双榜第一，Max Planck Institute，NeurIPS 2022
- [MotionDiffuser](30-papers/motiondiffuser-2306.03083.md): 扩散模型学习多 agent 轨迹联合分布，置换不变 denoiser + PCA 压缩 + 推理时可微约束采样（attractor/repeller），WOMD SOTA，CVPR 2023 Highlight
- [DCLM](30-papers/dclm-2406.11794.md): 固定模型只改数据，系统对比数据过滤策略的影响
- [Gopher](30-papers/gopher-2112.11446.md): DeepMind 280B 模型，重复 n-gram 过滤方法被 Llama 3 引用
- [Instruction Tuning with GPT-4](30-papers/instruction-tuning-with-gpt-4-2304.03277.md): 首次系统验证用 GPT-4 生成指令数据和比较数据来蒸馏开源 assistant
- [Self-Instruct](30-papers/self-instruct-2212.10560.md): instruction tuning 合成数据路线起点，用模型自己生成 instruction / instance 再对齐自己
- [Instruction Backtranslation](30-papers/instruction-backtranslation-2308.06259.md): 给无标注网页文本自动配指令，模型自评筛选高质量对，无蒸馏超越所有非蒸馏 LLaMA；ICLR 2024
- [Unnatural Instructions](30-papers/unnatural-instructions-2212.09689.md): 15 个种子样本 → LLM 全自动生成 24 万条指令数据，合成数据媲美人工众包的实证
- [WizardLM / Evol-Instruct](30-papers/wizardlm-evol-instruct-2304.12244.md): 用 LLM 迭代把简单指令进化成复杂版本，WizardLM-13B 在代码/数学上大幅超越 Vicuna，ICLR 2024
- [AlpaGasus](30-papers/alpagasus-2307.08701.md): 用 ChatGPT 对 Alpaca 52k 数据打分，只取 9k 高质量样本训练反超原版——"数据质量 > 数据数量"的早期实证
- [Stanford Alpaca](30-papers/stanford-alpaca.md): 把 Self-Instruct 工程化成低成本、可复现的开源 instruction-tuning recipe
- [Vicuna](30-papers/vicuna-open-source-chatbot.md): 用 ShareGPT 多轮对话把开源 assistant 从 instruction 模式推进到 chat 模式
- [LIMA](30-papers/lima-2305.11206.md): LLaMA 65B + 1,000 条精选 demonstrations，无 RLHF 也能激活强 assistant 行为，NeurIPS 2023
- [Deita](30-papers/deita-2312.15685.md): 用复杂度、质量、多样性三维自动筛选 6K/10K instruction tuning 数据，ICLR 2024
- [MagPie](30-papers/magpie-2406.08464.md): 只用 aligned LLM 的 chat template 触发自生成用户指令，百万级合成 alignment 数据，ICLR 2025
- [Scaling Laws for Neural Language Models (Kaplan et al.)](30-papers/scaling-laws-neural-lm-2001.08361.md): loss 与 N/D/C 各呈幂律；固定计算应优先扩大模型（Kaplan rule），被 Chinchilla 修正为等比例，2020
- [Chinchilla Scaling Laws](30-papers/chinchilla-2203.15556.md): 固定 FLOPs 下参数量与 token 数应等比例扩大（~20× rule），当时大模型普遍过大欠训；70B Chinchilla 以同等计算全面超越 280B Gopher，NeurIPS 2022
- [Scaling Data-Constrained Language Models](30-papers/scaling-data-constrained-lms-2305.16264.md): 数据受限场景下重复数据最多 4 epoch 几乎无损，超过后收益递减；最优分配应多加 epoch 少加参数，NeurIPS 2023
- [Are Emergent Abilities a Mirage?](30-papers/emergent-abilities-mirage-2304.15004.md): 涌现能力是评估指标非线性的人工产物，换用线性指标即变为平滑提升；数学证明 + 视觉模型实验，NeurIPS 2023 Oral
- [PaLM: Scaling Language Modeling with Pathways](30-papers/palm-2204.02311.md): Google 540B 密集 Transformer，Pathways 系统 6144 TPU v4 训练，提出 MFU 效率度量，CoT 推理和多语言 SOTA，2022
- [Dolma: an Open Corpus of Three Trillion Tokens](30-papers/dolma-2402.00159.md): AI2 发布的 3T token 开放英语预训练语料库，配套开源 Dolma Toolkit，OLMo 系列数据基础，数据策划过程最透明的同类语料库，ACL 2024
- [开放权重模型全景（2023–2026-05）](30-papers/open-weight-models-landscape.md): 从 Llama 2 到 Qwen3 / DeepSeek-R1 / OLMo 2 / Qwen3-Coder 的开放权重生态总览

## Comparisons

- [Human Feedback vs AI Feedback vs Verification](40-comparisons/human-feedback-vs-ai-feedback-vs-verification.md): 后训练三类监督信号源的目标、优劣和适用任务对比
- [参数量 vs 能力：2023–2026 的时间压缩](40-comparisons/parameter-vs-capability-over-time-2023-2026.md): 公开参数量模型里，达到相近可用能力所需参数量如何随时间下降
- [Parameters vs Context vs Memory vs Skills](40-comparisons/parameters-context-memory-skills-agent-learning.md): agent 系统里参数、上下文、外部记忆和 skill 的学习边界

## Open Questions

- [Active Questions](50-questions/active-questions.md): current unresolved questions worth revisiting while reading.

## Meta

- [论文名称词源](90-meta/glossary-names.md): Gopher/Dolma/DoReMi/LIMA/Phi 等名字的含义、梗和来源，按字母序排列
- [研究者简介](90-meta/glossary-people.md): wiki 高频作者，按研究方向分组（Scaling/数据/对齐/推理/小模型/系统）
- [Benchmark 速查表](90-meta/glossary-benchmarks.md): wiki 里出现的评测集，按能力分类（综合/常识/QA/推理/NLI/数学/代码），含测什么、评测方式、污染风险
- [数据集速查表](90-meta/glossary-datasets.md): 预训练语料、指令微调数据集、评测数据集，含规模、来源机构、被哪些模型使用
- [机构与实验室速查](90-meta/glossary-orgs.md): Google DeepMind/OpenAI/Anthropic/Meta/AI2 等机构历史、分拆关系、代表工作，以及常见混淆点
- [Knowledge Base Conventions](90-meta/conventions.md): page rules and maintenance expectations.
- [Lint Report 2026-04-28](90-meta/lint-20260428.md): wiki health check findings and action items.
- [Lint Report 2026-05-03](90-meta/lint-20260503.md): wiki health check findings and action items.
- [Log](log.md): chronological record of ingests and updates.
