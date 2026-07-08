# Wiki Index

This is the content-oriented entry point for the LLM wiki.

## Overview

- [Knowledge Base Overview](00-overview/knowledge-base-overview.md): repo purpose, structure, and operating model.
- [Neural Combinatorial Optimization for VRP](00-overview/neural-vrp.md): 用 attention encoder-decoder + REINFORCE 端到端学习 VRP 启发式求解器；AM→POMO→混合 OR 的演化路线及 2026 年前沿
- [AV Data Pipeline Architecture](00-overview/av-data-pipeline-architecture.md): 自动驾驶数据处理全链路架构概览
- [自动驾驶模型评测全栈概览](00-overview/av-model-evaluation.md): 感知/预测/规划/端到端各层评测指标、主流 benchmark 及其局限，开环 vs 闭环的根本区别
- [自动驾驶开放生态](00-overview/av-open-ecosystem.md): 数据集/模型权重/Leaderboard 全景，WOMD/Argoverse/nuScenes/nuPlan 开放程度和许可证，OpenDriveLab 生态
- [ODD：运行设计域](00-overview/odd-operational-design-domain.md): 地理/环境/交通/速度五维约束体系，SAE J3016/ISO 34503/SOTIF 标准关系，ODD 边界识别挑战，与训练数据和数据飞轮的关系
- [SAE J3016：驾驶自动化分级标准](00-overview/sae-j3016.md): L0–L5 六级定义，DDT/ODD/OEDR/ADS 术语体系，L2 vs L3 责任边界，L3 工程困境，全球监管采用现状
- [ISO 34503：ODD 分类标准](00-overview/iso-34503.md): 道路/环境/交通/速度/地理/连接/时间七维属性体系，SAE J3016 的精细化延伸，BSI PAS 1883 的国际化升级，可测量性与语法严谨性
- [SOTIF：预期功能安全（ISO 21448）](00-overview/sotif-iso-21448.md): 覆盖 ISO 26262 空白——ML 算法设计局限导致的功能不足（FI），四象限场景模型，16 类 OI 分类，触发条件分析，FI 缓解架构 Daruma
- [PNC Model Architecture](00-overview/pnc-model-architecture.md): PNC 神经网络模型的架构、规模与训练数据
- [Tesla Data Engine](00-overview/tesla-data-engine.md): Karpathy 在 Tesla AI Day 演讲中描述的数据飞轮范式
- [无图规划（HD Map Free）](00-overview/av-mapless-planning.md): 在线建图替代 HD Map 的技术路线，MapTR→VAD→SparseDrive 谱系，nuScenes/NAVSIM/Bench2Drive benchmark 对比，工业落地现状

## Roadmaps

- [LLM Learning Roadmap (2026-04-10)](10-roadmaps/llm-learning-roadmap-20260410.md): main post-2024 reading path focused on open models, reasoning, alignment, long context, and agent evaluation.
- [LLM 数据工程路线图](10-roadmaps/data-engineering-llm.md): 过滤、去重、数据混合配方、合成数据的学习路径
- [AV 数据工程路线图](10-roadmaps/data-engineering-av.md): 自动驾驶数据飞轮、标注体系、传感器融合的学习路径
- [PNC 规划模型学习路线图](10-roadmaps/pnc-model-learning-roadmap.md): 理解 PNC SharedEncoder+GeneralDecoder 架构的 13 篇论文，按依赖关系排列（Transformer→DETR→Wayformer→TNT→MTR→HiVT→BERT→DPO→Scaling Laws）

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
- [LLM 推理框架概览](20-concepts/llm-inference-frameworks.md): vLLM/SGLang/TensorRT-LLM/TGI/llama.cpp/Ollama/MLX/MLC LLM 的核心技术、名称来源、维护组织和选型决策树
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

**模型架构与推理**
- [Transformer 架构](20-concepts/transformer-architecture.md): Block 结构（Residual/LayerNorm/FFN/Pre-LN vs Post-LN），三类架构（Encoder-only/Decoder-only/Encoder-Decoder），token 前向传播完整 shape 路径
- [自回归生成](20-concepts/autoregressive-generation.md): teacher forcing vs 自回归的区别，Greedy/Beam/Temperature/Top-p/Top-k 采样策略，为什么推理比训练慢，引出 KV cache
- [KV Cache 与 Prompt Cache](20-concepts/kv-cache.md): Prefill vs Decode 阶段，K/V 缓存原理，内存公式与 7B 估算，Prompt Cache 前缀复用，PagedAttention（vLLM）
- [LoRA / QLoRA](20-concepts/lora-qlora.md): 低秩分解 W+BA，7B full fine-tune 显存拆解（约 84 GB），QLoRA NF4+double quant，单卡 48 GB 跑 65B

**微调与对齐**
- [Instruction Tuning](20-concepts/instruction-tuning.md): 指令微调的数据来源、演化路线与关键论文导读
- [RLHF](20-concepts/rlhf.md): InstructGPT、DPO、PPO、GRPO 的完整对比

**训练与优化**
- [Loss Functions](20-concepts/loss-functions.md): NLL、KL、DPO、REINFORCE 的选择决策树
- [Softmax 与交叉熵](20-concepts/softmax-and-cross-entropy.md): softmax 求导过程，交叉熵损失与 softmax 的链式法则，与 Actor 梯度更新的对比
- [强化学习基础](20-concepts/rl-fundamentals.md): 从监督学习的边界出发，覆盖 agent/环境/策略/V值/Q值/Advantage/REINFORCE/Actor-Critic，配老虎机和格子世界代码
- [Bellman 方程](20-concepts/bellman-equation.md): V 值的自引用递推关系，策略迭代收敛的数学基础，TD 误差的来源
- [参数调度：衰减、Warmup 与 Clip](20-concepts/parameter-scheduling.md): 指数/线性/余弦/阶梯/Warmup+余弦五种调度形式对比，以及 PPO clip 为什么不是衰减
- [REINFORCE](20-concepts/reinforce.md): 策略梯度基础算法，奖励/Return/Advantage 的区别
- [PPO 逐行讲解](20-concepts/ppo-explained.md): PPO 的梯度机制与监督学习的区别

**基础概念**
- [Tokenization](20-concepts/tokenization.md): BPE、WordPiece、SentencePiece
- [Word Embedding](20-concepts/word-embedding.md): 词向量基础
- [Attention 直觉：Self/Cross/Local 三种模式](20-concepts/attention-intuition.md): Q/K/V 的图书馆类比，三种 attention 变体的核心计算和 shape 规律，各场景中 QKV 的直觉解释
- [位置编码（PE）](20-concepts/positional-encoding.md): 正弦/余弦/可学习/0初始化/RoPE/ALiBi 各类 PE 的原理、外推性、参数量对比，驾驶场景里坐标编码的特殊设计
- [Attention 优化技术](20-concepts/attention-optimization.md): FlashAttention/RoPE/ALiBi/Latent Queries/Linear Attention——不改变 attention 语义、只改变计算方式的加速手段；改变 attend 范围的变体（GQA/MLA/Factorized 等）见 attention-intuition
- [Tensor 操作参考](20-concepts/tensor-operations.md): reshape/permute/expand/einsum 详解，内存布局原理，为什么 reshape 和 for 循环等价
- [矩阵书写惯例](20-concepts/matrix-notation-conventions.md): 行向量（代码/PyTorch）vs 列向量（数学/论文）两种惯例的对应关系，读论文时快速转换
- [MFU](20-concepts/mfu.md): 模型 FLOPs 利用率
- [Ablation Study](20-concepts/ablation-study.md): 消融实验方法论
- [Synthetic Data with Verification](20-concepts/synthetic-data-with-verification.md): 合成数据与验证
- [Minimax 博弈](20-concepts/minimax.md): 博弈论中对抗框架，GAN / RLHF reward hacking 的理论基础
- [随机森林（Random Forest）](20-concepts/random-forest.md): 集成学习方法，bagging + 特征随机性的决策树集成
- [高斯混合模型（GMM）](20-concepts/gaussian-mixture-model.md): K 个高斯加权叠加建模多峰分布，运动预测轨迹输出的主流方式，Winner-Takes-All loss 驱动多模态分化
- [分布式训练](20-concepts/distributed-training.md): 数据并行 vs 模型并行，AllReduce 梯度同步原理，DistributedDataParallel，数据切分提升 I/O 效率

**目标检测基础**
- [匈牙利算法（Hungarian Algorithm）](20-concepts/hungarian-algorithm.md): 二分图最优一一匹配，O(N³) 解指派问题，DETR 用它替代 anchor+NMS 实现端到端检测，也被 MapTR/BEVFormer 继承

**自动驾驶感知基础**
- [BEV：Bird's Eye View（鸟瞰图表示）](20-concepts/bev-bird-eye-view.md): 俯视坐标系的来龙去脉——经典 IPM vs 学习 BEV（Lift-Splat/BEVFormer），栅格/向量化/稀疏三种形式，TPV 三视图扩展，与 wiki 内 BEVFormer/MapTR/TPVFormer/UniAD 的关系

## Papers

- [Waymo Rider-Only Safety Study（7.1M miles）](30-papers/waymo-safety-rider-only-2312.12675.md): L4 商业部署安全性实证，三层结果指标+漏报调整方法论，police-reported 事故率 -55%、有伤害事故率 -80%，Phoenix/SF 统计显著，Traffic Injury Prevention 2024
- [ADS 功能不足分类与 Daruma 缓解架构（Fu et al., 2024）](30-papers/ads-fi-characterization-daruma-2404.09557.md): 首篇 ADS FI 系统性实证研究，16 类 OI 分类表（世界模型/交通规则/运动规划/ODD），FI 是系统故障 5 倍，Daruma 跨通道仲裁架构，NXP/TU/e/TNO，arXiv 2024
- [Deformable DETR: Deformable Transformers for End-to-End Object Detection](30-papers/deformable-detr-2010.04159.md): MSDeformAttn（稀疏采样 K=4 点 × 多尺度）替换标准 attention，收敛 10× 快于 DETR，小物体 AP_S +5.9，不需要 FPN，COCO AP 46.2（two-stage），MSDeformAttn 成为后续视觉 Transformer 的标准算子，SenseTime，ICLR 2021
- [DETR: End-to-End Object Detection with Transformers](30-papers/detr-2005.12872.md): 集合预测 + 匈牙利二分匹配消除 NMS 和 anchor，Transformer encoder-decoder + 100 object queries，COCO AP 42.0 与 Faster R-CNN 持平，大目标 AP_L +7.8 但小目标 -5.5，开创 detection transformer 范式，FAIR，ECCV 2020
- [PointNet：点云深度学习](30-papers/pointnet-1612.00593.md): 逐点 MLP + max pooling 对称函数直接消费原始点云，无需体素化，置换不变，O(N) 复杂度，分类/零件分割/语义分割三任务 SOTA，快 Subvolume 141 倍，Stanford，CVPR 2017
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
- [FPN: Feature Pyramid Networks for Object Detection](30-papers/fpn-1612.03144.md): top-down 路径 + lateral connections 融合 CNN 多层特征为统一通道多尺度金字塔，单图像取代图像金字塔，COCO SOTA，后续几乎所有检测/BEV 方法的标准特征提取器，FAIR，CVPR 2017
- [DINO: DETR with Improved DeNoising Anchor Boxes](30-papers/dino-2203.03605.md): 对比去噪训练（CDN）+ 混合查询选择 + Look Forward Twice，ResNet-50 12 epoch 49.4 AP（比 DN-DETR +6.0），SwinL 63.3 AP test-dev 首个登顶 COCO 的端到端 Transformer 检测器，IDEA Research + HKUST + Tsinghua，ICLR 2023
- [BEVFormer: BEV Representation from Multi-Camera Images](30-papers/bevformer-2203.17270.md): 200×200 BEV 查询 + 空间交叉注意力（Pillar投影→多相机采样）+ 时序自注意力（ego-motion对齐历史BEV），nuScenes val NDS 0.517 / test 0.569，camera BEV 感知标准基线，上海 AI Lab，ECCV 2022
- [MapTR: Online Vectorized HD Map Construction](30-papers/maptr-2208.14437.md): 等价置换建模消除车道线点集排列歧义，层次化 query decoder，nuScenes 45.9 mAP@25.1 FPS（nano），在线 HD Map 构建奠基之作，HKUST，ICLR 2023
- [Gen-LaneNet: A Generalized and Scalable Approach for 3D Lane Detection](30-papers/gen-lanenet-2003.10656.md): 虚拟 top-view 坐标系 anchor 解决 3D-LaneNet 特征对齐问题 + 两阶段解耦减少 3D 标注需求，发布 Apollo 3D Lane Synthetic（含 Balanced/Rarely Observed/Visual Variants 三划分），F-score +13%（光照泛化），Baidu Apollo，ECCV 2020
- [BEV-LaneDet: 3D Lane Detection Baseline](30-papers/bev-lanedet-2210.06006.md): Virtual Camera 同质化相机参数 + Key-Points Representation（BEV 网格逐格检测）+ STP 双尺度 MLP 特征投影；OpenLane F-Score 58.4（vs PersFormer 47.8），185 FPS TensorRT，HAOMO.AI，2022
- [PersFormer + OpenLane: 3D Lane Detection and Benchmark](30-papers/persformer-openlane-2203.11089.md): OpenLane——首个真实世界 3D 车道线 benchmark（200K 帧、14 类、最多 24 条/帧、Waymo 数据）；PersFormer——IPM+Deformable Attn 前视图→BEV 基线，F-Score 50.5，上海 AI Lab，ECCV 2022 Oral
- [MonoScene: Monocular 3D Semantic Scene Completion](30-papers/monoscene-2112.00726.md): 首个只用单目 RGB 图像完成 3D SSC，FLoSP 沿光线反投影 2D 特征 + 3D CRP 上下文先验 + Scene-Class Affinity Loss，camera-only occupancy 起点，Inria，CVPR 2022
- [TPVFormer: Tri-Perspective View for Vision-Based 3D Semantic Occupancy Prediction](30-papers/tpvformer-2302.07817.md): 三个互相垂直 TPV 平面替代 voxel，O(HW+DH+WD) 复杂度，ICA+CVHA transformer 从多视角图像提升特征，camera-only occupancy 经典 baseline，Tsinghua，CVPR 2023
- [VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion](30-papers/voxformer-2302.12251.md): 两阶段——Stage-1 深度估计驱动稀疏 voxel query proposal，Stage-2 MAE-like 稀疏→稠密 completion，SemanticKITTI IoU+20%/mIoU+18%，NYU/NVIDIA，CVPR 2023
- [Occ3D: A Large-Scale 3D Occupancy Prediction Benchmark](30-papers/occ3d-2304.14365.md): 三步自动标注 pipeline（致密化+遮挡推理+图像精化），Occ3D-nuScenes/Waymo benchmark，visibility mask 设计，CTF-Occ 模型，现行最常用 occupancy 评测标准，Tsinghua，NeurIPS 2023
- [MetaDrive](30-papers/metadrive-2109.12674.md): 可组合自动驾驶 RL 模拟平台，BIG 算法程序化生成 + Waymo/Argoverse 真实数据导入，300 FPS 轻量运行，ScenarioNet 直接前身，TPAMI 2022
- [ScenarioNet](30-papers/scenarionet-2306.12241.md): 统一场景描述格式整合 Waymo/nuScenes/nuPlan/L5/Argoverse，MetaDrive 模拟器支持闭环 RL/IL 和 AD stack 测试，NeurIPS 2023
- [TrafficGen](30-papers/trafficgen-2210.06609.md): 数据驱动交通场景生成，encoder-decoder + 自回归从 WOMD 学习车辆放置和长轨迹，生成数据改善 RL 安全性，ScenarioNet 场景嵌入工具，ICRA 2023
- [Waymo Open Motion Dataset（WOMD）](30-papers/waymo-open-motion-dataset.md): Waymo 运动预测数据集，103K 场景×20s，HD map + agent 状态序列，运动预测 benchmark 标准，无原始传感器数据
- [Argoverse Motion Forecasting](30-papers/argoverse-motion-forecasting.md): Argo AI 运动预测 benchmark，Brier-minFDE 为主指标（距离+置信度综合），速度自适应 MR 阈值，与 WOMD 互补
- [VectorNet: Encoding HD Maps and Agent Dynamics](30-papers/vectornet-2005.04259.md): 向量化 HD 地图和轨迹 + 层次化 GNN（polyline 子图 + 全局交互图）+ 图补全辅助任务，参数减少 70% 且 FLOPs 降 200×，Argoverse SOTA，Waymo/Google，CVPR 2020
- [TNT: Target-driveN Trajectory Prediction](30-papers/tnt-2008.08294.md): 三阶段流水线——目标点预测（softmax over 候选点）+ 目标条件轨迹估计 + NMS 式打分选 K 条；奠定"意图分解"范式，Argoverse/INTERACTION/SDD SOTA，Waymo + Google，CoRL 2020
- [Wayformer](30-papers/wayformer-2207.05844.md): 同质化 attention 架构家族，Early/Late/Hierarchical 三种融合策略系统对比，Early Fusion 最优，WOMD+Argoverse 双榜 SOTA，Waymo，ICRA 2023
- [MTR: Motion Transformer](30-papers/mtr-2209.13508.md): Motion Query Pair（静态意图锚点+动态搜索查询）驱动迭代轨迹精化，WOMD 边际/联合预测双榜第一，Max Planck Institute，NeurIPS 2022
- [MTR++: Multi-Agent Motion Prediction](30-papers/mtrpp-2306.17770.md): Symmetric Context Encoder（共享场景编码）+ Mutually-Guided Intention Querying（跨 agent 意图交流），Waymo Challenge 2022/2023 双冠，TPAMI 2024
- [MotionDiffuser](30-papers/motiondiffuser-2306.03083.md): 扩散模型学习多 agent 轨迹联合分布，置换不变 denoiser + PCA 压缩 + 推理时可微约束采样（attractor/repeller），WOMD SOTA，CVPR 2023 Highlight
- [GameFormer: 预测+规划联合博弈建模](30-papers/gameformer-2303.05760.md): level-k 博弈框架迭代精化预测与规划——每层考虑"他车如何回应上一层预测"，同时输出自车规划+他车预测，WOMD 联合预测+nuPlan 规划双覆盖，NTU，ICCV 2023
- [PLUTO: Pushing the Limit of Imitation Learning-based Planning](30-papers/pluto-2404.14327.md): 横纵解耦 Transformer + 对比模仿学习（CIL）+ 可微辅助 loss，首个在 nuPlan Val14 超越最强规则规划器（93.21 vs PDM-Closed 93.08），HKUST，arXiv 2024
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
- [Sparc3D: Sparse Representation and Construction for High-Resolution 3D Shapes Modeling](30-papers/sparc3d-2505.14521.md): 稀疏可变形 Marching Cubes（Sparcubes）+ 纯稀疏卷积 VAE（Sparconv-VAE），1024³ 水密重网格化 30 秒完成，消除 3D VAE 模态不匹配，训练 4× 加速，与 TRELLIS latent diffusion 配合做高保真 3D 生成，NTU + Math Magic + Imperial College，arXiv 2025-05（非主线，3D 资产生成方向）
- [开放权重模型全景（2023–2026-05）](30-papers/open-weight-models-landscape.md): 从 Llama 2 到 Qwen3 / DeepSeek-R1 / OLMo 2 / Qwen3-Coder 的开放权重生态总览

- [The Energy Footprint of Humans and Large Language Models（Luccioni et al., CACM 2024）](30-papers/energy-footprint-humans-llm-cacm.md): 以"写 250 词"为基准比较 LLM 推理能耗与人类代谢：单次推理约 0.00037 kWh（Llama 65B），人类完成同等任务代谢高出 300 倍；纠正"LLM 一定更费电"的简单叙事，配套 FAccT 2024 正式论文（2311.16863），HuggingFace + CMU

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
- [Lint Report 2026-05-15](90-meta/lint-20260515.md): 105 files, 0 orphans, 0 dead links; 2 papers missing 现状与影响, 7 cross-link gaps.
- [Lint Report 2026-05-20](90-meta/lint-20260520.md): ~112 files, 0 orphans, 0 missing 现状与影响; 7 files missing from index (fixed), 4 AD benchmark glossary gaps (fixed).
- [Log](log.md): chronological record of ingests and updates.
