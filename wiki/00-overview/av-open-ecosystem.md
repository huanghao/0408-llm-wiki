# 自动驾驶开放生态：数据集、模型、Leaderboard

自动驾驶领域的开放程度远低于 LLM——大多数数据和代码来自产业界而非学术界，数据集版权复杂，Leaderboard 分散在各家机构。这篇文档梳理 2019-2026 年主流的开放数据集、开放模型权重、公开排行榜，以及它们之间的关联。

---

## 开放数据集全景

### 感知类（原始传感器数据）

| 数据集 | 机构 | 时间 | 传感器 | 规模 | 核心任务 |
|--------|------|------|--------|------|---------|
| **KITTI** | KIT | 2012 | 单目+LiDAR | 14K 帧 | 3D 检测、深度估计 | 
| **nuScenes** | Motional | 2019 | 6 摄像头+LiDAR+毫米波雷达 | 1000 场景×20s | 3D 检测、跟踪、预测 |
| **Waymo Open Dataset** | Waymo | 2019 | 5 摄像头+5 LiDAR | 1000 段×20s | 2D/3D 检测、跟踪 |
| **Argoverse 1** | Argo AI | 2019 | 7 摄像头+LiDAR+HD map | 333K 场景 | 运动预测 |
| **Argoverse 2** | Argo AI | 2022 | 7 摄像头+LiDAR+HD map | 250K 场景 | 运动预测、感知 |
| **nuPlan** | Motional | 2021 | 8 摄像头+LiDAR | 1500h | 规划（闭环）|
| **Lyft L5** | Lyft | 2020 | 摄像头+LiDAR | 170K 场景 | 运动预测 |

**开放程度**：nuScenes、Argoverse 完全开放（注册即可下载）；Waymo Open Dataset 需要申请，审核通常几天内通过；nuPlan 完全开放。

### 运动预测专用（无原始传感器）

| 数据集 | 机构 | 时间 | 规模 | 特点 |
|--------|------|------|------|------|
| **WOMD（Waymo Open Motion Dataset）** | Waymo | 2021 | 103K 场景×20s | 最大规模，Interactive Split，无原始传感器 |
| **INTERACTION Dataset** | INTERACTION | 2019 | 40K 场景 | 复杂交互场景（合并/绕行），高精地图 |
| **ETH/UCY** | 学术 | 2007-2009 | 行人轨迹 | 行人预测经典 benchmark，体量小但历史久 |

---

## 主要 Leaderboard

### 运动预测

| Leaderboard | 数据集 | 主指标 | 更新频率 | 地址 |
|------------|--------|--------|---------|------|
| **WOMD Motion Prediction** | WOMD | mAP | 持续开放 | waymo.com/open |
| **WOMD Interaction Prediction** | WOMD Interactive Split | mAP（joint） | 持续开放 | waymo.com/open |
| **Argoverse 1 Motion** | Argoverse 1 | minFDE | 已关闭新提交 | eval.ai |
| **Argoverse 2 Motion** | Argoverse 2 | Brier-minFDE | 持续开放 | eval.ai |

**WOMD 两个子轨道的区别**：
- Motion Prediction（边际预测）：独立评测每个 agent，N 个 agent 的预测彼此独立
- Interaction Prediction（联合预测）：评测成对的交互 agent，需要预测两个 agent 的 joint 轨迹，考核它们之间是否合理互动

### 感知

| Leaderboard | 数据集 | 主指标 | 地址 |
|------------|--------|--------|------|
| **nuScenes 3D Detection** | nuScenes | NDS | nuscenes.org |
| **nuScenes Prediction** | nuScenes | minADE | nuscenes.org |
| **Waymo Detection** | Waymo OD | mAP L1/L2 | waymo.com/open |

### 规划/端到端

| Leaderboard | 数据集 | 主指标 | 备注 |
|------------|--------|--------|------|
| **nuPlan** | nuPlan | 综合分数 | reactive 仿真，计算代价高；需要 HD Map |
| **NAVSIM** | nuScenes 子集 | PDM-Score | 非反应式，低成本，2024 年兴起；无图可用 |
| **nuScenes E2E 规划** | nuScenes | L2 + collision rate | 开环评测；**指标失真**（见注）|
| **Bench2Drive** | CARLA 仿真 | 路线完成率+碰撞 | 端到端闭环，无图，arXiv 2604.01259 |
| **CARLA Leaderboard** | 纯仿真 | 路线完成率+碰撞 | 端到端系统评测 |

⚠️ **nuScenes E2E 规划指标说明**：BEV-Planner（2024，arXiv 2406.02445）证明 nuScenes 开环规划的 L2 + collision 指标严重失真——匀速直行基线可接近 SOTA，原因是数据集中直道占比过高。2022-2023 年在此榜上的排名不反映真实规划能力。NAVSIM 的 PDM-Score 是目前更可靠的替代。

### 年度竞赛（附奖金）

Waymo、nuScenes 等每年在 CVPR/ECCV Workshop 举办竞赛：
- **Waymo Open Dataset Challenge**：每年 CVPR，边际/联合/仿真 agent 等多个子赛道，MTR 是 2022 年冠军
- **nuScenes Detection/Prediction Challenge**：每年 CVPR
- **Argoverse Motion Forecasting Competition**：每年 NeurIPS

---

## 开放模型权重

自动驾驶领域的模型权重开放程度远低于 LLM，以下是主要的开源模型：

| 模型 | 机构 | 任务 | 代码/权重 | 备注 |
|------|------|------|----------|------|
| **UniAD** | OpenDriveLab | 端到端（感知+预测+规划） | github.com/OpenDriveLab/UniAD | 代码+权重完全开放 |
| **MTR** | Max Planck | 运动预测 | github.com/sshaoshuai/MTR | 代码+权重开放 |
| **Wayformer** | Waymo | 运动预测 | 无官方权重 | 论文描述架构，Waymo 未开源 |
| **MotionDiffuser** | CMU/Waymo | 运动预测 | 部分开源 | - |
| **SparseDrive** | - | 端到端 | 开源 | UniAD 的轻量替代 |
| **VAD** | Horizon Robotics | 端到端 | 开源 | 向量化场景表示 |
| **DriveDreamer** | - | 端到端 | 部分开源 | - |
| **NAVSIM baselines** | 多方 | 端到端评测 | 开源 | TransFuser、Hydra-MDP 等 |

**和 LLM 开源生态的对比**：AV 领域没有类似 HuggingFace 的统一模型中心，模型分散在各机构 GitHub，权重格式不统一（通常是 PyTorch checkpoint），没有统一的部署框架。

---

## OpenDriveLab：目前最活跃的学术开放生态

上海 AI Lab 主导的 OpenDriveLab 是目前 AV 领域学术开源最活跃的机构：

- **UniAD**（CVPR 2023 Best Paper）：完整开源
- **DriveX / GenAD**：UniAD 后续工作
- **OpenLane-V2**：车道图感知 benchmark
- **DriveLM**：VLM 驱动的驾驶数据集
- GitHub：github.com/OpenDriveLab

---

## 数据集使用注意事项

**许可证差异**：
- nuScenes：CC BY-NC-SA（非商用），学术可以自由用
- Waymo Open Dataset：研究专用，明确禁止商业使用，发表论文需标注 Waymo
- Argoverse：MIT 许可（相对宽松）
- nuPlan：CC BY-NC-SA

**数据下载方式**：
- Waymo：通过 Google Cloud Storage 下载，需要 GCP 账户，数据量较大（约 1TB+）
- nuScenes：直接从官网下载，分 mini/trainval/test 几个版本
- Argoverse：直接从 argoverse-api 下载，支持流式读取

---

## 2026 年视角：哪些值得重点看，哪些可以跳过

> 以下是主观判断，基于"2026 年如果要进入 AV 领域做研究/工程，哪些是必须懂的，哪些已经被历史淘汰"的标准。

### 数据集：留哪些，跳哪些

**重点掌握（主流，还在用）**

- **WOMD**：运动预测的事实标准。mAP、minSADE 是论文必须报告的指标，不了解 WOMD 等于没有进入这个领域。
- **nuScenes**：感知评测的事实标准，NDS 指标、BEV 感知范式都从这里来。2026 年感知论文仍以 nuScenes 为主要 benchmark。
- **nuPlan + NAVSIM**：规划评测的两个入口。nuPlan 是"重量级闭环"，NAVSIM 是"轻量开环近似"——两个都要知道，但做研究选 NAVSIM 更实际（计算成本低 10 倍以上）。
- **Argoverse 2**：运动预测的次主流 benchmark，Brier-minFDE 指标值得了解；Argoverse 1 已基本退出，不用专门学。

**可以了解理念但不需要深入**

- **KITTI**：2012 年的数据集，已经严重过时，只在学 3D 检测基础概念时会碰到，不用专门研究。
- **Lyft L5**：已停止维护，研究论文引用越来越少，可跳过。
- **INTERACTION / ETH-UCY**：行人预测的历史 benchmark，WOMD 和 Argoverse 覆盖了更复杂的交通场景，这两个只在做行人专项研究时才需要看。

---

### 模型：顶会 AV 著名模型全景（2021–2026）

按任务分类，星号（⭐）表示"必须知道的里程碑"：

#### 运动预测

| 模型 | 机构 | 年份 | 为什么重要 | 推荐程度 |
|---|---|---|---|---|
| ⭐ **Wayformer** | Waymo | ICRA 2023 | 系统性对比 Early/Late/Hierarchical Fusion，定义了"同质化 attention"基线，方法论价值高 | **必看**（架构参考）|
| ⭐ **MTR** | Max Planck | NeurIPS 2022 Oral | Motion Query Pair 思路，2022 年 WOMD 双榜第一，代码开源；影响了之后多个工作的 query-based 设计 | **必看**（方法+结果）|
| ⭐ **MotionDiffuser** | Waymo | CVPR 2023 Highlight | 第一个在运动预测上用扩散模型建联合分布，约束采样框架影响深远 | **必看**（范式转变）|
| **JFP** | Waymo | CoRL 2022 | mAP 指标上长期 SOTA，因子化联合分布；但代码未开源，只能看论文 | 推荐了解 |
| **SceneTransformer** | Waymo | ICLR 2022 | 最早系统做联合预测的 Transformer，但被 MTR/MotionDiffuser 超过 | 了解历史 |
| **MultiPath++** | Waymo | RA-L 2022 | anchor-based 方法的代表，理解"为什么需要 anchor"时参考 | 了解理念 |

#### 端到端 / 规划

| 模型 | 机构 | 年份 | 为什么重要 | 推荐程度 |
|---|---|---|---|---|
| ⭐ **UniAD** | OpenDriveLab | CVPR 2023 Best Paper | 五模块 query 接口串联，规划导向端到端的奠基之作，代码权重完全开源 | **必看**（架构+代码）|
| ⭐ **NAVSIM baselines** | 多方 | 2024 | 包含 TransFuser、Hydra-MDP 等，提供可直接复现的端到端评测基线 | **必看**（可复现）|
| **VAD** | Horizon | ICCV 2023 | 向量化场景表示做端到端，比 UniAD 轻量；Horizon 内部产品线有影响 | 推荐了解 |
| **SparseDrive** | 学术 | 2024 | UniAD 的稀疏化轻量版，工程价值高 | 推荐了解 |
| **DriveDreamer / DrivX** | 多方 | 2023–2024 | 世界模型 / 生成式端到端，代表新方向但还不成熟 | 了解趋势 |
| **CARLA baselines** | 多方 | 持续 | CARLA 仿真里的各种 IL/RL baselines，与真实数据 gap 大，主要学方法论 | 了解理念 |

#### 感知

| 模型 | 机构 | 年份 | 为什么重要 | 推荐程度 |
|---|---|---|---|---|
| ⭐ **BEVFormer** | OpenDriveLab | ECCV 2022 | BEV 感知范式的奠基之作，时序 attention + 空间 cross-attention，nuScenes SOTA | **必看**（BEV 范式）|
| **DETR3D / PETR** | 多方 | 2021–2022 | Camera-only 3D 检测的早期主流方法，BEVFormer 出来后逐渐被取代 | 了解历史 |
| **BEV-Fusion** | MIT | CVPR 2022 | LiDAR+Camera 融合做 BEV，仍是融合感知的重要参考 | 推荐了解 |

---

### 方法/方向：哪些还是主流，哪些已经边缘化

**2026 年仍主流**：
- **BEV 统一表示**：已成感知和预测的标准范式，不会被取代
- **Transformer + query-based 预测**：MTR 开创的路线，仍是运动预测主流
- **扩散模型 for 场景生成和预测**：MotionDiffuser 开创，2024–2025 年持续有新工作
- **NAVSIM 评测框架**：轻量开环评测，正在取代 nuPlan 成为端到端研究的主流评测方式
- **端到端（感知→预测→规划）**：UniAD 之后的主流方向，但如何做"有保障的端到端"仍是开放问题

**正在边缘化**：
- **anchor-based 预测（MultiPath/MultiPath++）**：被 query-based 和扩散模型方法超过，不再是研究重点
- **CARLA 仿真为主的端到端**：与真实数据 gap 太大，逐渐被 nuPlan/NAVSIM 取代
- **单传感器纯 LiDAR 感知**：已被多模态融合和 Camera-only 路线分流，纯 LiDAR 只在特定硬件场景下有意义

**新兴但尚未成熟**：
- **VLM/LLM 驱动的驾驶**（DriveLM、DriveVLM）：2024–2025 年热点，能力提升明显但离部署还远
- **世界模型**（DriveDreamer、GAIA-1）：生成式预测，概念领先但评测体系不完善
- **Foundation Model for AV**：借鉴 LLM 的大规模预训练思路，仍处于探索阶段

---

### 推荐阅读顺序（从零开始）

1. **数据集**：先读 nuScenes 论文了解感知数据集结构 → 读 WOMD 了解预测数据集结构 → 读 NAVSIM 了解规划评测逻辑
2. **感知范式**：BEVFormer（BEV 表示的来龙去脉）
3. **预测**：Wayformer（架构基础）→ MTR（query-based 方法）→ MotionDiffuser（扩散模型方向）
4. **端到端**：UniAD（必读，完整端到端系统）→ NAVSIM baselines（可复现的评测）
5. **可跳过**：KITTI、Lyft L5、MultiPath（只在需要了解历史时看）、CARLA baselines（了解思路即可）

---

## 和 wiki 内其他概念的关联

- [自动驾驶模型评测全栈概览](./av-model-evaluation.md)：评测指标体系和这些 benchmark 的关系
- [WOMD](../30-papers/waymo-open-motion-dataset.md)：最主流的运动预测数据集详细介绍
- [Argoverse Motion Forecasting](../30-papers/argoverse-motion-forecasting.md)：Argoverse 数据集和指标详细介绍
- [nuScenes](../30-papers/nuscenes-1903.11027.md)：感知评测主流数据集
- [nuPlan](../30-papers/nuplan-2106.11810.md)：规划闭环评测 benchmark
- [NAVSIM](../30-papers/navsim-2406.15349.md)：新兴的轻量评测框架
