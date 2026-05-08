# Waymo Open Motion Dataset（WOMD）

## 是什么

WOMD 是 Waymo 于 2021 年发布的大规模运动预测数据集，专注于**交通参与者的未来轨迹预测**，是目前规模最大、场景最丰富的运动预测公开数据集之一。

- 论文：Large Scale Interactive Motion Forecasting for Autonomous Driving: The Waymo Open Motion Dataset（Ettinger et al., 2021）
- 数据地址：https://waymo.com/open/data/motion/

---

## 核心数据规格

| 项目 | 数据 |
|------|------|
| 场景总数 | 约 103,354 个场景 |
| 单场景时长 | 20 秒（前 1 秒历史 + 后 8 秒需预测） |
| 采样频率 | 10 Hz（100ms/帧） |
| 地理覆盖 | 美国 6 个城市（旧金山、凤凰城、西雅图等） |
| 包含 HD map | 是（车道、路口、停止线、人行横道等） |
| 参与者类型 | 车辆、行人、自行车 |
| 每场景参与者数 | 平均约 30 个，最多超 100 个 |

---

## 和其他数据集的定位区别

这是理解 WOMD 的关键——它和 nuScenes、nuPlan 在**目标任务**上根本不同：

| | WOMD | nuScenes | nuPlan |
|---|---|---|---|
| **核心任务** | 运动预测（预测他车轨迹） | 感知（检测/分割/跟踪） | 规划（ego 车怎么走） |
| **视角** | 他车中心（预测周围所有 agent） | ego 中心（感知 ego 周围） | ego 中心（规划 ego 行为） |
| **传感器数据** | 无原始传感器数据（只有 3D 标注状态） | 6 摄像头 + LiDAR + 毫米波雷达 | 8 摄像头 + LiDAR |
| **标注内容** | 位置/速度/朝向/尺寸/类型序列 | 3D 检测框、语义分割、属性 | 驾驶日志 + 人工标注意图 |
| **数据规模** | 103K 场景 × 20s | 1000 个场景 × 20s | 1500 小时驾驶日志 |
| **典型用途** | 训练/评测运动预测模型 | 训练/评测感知模型 | 训练/评测规划模型 |

**WOMD 没有原始摄像头和 LiDAR 点云**——它直接提供的是已经 3D 标注好的各 agent 状态序列，加上高精地图。这让它更适合研究"给定感知结果，如何预测未来"，而不是"如何从原始传感器数据感知"。

---

## 数据格式

每个场景包含：

**HD Map**（向量化格式）：
- 车道中心线（折线段序列）
- 道路边界
- 停止线、人行横道
- 交通灯状态序列（随时间变化）

**Agent 状态序列**：每个 agent 每帧包含：
- 位置 (x, y)，世界坐标系
- 速度 (vx, vy)
- 朝向角
- 尺寸（长/宽/高）
- 类型（车辆/行人/自行车）
- 是否有效（遮挡或超出范围时无效）

**预测目标**：官方指定每个场景中 8 个"感兴趣的 agent"，需要预测它们从第 1s 到第 8s 的未来轨迹。

---

## 官方 Benchmark：Waymo Motion Prediction Challenge

**Leaderboard 和数据集的关系**：WOMD 是数据集（训练 + 验证 + 测试集），Leaderboard 是基于这份测试集的持续在线排行榜。Waymo 把测试集的 GT 标注保留在服务器端，研究者把模型预测结果提交到 [waymo.com/open](https://waymo.com/open) 的评测系统，服务器计算指标后写入公开排行榜。发表论文时"在 WOMD Leaderboard 达到 SOTA"就是指在这个排行榜上的排名。

每年 Waymo 还以竞赛形式举办 **Waymo Open Motion Dataset Challenge**（通常在 CVPR workshop），设置奖金和时间截止，从 Leaderboard 里选出排名靠前的参赛队，邀请获奖者在 workshop 做报告。常规论文提交和年度竞赛用的是同一套测试集和评测服务器，区别只在于时间窗口和是否有奖项。

**挑战赛入口页**：`waymo.com/open/challenges/` 是历年竞赛的汇总主页。竞赛结束后，该年度的子赛道页面和排行榜仍保留可查，但主页默认突出显示"当前活跃"内容，已结束的年度需要点进各子赛道才能看到结果——这是"找不到 2025 年结果"的原因，并非数据消失。2026 年官方明确不举办正式竞赛，但持续 Leaderboard 仍接受提交。

历年竞赛赛道演变：
- **2020–2021**：3D 检测、跟踪、Motion Prediction、Interaction Prediction
- **2022**：新增 Occupancy/Flow Prediction、3D Camera-Only Detection
- **2023**：新增 Sim Agents（仿真真实性评测）、2D Video Panoptic Segmentation
- **2024**：Motion Prediction、Sim Agents、Occupancy/Flow、3D 语义分割
- **2025**：E2E Challenge（视觉端到端驾驶）、Scenario Generation、Sim Agents、Interaction Prediction；奖项已发放，结果页仍可访问
- **2026**：不举办正式竞赛，Leaderboard 持续开放提交

Leaderboard 当前主要子轨道：
- **Motion Prediction**（边际预测）：独立预测每个 agent，不建模 agent 间交互
- **Interaction Prediction**（联合预测/Interactive Split）：针对成对交互 agent 的场景，预测 joint 轨迹；MotionDiffuser、MTR 等论文报的"Interactive Split"结果就是这个子轨道
- **Sim Agents Challenge**（2023 年新增）：评测模型作为仿真 agent 驾驶的真实性，而非静态预测精度

主要评测指标：

**minADE**（最小平均位移误差）：预测 K 条轨迹（K=6），取和真实轨迹平均距离最近的那条，计算每帧的平均距离误差。

**minFDE**（最小终点位移误差）：K 条预测中终点最接近真实终点的那条的终点误差。

**MR**（Miss Rate）：K 条预测中最接近的那条，终点误差超过 2m 的场景比例。

**mAP**（WOMD 定义的 mAP）：按预测的置信度排序计算各轨迹类型（直行/左转/右转等）的 AP，衡量预测的多样性和置信度校准。

Interactive Split 额外指标（用于联合预测）：**minSADE**（场景级联合 ADE）、**minSFDE**（场景级联合 FDE）、**SMissRate**、**Overlap**（预测轨迹碰撞率）——这些是 MotionDiffuser 和 MTR 论文里报告的主要指标。

---

## 为什么在 AV 领域影响力大

**规模**：103K 场景远超早期数据集（nuScenes 1K、Argoverse 1 约 33K），提供了训练数据量上的质的飞跃。

**交互场景比例高**：Waymo 在采集时刻意筛选了包含复杂交互（变道、并道、行人穿越路口）的场景，比随机采集的数据集包含更多挑战性情形。

**多 agent 联合预测**：早期运动预测只预测单个 agent，WOMD 推动了联合预测多个 agent 的研究方向（MTR、MotionDiffuser 等工作的主要 benchmark）。

**高精地图配套完整**：地图信息质量高，是研究"如何利用地图结构约束预测"的标准数据集。

---

## 在 wiki 涉及的工作中的角色

- **TrafficGen**（2210.06609）：从 WOMD 的 50K 场景子集训练场景生成模型，用前 9 帧学习车辆放置和轨迹生成，生成的场景再导入 MetaDrive 训练 RL 智能体
- **ScenarioNet**（2306.12241）：将 WOMD 格式统一到 ScenarioNet 通用格式，和 nuPlan/nuScenes/Argoverse 并列支持，用 TrafficGen 编码器提取 WOMD 场景 embedding 做 domain gap 分析
- **rStar-Math**（2501.04519）等运动预测工作：WOMD 是运动预测模型（MTR、MotionDiffuser 等）的核心 benchmark，minADE/minFDE/MR 是评测标准

---

## 和 wiki 内其他概念的关联

- [TrafficGen](./trafficgen-2210.06609.md)：以 WOMD 为训练数据的场景生成模型
- [ScenarioNet](./scenarionet-2306.12241.md)：将 WOMD 纳入统一场景描述格式
- [nuScenes](./nuscenes-1903.11027.md)：感知任务的对应数据集，与 WOMD 在任务和格式上互补
- [nuPlan](./nuplan-2106.11810.md)：规划任务的对应数据集
- [自动驾驶模型评测全栈概览](../00-overview/av-model-evaluation.md)：WOMD 在预测评测层的定位
