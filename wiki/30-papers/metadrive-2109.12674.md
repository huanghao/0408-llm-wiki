# MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning（Li et al., 2022）

一句话总结：MetaDrive 是一个可组合的自动驾驶 RL 模拟平台，通过程序化生成（BIG 算法）和真实数据导入（Waymo/Argoverse）生成无限多样的驾驶场景，支持单智能体泛化、安全 RL 和多智能体学习，300 FPS 轻量运行，是 ScenarioNet 的直接前身。

## 基本信息

- 论文：MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning
- 作者：Quanyi Li, Zhenghao Peng, Lan Feng, Qihang Zhang, Zhenghai Xue, Bolei Zhou
- 机构：CUHK / Centre for Perceptual and Interactive Intelligence（Li, Zhang, Xue）；ETH Zurich（Feng）；UCLA（Peng, Zhou）
- 发表：IEEE Transactions on Pattern Analysis and Machine Intelligence（TPAMI），2022
- arXiv：2109.12674（首次提交 2021-09）
- 代码：https://metadriverse.github.io/metadrive

## 核心问题

RL 在自动驾驶中的泛化问题：**一个在某个城市或场景中训练好的智能体，换到新场景往往失败**——根本原因是现有模拟器的训练场景多样性严重不足（固定地图、固定交通流）。

同期主流模拟器的两类缺陷：

| 类型 | 代表 | 问题 |
|---|---|---|
| 高保真渲染（CARLA、SUMMIT、AirSim） | 逼真但慢，场景固定，难以大规模随机化 | 与实际 RL 训练效率矛盾 |
| 轻量但功能单一（Highway-env、DriverGym） | 缺乏多智能体支持或真实数据导入 | 无法研究泛化和安全问题 |

MetaDrive 的核心设计原则：**可组合性（Compositionality）**——把道路网络、交通流、障碍物、智能体控制解耦为独立组件（Manager），可任意组合生成新场景，单个 100 MB 实例即可以 300 FPS 运行。

## 方法：系统设计

### 核心抽象：Object + Policy + Manager

**Object**：场景中所有实体（车辆、障碍物、路灯、道路结构）的统一基类。每个 Object 绑定物理模型（Bullet 引擎刚体）和渲染模型（Panda3D），通过 `object.step()` / `object.set_state()` 控制。参数空间 $\Omega$ 定义 Object 的可随机化范围（车道宽度、车辆摩擦系数等）。

**Policy**：接收 Object 状态和环境状态、返回动作的函数。内置策略：
- **IDM policy**：规则驾驶，模拟真实交通流
- **Replay Traffic Manager**：严格回放数据集中的轨迹
- **EnvInputPolicy**：接受外部 RL 动作（ego 车辆默认策略）

**Manager**：管理一类 Object 的生命周期（生成/回收/状态获取）。四类基础 Manager 顺序叠加即构成完整场景：

```
Map Manager        → 生成/加载地图（PG 或真实数据）
Traffic Manager    → 生成/控制背景交通流
Object Manager     → 随机散布障碍物（安全关键场景）
Agent Manager      → 管理 RL 智能体（单/多智能体）
```

通过 `engine.register_manager()` / `engine.update_manager()` 可在运行时替换或添加 Manager，实现"同一地图，不同任务"的组合。

### 程序化地图生成：BIG 算法（Section 4.1）

MetaDrive 定义 7 种基础路块（Block）：直路（Straight）、匝道（Ramp）、岔路（Fork）、环形（Roundabout）、弯道（Curve）、T 形交叉（T-Intersection）、十字交叉（Intersection）。每个 Block 有若干 socket（连接口）和参数空间 $\Omega$（车道数、宽度、曲率等）。

**BIG（Block Incremental Generation）算法**（Algorithm 1）：
1. 随机选路块类型 $T$，在 $\Omega_T$ 上随机采样参数 $\omega$，得到 $G_\omega$
2. 将新路块旋转至与已有网络某 socket 对接
3. 若新路块与已有网络无交叉 → 追加；否则丢弃，重试（最多 T 次）
4. 重复直到达到目标路块数 N

一行配置即可生成不同地图：`config["map_config"]["type"] = "block_num"` 指定路块数，或 `"block_sequence"` 指定精确路块序列。

### 真实地图导入（Section 4.1）

MetaDrive 支持直接导入 Waymo（20,000+ 场景）和 Argoverse（~100 场景）的车道中心线数据，转换为 Frenet 坐标系，实现精确的真实场景重建（Figure 5/6）。

### 传感器与观测（Section 4.3）

- **低级观测**：240 维 2D 伪激光雷达点云（50m 探测范围，Gaussian 噪声）、目标车辆自身状态向量、导航信息（到目标的 checkpoint 序列）
- **高级观测**：BEV 语义地图、RGB 摄像头、深度摄像头
- **性能**：单智能体场景 300 FPS，40 智能体多智能体场景 60 FPS（2 CPU + 8 并行 rollout workers）

## 关键结果 / 数据

### RL Benchmark 四个任务（Section 5）

**评测指标**：Success Rate（到达终点）、Traffic Rule Violation Rate（出界率）、Crash Rate（碰撞率）

**任务 1：泛化到未见 PG 场景（Section 5.2 / Figure 8）**

训练 SAC 和 PPO，训练场景数 N 从 1 增至 1000：
- N=1 时训练-测试性能差距显著（过拟合）
- N 增大后测试 Success Rate 持续上升、违规率和碰撞率下降
- **结论：场景多样性是泛化的关键，MetaDrive 的程序化生成有效支持泛化研究**

**任务 2：泛化到未见真实场景（Section 5.3 / Figure 9）**

- 真实 Waymo 场景比 PG 场景更难（噪声大，无效场景多）
- PG 训练的 PPO 在真实场景上泛化差（sim-to-real gap 无法通过合成多样性消除）
- 混合训练（真实数据比例 0%→100%）：提高真实数据比例持续提升真实场景测试性能
- **结论：sim-to-real gap 需要真实数据，合成多样性无法替代**

**任务 3：安全探索（Section 5.4 / Table 3）**

| 方法 | Cumulative Reward | Cumulative Cost | Success Rate |
|---|---|---|---|
| SAC-RS（Reward Shaping） | 327.13 | 3.38 | **0.801** |
| PPO-RS | 197.27 | 3.33 | 0.207 |
| SAC-Lag（安全约束） | 324.23 | **1.90** | 0.714 |
| PPO-Lag | 269.51 | 1.82 | 0.477 |
| CPO | 194.06 | **1.71** | 0.210 |
| BC（离线） | 101.63 | 1.00 | 0.01 |
| CQL（离线） | 156.4 | 6.82 | 0.11 |

SAC-RS 成功率最高但安全违规较多；SAC-Lag 在成功率和安全性之间取得较好平衡。

**任务 4：多智能体 RL（Section 5.5 / Table 4）**

在 5 种 MARL 环境（Roundabout、Intersection、Tollgate、Bottleneck、Parking Lot）上对比 IPPO、MF-CCPPO、CoPO 等：
- **CoPO**（Coordinated Policy Optimization）在大多数环境上表现最优（Tollgate 79.66%，Roundabout 73.65%）
- 40 智能体并发运行 60 FPS，远超同类 MARL 驾驶模拟器

## 局限性

论文 Section 6 明确列出：

- **渲染保真度低**：为换取速度和可组合性，MetaDrive 放弃了 CARLA 级别的光照/天气/材质模拟；传感器仿真基于合成模型，真实摄像头数据保真度受限
- **行人和自行车缺失**：生成场景不包含非机动车交通参与者（仅有车辆和静态障碍物）
- **无系统化角落场景描述**：缺乏类似 OpenSCENARIO 的语言来描述精确的近事故场景，难以大规模生成安全关键 corner cases
- **单线程 Python GIL 限制**：与 IsaacGym 等 GPU-based 模拟器相比样本效率仍有差距

## 现状与影响

一句话定性：**MetaDrive 是"泛化性 RL + 自动驾驶"方向的标准基准平台——它定义了可组合合成场景 + 真实数据导入的双轨范式，直接演化为 ScenarioNet（2023）以支持更大规模的多数据集真实场景，至今仍是该方向引用最多的开源模拟器之一。**

- **ScenarioNet 的直接前身**：ScenarioNet（Li et al. 2023，NeurIPS）在 MetaDrive 的物理引擎和渲染后端基础上，引入统一场景描述格式（USD），将支持数据集从 Waymo/Argoverse 扩展到 nuScenes/nuPlan/L5，场景规模从数万扩展到百万级
- **MARL 研究基准**：CoPO（Peng et al. 2021）、SafeDriving 等后续工作直接在 MetaDrive 上提出并评测，MetaDrive 的多智能体环境已成为学界常见 baseline
- **安全 RL 基准**：MetaDrive 的 Safe Exploration 任务是少数同时支持连续动作空间 + 成本约束 + 场景多样化的安全 RL 评测环境
- **可组合性影响**：MetaDrive 提出的 Manager 抽象（Map/Traffic/Object/Agent 分离）被 ScenarioNet 继承，并在后续工作中被用来实现 AD stack（Openpilot）的闭环测试

## 和 wiki 内其他概念的关联

- [ScenarioNet](./scenarionet-2306.12241.md)：MetaDrive 的直接演化版本；ScenarioNet 保留 MetaDrive 的 Manager 架构和物理引擎，新增统一场景描述格式和五大真实数据集支持，将平台定位从"泛化 RL 基准"扩展到"多数据集训练基础设施 + AD stack 测试"
- [nuScenes](./nuscenes-1903.11027.md)：MetaDrive 通过 ScenarioNet 间接支持 nuScenes；MetaDrive 原版支持 Waymo 和 Argoverse
- [nuPlan](./nuplan-2106.11810.md)：MetaDrive 的闭环模拟思路与 nuPlan 的闭环规划评测互补；ScenarioNet 把两者打通
- [RLHF / PPO](../20-concepts/rlhf.md)：MetaDrive 实验中使用 PPO 和 SAC 作为基准算法，提供了 displacement reward + terminal reward 的 reward shaping 设计参考

## 值得看的部分 / 相关资料

- **Section 3（系统设计）**：Object/Policy/Manager 三层抽象的完整描述，理解 MetaDrive 可组合性的核心；Figure 2 展示了通过替换 Manager 产生不同场景的四种组合
- **Section 4.1（BIG 算法）**：Algorithm 1 伪码 + Figure 3（路块类型 + 不同路块数的生成地图）——理解 PG 地图生成机制的最快入口
- **Section 5.2–5.3（泛化实验）**：Figure 8/9 展示训练场景数量对泛化的影响，以及 sim-to-real gap 的量化证据——"合成多样性 ≠ 真实泛化"的清晰实证
- **Table 3（安全 RL）**：SAC-RS vs SAC-Lag vs CPO 的成功率/安全违规对比，是 constrained RL 基准结果的参考
- **Table 4（MARL）**：5 种多智能体环境上的 IPPO/MF-CCPPO/CoPO 对比
- 相关工作：
  - Li et al. 2023, *ScenarioNet*（arXiv:2306.12241，NeurIPS 2023）——MetaDrive 的多数据集扩展版，wiki 有对应文档
  - Peng et al. 2021, *Learning to Simulate Self-Driven Particles System with Coordinated Policy Optimization（CoPO）*（NeurIPS 2021）——MetaDrive 多智能体任务上的代表性算法
