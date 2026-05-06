# ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling（Li et al., 2023）

一句话总结：ScenarioNet 定义统一场景描述格式，把 Waymo、nuScenes、nuPlan、Lyft L5、Argoverse 等异构驾驶数据集整合进同一平台，通过 MetaDrive 模拟器支持闭环 RL/IL 训练、多智能体学习和 AD stack 测试，NeurIPS 2023 Datasets and Benchmarks。

## 基本信息

- 论文：ScenarioNet: Open-Source Platform for Large-Scale Traffic Scenario Simulation and Modeling
- 作者：Quanyi Li, Zhenghao Peng, Lan Feng, Zhizheng Liu, Chenda Duan, Wenjie Mo, Bolei Zhou
- 机构：University of Edinburgh（Li）；ETH Zurich（Peng, Feng, Mo）；UCLA（Liu, Duan, Zhou）
- 发表：NeurIPS 2023 Datasets and Benchmarks Track
- arXiv：2306.12241
- 代码/主页：https://metadriverse.github.io/scenarionet

## 核心问题

自动驾驶研究中的两个关键瓶颈：

1. **数据格式异构**：Waymo、nuScenes、nuPlan、Lyft L5、Argoverse 等大型驾驶数据集格式各异，由不同机构发布，难以跨数据集聚合训练。研究者想同时用 nuPlan 和 Waymo 数据训练需要从头构建桥接层。
2. **模拟器与数据集耦合**：现有模拟器往往与特定数据集绑定（nuPlan-devkit 绑 nuPlan，DriverGym 绑 L5，Nocturne 绑 Waymo），缺乏 3D rendering 和 AD stack 接入能力；而具备 3D 渲染的模拟器（如 CARLA）又基于合成场景，无法直接导入真实世界驾驶数据。

核心洞察：**驾驶数据集中的 HD map + 对象轨迹已经完整描述了一个交通场景的"数字孪生"——只需统一格式，即可在同一模拟器中重放并用于训练和评测**，无需依赖原始传感器数据。

## 方法：系统设计

### 三层架构（Figure 3）

```
III. Application Layer   单智能体学习 | 多智能体学习 | 场景生成 | AD Stack 测试
         ↑ Simulation ↓
II. System Layer         过滤 / 合并 / 分割 / 采样 / 健全性检查
         ↑ Convert ↓
I. Data Layer            MetaDrive 合成场景 | 安全关键场景 | Waymo | nuPlan/nuScenes | L5 | 更多数据集
```

数据层通过 Conversion 将各来源格式统一为内部场景描述（`scenario_id.pkl`），系统层提供数据库操作工具，应用层接入 MetaDrive 模拟器执行各类 ML 任务。

### 统一场景描述（Unified Scenario Description）

每个场景文件是一个嵌套字典（4 个顶级 key）：

| Key | 内容 |
|---|---|
| `map_features` | 车道 / 车道线的多段线、类型、连通性 |
| `objects` | 每个 object 的 position / heading / velocity / size / type，以 object-centric 方式存储完整时间序列 |
| `traffic_light` | 红绿灯状态序列（Red/Yellow/Green/Unknown） |
| `metadata` | 来源数据集、时间间隔、坐标系、统计信息（对象数、移动距离等） |

关键设计：`[object_id]["valid"]` 字段标记对象在每一帧是否出现，模拟器据此决定创建或销毁对应实体，实现精确的数字孪生重放。

### 数据库操作

- **Conversion**：调用各数据集官方 API 填充统一格式，产出 `dataset_summary.pkl` + `dataset_mapping.pkl` + 若干 `scenario_id.pkl`
- **Filtering**：按条件筛选（如 ego 车移动距离 > 10m、交通参与者 > 200）
- **Merging**：合并多个数据库形成更大数据集
- **Splitting / Sampling**：copy-free 设计，只复制 summary 文件不复制原始数据
- **New Dataset Support**：只需实现一个 `convertor_function`（填充统一格式各字段），调用 `write_to_directory` 即可完成并行转换

### MetaDrive 模拟器集成（Section 3.3）

- **物理引擎**：Bullet 物理引擎，支持 500 FPS（含 100+ 交互对象）
- **渲染**：2D Pygame（BEV）+ 3D OpenGL（Panda3D），摄像机优化：CUDA/OpenGL 桥接使 1920×1080 图像从 11 FPS 提升到 300 FPS
- **传感器**：伪激光雷达（2D）、BEV 图像、RGB 摄像头、深度摄像头、语义摄像头
- **控制策略接口**：`ReplayPolicy`（回放轨迹）、`IDMPolicy`（闭环响应，自动避让）、`EnvInputPolicy`（接受外部 RL 动作）——三者可混合，支持单智能体 / 多智能体 / AD stack 测试
- **ROS bridge**：接入 Autoware 等开源 AD stack，实现真实驾驶场景中的端到端测试

## 关键结果 / 数据

### 数据库规模（Table 2）

| 数据集 | 场景数 | 平均轨迹长度 | 平均车辆数 | 交叉口比例 |
|---|---|---|---|---|
| Waymo | 70,000+ | 136.55m | 89.93 辆 | 71% |
| nuPlan（Boston 子集） | 50,261 | 95.48m | 53.96 辆 | 57% |
| PG（程序生成） | 50,000 | 226.07m | 9.81 辆 | 36% |

每个场景时长 20 秒；Waymo 场景交通最密集，PG 场景轨迹最长但车辆最少。

### 模拟器能力对比（Table 1）

ScenarioNet 是唯一同时支持以下全部特性的平台：Waymo/L5/nuPlan/nuScenes/Argoverse 多数据集、RL、IL、多智能体 RL & IL、场景生成、AD Stack 测试、3D 渲染。

### 跨数据集 t-SNE 分析（Section 4.2 / Figure 4）

用 TrafficGen 编码器提取嵌入，t-SNE 可视化 3000 个场景：
- PG（合成）与真实世界场景（Waymo/nuPlan）存在明显 **domain gap**
- Waymo 和 nuPlan 之间也有差异：Waymo 场景更多复杂交叉口，nuPlan 场景地图结构更简单
- 结论：**单一数据来源不足以覆盖所有交通情况，需要聚合多来源数据**

### 跨数据集 RL 泛化（Section 4.3 / Figure 5）

在 nuPlan-test 上评测不同训练集训练的 PPO 智能体：

- `PG-train` 训练的智能体 Success Rate 低，Timeout Rate 高——合成场景无法学习高速驾驶技能
- `nuPlan-train` + curriculum 训练的智能体 Success Rate 最高（~0.75）
- `Combine（PG+nuPlan）+ curriculum`：Success Rate 最高（~0.80），因 PG 的弯道轨迹提升了曲率跟踪能力
- **结论：真实数据对闭合 sim-to-real gap 是必要的；合成数据可补充多样性但不能替代真实数据**

### 多智能体学习（Section 4.4 / Table 3）

在 Waymo 数据集上比较 TD3、PPO、CoPO（RL）和 GAIL、AIRL（IL）：
- PPO（Route Completion 0.740，Success Rate 0.570）和 CoPO 表现优于 GAIL/AIRL
- displacement reward（GT 轨迹投影）是强监督信号，RL 方法可利用环境交互优化，GAIL/AIRL 需学习判别器故表现弱

### AD Stack 测试（Section 4.5）

通过 ROS bridge 接入 Openpilot（开源端到端 AD 系统）：
- Openpilot 在重建的真实场景中成功实现车道保持和在交叉口前停车
- 验证了 ScenarioNet 可作为开源 AD stack 的闭环测试平台

## 局限性

论文 Section 5 明确列出：

- **3D 资产匮乏**：车辆、行人等 3D 模型从网络收集，可能与真实传感器数据外观不一致；未来计划用神经渲染从驾驶日志重建真实纹理
- **单线程模拟**：受 Python GIL 限制，模拟是单线程的；IsaacGym 等 GPU-based 模拟器样本效率更高，需结合 Ray/Rllib 并行化
- **不支持可微分模拟**：无法对奖励函数直接求导（如 PODS 方法所需），未来计划引入自行车动力学模型
- **Domain gap 未消除**：通过 ScenarioNet 所有测试场景不代表可安全部署到真实世界；训练模型在真实部署中仍有 domain gap

## 现状与影响

一句话定性：**ScenarioNet 是自动驾驶 ML 训练基础设施层面的重要开放平台——它填补了"异构驾驶数据集统一可用于闭环 RL/IL 训练"的空白，与 nuScenes / nuPlan 等数据集的关系是"工具链"而非竞争，至今仍是 MetaDrive 生态的核心组件。**

- **MetaDrive 生态延伸**：ScenarioNet 是 MetaDrive（Li et al. 2021）的直接升级——MetaDrive 专注合成场景，ScenarioNet 引入真实世界数据支持，两者共用同一物理引擎和渲染后端
- **跨数据集训练的基础设施**：在 ScenarioNet 之前，研究者在 nuPlan 和 Waymo 上同时训练需要大量工程桥接；ScenarioNet 把这个过程标准化，使多来源混合训练成为"开箱即用"的功能
- **AD stack 测试的新范式**：通过 ROS bridge + 真实场景数字孪生，ScenarioNet 提供了一条"把真实驾驶日志变成 AD 系统测试用例"的路径，与 nuPlan 的闭环评测思路互补（nuPlan 侧重 planner API 评测，ScenarioNet 侧重策略学习和端到端 AD 测试）
- **WOMD 场景生成基础**：论文的场景生成实验基于 TrafficGen（Feng et al. 2022），ScenarioNet 提供的统一数据格式使跨数据集场景生成训练成为可能

## 和 wiki 内其他概念的关联

- [nuScenes](./nuscenes-1903.11027.md)：ScenarioNet 支持的五大数据集之一；nuScenes 的 HD map 和对象轨迹在 ScenarioNet 中被转换为统一格式，用于闭环训练和多视角渲染
- [nuPlan](./nuplan-2106.11810.md)：ScenarioNet 支持的另一核心数据集；实验中 nuPlan Boston 子集用于 RL 泛化对比，证明真实数据对闭合 sim-to-real gap 的必要性
- [t-SNE and Embedding Visualization](../20-concepts/tsne-dimensionality-reduction.md)：ScenarioNet 用 t-SNE 可视化 TrafficGen 编码的场景嵌入，揭示合成场景与真实场景、不同真实数据集之间的 domain gap（Figure 4）
- [RLHF / PPO](../20-concepts/rlhf.md)：实验中使用 PPO 训练单智能体和多智能体驾驶策略，ScenarioNet 提供闭环训练环境，GT 轨迹位移投影作为 dense reward

## 值得看的部分 / 相关资料

- **Section 3.1（统一场景描述格式）**：Figure 2 给出完整的 JSON 字典结构示例，是理解 ScenarioNet 数据模型的最快入口；object-centric vs frame-centric 的设计取舍在此有清晰说明
- **Figure 3（系统架构图）**：三层架构（Data → System → Application）一览，data conversion 和 simulation 两条数据流的关系一目了然
- **Table 1（模拟器能力对比）**：7 个现有模拟器与 ScenarioNet 的功能矩阵，快速理解 ScenarioNet 的定位
- **Section 4.3（跨数据集 RL 泛化）+ Figure 5**：真实 vs 合成训练数据的泛化差距实验，结论"合成数据无法替代真实数据用于高速驾驶技能学习"是 AV 数据工程的重要实证
- **Figure 4（t-SNE 场景嵌入）**：直观展示三类数据库（Waymo/nuPlan/PG）的分布差异，PG 与真实场景的 domain gap 清晰可见
- 相关工作：
  - Li et al. 2021, *MetaDrive: Composing Diverse Driving Scenarios for Generalizable Reinforcement Learning*（arXiv:2109.12674）——ScenarioNet 的前身，合成场景版本
  - Feng et al. 2022, *TrafficGen*（arXiv:2210.06609）——ScenarioNet 场景生成实验使用的生成模型
  - Caesar et al. 2021, *nuPlan*（arXiv:2106.11810）——ScenarioNet 最重要的真实数据来源之一，wiki 有对应文档
