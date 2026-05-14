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

1. **数据格式异构**：Waymo、nuScenes、nuPlan、Lyft L5、Argoverse 等大型驾驶数据集格式各异，由不同机构发布，难以跨数据集聚合训练。这里的"格式异构"不仅是文件结构不同，还包括：各数据集对 HD map 的表达方式不同（有的存车道中心线，有的存多边形），对象 ID 的命名规则不同，时间轴的组织方式（frame-centric vs object-centric）不同（frame-centric：每一帧记录"当前时刻所有对象的状态"，查某对象的完整轨迹要跨帧拼接；object-centric：每个对象独立记录自己所有时刻的状态序列，直接取一个 object 即得完整轨迹——ScenarioNet 统一采用 object-centric，和 `objects[id]["valid"]` 的设计一致），以及传感器采样频率、坐标系基准也不同。ScenarioNet 的解法是**只提取"中级表示"**（HD map 几何 + 对象轨迹），丢弃原始点云/图像，换取跨数据集统一。研究者想同时用 nuPlan 和 Waymo 数据训练需要从头构建桥接层。
2. **模拟器与数据集耦合**：现有模拟器往往与特定数据集绑定（nuPlan-devkit 绑 nuPlan，DriverGym 绑 L5，Nocturne 绑 Waymo），缺乏 3D rendering 和 AD stack 接入能力；而具备 3D 渲染的模拟器（如 CARLA）又基于合成场景，无法直接导入真实世界驾驶数据。

核心洞察：**驾驶数据集中的 HD map + 对象轨迹已经完整描述了一个交通场景的"数字孪生"——只需统一格式，即可在同一模拟器中重放并用于训练和评测**，无需依赖原始传感器数据。

> **ScenarioNet 的核心是数据管理和模拟平台，不是生成模型。** 它解决的问题是"把各来源数据统一成可以跑 RL/IL 训练的环境"，而不是"如何合成新的驾驶场景"。论文中也包含"场景生成"应用实验（Application Layer 的一个方向），但那里用的生成模型是 **TrafficGen**（Feng et al. 2022，独立论文），ScenarioNet 只是提供了统一数据格式让 TrafficGen 可以跨数据集训练和应用。如果想了解驾驶场景生成方法的细节（如何放置车辆、如何生成轨迹、生成数据质量评测），请看 [TrafficGen wiki 文档](./trafficgen-2210.06609.md)。

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

- **物理引擎**：Bullet 物理引擎（开源 C++ 物理库，被 Blender/PyBullet 广泛使用），负责车辆动力学模拟（加速/转向/摩擦）和碰撞检测，支持 500 FPS（含 100+ 交互对象）
- **渲染**：
  - 2D Pygame（BEV 俯视图，轻量调试用）
  - 3D Panda3D（开源 Python 3D 游戏引擎，提供 OpenGL 渲染后端）——不追求光线追踪级别的真实感，而是用"够用的 3D"换取高速运行
  - 摄像机优化：通常 CPU/GPU 之间传图像需要经过系统内存，很慢（11 FPS）。ScenarioNet 通过 CUDA/OpenGL 互操作（interop），让 OpenGL 渲染结果直接留在 GPU 显存（VRAM）里，用 CUDA 读取后转成 PyTorch tensor，全程不经过 CPU——这是从 11 FPS 跳到 300 FPS 的核心原因
- **传感器**：伪激光雷达（2D 点云，240 射线）、BEV 图像、RGB 摄像头、深度摄像头、语义摄像头
- **控制策略接口**：
  - `ReplayPolicy`：严格回放数据集中记录的轨迹（用于重现真实事故或验证重建精度）
  - `IDMPolicy`（Intelligent Driver Model）：基于前车距离/速度自动控制，产生响应式交通流，避免背景车辆"穿模"
  - `EnvInputPolicy`：接受外部 RL 动作（ego 车辆默认），三者可混合使用
- **ROS bridge**：ROS（Robot Operating System）是工业界 AD 栈的通信标准。ScenarioNet 提供 ROS 接口，让 Autoware 等开源 AD stack 以为自己在真实车上运行，实际上收到的传感器数据来自 MetaDrive 模拟的真实场景

## 关键结果 / 数据

### 数据库规模（Table 2）

以下指标均来自论文原文 Table 2，是了解一个驾驶数据集时的典型观测维度：

| 数据集 | 场景数 | 平均轨迹长度 | 平均车辆数 | 交叉口比例 | Construction Ratio |
|---|---|---|---|---|---|
| Waymo | 70,000+ | 136.55m | 89.93 辆 | 71% | 0.0 |
| nuPlan（Boston 子集） | 50,261 | 95.48m | 53.96 辆 | 57% | 1.0 |
| PG（程序生成） | 50,000 | 226.07m | 9.81 辆 | 36% | 0.39 |

> **PG（Procedural Generation，程序化生成）**：MetaDrive 的默认合成场景来源——用算法按规则随机拼接路块（直道/弯道/交叉口等）生成无限多样的地图，同时用 IDM 模型生成背景交通流。PG 场景完全合成，没有真实世界的噪声和稀有事件，但可以任意生成、无版权限制。与之对比，Waymo/nuPlan 是从真实驾驶日志中提取的场景。
>
> **各指标的含义**：
> - **平均轨迹长度**：ego 车辆在场景中平均移动的距离，间接反映场景复杂度和车速——PG 场景路程最长，因为合成场景车速高、无拥堵
> - **平均车辆数**：场景中同时存在的交通参与者数量，反映交通密度——Waymo 场景最密集（城市路口），PG 场景稀疏
> - **交叉口比例**：包含交叉路口的场景比例，反映复杂交通决策场景的覆盖率
> - **Construction Ratio**：含施工锥/路障的场景比例，反映安全关键场景的覆盖
>
> 其他有价值但本论文未列出的场景观测指标（供参考）：平均场景时长、行人/自行车密度、天气/光照分布、事故/近事故率、速度分布（高速/低速路段比例）、夜间场景比例。

每个场景时长 20 秒；Waymo 场景交通最密集，PG 场景轨迹最长但车辆最少。

### 模拟器能力对比（Table 1）

ScenarioNet 是唯一同时支持以下全部特性的平台：Waymo/L5/nuPlan/nuScenes/Argoverse 多数据集、RL、IL、多智能体 RL & IL、场景生成、AD Stack 测试、3D 渲染。

### 跨数据集 t-SNE 分析（Section 4.2 / Figure 4）

用 TrafficGen 编码器提取场景嵌入（将场景的地图结构和交通流编码为一个向量），t-SNE 可视化 3000 个场景：

- **PG（合成）与真实世界场景（Waymo/nuPlan）存在明显 domain gap**：在 t-SNE 图中，PG 场景聚集在左侧，Waymo/nuPlan 场景散布在右侧，说明合成场景的地图结构和交通行为与真实世界系统性不同。具体来看，合成场景的路网拓扑简单规整（矩形路块拼接），而真实场景有不规则的城市路网、复杂交叉口几何、多种车道变换；合成交通流（IDM）按固定规则行驶，真实交通有加塞、跟随错误等人类行为
- Waymo 和 nuPlan 之间也有差异：Waymo 场景更多复杂交叉口，nuPlan 场景地图结构更简单
- 结论：**单一数据来源不足以覆盖所有交通情况，需要聚合多来源数据**

### 跨数据集 RL 泛化（Section 4.3 / Figure 5）

在 nuPlan-test 上评测不同训练集训练的 PPO 智能体：

> **评测指标说明**：
> - **Success Rate**：智能体成功到达目的地的场景比例（episode 内到达终点且未超时/出界）
> - **Timeout Rate**：智能体在规定步数内未到达目的地的比例——反映策略是否学会"快速前进"，在高速场景中合成数据训练的智能体 Timeout Rate 高，因为它不会驾驶快车
> - **Out of Road Rate**：智能体驶出可行驶区域的比例
>
> **curriculum 训练（课程学习）**：先从简单场景开始训练，智能体达到一定成功率后才切换到更难的场景。ScenarioNet 中按 `track_length × cumulative_curvature` 计算难度分，分 100 个等级，每级 400 个场景，成功率到 75% 才升级。与随机采样训练相比，课程学习显著降低内存占用（每个 worker 只加载当前难度子集）并加速收敛。

- `PG-train` 训练的智能体 Success Rate 低，Timeout Rate 高——合成场景车速低、无真实拥堵，智能体未学会高速行驶技能
- `nuPlan-train` + curriculum 训练的智能体 Success Rate 最高（~0.75）
- `Combine（PG+nuPlan）+ curriculum`：Success Rate 最高（~0.80）——PG 场景含大量弯道，模型学会了更好的曲率跟踪（沿弯道行驶而不走直线捷径出界），在真实场景中也有帮助
- **结论：真实数据对闭合 sim-to-real gap 是必要的；合成数据可补充多样性但不能替代真实数据**

> **sim-to-real gap（仿真到真实的差距）**：在模拟器中训练的策略，部署到真实世界（或真实数据重建的场景）时性能大幅下降的现象。原因是模拟器对物理/交通行为的简化（传感器噪声、驾驶风格、不规则路网等）与真实世界不同，模型过拟合到模拟器的"假设"。

### 多智能体学习（Section 4.4 / Table 3）

在 Waymo 数据集上比较多种算法：

> **算法简介**：
> - **TD3 / PPO / CoPO**：强化学习（RL）方法——智能体通过与模拟环境交互、根据奖励函数更新策略。TD3（Twin Delayed DDPG）是 off-policy 连续动作 RL；PPO 是 on-policy 策略梯度；CoPO（Coordinated Policy Optimization）是专为多智能体驾驶设计的方法，兼顾个体奖励和群体协调
> - **GAIL / AIRL**：模仿学习（IL, Imitation Learning）方法——不给奖励函数，直接从专家轨迹（这里是 GT 驾驶轨迹）学习行为。GAIL/AIRL 需要训练一个判别器来区分"专家行为"和"智能体行为"，判别器本身需要额外训练，且无法直接利用环境交互信息

- PPO（Route Completion 0.740，Success Rate 0.570）和 CoPO 表现优于 GAIL/AIRL

> **displacement reward（位移奖励）**：以智能体在 GT 轨迹上的投影位移为奖励——即"沿着 GT 轨迹前进了多远"。这是一种 dense reward（每步都有奖励），比 sparse reward（只在到达终点时才有奖励）更易学习。RL 方法可直接优化这个奖励并与环境交互，而 GAIL/AIRL 的奖励来自判别器，信号质量不如直接位移投影稳定，所以 RL 在此任务上表现更好。

### AD Stack 测试（Section 4.5）

通过 ROS bridge 接入 Openpilot：

> **Openpilot**：Comma.ai 开发的开源端到端自动驾驶系统，支持多款量产车辆。它直接从摄像头图像输出方向盘/油门/刹车控制指令（end-to-end），不依赖精确 HD map。ScenarioNet 用它作为"开源 AD stack"的代表，在真实场景重建的模拟环境中测试其能力。

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
- **WOMD（Waymo Open Motion Dataset）场景生成基础**：论文的场景生成实验基于 TrafficGen（Feng et al. 2022），ScenarioNet 提供的统一数据格式使跨数据集场景生成训练成为可能

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

---

## 附录：统一场景描述数据结构详解

每个场景文件是一个 `.pkl`（Python pickle）序列化的嵌套字典。以下是完整字段说明和典型取值。

---

### 顶层结构

```python
scenario = {
    "map_features":    {...},   # 静态地图元素
    "objects":         {...},   # 动态交通参与者
    "traffic_light":   {...},   # 信号灯状态
    "metadata":        {...},   # 场景元信息
}
```

---

### map_features

地图元素以 ID 为键的字典，每个元素是一条折线（polyline）：

```python
scenario["map_features"] = {
    "lane_0": {
        "type": "LANE_SURFACE_STREET",   # 车道类型（枚举）
        # 可能的值：
        #   LANE_SURFACE_STREET   普通道路车道
        #   LANE_BIKE_LANE        自行车道
        #   LANE_SURFACE_UNSTRUCTURED  非结构化道路
        #   ROAD_EDGE_BOUNDARY    道路边界
        #   ROAD_LINE_SOLID_SINGLE_WHITE  实线车道线
        #   ROAD_LINE_BROKEN_SINGLE_WHITE 虚线车道线
        #   CROSSWALK             人行横道
        #   SPEED_BUMP            减速带

        "polyline": np.array([          # shape: [N, 2]，N 个折线顶点的 (x, y) 坐标
            [12.3, 45.6],               # 世界坐标系，单位：米
            [13.1, 45.8],
            [13.9, 46.0],
            ...
        ]),

        "speed_limit_mph": 30.0,        # 限速（英里/小时），部分数据集有，无则为 None

        # 仅 LANE 类型有以下连通性字段：
        "entry_lanes": ["lane_3", "lane_5"],   # 进入此车道的上游车道 ID 列表
        "exit_lanes":  ["lane_1"],             # 离开此车道的下游车道 ID 列表
        "left_neighbors":  ["lane_7"],         # 左侧相邻车道 ID
        "right_neighbors": ["lane_8"],         # 右侧相邻车道 ID
    },
    "lane_1": {...},
    "road_edge_0": {...},
    ...
}
```

**典型规模**（一个 Waymo 场景）：约 50-200 个地图元素，每条折线 5-50 个顶点。

---

### objects

所有动态交通参与者，以 object ID 为键，每个对象存储完整时间序列（object-centric）：

```python
scenario["objects"] = {
    "ego":  {                               # "ego" 是自车的固定 ID
        "type": "VEHICLE",                  # 对象类型：VEHICLE / PEDESTRIAN / CYCLIST / OTHERS
        "state": {
            "position": np.array([          # shape: [T, 3]，T 个时间步，每步 (x, y, z)
                [100.2, 200.5, 0.0],        # z 通常为 0（2D 场景），单位：米
                [100.8, 200.9, 0.0],
                ...
            ]),
            "heading": np.array([           # shape: [T]，单位：弧度
                1.57,                       # 朝向角，0 = 东，π/2 = 北（右手坐标系）
                1.58,
                ...
            ]),
            "velocity": np.array([          # shape: [T, 2]，(vx, vy)，单位：米/秒
                [2.1, 0.3],
                [2.2, 0.3],
                ...
            ]),
            "length": np.array([...]),      # shape: [T]，车辆长度，单位：米（通常不变）
            "width":  np.array([...]),      # shape: [T]，车辆宽度，单位：米
            "height": np.array([...]),      # shape: [T]，车辆高度，单位：米
            "valid":  np.array([            # shape: [T]，bool，该帧对象是否存在
                True, True, True,           # 遮挡/出界时为 False，模拟器据此销毁实体
                ...
            ]),
        }
    },
    "obj_0": {
        "type": "VEHICLE",
        "state": { ... }                    # 结构同 ego，但 ID 不固定
    },
    "obj_1": {
        "type": "PEDESTRIAN",
        "state": { ... }
    },
    ...
}
```

**时间步 T 的说明**：
- Waymo/WOMD：T=91（9 秒 × 10Hz，前 10 帧为历史，后 80 帧为预测窗口）
- nuPlan：T=20（20 秒 × 1Hz，或按配置）
- MetaDrive PG：T 由场景时长决定

**典型规模**（一个 Waymo 场景）：约 30-150 个 object，每个对象 91 帧，"ego" 始终存在。

---

### traffic_light

以信号灯 ID 为键，存储每个时间步的状态：

```python
scenario["traffic_light"] = {
    "signal_0": {
        "state": {
            "object_state": np.array([      # shape: [T]，每帧的信号灯状态（整数枚举）
                0,   # LANE_STATE_UNKNOWN
                1,   # LANE_STATE_ARROW_STOP
                2,   # LANE_STATE_ARROW_CAUTION
                3,   # LANE_STATE_ARROW_GO
                4,   # LANE_STATE_STOP
                5,   # LANE_STATE_CAUTION
                6,   # LANE_STATE_GO
                7,   # LANE_STATE_FLASHING_STOP
                8,   # LANE_STATE_FLASHING_CAUTION
            ]),
        },
        "lane": "lane_5",                   # 受此信号灯控制的车道 ID
    },
    "signal_1": {...},
    ...
}
```

---

### metadata

场景级别的统计信息和来源信息：

```python
scenario["metadata"] = {
    "id":          "scenario_0001abc",      # 场景唯一 ID（字符串）
    "dataset":     "waymo",                 # 来源数据集名称
                                            # 可能值：waymo / nuscenes / nuplan / lyft / argoverse / pg
    "coordinate":  "standard",             # 坐标系（standard = 右手系，x 东 y 北）
    "timestep":    0.1,                    # 时间步长（秒），Waymo=0.1s，nuPlan=1.0s
    "sdc_id":      "ego",                  # 自车（self-driving car）的 object ID
    "object_summary": {                    # 各 object 的统计摘要
        "ego":   {
            "type":               "VEHICLE",
            "track_length":       91,           # 该 object 有效帧数
            "moving_distance":    45.3,          # 总移动距离（米）
            "valid_length":       89,            # valid=True 的帧数（<track_length 代表有遮挡）
        },
        "obj_0": {...},
        ...
    },
    "current_time_index": 10,              # 当前时刻在时间序列中的索引（用于区分历史/未来）
    "number_summary": {                    # 对象数量统计
        "num_objects":       45,
        "num_vehicles":      38,
        "num_pedestrians":   5,
        "num_cyclists":      2,
        "num_others":        0,
        "num_traffic_lights": 4,
        "num_map_features":  120,
    },
}
```

---

### 读取示例

```python
import pickle

# 加载场景
with open("scenario_0001abc.pkl", "rb") as f:
    scenario = pickle.load(f)

# 获取自车的位置序列
ego_positions = scenario["objects"]["ego"]["state"]["position"]  # [T, 3]

# 获取第 10 帧（当前帧）所有有效对象的位置
t = scenario["metadata"]["current_time_index"]   # = 10
for obj_id, obj in scenario["objects"].items():
    if obj["state"]["valid"][t]:
        pos = obj["state"]["position"][t]         # (x, y, z)
        print(f"{obj_id}: {pos}")

# 获取所有车道中心线
for feat_id, feat in scenario["map_features"].items():
    if feat["type"].startswith("LANE"):
        polyline = feat["polyline"]   # [N, 2]
```
