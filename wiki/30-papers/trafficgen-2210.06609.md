# TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios（Feng et al., 2022）

一句话总结：TrafficGen 是一个数据驱动的交通场景生成模型，用 encoder-decoder + 自回归采样从 Waymo 真实驾驶日志中学习生成多样、真实的交通快照和长轨迹，可用于增强训练数据、修复片段轨迹和改善 RL 智能体安全性，ICRA 2023。

## 基本信息

- 论文：TrafficGen: Learning to Generate Diverse and Realistic Traffic Scenarios
- 作者：Lan Feng, Quanyi Li, Zhenghao Peng, Shuhan Tan, Bolei Zhou（Feng/Li/Peng 同等贡献）
- 机构：ETH Zurich（Feng）；The University of Edinburgh（Li）；UCLA（Peng, Zhou）；UT Austin（Tan）
- 发表：ICRA 2023（IEEE International Conference on Robotics and Automation）
- arXiv：2210.06609（首次提交 2022-10）
- 代码/主页：https://metadriverse.github.io/trafficgen

## 核心问题

自动驾驶系统需要在大量多样且真实的交通场景中评测 AI 安全性，但现有来源有两类缺陷：

1. **真实数据集中的轨迹是碎片化的**：Waymo Open Dataset 中仅约 30% 的轨迹超过 10 秒，只有 12% 覆盖完整场景，被其他车辆遮挡导致轨迹截断，难以支撑完整的闭环测试。
2. **程序化生成（PG）脱离真实分布**：CARLA 等模拟器用规则和启发式方法生成交通场景，缺乏真实世界复杂路口行为、密度分布和驾驶风格，存在明显 sim-to-real gap。

核心洞察：**已有真实驾驶数据集（如 WOMD）包含的道路结构和交通行为已经足够让模型学习真实交通分布；只需一个能从 HD map 条件生成新场景的生成模型，就能无限扩展真实数据覆盖而不依赖新的数据采集。**

> **WOMD（Waymo Open Motion Dataset）**：Waymo 发布的大规模运动预测数据集，包含约 10 万个 20 秒驾驶场景，每个场景包含 HD map 和所有参与者的完整状态序列（位置/速度/朝向/类型）。与 nuScenes/nuPlan 相比，WOMD 聚焦于交通流运动预测（而非感知或规划），是 TrafficGen/ScenarioNet 等工作的核心训练来源。

## 方法：TrafficGen 的两阶段生成

### 问题形式化

一个交通场景定义为 $\tau = (\mathbf{m}, \mathbf{s}_{1:T})$，其中 $\mathbf{m}$ 是 HD map，$\mathbf{s}_{1:T}$ 是所有 $N$ 辆车在 $T$ 个时间步的状态序列。给定一个已有场景 $\tau$（或空场景），TrafficGen 的目标是生成新场景 $\tau'$——具有相似的场景结构但不同的车辆分布和更长的时间步。

整体是一个 **encoder-decoder** 架构，分两阶段：（1）车辆放置（决定在哪里放多少辆车），（2）轨迹生成（决定每辆车怎么走）。

**架构伪代码（展示层次和维度）：**

```python
# 输入
map_vectors: [I, d_map]    # I 个地图向量，每个 d_map 维（位置+车道类型+交通灯+车辆状态）
                            # I 可达 1000，d_map ≈ 30

# ── 编码器：MCG（Multi-Context Gating） ──────────────────────────────
# 5 个堆叠的 MCG block，每个 block：
#   1. max-pool 所有向量 → context c: [d_hidden]
#   2. 每个向量和 c 做 gating：v_i' = MLP(concat(v_i, c))
for block in range(5):
    c = max_pool(map_vectors)              # [d_hidden]，全局摘要
    map_vectors = MLP(concat(map_vectors, c.expand(I)))  # [I, d_hidden]
# 输出：
local_features: [I, d_hidden]   # 每个地图区域的特征（融合了全局信息）
global_context: [d_hidden]      # 整个场景的全局 context（用于轨迹生成）

# ── 解码器第一阶段：车辆放置（自回归） ──────────────────────────────
placed_vehicles = []
for step in range(N_max_vehicles):
    # 1. 对每个地图区域打分，softmax 得到放车概率分布
    placement_logits = MLP_place(local_features)   # [I]
    region_probs = softmax(placement_logits)        # [I]

    # 2. 采样一个区域
    chosen_region = sample(region_probs)            # 标量，某个地图区域的索引

    # 3. 在该区域内，用 GMM 采样车辆属性（K=10 个高斯分量）
    pos    = sample(GMM_pos(local_features[chosen_region]))     # (x, y)
    heading = sample(GMM_heading(local_features[chosen_region])) # 角度
    speed  = sample(GMM_speed(local_features[chosen_region]))   # 标量
    size   = sample(GMM_size(local_features[chosen_region]))    # (l, w, h)

    placed_vehicles.append((chosen_region, pos, heading, speed, size))

    # 4. 把新车信息写回地图向量，重新编码（自回归：当前步影响下一步）
    map_vectors[chosen_region].update(vehicle_info)
    local_features, global_context = encoder(map_vectors)  # 重新编码

# ── 解码器第二阶段：轨迹生成（滚动解码） ─────────────────────────────
# 每辆车每步生成 K=10 条候选轨迹 + 概率
all_trajectories = []
current_states = placed_vehicles  # 初始状态

for t in range(0, T, l):   # 每 l 秒滚动一次（l=3s 最优）
    # 用全局 context 预测接下来 l 步的 K 条轨迹
    traj_candidates: [N, K, l, 4]  # N 辆车，K 条候选，l 步，每步(x,y,heading,speed)
    traj_probs:      [N, K]        # 每条候选的概率

    traj_candidates, traj_probs = MotionModel(global_context, current_states)

    # 采样每辆车的轨迹（选概率最高的或按概率采样）
    chosen_traj = sample(traj_candidates, traj_probs)  # [N, l, 4]

    # 执行前 l 步，更新状态，再次编码
    current_states = chosen_traj[:, -1, :]             # 取最后一帧作为新状态
    global_context = encoder(map_vectors, current_states).global_context
    all_trajectories.append(chosen_traj)

output_trajectories: [N, T, 4]   # 最终：N 辆车，T 步完整轨迹
```

层次小结：
- 编码器把 I 个地图向量（最多 1000 个）压缩成局部特征 + 全局 context
- 车辆放置是自回归的——每放一辆车就重新编码，N 辆车 = N 次前向传播
- 轨迹生成是滚动的——每 3 秒重新看一次当前状态，而不是一次性预测 20 秒

### 编码器：向量化地图与 Multi-Context Gating（MCG）

**向量化表示**：HD map 中每条车道被离散化为若干向量，每个向量代表地图上一个小区域（5m×5m 矩形），包含：
- 该区域的起点 $p_i^s$、终点 $p_i^e$（定义局部坐标系）
- 车道类型 $t_i$、交通灯状态 $u_i$
- 该区域内的车辆信息：是否有车 $m_i$、局部位置 $q_i^v$、朝向差 $h_i$、速度 $vel_i$、尺寸 $bbox_i$

完整向量表示 $v_i = (p_i^s, p_i^e, t_i, u_i) \oplus (m_i, q_i^v, h_i, vel_i, bbox_i)$（式 1）

**向量化表示 vs BEV/Occ**：

TrafficGen 的向量化地图表示和 BEV（Bird's Eye View）感知、Occ（Occupancy）地图在表达"空间信息"这件事上有根本区别：

| | 向量化（TrafficGen） | BEV 感知 | Occ 地图 |
|---|---|---|---|
| **表示形式** | 语义折线段（车道中心线等结构化元素） | 密集栅格特征图 | 每个体素是否被占据 |
| **输入来源** | HD map（预先制作的精确地图） | 实时传感器（摄像头/LiDAR） | 实时传感器 |
| **分辨率** | 不固定，沿道路稀疏采样（5m 间隔） | 固定栅格（如 200×200，0.5m/格） | 固定体素（如 200×200×16） |
| **语义信息** | 高（车道类型、交通灯、连通关系明确编码） | 中（通过特征学习隐式包含） | 低（主要是"有/无"占据） |
| **计算效率** | 高（只有 ~1000 个向量） | 低（密集特征图，计算量大） | 低（三维体素，更大） |
| **适用场景** | 已知地图区域的运动预测/场景生成 | 未知区域的实时感知 | 精细障碍物感知、驾驶空间自由度 |

简单说：**向量化表示利用了 HD map 的先验结构**，只对道路有意义的地方建模，非常紧凑；BEV/Occ 是"不假设任何结构"的密集表示，适合处理 HD map 之外的任意障碍物和动态信息。两者常配合使用：HD map 向量化提供静态结构（道路/车道），BEV/Occ 提供动态感知（实时障碍物/可行驶区域）。

**Multi-Context Gating（MCG）**：复杂地图可包含多达 1000 个向量，直接做 cross-attention 计算量太大。MCG 是 cross-attention 的近似——不是让每个元素关注所有其他元素，而是先将整个向量集合压缩成一个 context 向量 $c$（max-pool），再用这个 context 向量更新每个局部向量的表示。经过 5 个堆叠的 MCG block 后，输出融合了全局场景信息的局部特征 $\mathbf{v}' = \{v_i'\}$ 和全局 context $c'$。

> **为什么不直接用 Transformer 做 cross-attention？** cross-attention 复杂度是 $O(I^2)$，$I$ 可达 1000。MCG 把复杂度降到 $O(I)$——每个向量只和一个全局 context 交互，计算量与向量数量线性增长而非平方增长。这是工程性权衡：损失了一定的局部细节，换取可以处理任意大小地图的能力。

### 解码器第一阶段：车辆放置（Vehicle Placement）

给定融合后的特征 $\mathbf{v}'$，对每个地图区域 $j$ 生成一个权重 $w_j = \text{MLP}_{\text{place}}(v_j')$，然后按 $w_j$ 的归一化分布采样区域 $i$（式 2-3），在被选中的区域内再采样车辆属性：

- 位置：$K$ 个二元高斯分布的混合模型（GMM），采样 $q_i \sim \text{GMM}_{\text{pos},i}$（式 4-6）
- 朝向、速度、尺寸：同样建模为 $K=10$ 个高斯混合分布，分别为 $\text{GMM}_{\text{heading}}$、$\text{GMM}_{\text{speed}}$、$\text{GMM}_{\text{size}}$

**自回归采样**：每次采样一辆车 → 将新车信息编入局部向量 → 重新编码得到更新后的 $\mathbf{v}'$ → 采样下一辆车，直到车辆数量达到预设上限。Figure 5 的热力图直观展示了每步采样后空间概率分布的变化。

**训练**：使用 **random mask** 策略（受 BERT 启发）——随机遮住地图上部分车辆，让模型预测被遮住的区域是否有车（BCE 损失）和车辆属性（GMM 负对数似然），同时保留其他车辆的信息作为 context。这个设计使模型既能从空地图生成新场景，也能在已有部分车辆的场景上进行增强/修复。

### 解码器第二阶段：轨迹生成（Trajectory Generation）

将全局 context $c'$ 输入一个 motion forecasting 模型，在每个时间步为每辆车生成 $K=10$ 条候选轨迹及其概率（式 8）。

**关键设计——实时采样（real-time sampling）**：传统轨迹预测模型以过去轨迹作为输入（e.g., Multipath++），直接用于长轨迹生成时会因历史累计误差（distributional shift）失效。TrafficGen 用全局 context $c'$ 代替历史轨迹输入，并以 $l$ 步为间隔滚动解码：预测 $l$ 步 → 执行前 $l$ 步 → 用新状态更新 context → 再预测 $l$ 步（Figure 4）。

- 使用 $l=3s$ 的采样间隔时性能最佳（Table II 中 Mean ADE 1.54m，Mean FDE 4.59m，SCR 6.4%）
- 训练用 WOMD 前 9 秒轨迹（50,000 个场景的前 9 帧），MSE loss 对最接近 GT 的预测候选计算

> **ADE/FDE**：Average Displacement Error（平均位移误差，整条轨迹每步预测位置与 GT 的平均距离）和 Final Displacement Error（最终位移误差，轨迹终点预测与 GT 的距离）。是轨迹预测的标准评测指标，ADE 反映整体准确性，FDE 反映长期预测能力。
>
> **SCR（Scenario Collision Rate）**：场景碰撞率，场景中发生碰撞的车辆占比（IOU 超阈值视为碰撞）。越低越好，反映生成轨迹的交互合理性——如果车辆之间相互穿透说明生成的轨迹不符合物理约束。

## 关键结果 / 数据

### 车辆放置对比（Table I，MMD 指标）

用最大均值差异（MMD）衡量生成车辆属性分布与真实场景分布的距离（越小越好）：

| 方法 | Pos | Heading | Speed | Size |
|---|---|---|---|---|
| SceneGen with VectorRep | 0.1362 | 0.1307 | 0.1772 | 0.1190 |
| SceneGen | 0.1452 | 0.1387 | 0.1860 | 0.1286 |
| **TrafficGen** | **0.1192** | **0.1189** | **0.1602** | **0.0932** |

TrafficGen 在全部四个属性上均优于竞争方法 SceneGen，说明生成的车辆位置、朝向、速度、尺寸分布更接近真实数据。

> **MMD（最大均值差异，Maximum Mean Discrepancy）**：衡量两个分布之间距离的统计量。对两组样本 $p$（生成）和 $q$（真实），$\text{MMD}^2(p,q) = \mathbb{E}_{x,x' \sim p}[k(x,x')] + \mathbb{E}_{y,y' \sim q}[k(y,y')] - 2\mathbb{E}_{x \sim p, y \sim q}[k(x,y)]$，其中 $k$ 是高斯核函数。MMD=0 当且仅当两个分布完全相同。直觉：两组样本内部的相似度之和，减去两组之间的相似度——两组越像，第三项越大，MMD 越小。

### 轨迹生成性能（Table II）

| 采样间隔 | Mean ADE (m) | Mean FDE (m) | SCR (%) |
|---|---|---|---|
| 9s | 1.55 | 4.62 | 7.5 |
| **3s** | **1.54** | **4.59** | **6.4** |
| 1s | 1.56 | 4.64 | 4.9 |

采样间隔 3s 在 ADE/FDE 上最优；1s 的碰撞率最低（4.9%）但误差略高。论文推荐 3s 作为默认设置。

### 消融实验（Table III）

遮住车辆信息（A）和红绿灯信息（B）分别使 MMD 上升，说明两类 context 都对车辆放置有贡献。去掉向量化地图表示（C，改用图像输入）也使性能下降，验证了向量化 HD map 表示的重要性。

### RL 安全性提升（Table IV）

在 MetaDrive 模拟器中训练 PPO 智能体，比较不同训练数据集的效果：

| 训练集 | Success Rate ↑ | Safety Violation ↓ |
|---|---|---|
| Real Data（Waymo）| 0.60 ±0.11 | 3.65 ±0.71 |
| Heuristic Data（PG）| 0.31 ±0.04 | 2.82 ±0.71 |
| **Generated Data（TrafficGen）** | **0.62 ±0.06** | 3.01 ±0.26 |
| Augmented Data（Generated + Real）| 0.61 ±0.04 | **2.14 ±0.32** |

关键发现：
- PG（启发式）数据训练的智能体成功率最低（0.31），说明合成场景无法替代真实分布
- TrafficGen 生成数据的成功率略高于真实数据（0.62 vs 0.60），说明生成数据可以作为真实数据的有效替代
- **Real + Generated（增强）数据安全违规最少（2.14）**：真实场景中高密度交通让智能体学会了更保守的驾驶，而 TrafficGen 生成的场景密度更高（通过设置更大的车辆生成数 N），提供了更具挑战性的训练条件

## 三种应用场景（Figure 1）

1. **同一地图生成多样场景**：给定同一 HD map，每次用不同随机种子采样，生成完全不同的车辆布局和轨迹，大幅扩展测试用例数量
2. **增强已有场景（Augmentation）**：在已有场景基础上增加更多车辆，提高场景复杂度和交通密度，用于训练在密集交通中的安全驾驶能力
3. **修复/延长轨迹（Inpainting）**：对碎片化的真实轨迹（只有前几秒）进行延长，填补片段化数据集（Waymo 中 70% 的轨迹 <10 秒）的空缺

## 局限性

论文 Section V 隐含、附录可推导出：

- **生成轨迹的物理真实性受限**：SCR（碰撞率）最低 4.9%，说明仍有约 5% 的场景存在车辆碰撞，不适合直接作为 GT 轨迹使用，需要结合 IDM 等规则修正
- **仅支持 Waymo 格式**：训练数据仅来自 WOMD，生成的场景分布偏向 Waymo 覆盖的地理区域（美国城市）；不同城市/国家的驾驶行为差异未被覆盖
- **长时轨迹累计误差**：尽管实时采样降低了分布偏移，FDE 仍随时间增长，长时生成（>20s）的轨迹质量未经验证
- **车辆类型覆盖有限**：实验主要关注机动车，行人和自行车的生成能力未充分评测
- **计算成本**：自回归车辆放置需要逐车重新编码（N 辆车 = N 次前向传播），在大场景（N=50+）下生成速度较慢

## 现状与影响

一句话定性：**TrafficGen 是自动驾驶数据增强方向的代表性工作——它证明了"从真实驾驶数据中学习生成真实场景"可以显著优于规则生成，且生成数据可以直接改善 RL 智能体的安全性；ScenarioNet 直接使用 TrafficGen 的编码器提取场景 embedding 用于 domain gap 分析，是 MetaDrive 生态中场景生成的标准工具。**

- **ScenarioNet 的场景生成模块**：ScenarioNet（NeurIPS 2023）的 Section 4.2 t-SNE 分析中，用 TrafficGen 编码器将 Waymo/nuPlan/PG 三类场景编码为 embedding 向量，直观展示跨数据集的 domain gap——TrafficGen 的向量化场景表示成为 ScenarioNet 跨数据集分析的工具
- **MetaDrive 生态中的数据增强标准路线**：TrafficGen 与 MetaDrive（模拟器）和 ScenarioNet（数据管理）构成完整的数据飞轮：真实数据 → TrafficGen 生成更多场景 → MetaDrive 闭环训练 → 更好的 RL 智能体
- **后续工作的基础**：TrafficGen 的向量化 HD map 表示方案（基于 VectorNet 思路，用 MCG 代替全量 attention）被后续场景生成工作（如 UniSim、CTG++）借鉴；实时采样的轨迹生成框架也被多个运动预测工作采用
- **局限已被后续工作推进**：2023-2025 年出现了多个扩展版本，支持更多数据集、条件生成（安全关键场景）、语言条件控制等；基于扩散模型的场景生成（如 DiffScene）在多样性上有进一步提升

## 和 wiki 内其他概念的关联

- [ScenarioNet](./scenarionet-2306.12241.md)：TrafficGen 是 ScenarioNet 场景生成实验（Section 4.2）的核心工具。ScenarioNet 用 TrafficGen 编码器提取场景 embedding，做 t-SNE 可视化分析 Waymo/nuPlan/PG 的 domain gap
- [MetaDrive](./metadrive-2109.12674.md)：TrafficGen 的附录（Section VI）详细描述了如何将生成场景导入 MetaDrive 模拟器——把生成的车辆状态序列转换为 MetaDrive 的 IDM 控制背景车辆，构建可交互的 reactive 环境
- [t-SNE and Embedding Visualization](../20-concepts/tsne-dimensionality-reduction.md)：ScenarioNet 使用 TrafficGen 的编码器产出的 embedding 做 t-SNE 分析，TrafficGen 的向量化地图表示学到的 embedding 质量直接影响 domain gap 分析的可信度
- [RLHF / PPO](../20-concepts/rlhf.md)：Table IV 的 RL 实验使用 PPO 训练 MetaDrive 中的 ego 驾驶智能体，reward = displacement reward + speed reward + terminal reward（见附录式 10），TrafficGen 生成的高密度场景提供了更具挑战的训练环境

## 值得看的部分 / 相关资料

- **Section III（Method）全文**：3 页，是理解 TrafficGen 架构的完整入口。Figure 2 是架构图，清晰展示从向量化地图输入到 MCG 编码、Vehicle Placement 解码、Trajectory Generation 解码的完整流程；Figure 3 解释向量化表示的局部坐标系；Figure 4 展示实时滚动采样的直觉
- **Figure 5（自回归放置过程热力图）**：从空地图开始，每步采样后的空间概率分布变化——直观展示模型"学会了"交通密度分布的感知（交叉口/干线上概率高）
- **Table IV（RL 安全性对比）**：Heuristic/Real/Generated/Augmented 四类训练数据效果对比，是"生成数据 vs 真实数据"的直接实证
- **附录 Section VI（Building Reactive Traffic Scenario）**：TrafficGen 生成场景 → MetaDrive 导入的工程细节，包括 road network 导入、IDM actuated 背景车辆设置、reward 函数设计（式 10）
- **附录 Section VII（Procedural Generation Baseline）**：对比用的 PG 方案细节，解释为何 PG 数据导致 RL 智能体在真实场景上失败
- 相关工作：
  - Li et al. 2023, *ScenarioNet*（arXiv:2306.12241，NeurIPS 2023）——直接使用 TrafficGen 作为场景生成组件，wiki 有对应文档
  - Li et al. 2021, *MetaDrive*（arXiv:2109.12674，TPAMI 2022）——TrafficGen 生成场景的导入目标模拟器，wiki 有对应文档
  - Tan et al. 2021, *SceneGen*（CVPR 2021）——TrafficGen 的直接比较对象，车辆放置基线方法
  - Gao et al. 2020, *VectorNet*（CVPR 2020）——TrafficGen 向量化 HD map 表示的技术来源
