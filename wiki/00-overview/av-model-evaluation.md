# 自动驾驶模型评测：预测与规划为核心

自动驾驶的评测体系很大，但对于以 Wayformer 风格为代表的**向量化场景编码 + Transformer Decoder 轨迹生成**这类模型，预测和规划层的评测是核心。本文从这个视角出发，重点介绍这类模型用什么指标评测、为什么这样设计，感知层从略。

---

## 坐标系视角：Ego-centric vs Scene-centric vs Agent-centric

运动预测模型需要决定"从谁的视角来描述场景"，这个选择影响架构设计、计算效率和多 agent 建模能力。

**Ego-centric（自车中心）**：所有输入都转换到**自车（autonomous vehicle）的坐标系**下描述。"前方 10 米"指的是相对自车前方 10 米。

- 典型用途：规划模块（自车视角天然合理）
- 优点：坐标系语义清晰，感知数据天然是自车坐标系输出
- 缺点：自车移动时整个场景坐标都要跟着变换，多帧之间不一致

**Agent-centric（预测目标中心）**：以**被预测的 agent** 为坐标原点描述场景。Wayformer 用的就是这种——预测 agent-1 时，整个场景转换到 agent-1 的坐标系；预测 agent-2 时，再转换到 agent-2 的坐标系。

- 典型用途：运动预测模型（Wayformer、MTR）
- 优点：对每个 agent 的预测任务是对称的，模型不需要知道"自车在哪"
- 缺点：**N 个 agent = N 次前向传播**，场景里的公共信息（道路、交通灯）被重复编码 N 次。agent-1 和 agent-2 互相是邻居时，interaction 数据有冗余

**Scene-centric（场景中心）**：用**统一的世界坐标系**描述整个场景，所有 agent 的轨迹都在同一坐标系下。

- 典型用途：UniAD、MotionDiffuser 等联合预测模型
- 优点：**一次前向传播预测所有 agent**，公共特征（道路、场景结构）只编码一次，可以建模多 agent 间的联合分布（A 刹车导致 B 也刹车）
- 缺点：模型需要处理变化的 agent 数量和位置，不能简单用固定形状的 tensor；世界坐标系的旋转不变性需要额外处理

**Map-centric（地图中心）**：以地图元素（车道段、路口）为中心组织特征，适合重地图推理的模型。不常见，通常作为 scene-centric 的一个子类。

| | Ego-centric | Agent-centric | Scene-centric |
|---|---|---|---|
| 坐标原点 | 自车 | 被预测的 agent | 世界坐标系 |
| 典型模型 | PNC 规划 Decoder | Wayformer、MTR | UniAD、MotionDiffuser |
| 前向传播次数 | 1 次 | N 次（N=agent 数） | 1 次 |
| 多 agent 交互 | 不适用（单 ego） | 弱（独立预测） | 强（联合建模） |
| 计算效率 | 高 | 低（重复编码） | 高 |
| 坐标系不变性 | 需处理 | 自然对齐 | 需处理 |

**Wayformer → UniAD 的演进**就体现了 agent-centric 到 scene-centric 的转变：Wayformer 独立预测每个 agent 效率低，UniAD 用 scene-centric 的 MotionFormer 一次处理所有 agent，同时建模多 agent 交互。

---

## 这类模型做什么

以 PNC SharedEncoder + 各 Decoder 的架构为例，模型做两件事：

**运动预测（Motion Prediction）**：给定场景（HD map + 历史轨迹），预测周围其他 agent（车辆、行人）的未来轨迹。输出是多条候选轨迹 + 每条的置信度（GMM 或 K 条 weighted trajectories）。

**运动规划（Motion Planning）**：给定同样的场景输入，预测**自车（ego）**的未来轨迹。输入是感知结果，输出是多条候选轨迹供规则层选择。

两者在架构上几乎完全一样（都是 scene encoder → cross-attention decoder），区别只是：预测的目标是他车，规划的目标是自车。Wayformer 本身是预测模型；PNC 的 `prediction_decoder` 做预测，其他 Decoder 做规划。

---

## 开环评测：离线回放打分

**什么是开环**：把历史驾驶数据当作测试集，模型输出轨迹，和真实轨迹比较打分，场景不受模型影响。相当于"做题"——答案已经确定了。

**主流评测流水线**（以 WOMD、Argoverse 为代表）：

```
历史轨迹（1s）+ HD map + 场景 agent 信息
          ↓
      模型推理
          ↓
   K 条预测轨迹 + 置信度
          ↓
   和真实轨迹对比计算指标
```

---

## 核心指标详解

### 边际预测 vs 联合预测：两个不同的子任务

在进入指标之前，需要先区分两类预测任务——WOMD 为此设立了两个独立的 Leaderboard。

**边际预测（Marginal Prediction）**：独立预测每个 agent 的未来轨迹，不考虑 agent 之间的联合一致性。minADE/minFDE/MR/mAP 都是边际预测指标，Wayformer 报告的就是这类结果。

**联合预测（Joint / Interactive Prediction）**：评测**成对交互 agent** 的联合轨迹。要求两个 agent 的预测轨迹在物理和逻辑上是一致的——比如 A 车让行时，B 车才会通过，不能两个 agent 各自独立预测出"都继续直行"的冲突结果。

联合预测的额外指标（MotionDiffuser、MTR 等论文报告的主要结果）：

| 指标 | 含义 |
|------|------|
| **minSADE**（Scene-level ADE） | K 个 joint 预测里，选整个场景（两 agent）平均位移最小的那组 |
| **minSFDE**（Scene-level FDE） | K 个 joint 预测里，选整个场景终点误差最小的那组 |
| **SMissRate** | joint 预测中最好组的场景级 FDE 超阈值的比例 |
| **Overlap** | 两个 agent 的预测轨迹互相碰撞的比率（物理合理性） |

这些指标为什么之前没提？因为：边际预测是**更广泛使用的基准**（所有模型都报），联合预测是**更高难度的子任务**（只有专门做 joint prediction 的工作报）。Wayformer 是边际预测专用架构，不参与联合预测排行榜。

---

### 指标演化历史

了解指标来历有助于判断哪些指标仍是主流、哪些已经过时。

| 指标 | 起源时间 | 来源/场合 | 当前状态 |
|------|---------|----------|---------|
| **ADE/FDE（单条）** | ~2016 | Social Force、Social LSTM 等早期行人预测论文 | 已过时——多模态时代的基础，现在很少单独报告 |
| **minADE/minFDE（K=5/10）** | 2018-2019 | DESIRE、MTP、Trajectron 等 | 仍是主流，K=6 是 WOMD/Argoverse 标准 |
| **MR（Miss Rate）** | 2020 | WOMD 论文（Ettinger et al.）随数据集发布 | 主流，WOMD 排行榜固定指标 |
| **mAP（轨迹级）** | 2021 | WOMD 官方竞赛引入 | WOMD 主排行榜指标，仍是主流 |
| **Brier-minFDE** | 2021 | Argoverse 2 论文（Wilson et al.）随数据集发布 | Argoverse 2 主指标，仍是主流 |
| **minSADE/minSFDE** | 2021 | WOMD Interactive Split 随数据集发布 | 联合预测主流指标 |
| **Overlap** | 2021 | WOMD Interactive Split | 联合预测辅助指标 |
| **PDM-Score** | 2024 | NAVSIM 论文 | 端到端评测新兴指标，快速成为主流 |

**ADE/FDE 为什么被取代**：2018 年之前行人预测论文普遍用单条 ADE/FDE，因为当时模型只输出一条轨迹。2019 年开始多模态输出成为标配，单条指标无法衡量覆盖能力，min-over-K 成为标准。如今看到论文只报 ADE/FDE（不带 min-over-K），通常是较老的工作或消融实验的简化版本。

---

### ADE / FDE：最基础的距离误差

**ADE（Average Displacement Error）**：预测轨迹和真实轨迹，在每个时间步上的欧氏距离取平均。

```
ADE = (1/T) × Σ_t ||ŷ_t - y_t||₂
```

**FDE（Final Displacement Error）**：只看轨迹终点的误差。

```
FDE = ||ŷ_T - y_T||₂
```

ADE 反映全程预测准确性，FDE 更关注长期意图（能不能预测对最终去哪）。FDE 通常比 ADE 大，因为越往后误差越大。

**为什么还不够**：这两个指标假设有唯一正确的未来轨迹。但真实驾驶有多种合理选择——同一个路口，直行和右转都是合理的。单条轨迹预测必然不完整。

---

### minADE / minFDE：多模态预测的标准指标

现代模型都输出 K 条候选轨迹（K=6 或 K=64 常见）。min-over-K 指标：在 K 条轨迹里选最接近真实轨迹的那条，计算它的 ADE 或 FDE。

```
minADE_K = min_{k=1..K} ADE(ŷ^k, y)
minFDE_K  = min_{k=1..K} FDE(ŷ^k, y)
```

**直觉**：测的是"你预测的 K 条轨迹里，最好的那条有多好"——即覆盖能力（coverage）。min-over-K 而不是 mean-over-K，根本原因是 GT 不唯一：同一个场景直行和左转都合理，GT 只是"这次实际发生的"，其他 K-1 条不能说"都是错的"。详见 [GMM：GT 不唯一时监督学习还能做吗](../20-concepts/gaussian-mixture-model.md#gt-不唯一时监督学习还能做吗)。

**WOMD 上的典型数值**（Wayformer，K=6）：

| 指标 | Wayformer Early Fusion | 含义 |
|------|----------------------|------|
| minADE | 0.545m | 最近的那条，整体平均每步差 0.55m |
| minFDE | 1.126m | 最近的那条，终点差 1.13m |
| MR | 12.3% | 12.3% 的场景最近条也超过 2m 阈值 |

**局限**：模型可以通过撒很多条轨迹（大 K）来让 min 指标好看，但这些轨迹的置信度未被考核。

---

### MR（Miss Rate）：最近条是否"足够近"

Miss Rate 是 minFDE 的二值化版本：

```
MR = P(minFDE_K > threshold)
```

WOMD 的阈值通常是 2m，Argoverse 是根据速度自适应的阈值。

**用途**：捕捉那些"连最好的预测也差很多"的场景——这类场景可能是真正困难的 corner case。

---

### mAP（WOMD 定义）：衡量置信度校准

WOMD 的 mAP 不是传统 CV 的 mAP，是**轨迹级别的 mean Average Precision**：

- 将每条预测轨迹按置信度排序
- 按轨迹类型（直行/左转/右转/变道/静止/其他）分别计算 AP
- 各类型 AP 取均值

**衡量什么**：模型是否能把高置信度的轨迹分配给真正正确的那条，而不是平均分配或乱分配。

```
Wayformer Early Fusion WOMD mAP: 0.412
```

**和 minFDE 的区别**：minFDE 不管置信度，有一条近就行；mAP 要求你给正确的那条分配高置信度。

---

### Brier-minFDE：同时惩罚距离和置信度

Brier-minFDE 把轨迹质量和置信度校准合在一个指标里：

```
Brier-minFDE = (1 - p_best)² + minFDE
```

其中 `p_best` 是最接近真实轨迹的那条的预测概率。

**含义**：即使最近的那条轨迹距离近，如果它的概率低，也会被惩罚。迫使模型把高概率分给最接近 GT 的轨迹。

**Argoverse 2 的主排行榜指标**就是 Brier-minFDE（也叫 b-minFDE），因为它同时考核了覆盖能力和置信度。

```
Wayformer Early Fusion Argoverse Brier-minFDE: 1.7451
```

---

### Overlap（碰撞率）：物理合理性

预测的轨迹不能让车辆互相穿透。WOMD 有一个 Overlap 指标，检测预测轨迹是否和其他 agent 的真实位置发生碰撞。

```
Wayformer Early Fusion Overlap: 0.127
```

约 12.7% 的预测轨迹在场景里存在某种程度的 overlap——这是模型不建模 agent 间交互的代价（每个 agent 独立预测，不保证互斥性）。

---

### 规划专用指标（L2 + 安全）

对于规划模型（预测 ego 轨迹），在 nuScenes planning track 等 benchmark 上用的是：

**L2 位移误差**（等同于 ADE，但针对 ego）：

```
规划 L2 = (1/T) × Σ_t ||ŷ_ego_t - y_ego_t||₂
```

通常报 1s/2s/3s 的 L2，以观察误差随时间增长的趋势。

**碰撞率**（Collision Rate）：规划轨迹和场景中其他 agent（使用真实位置）是否碰撞，按时间窗口（1s/2s/3s）分别统计。

**两者的关系**：L2 低不等于碰撞率低。一个"精确模仿专家"的模型可能 L2 很低，但在边缘场景下不如一个更保守（L2 略高但不碰撞）的模型。这是开环评测的根本问题。

---

## 指标对照总结

| 指标 | 类型 | 衡量什么 | 适用场景 | 局限 |
|------|------|---------|---------|------|
| ADE | 单轨迹距离 | 全程平均误差 | 确定性预测 | 不适合多模态 |
| FDE | 单轨迹距离 | 终点误差 | 长期意图 | 不适合多模态 |
| minADE_K | 多模态 | 最好条的全程误差 | 多模态预测 | 不考核置信度 |
| minFDE_K | 多模态 | 最好条的终点误差 | 主流排行榜 | 不考核置信度 |
| MR | 多模态 | 最好条是否超阈值 | Corner case | 只看终点 |
| mAP (WOMD) | 置信度 | 置信度排序质量 | 置信度校准 | 复杂，不直观 |
| Brier-minFDE | 综合 | 距离 + 置信度 | Argoverse 主指标 | 对置信度敏感 |
| Overlap | 安全 | 物理碰撞 | 预测合理性 | 不考核整体质量 |
| L2（规划）| 规划 | ego 轨迹误差 | 规划开环 | 惩罚正确的保守行为 |
| 碰撞率（规划）| 规划安全 | ego 和他车碰撞 | 规划安全性 | 结合 L2 才完整 |

---

## 四层评测：叫法、定义和各层指标

### 叫法的统一

"开环/闭环"和"仿真/实车"是两个独立维度，常被混用。业界标准叫法：

| 层 | 标准叫法 | 因果维度 | 环境维度 | 定义 |
|---|--------|---------|---------|------|
| **1** | 离线开环（Offline Open-loop） | 开环 | 离线数据 | 回放历史数据，模型输出不影响场景 |
| **2** | 仿真闭环（Simulation Closed-loop） | 闭环 | 虚拟仿真 | 模型真正驱动 ego，其他 agent 做出反应 |
| **3** | 影子模式（Shadow Mode / Online Open-loop） | 开环 | 真实路 | 实车跑真实路，AD 不接管，只记录输出和对比 |
| **4** | 实车路测（Real-world Closed-loop） | 闭环 | 真实路 | AD 真正接管，ego 行为影响周围 agent |

**开环/闭环的核心区别**是"模型输出是否影响后续场景"，**不是**"训练中 vs 训练后"。开环和闭环都是评测，不是训练过程——区别在于数据是历史录制还是模型实际执行。

**影子模式是开环**：实车跑真实路，但 AD 不接管车辆，其他 agent 的行为不受影响。ego 的行为由人类驾驶员决定，AD 系统只在后台计算"如果我来驾驶会输出什么"，然后和人类驾驶员的实际行为对比。

---

### 第一层：离线开环指标

已在上文详细介绍。核心指标汇总：

| 指标 | 适用任务 | 主要使用场景 |
|------|---------|------------|
| minADE_K | 边际预测 | WOMD/Argoverse，最通用 |
| minFDE_K | 边际预测 | WOMD/Argoverse，主要排行榜 |
| MR | 边际预测 | WOMD/Argoverse corner case |
| mAP（WOMD 定义） | 边际预测 | WOMD 官方主指标 |
| Brier-minFDE | 边际预测 | Argoverse 2 官方主指标 |
| Overlap | 边际预测 | WOMD，物理合理性 |
| minSADE/minSFDE | 联合预测 | WOMD Interactive Split |
| SMissRate | 联合预测 | WOMD Interactive Split |
| EPA（Expected Prediction Accuracy） | 边际预测 | WOMD 复合指标，部分论文报告 |
| L2 位移误差 | 规划 | nuScenes planning，ego 轨迹质量 |
| Collision Rate（离线） | 规划 | nuScenes planning，用真实 agent 位置计算 |
| OffRoadRate | 规划 | 轨迹离开可行驶区域的比例 |

**遗漏说明**：离线开环层还有 OffRoadRate（轨迹越界）、EPA（WOMD 复合指标）等，本文档主要覆盖排行榜主流指标，不追求穷举。

---

### 第二层：仿真闭环指标

模型在仿真器（nuPlan、CARLA、MetaDrive）里真正驱动 ego，其他 agent 响应 ego 的行为。

**核心指标**：

| 指标 | 含义 | 越高/低越好 |
|------|------|-----------|
| **路线完成率（Route Completion）** | 完成预定路线的百分比，防止"站着不动"策略 | ↑ 越高越好 |
| **碰撞率（Collision Rate）** | 和其他 agent、静态障碍物发生碰撞的次数/比例 | ↓ 越低越好 |
| **交规违反率（Traffic Infraction）** | 闯红灯、压实线、超速等 | ↓ 越低越好 |
| **舒适度（Comfort）** | 加速度、角速度、加加速度（jerk）超标次数 | ↓ 越低越好 |
| **nuPlan 综合分** | 以上各项加权，分 Reactive/Non-reactive 两种仿真 | ↑ 越高越好 |
| **PDM-Score（NAVSIM）** | 无碰撞×可行驶区域×行驶方向×舒适度×进度的几何平均 | ↑ 越高越好 |

**碰撞率 ≠ 安全**：一个永远刹车不动的模型碰撞率为 0 但路线完成率也为 0。必须同时看碰撞率和路线完成率，两者共同约束才有意义。

**nuPlan 的两种仿真模式**：
- **Non-reactive**：其他 agent 按历史数据回放，不响应 ego（类似 NAVSIM 的思路）
- **Reactive**：其他 agent 用 IDM 等规则响应 ego 的行为，但规则仿真和真实驾驶员行为有差距

---

### 第三层：影子模式指标

实车跑真实路，AD 系统不接管，在后台计算"如果我接管了会怎样"。核心是对比 AD 系统的输出轨迹和人类驾驶员的实际轨迹。

**核心指标**：

| 指标 | 含义 |
|------|------|
| **轨迹差异（Shadow Trajectory Divergence）** | AD 输出轨迹和人类实际轨迹的差距（L2 或方向差异） |
| **安全事件触发率** | AD 系统判断"如果我在控制，这里会需要紧急干预"的比率 |
| **舒适度达标率** | AD 输出轨迹的加速度、jerk 是否在可接受范围内 |
| **规则符合率** | AD 输出是否会产生交规违反 |

**影子模式的价值**：比仿真更真实（真实交通），比路测更安全（不接管不影响行车安全），是"用真实场景验证 AD 能力"的低成本方式。Tesla 的数据飞轮大量依赖影子模式来发现 hard case。

**局限**：影子模式只能评测"和人类的差距"，不能评测"在极端场景下是否安全"——因为 ego 不接管，极端场景里人类驾驶员已经做了处理，AD 的输出只是"假设性的"。

---

### 第四层：实车路测指标

AD 真正接管，是唯一能评测真实因果效果的层次。

**核心指标**：

| 指标 | 含义 | 行业参考值 |
|------|------|----------|
| **接管/干预次数（Interventions）** | 安全员需要人工干预的次数 | Waymo 2023: ~0.05 次/万英里（无人驾驶区域）|
| **DMPH（Disengagements per Million Hours）** | 每百万小时的接管次数，标准化指标 | 各公司数据不可比，口径不一 |
| **Miles per Intervention** | 每次接管之间的平均里程，直接反映自动化能力 | 越高越好 |
| **Critical Event Rate** | 紧急制动、危险接近等严重事件的比率 | 越低越好 |
| **碰撞率（实车）** | 实际发生碰撞的次数/万英里 | 人类驾驶约 1-2 次/百万英里 |

**为什么各公司数据不可比**：接管的定义不同（"驾驶员主动接管" vs "系统触发降级"），路测区域不同（高速公路 vs 城市复杂路况），气候条件不同。Waymo 在凤凰城的表现不能直接和百度在北京的比较。

---

### 四层评测的工程管线

```
离线开环（minADE/minFDE）
  ← 每次训练后自动跑，小时级，过滤明显退步的版本

仿真闭环（nuPlan/NAVSIM/内部仿真器）
  ← 重要版本跑，天级，验证规划安全性

影子模式（实车但不接管）
  ← 上线前，周-月级，用真实交通验证

实车路测（AD 接管）
  ← 产品发布前，持续收集，最终裁判
```

PNC 模型目前主要依赖离线开环（ADE/FDE on val set）+ 规则层仿真。仿真闭环和影子模式的投入程度取决于场景优先级和工程资源。

---

## 这类模型的评测难点

**多模态覆盖 vs 置信度校准的矛盾**：

多撒轨迹（K=64）能让 minFDE 好看，但会让 mAP 和 Brier-minFDE 变差。工程上的权衡：推理时用轨迹聚合（trajectory aggregation）把 K=64 的输出压缩到 K=6，同时兼顾覆盖和置信度。Wayformer 的 trajectory aggregation 在 Section 3.4 有详细描述。

**开环和闭环的 metric gap**：

minADE/minFDE 低，不代表规划好。这是这类模型最核心的评测困境。一个直觉解释：模型可以学会"平均来说人类驾驶员会怎么走"（低 L2），但"遇到罕见危险时应该怎么走"（低碰撞率）是另一回事。

**ego 预测 vs 他车预测的不对称性**：

预测他车时，"最近条"能覆盖到就算好，因为规划模块会做安全决策。预测 ego 时，不仅要近，还要安全可行、舒适、符合交规。评测 ego 轨迹时通常需要比预测他车更多的约束项。

---

## 参考 Benchmark 的典型分数区间

（供内部评测对齐用，数据来自公开排行榜）

**WOMD（K=6，1M steps 训练）**：

| 方法 | minADE ↓ | minFDE ↓ | MR ↓ | mAP ↑ |
|------|---------|---------|------|-------|
| SceneTransformer | 0.612 | 1.212 | 0.156 | 0.279 |
| MultiPath++ | 0.556 | 1.158 | 0.134 | 0.409 |
| Wayformer Early Fusion | **0.545** | **1.126** | **0.123** | 0.412 |
| MTR（2023）| ~0.50 | ~1.07 | ~0.11 | ~0.45 |

**Argoverse 2（K=6）**：

| 方法 | Brier-minFDE ↓ | MR ↓ | minADE ↓ |
|------|--------------|------|---------|
| TNT | 1.7564 | 0.1350 | 0.7659 |
| Wayformer Early Fusion | **1.7451** | **0.0192** | 0.7672 |

---

## 和 wiki 内其他概念的关联

- [自动驾驶开放生态](./av-open-ecosystem.md)：数据集、模型权重、Leaderboard 的统一全景，包括各数据集的许可证和下载方式
- [Wayformer](../30-papers/wayformer-2207.05844.md)：本文重点模型，WOMD/Argoverse 双榜 SOTA，指标数据来源
- [WOMD](../30-papers/waymo-open-motion-dataset.md)：主要 benchmark 数据集，minADE/minFDE/MR/mAP/Interactive Split 指标来源
- [Argoverse Motion Forecasting](../30-papers/argoverse-motion-forecasting.md)：Brier-minFDE 指标来源，与 WOMD 互补
- [NAVSIM](../30-papers/navsim-2406.15349.md)：开环和闭环之间的中间方案，PDM-Score
- [nuPlan](../30-papers/nuplan-2106.11810.md)：规划层闭环评测的主流 benchmark
- [PNC 模型架构](./pnc-model-architecture.md)：本文描述的被评测模型的内部架构参考
