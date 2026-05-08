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

## 开环 vs 闭环：工业界的实际做法

**开环的定位**：快速迭代的代理指标。一次模型改动几小时内就能出 minADE/minFDE，足以过滤明显退步的版本。但开环指标和真实驾驶质量的相关性弱（UniAD 等工作都观察到两者可以不相关）。

**为什么仍然使用开环**：工程上的不可替代性——闭环仿真（nuPlan、CARLA）需要数小时到数天，路测代价更高。开环指标作为"门槛"而非"终点"。

**闭环评测的分层**：

```
开环（minADE/minFDE）← 每次迭代必跑，小时级
      ↓ 通过才进入
仿真闭环（nuPlan / NAVSIM）← 重要版本跑，天级
      ↓ 通过才进入
影子模式（实车不介入，在线评测）← 上线前，周级
      ↓ 通过才进入
实车路测（小范围受控）← 产品发布前
```

PNC 模型目前主要依赖开环（ADE/FDE on val set）+ 规则层仿真，闭环评测的投入程度取决于场景优先级。

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

- [Wayformer](../30-papers/wayformer-2207.05844.md)：本文重点模型，WOMD/Argoverse 双榜 SOTA，指标数据来源
- [WOMD](../30-papers/waymo-open-motion-dataset.md)：主要 benchmark 数据集，minADE/minFDE/MR/mAP 指标定义来源
- [NAVSIM](../30-papers/navsim-2406.15349.md)：开环和闭环之间的中间方案，PDM-Score
- [nuPlan](../30-papers/nuplan-2106.11810.md)：规划层闭环评测的主流 benchmark
- [PNC 模型架构](./pnc-model-architecture.md)：本文描述的被评测模型的内部架构参考
