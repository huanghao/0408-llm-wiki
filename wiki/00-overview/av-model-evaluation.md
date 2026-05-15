# 自动驾驶模型评测：预测与规划为核心

对于以 Wayformer/PNC 风格为代表的**向量化场景编码 + Transformer Decoder 轨迹生成**这类模型，预测和规划层的评测是核心。感知层从略。

---

## 这类模型做什么

以 PNC SharedEncoder + 各 Decoder 的架构为例，模型做两件事：

**运动预测（Motion Prediction）**：给定场景（HD map + 历史轨迹），预测周围其他 agent（车辆、行人）的未来轨迹。输出是多条候选轨迹 + 每条的置信度（GMM 或 K 条 weighted trajectories）。

**运动规划（Motion Planning）**：预测**自车（ego）**的未来轨迹。输出是多条候选轨迹供规则层选择。

两者架构几乎一样（scene encoder → cross-attention decoder），区别只是目标：预测针对他车，规划针对 ego。

---

## 坐标系视角：Ego-centric vs Scene-centric vs Agent-centric

模型需要决定"从谁的视角描述场景"，这影响架构、计算效率和多 agent 建模能力。

| | Ego-centric | Agent-centric | Scene-centric |
|---|---|---|---|
| 坐标原点 | 自车 | 被预测的 agent | 世界坐标系 |
| 典型模型 | PNC 规划 Decoder | Wayformer、MTR | UniAD、MotionDiffuser |
| 前向传播次数 | 1 次 | N 次（N=agent 数） | 1 次 |
| 多 agent 交互 | 不适用 | 弱（独立预测） | 强（联合建模） |
| 公共特征重复编码 | — | 是（每个 agent 各一次）| 否 |

**Ego-centric**：自车坐标系，感知数据天然对齐，适合规划模块。  
**Agent-centric**（Wayformer/MTR）：每个 agent 为中心，N 个 agent = N 次前向传播，场景公共信息重复编码 N 次。  
**Scene-centric**（UniAD）：统一世界坐标系，一次前向传播处理所有 agent，可建模 agent 间联合分布（A 刹车→B 也刹车）。

Wayformer → UniAD 的演进就体现了 agent-centric 到 scene-centric 的转变。

---

## 四层评测框架

"开环/闭环"和"仿真/实车"是两个独立维度，常被混用。业界标准叫法：

| 层 | 标准叫法 | 因果维度 | 环境 | 定义 |
|---|--------|---------|------|------|
| 1 | **离线开环**（Offline Open-loop） | 开环 | 历史数据 | 回放录制数据，模型输出不影响场景 |
| 2 | **仿真闭环**（Simulation Closed-loop） | 闭环 | 虚拟仿真 | 模型驱动 ego，其他 agent 做出反应 |
| 3 | **影子模式**（Shadow Mode / Online Open-loop） | 开环 | 真实路 | 实车跑真实路，AD 不接管，后台记录输出 |
| 4 | **实车路测**（Real-world Closed-loop） | 闭环 | 真实路 | AD 真正接管，ego 行为影响周围 agent |

**开环/闭环的核心区别**是"模型输出是否影响后续场景"，**不是**"训练中 vs 训练后"——两种都是评测，都在训练结束后进行。

**影子模式是开环**：AD 不接管，ego 由人类驾驶，其他 agent 行为不受 AD 影响，AD 只在后台输出"如果我控制会怎样"。

### 评测结果如何帮助模型提升

这里有个关键区分：评测**指标**不直接进入梯度，进入梯度的是**loss 函数**。两者的关系：

**层 1（离线开环）→ 直接对应 loss**

| 离线开环指标 | 对应的训练 loss |
|------------|--------------|
| minADE/minFDE | GMM 负对数似然（回归）+ Winner-Takes-All 分类 |
| mAP / Brier-minFDE | 分类 loss（让最优匹配条的置信度最高）|
| Overlap（碰撞） | UniAD 有 collision loss，但大多数预测模型的 loss 里没有碰撞项 |

评测指标是 loss 的外部验证——loss 下降不一定等于指标提升，因为 loss 的设计可能不完美。

**层 2-4（闭环/影子/路测）→ 通过数据飞轮间接提升**

这三层的评测结果**不可导**，无法直接进入训练，但通过以下方式推动下一轮训练数据改善：

1. **发现 hard case**：影子模式里 AD 输出了危险轨迹（人类驾驶员当时处理了），这个 case 被标记为高价值样本，加回训练集并提高权重
2. **发现系统性错误**：仿真闭环发现"在路口总犯同一类错误"→ 排查该类场景在训练集的覆盖情况 → 补充数据后重训
3. **接管数据直接入训**：每次安全员接管后的片段是高价值训练数据（Tesla/Waymo 数据飞轮的核心）

**评测不反传梯度，但评测→发现问题→数据改善→重训，构成整个能力迭代闭环。**

---

## 第一层指标：离线开环

### 边际预测 vs 联合预测

WOMD 为此设立了两个独立 Leaderboard：

**边际预测（Marginal Prediction）**：独立预测每个 agent，不考虑 agent 间联合一致性。所有主流模型都报这类结果。

**联合预测（Joint / Interactive Prediction）**：评测**成对交互 agent** 的联合轨迹，要求两个 agent 的预测物理上一致——不能双方都预测"继续直行"而发生碰撞。只有专门做 joint prediction 的工作（MotionDiffuser、MTR Interactive 等）报这类结果。

---

### 边际预测指标

**ADE / FDE**（~2016，已过时）：单条轨迹的全程/终点误差。多模态时代之前的标准，现在只在消融实验简化版里见到。

**minADE / minFDE**（2018-2019，仍主流）：K 条轨迹里选最接近 GT 的那条，计算 ADE 或 FDE。

```
minADE_K = min_{k=1..K} (1/T) × Σ_t ||ŷ_t^k - y_t||₂
minFDE_K = min_{k=1..K} ||ŷ_T^k - y_T||₂
```

测的是覆盖能力（K 条里最好的有多好），不考核置信度。min-over-K 而不是 mean-over-K 的原因：GT 不唯一，其他 K-1 条不能说"都是错的"。详见 [GMM：GT 不唯一时监督学习还能做吗](../20-concepts/gaussian-mixture-model.md#gt-不唯一时监督学习还能做吗)。

**MR（Miss Rate）**（2020，WOMD 随数据集发布，仍主流）：minFDE 的二值化版本——最近条的终点误差是否超阈值（WOMD: 2m，Argoverse: 速度自适应）。捕捉"连最好的预测也差很多"的 corner case。

```
MR = P(minFDE_K > threshold)
```

**mAP（轨迹级）**（2021，WOMD 官方竞赛引入，仍主流）：不是传统 CV 的 mAP，但思路相同。

**AP（Average Precision）是什么**：AP 是精度-召回率曲线下的面积，用来衡量"排序质量"——模型把真正好的结果排在前面的能力。直觉：如果你有 100 个预测，让你按置信度从高到低排列，AP 衡量的是"置信度高的那些是否确实更准"。

**轨迹级 mAP 的计算**：

1. 把模型输出的所有预测轨迹按**置信度**从高到低排列
2. 按**轨迹类型**分类（直行/左转/右转/变道/静止/其他，通过终点位置判断）
3. 对每种类型，遍历排好序的预测列表：
   - 对每个预测，判断它是否"命中"——终点在 GT 终点的 2m 范围内
   - 统计当前精度（Precision）= 已命中数 / 已看过的预测数
   - 统计当前召回率（Recall）= 已命中数 / 总真实轨迹数
4. 计算 Precision-Recall 曲线下面积，得到该类型的 AP
5. 各类型 AP 取均值，得到 mAP

**为什么用 mAP 而不是 minFDE**：minFDE 不管置信度，只要有一条近就行——模型可以输出 64 条轨迹把所有方向都撒到，minFDE 会很好看，但置信度完全乱。mAP 要求"高置信度的轨迹必须真的准"，惩罚乱分配置信度的行为。

```
Wayformer Early Fusion WOMD mAP: 0.412
```

衡量置信度排序质量——能否把高置信度分配给正确的那条。

**Brier-minFDE**（2021，Argoverse 2 论文引入，仍主流）：把距离和置信度合在一个指标里。

```
Brier-minFDE = (1 - p_best)² + minFDE
```

`p_best` 是最接近 GT 那条的预测概率。即使距离近，概率低也会被惩罚。Argoverse 2 官方主指标。

**Overlap**（2021，WOMD 引入）：预测轨迹和其他 agent 真实位置发生碰撞的比率。衡量物理合理性——agent-centric 独立预测时常见，因为不保证多 agent 轨迹互斥。

**EPA（Expected Prediction Accuracy）**：WOMD 的复合指标，加权合并 ADE/FDE/MR，部分论文报告，非主流。

**规划专用（离线）**：

- **L2 位移误差**：ego 轨迹和专家轨迹的欧氏距离，通常报 1s/2s/3s，nuScenes planning 常用
- **Collision Rate（离线）**：规划轨迹和真实 agent 位置的碰撞，按时间窗口统计
- **OffRoadRate**：轨迹离开可行驶区域的比例

L2 低不等于碰撞率低——"精确模仿专家"的模型在边缘场景下可能不安全。

---

### 联合预测指标（WOMD Interactive Split）

| 指标 | 含义 |
|------|------|
| **minSADE**（Scene-level ADE） | K 个 joint 预测里，整个场景平均位移最小的那组 |
| **minSFDE**（Scene-level FDE） | K 个 joint 预测里，整个场景终点误差最小的那组 |
| **SMissRate** | 最好 joint 预测的场景级 FDE 超阈值比例 |
| **Overlap** | 两 agent 预测轨迹互相碰撞的比率 |

---

### 指标演化时间线

| 指标 | 引入时间 | 来源 | 当前状态 |
|------|---------|------|---------|
| ADE/FDE（单条） | ~2016 | Social LSTM 等早期行人预测 | 已过时 |
| minADE/minFDE（K=5/10） | 2018-2019 | DESIRE、MTP、Trajectron 等 | 主流，K=6 是标准 |
| MR | 2020 | WOMD 论文（Ettinger et al.） | 主流 |
| mAP（轨迹级）、minSADE/minSFDE | 2021 | WOMD 官方竞赛 + Interactive Split | 主流 |
| Brier-minFDE | 2021 | Argoverse 2 论文（Wilson et al.） | Argoverse 主指标 |
| PDM-Score | 2024 | NAVSIM 论文 | 快速成为端到端主流 |

---

## 第二层指标：仿真闭环

模型在仿真器（nuPlan、NAVSIM、CARLA、MetaDrive）里真正驱动 ego，其他 agent 用规则（IDM 等）响应 ego 行为。

| 指标 | 含义 | 方向 |
|------|------|------|
| **路线完成率** | 完成预定路线的百分比，防止"站着不动"策略 | ↑ |
| **碰撞率** | 和 agent、障碍物碰撞的次数/比例 | ↓ |
| **交规违反率** | 闯红灯、压实线、超速等 | ↓ |
| **舒适度（Comfort）** | 加速度、jerk 超标次数 | ↓ |
| **nuPlan 综合分** | 以上各项加权，Reactive/Non-reactive 两种模式 | ↑ |
| **PDM-Score（NAVSIM）** | 无碰撞×可行驶区域×行驶方向×舒适度×进度的几何平均 | ↑ |

碰撞率 ≠ 安全：永远刹车碰撞率 = 0 但路线完成率也 = 0，必须同时看两个指标。

nuPlan 分 Reactive（IDM 响应 ego）和 Non-reactive（按历史回放）两种模式，NAVSIM 是非反应式的轻量近似。

---

## 第三层指标：影子模式

实车跑真实路，AD 不接管，后台记录 AD 输出，和人类驾驶员实际行为对比。

| 指标 | 含义 |
|------|------|
| **轨迹差异（Shadow Trajectory Divergence）** | AD 输出 vs 人类实际轨迹的 L2 或方向差 |
| **安全事件触发率** | AD 判断"如果我控制，这里需要紧急干预"的比率 |
| **舒适度达标率** | AD 输出轨迹的 jerk/加速度是否在可接受范围 |
| **规则符合率** | AD 输出是否会产生交规违反 |

价值：比仿真更真实（真实交通），比路测更安全（不接管）。Tesla 数据飞轮大量依赖影子模式发现 hard case。

局限：无法评测"极端场景下是否安全"——极端情况人类已经处理，AD 输出只是假设性的。

---

## 第四层指标：实车路测

AD 真正接管，唯一能评测真实因果效果的层次。

| 指标 | 含义 | 参考 |
|------|------|------|
| **接管次数（Interventions）** | 安全员人工接管的次数 | Waymo 2023: ~0.05 次/万英里 |
| **Miles per Intervention** | 每次接管间的平均里程 | 越高越好 |
| **DMPH** | Disengagements per Million Hours，标准化 | 各公司口径不一，不可直接比较 |
| **Critical Event Rate** | 紧急制动、危险接近等严重事件比率 | 越低越好 |
| **碰撞率（实车）** | 实际碰撞次数/万英里 | 人类驾驶: ~1-2 次/百万英里 |

各公司数据不可比：接管定义不同、路测区域不同、气候不同。Waymo 凤凰城的数字不能和百度北京直接比。

---

## 工程管线：四层的节奏和作用

```
离线开环（minADE/minFDE）
  ← 每次训练后自动跑，小时级
  ← 作用：过滤明显退步的版本，是训练 loss 的外部验证

仿真闭环（nuPlan/NAVSIM/内部仿真器）
  ← 重要版本跑，天级
  ← 作用：验证规划安全性，发现系统性错误，触发数据补充

影子模式（实车但不接管）
  ← 上线前，周-月级
  ← 作用：发现仿真覆盖不到的 hard case，高价值样本回流训练集

实车路测（AD 接管）
  ← 产品发布前，持续积累
  ← 作用：最终裁判，接管数据直接进训练集，构成数据飞轮
```

PNC 模型目前主要依赖离线开环（ADE/FDE on val set）+ 规则层仿真。仿真闭环和影子模式的投入程度取决于场景优先级。

---

## 评测难点

**多模态覆盖 vs 置信度校准的矛盾**：多撒轨迹（K=64）让 minFDE 好看，但让 mAP 和 Brier-minFDE 变差。用 trajectory aggregation 把 K=64 压缩到 K=6，兼顾两者。

**开环和闭环的 metric gap**：minADE/minFDE 低不代表规划好。模型可以学会"平均情况下怎么走"（低 L2），但无法保证"罕见危险场景下安全"（低碰撞率）。

**GT 不唯一的根本矛盾**：离线开环里 GT 只是"这次实际发生的"，其他合理未来都没有 GT，min-over-K 是权宜之计。详见 [GMM 文档](../20-concepts/gaussian-mixture-model.md#gt-不唯一时监督学习还能做吗)。

---

## 参考 Benchmark 的典型分数

（供内部评测对齐用，数据来自公开排行榜）

**WOMD 边际预测（K=6）**：

| 方法 | minADE ↓ | minFDE ↓ | MR ↓ | mAP ↑ |
|------|---------|---------|------|-------|
| SceneTransformer | 0.612 | 1.212 | 0.156 | 0.279 |
| MultiPath++ | 0.556 | 1.158 | 0.134 | 0.409 |
| Wayformer Early Fusion | **0.545** | **1.126** | **0.123** | 0.412 |
| MTR（2022）| ~0.50 | ~1.07 | ~0.11 | ~0.45 |

**Argoverse 2（K=6）**：

| 方法 | Brier-minFDE ↓ | MR ↓ | minADE ↓ |
|------|--------------|------|---------|
| TNT | 1.7564 | 0.1350 | 0.7659 |
| Wayformer Early Fusion | **1.7451** | **0.0192** | 0.7672 |

---

## 和 wiki 内其他概念的关联

- [自动驾驶开放生态](./av-open-ecosystem.md)：数据集、模型权重、Leaderboard 全景，许可证和下载方式
- [Wayformer](../30-papers/wayformer-2207.05844.md)：本文重点模型，WOMD/Argoverse 双榜 SOTA，Factorized Attention 和 Latent Queries 加速技术
- [Attention 优化技术](../20-concepts/attention-optimization.md)：Factorized/Axial Attention 和 Latent Queries 是 Wayformer 等运动预测模型的加速核心
- [WOMD](../30-papers/waymo-open-motion-dataset.md)：主要 benchmark，minADE/minFDE/MR/mAP/Interactive Split 指标来源
- [Argoverse Motion Forecasting](../30-papers/argoverse-motion-forecasting.md)：Brier-minFDE 指标来源
- [NAVSIM](../30-papers/navsim-2406.15349.md)：仿真闭环的轻量替代，PDM-Score
- [nuPlan](../30-papers/nuplan-2106.11810.md)：规划层仿真闭环主流 benchmark
- [高斯混合模型（GMM）](../20-concepts/gaussian-mixture-model.md)：轨迹输出格式，GT 不唯一的根本讨论
- [PNC 模型架构](./pnc-model-architecture.md)：被评测模型的内部架构参考
