# Argoverse Motion Forecasting Dataset

## 是什么

Argoverse 是 Argo AI（后被 Ford 收购）发布的自动驾驶数据集系列，包含运动预测（Motion Forecasting）和 3D 感知两个子集。运动预测部分是运动预测领域使用最广泛的公开 benchmark 之一，Wayformer 等论文的 Argoverse 结果均指此部分。

目前有两个版本：
- **Argoverse 1**（2019）：30 万个场景，5s 总时长（2s 历史 + 3s 预测）
- **Argoverse 2 Motion**（2022）：25 万个场景，11s 总时长（5s 历史 + 6s 预测），场景更丰富，图版更新

---

## 数据规格对比

| 项目 | Argoverse 1 | Argoverse 2 Motion |
|------|------------|-------------------|
| 场景数 | ~333K | ~250K |
| 历史时长 | 2s（20 帧，10Hz） | 5s（50 帧，10Hz） |
| 预测时长 | 3s（30 帧） | 6s（60 帧） |
| 地理 | 匹兹堡、迈阿密 | 6 个美国城市 |
| HD map | 提供（车道中心线 + 连通性） | 提供（更详细，含人行横道等） |
| 预测目标 | 1 个 focal agent（自车附近最相关的 agent） | 最多 5 个 scored agent |
| 参与者类型 | 主要车辆 | 车辆、行人、摩托车等 |

**和 WOMD 的定位区别**：Argoverse 的 focal agent 设计（每个场景只考核 1-5 个最重要的 agent）让任务更聚焦；WOMD 需要预测场景中所有 agent。Argoverse 历史更长（社区积累了更多基线），WOMD 规模更大（103K vs 333K，但 WOMD 单场景时间更长）。

---

## 主要评测指标

Argoverse 1 和 2 的指标略有差异，但核心一致。

### minADE（minimum Average Displacement Error）

预测 K 条候选轨迹（K=6 标准），取和真实轨迹平均距离最近的那条，计算每个时间步的欧氏距离均值：

```
minADE_K = min_{k=1..K} [ (1/T) × Σ_t ||ŷ_t^k - y_t||₂ ]
```

单位：米（m）。越小越好。Wayformer Early Fusion on Argoverse 1: **0.7672m**。

### minFDE（minimum Final Displacement Error）

K 条轨迹中，终点最接近真实终点的那条，计算终点的欧氏距离：

```
minFDE_K = min_{k=1..K} ||ŷ_T^k - y_T||₂
```

单位：米（m）。越小越好。比 minADE 更关注"最终去哪"的长期意图准确性。

### MR（Miss Rate）

K 条预测中最接近的那条，终点误差超过阈值的场景比例。

Argoverse 的阈值不是固定的 2m，而是**速度自适应**：高速移动的 agent 允许更大的误差（agent 的预期移动距离的一定比例）。这比 WOMD 的固定阈值更合理。

```
MR = P(minFDE_K > threshold(speed))
```

越小越好。Wayformer Early Fusion: **0.0192**（约 1.9% 的场景连最近条也超阈值）。

### Brier-minFDE（Argoverse 2 主指标）

**Brier-minFDE 是 Argoverse 2 的主排行榜指标**，同时衡量轨迹质量和置信度校准：

```
Brier-minFDE = (1 - p_best)² + minFDE_K
```

其中 `p_best` 是 K 条轨迹中，最接近 GT 的那条的预测概率。

**为什么加 `(1 - p_best)²` 项**：minFDE 只管"最近条有多近"，不管这条的置信度是多少。模型可以通过平均分配概率（每条都给 1/K）来回避置信度评估。Brier 项惩罚"最好条概率低"的情况——如果你给最接近 GT 的那条只分配了 0.1 的概率，$(1 - 0.1)^2 = 0.81$ 的惩罚直接加到 FDE 上。

**直觉**：Brier-minFDE 要求模型不仅要有一条好轨迹，还要"知道哪条是好的"并给它高置信度。

Wayformer Early Fusion on Argoverse 1: **1.7451**（minFDE=1.7451 时 Brier 项≈0，意味着最好条的概率接近 1，校准很好）。

---

## 和 WOMD 指标体系的对比

| 指标 | Argoverse | WOMD |
|------|-----------|------|
| 主指标 | Brier-minFDE | mAP |
| 覆盖指标 | minADE/minFDE | minADE/minFDE |
| 未命中率 | MR（速度自适应阈值） | MR（固定 2m 阈值） |
| 置信度评估 | Brier 项（显式） | mAP（隐式，通过 AP 曲线） |
| 预测目标数 | 1-5 个 focal agent | 最多 8 个 interested agent |

Argoverse 的 Brier-minFDE 把置信度评估集成进一个数字，直观简洁；WOMD 的 mAP 通过完整的 precision-recall 曲线评估，信息更丰富但不如 Brier 直观。

---

## DCMS 是什么

Wayformer 论文 Table 1 中 Argoverse 排行榜上的"次好方法"是 **DCMS**（Diverse Conditional Motion Set）——Waymo 内部的一个工作，采用条件化多模态轨迹生成方法，在 Argoverse 上取得当时次好的 Brier-minFDE 1.7564，被 Wayformer 超越（1.7451）。DCMS 本身没有公开论文，是 Argoverse 排行榜上的匿名提交。

---

## 和 wiki 内其他概念的关联

- [自动驾驶模型评测全栈概览](../00-overview/av-model-evaluation.md)：运动预测评测层的指标体系，Argoverse 是主要 benchmark 之一
- [Wayformer](./wayformer-2207.05844.md)：在 Argoverse 1 上取得 SOTA，Brier-minFDE 1.7451
- [Waymo Open Motion Dataset（WOMD）](./waymo-open-motion-dataset.md)：另一主流运动预测 benchmark，与 Argoverse 互补
