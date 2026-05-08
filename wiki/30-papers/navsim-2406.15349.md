# NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking

一句话总结：NAVSIM 提出一种"非反应式仿真"评测框架，用真实数据代替仿真器，在开环和闭环之间找到一个可大规模使用的中间地带，成为端到端自动驾驶的主流评测 benchmark，NeurIPS 2024 Datasets and Benchmarks。

## 基本信息

- 论文：NAVSIM: Data-Driven Non-Reactive Autonomous Vehicle Simulation and Benchmarking
- 作者：Daniel Dauner, Marcel Hallgarten, Tianyu Li, Xinshuo Weng, Zhiyu Huang 等（跨 Waymo、NVIDIA、Tübingen 等多家机构）
- 发表：NeurIPS 2024 Datasets and Benchmarks Track
- arXiv：2406.15349
- 代码/排行榜：https://github.com/autonomousvision/navsim

---

## 核心问题：开环和闭环评测各有什么缺陷？

**开环评测（L2 误差等）的问题**：

模型的输出不影响场景演化。如果模型比专家更早刹车避险，开环评测会因为"偏离专家轨迹"而惩罚这个正确决策。L2 误差和真实驾驶安全性的相关性很弱——有工作发现两者几乎不相关。

**仿真闭环评测（CARLA 等）的问题**：

- 仿真器和真实世界存在外观和动力学差距（sim-to-real gap）
- 构建高质量 reactive 仿真器本身是未解难题
- 计算成本高，不适合大规模快速迭代

NAVSIM 的切入点：能不能用**真实数据**来做近似闭环评测，不需要完整仿真器？

---

## 核心设计：非反应式仿真（Non-Reactive Simulation）

**基本思路**：

取真实驾驶录像，冻结场景中所有其他 agent（行人、车辆）的轨迹不变，只把 ego 车辆替换成模型的输出，然后在这个"部分仿真"的场景里检测碰撞、道路偏离等问题。

```
真实场景录像
  ├── 其他 agent 轨迹：固定不变（来自真实数据）
  └── Ego 轨迹：替换为模型输出
                    ↓
在替换后的场景里计算：碰撞？偏离车道？进度？舒适度？
```

**为什么叫"非反应式"**：其他 agent 不对 ego 的行为做出反应——ego 突然刹车，前面的车不会相应减速。这是和完整闭环仿真（reactive simulation）的根本区别。

**优势**：不需要仿真器，可以直接用 nuPlan 等真实数据集的大量场景。评测结果更接近真实驾驶，比纯开环的 L2 指标更有意义。

**局限**：其他 agent 不响应 ego，极端情况下（ego 做了完全异常的操作）评测结果会失真；只做短时间窗口（几秒），长时序行为无法评测。

---

## 数据集

NAVSIM 使用 **nuPlan** 数据集的真实驾驶数据（1500 小时，覆盖多个城市），从中筛选出挑战性场景子集（navtrain/navtest split）。

这是 NAVSIM 相比 nuPlan 官方 benchmark 的关键差异：nuPlan 用 reactive 仿真器跑完整规划任务，NAVSIM 用同一份数据做非反应式短时评测，大幅降低评测成本，可以跑大规模消融实验。

---

## PDM-Score：综合评测指标

NAVSIM 的核心指标 **PDM-Score** 由 5 个子指标加权组成，每个子指标的取值范围是 0 到 1（越高越好）：

| 子指标 | 含义 | 权重 |
|--------|------|------|
| **No at-fault collision** | 在短时仿真窗口内不发生碰撞 | 最高 |
| **Drivable area compliance** | 轨迹保持在可驾驶区域内（不压道路边界） | 高 |
| **Driving direction compliance** | 沿正确行驶方向行驶（不逆行） | 高 |
| **Comfort** | 加速度、角速度等不超过舒适阈值 | 中 |
| **Progress** | 在场景时间窗口内完成了多少路程（防止"站着不动"策略刷分） |  中 |

PDM-Score = 各子指标乘积的加权几何平均，任何一项为 0 会拉低总分（不能通过某项极好来补偿某项极差）。

---

## 关键结论

**TransFuser 能和 UniAD 打平**（在挑战性场景上）：NAVSIM 发现，一个参数量相对较小的 CNN 模型 TransFuser 在 PDM-Score 上可以媲美体量大得多的 UniAD（端到端 Transformer）。这说明：过去开环评测里 UniAD 的优势，部分来自"更好地模仿专家轨迹外观"，而不是"更好的驾驶能力"。

**CVPR 2024 挑战赛**：143 支团队参赛，463 份提交，是目前自动驾驶 benchmark 中参与度最高的竞赛之一。

---

## 和其他评测方式的对比

| 评测方式 | 数据来源 | 其他 agent | 成本 | 和安全的对应 |
|---------|---------|-----------|------|------------|
| 开环（L2 误差） | 真实数据 | 固定（回放） | 极低 | 弱 |
| NAVSIM（非反应式） | 真实数据 | 固定（回放） | 低 | 中 |
| nuPlan 闭环 | 真实数据 + 仿真 | Reactive（响应 ego） | 高 | 较强 |
| CARLA 闭环 | 纯仿真 | Reactive | 高 | 中（存在 sim gap） |
| 真实路测 | 真实 | 真实响应 | 极高 | 完整 |

NAVSIM 的定位：**以真实数据的质量 + 接近闭环的指标体系，在极低成本下大规模评测**——这是它快速成为主流的原因。

---

## 现状与影响

**已成为端到端 AD 的标准 benchmark 之一**：SparseDrive、DiffusionDrive、Hydra-MDP 等 2024-2025 年的主要端到端工作都在 NAVSIM 上汇报结果。

**局限推动后续工作**：非反应式仿真的局限（其他 agent 不响应）催生了 NAVSIM-v2 等扩展，尝试引入部分 reactive 行为；也有工作研究如何把 NAVSIM 的评测理念和 nuPlan 的 reactive 仿真结合。

---

## 和 wiki 内其他概念的关联

- [自动驾驶模型评测全栈概览](../00-overview/av-model-evaluation.md)：NAVSIM 在端到端评测层的定位和与其他方式的对比
- [nuPlan](./nuplan-2106.11810.md)：NAVSIM 使用 nuPlan 数据，是 nuPlan 官方 reactive 仿真的轻量替代
- [MetaDrive](./metadrive-2109.12674.md)：类似定位的轻量评测平台，专注 RL 训练，NAVSIM 更专注 E2E 模型评测
