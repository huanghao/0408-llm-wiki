# UniAD: Planning-oriented Autonomous Driving（Hu et al., 2023，CVPR Best Paper）

一句话总结：UniAD 提出"规划导向"的端到端自动驾驶框架，用统一的 query 接口把感知（跟踪+建图）、预测（运动预测+占据预测）、规划串成一个网络，每个模块都为最终规划服务，在 nuScenes 上全面超越此前所有模块化和端到端方法，CVPR 2023 Best Paper。

## 基本信息

- 论文：Planning-oriented Autonomous Driving
- 作者：Yihan Hu\*、Jiazhi Yang\*、Li Chen\*†、Keyu Li\*、Chonghao Sima、Xizhou Zhu 等（\* 同等贡献，† 项目负责人）
- 机构：OpenDriveLab & Shanghai AI Laboratory、武汉大学、商汤研究院
- 发表：CVPR 2023 Best Paper Award
- arXiv：2212.10156
- 代码：https://github.com/OpenDriveLab/UniAD

---

## 核心问题

自动驾驶系统通常被设计成三种形态（Figure 1）：

1. **独立模型（Standalone）**：感知/预测/规划各自独立训练，部署时串联，误差累积
2. **多任务学习（MTL）**：共享 backbone + 各任务独立 head，缺乏跨模块协调，负迁移风险
3. **端到端（直接优化规划）**：不显式建模感知和预测，可解释性差，安全无法保证

UniAD 的核心论点：**系统设计应该以规划为导向**——感知和预测模块不应该独立优化，而是应该提供有利于规划的中间表示，所有任务通过统一 query 接口互相传递信息，最终都服务于规划决策。

**关键问题**：在端到端框架里，哪些感知和预测任务是规划安全所必需的？哪些任务的协同对规划贡献最大？

---

## 方法：五个 Transformer Decoder 模块串联

### 整体架构（Figure 2）

```
多视角摄像头图像
      ↓ BEVFormer（BEV encoder）
  BEV 特征 B [H×W×C]
      ↓
  TrackFormer ──→ 跟踪 query Q_A（每个 agent 一个 query）
      ↓                    ↓
  MapFormer  ──→ 地图 query Q_M（每个地图元素一个 query）
      ↓          ↓
  MotionFormer（Q_A + Q_M → 运动预测，多模态轨迹）
      ↓
  OccFormer（运动 query + BEV → 未来占据预测）
      ↓
  Planner（ego-vehicle query → 规划轨迹 + 碰撞优化）
```

所有模块都是 Transformer decoder 结构，以 query 作为接口相互传递信息——上游模块的 query 输出直接作为下游模块的 key/value 输入，不需要显式的后处理或格式转换。

### TrackFormer：联合检测与跟踪

用 DETR 风格的 detection query + track query 同时做检测和跨帧跟踪。detection query 检测新出现的 agent，track query 保持已有 agent 的时序一致性表示。输出 $N_a$ 个 agent query $Q_A$（每个 agent 一个向量，维度 D）。

额外引入**ego-vehicle query**：专门对自车建模的 query，在 MotionFormer 中与其他 agent 的 query 一起参与交互，使规划能感知周围 agent 的意图。

### MapFormer：在线地图构建

基于 Panoptic SegFormer，用地图 query $Q_M$ 稀疏表示车道、路口等语义地图元素（不依赖 HD map）。输出的 $Q_M$ 在 MotionFormer 中为运动预测提供地图上下文。

### MotionFormer：多 agent 联合运动预测

输入：agent query $Q_A$、地图 query $Q_M$、BEV 特征 B。

每层捕获三种交互：agent-agent、agent-map、agent-goal point（deformable attention 关注轨迹终点周围的 BEV 特征）。输出每个 agent 的 K=6 条多模态候选轨迹。

关键设计：**scene-centric 预测**——把所有 agent 的轨迹一次性在统一坐标系下预测，而不是每个 agent 各自建立 ego-centric 坐标系再独立预测。这避免了 Wayformer 那种为每个 agent 重复编码整个场景的计算开销。

**非线性平滑器**：由于输入是 BEV 感知结果（有误差），直接回归轨迹会产生物理上不合理的高曲率路径。训练时对 GT 轨迹做非线性平滑（式4-5），约束轨迹满足运动学约束（jerk、曲率等）。

### OccFormer：未来占据预测

OccFormer 预测未来多时间步的 BEV 占据图（每个格子是否被占据、被哪个 agent 占据）。用 agent 特征和 BEV 密集特征的 pixel-agent 交互建模。输出 $T_o$ 步的占据预测 $\hat{O}$，供 Planner 做碰撞避免。

### Planner：规划 + 碰撞优化

输入：ego-vehicle query（来自 MotionFormer，已编码了自车与周围 agent 的交互）+ command embedding（直行/左转/右转）+ BEV 特征。

输出初始规划轨迹 $\hat{\tau}$，然后用 Newton 方法基于占据预测 $\hat{O}$ 做推理时优化：

```
τ* = argmin f(τ, τ̂, Ô)
   = λ_coord ||τ - τ̂||₂  （贴近原始预测）
   + λ_obs Σ_t D(τ_t, Ô^t)  （远离占据区域）
```

这使规划在避免模仿专家轨迹的同时，也能主动规避预测到的障碍物。

### 两阶段训练

1. **第一阶段**：只训练感知部分（TrackFormer + MapFormer），6 个 epoch
2. **第二阶段**：端到端训练所有模块，20 个 epoch

实验表明两阶段训练比直接端到端训练更稳定。

---

## 关键结果

所有实验在 nuScenes 数据集上进行，输入为 6 摄像头图像（无 LiDAR）。

### 消融实验（Table 2）：每个模块对规划的贡献

| 实验 | 模块 | 规划 avg.L2 ↓ | 规划 avg.Col ↓ |
|------|------|-------------|--------------|
| 0（基线 MTL）| 全部 | 1.154 | 0.941 |
| 10（仅规划）| 规划 | 1.131 | 0.773 |
| 11（+运动）| 规划+运动 | 1.014 | 0.717 |
| **12（全部）** | **全部** | **1.004** | **0.430** |

关键发现：
- 单独端到端规划（ID-10）已经比 MTL 基线（ID-0）好
- 加入运动预测（ID-11）进一步改善
- **加入占据预测（ID-12）对碰撞率的改善最显著**（0.717→0.430，降低 40%）
- 说明两类预测（运动+占据）对安全规划缺一不可

### 运动预测（Table 5）：对比视觉端到端方法

| 方法 | minADE ↓ | minFDE ↓ | MR ↓ | EPA ↑ |
|------|---------|---------|------|-------|
| PnPNet | 1.15 | 1.95 | 0.226 | 0.222 |
| ViP3D | 2.05 | 2.84 | 0.246 | 0.226 |
| Constant Vel. | 2.13 | 4.01 | 0.318 | — |
| **UniAD** | **0.71** | **1.02** | **0.151** | **0.456** |

相比 PnPNet：minADE 降低 38.3%，minFDE 降低 47.7%——这是因为联合感知和预测带来了更好的 agent 表示。

### 规划（Table 7）：对比视觉规划方法

| 方法 | L2-1s | L2-2s | L2-3s | Col-1s | Col-2s | Col-3s |
|------|-------|-------|-------|--------|--------|--------|
| NMP | — | — | 2.31 | — | — | 1.92 |
| FF² | 0.55 | 1.20 | 2.54 | 0.06 | 0.17 | 1.07 |
| EO² | 0.67 | 1.36 | 2.78 | 0.04 | 0.09 | 0.88 |
| ST-P3 | 1.33 | 2.11 | 2.90 | 0.23 | 0.62 | 1.27 |
| **UniAD** | **0.48** | **0.96** | **1.65** | **0.05** | **0.17** | **0.31** |

对比 ST-P3：avg.L2 降低 51.2%，avg.Col 降低 56.3%，而且超过多个 LiDAR-based 方法。

---

## 局限性

1. **计算代价高**：五个模块串联，训练需要大量计算资源，车端轻量化部署困难（论文 Future Work 明确提及）
2. **长尾场景**：失败案例主要出现在大型卡车/拖挂车场景——这类 agent 在训练数据中少见，运动预测和占据预测都不准确
3. **上游感知误差的传播**：尽管 query 接口能一定程度上软化误差传播（比硬阈值后处理好），感知严重失败时仍会影响规划
4. **HD map free 设计的代价**：MapFormer 做在线建图，比用 HD map 的方法在地图质量上有差距，影响下游任务
5. **目前仅纯视觉**：输入只有摄像头，没有用 LiDAR；加入 LiDAR 的版本没有在本文中探索

---

## 现状与影响

**一句话定性**：UniAD 是端到端自动驾驶领域的里程碑——它系统地证明了"规划导向的任务协同设计"优于独立模块，并提供了一套可复现的基线，CVPR 2023 Best Paper 奖项使其成为整个领域的方向性参考。

**直接影响**：
- **SparseDrive、VAD、DriveDreamer** 等大量 2023-2025 年工作以 UniAD 为基线对比
- **OpenDriveLab** 基于 UniAD 发展出 DriveX、GenAD 等系列工作
- nuScenes planning track 成为主流 benchmark，UniAD 的指标成为标准参考点
- Query-based 接口设计影响了后续多个端到端框架的设计

**NAVSIM 上的发现**：NAVSIM（2024）重新在非反应式仿真上评测 UniAD，发现简单的 TransFuser 可以和 UniAD 在 PDM-Score 上持平——说明 UniAD 在 nuScenes 开环评测上的优势，有部分来自更好地模仿专家轨迹外观，而不完全是更好的驾驶能力。

**局限推动后续工作**：计算代价问题推动了 SparseDrive（稀疏表示）、VAD（向量化场景）等更轻量的端到端方案；纯视觉限制推动了后续 camera+LiDAR 融合的端到端工作。

---

## 和 wiki 内其他概念的关联

- [自动驾驶模型评测全栈概览](../00-overview/av-model-evaluation.md)：UniAD 是端到端规划评测层的主要参考点，nuScenes planning L2/Col 指标的基准
- [Wayformer](./wayformer-2207.05844.md)：同类的运动预测工作，Wayformer 是 agent-centric 独立预测，UniAD 是 scene-centric 联合预测；NAVSIM 中两者被横向比较
- [NAVSIM](./navsim-2406.15349.md)：UniAD 在 NAVSIM 的非反应式闭环评测中被 TransFuser 追平，引发对开环评测有效性的重新审视
- [nuScenes](./nuscenes-1903.11027.md)：UniAD 所有实验的数据集
- [nuPlan](./nuplan-2106.11810.md)：规划闭环 benchmark，UniAD 的后续工作在 nuPlan 上有更多评测
- [PNC 模型架构](../00-overview/pnc-model-architecture.md)：PNC 是模块化设计（Encoder+多Decoder），UniAD 是端到端一体化；两者代表不同的系统设计哲学

## 值得看的部分 / 相关资料

- **Table 2（消融实验）**：是整篇论文最重要的表——逐步加入每个模块，直观展示各模块对规划的贡献，尤其 OccFormer 对碰撞率的关键作用
- **Figure 2（架构图）**：Query 接口的流向清晰，是理解"规划导向设计"的直觉入口
- **Section 2.4（Planning）**：规划模块的碰撞优化公式，解释为什么需要 OccFormer 输出
- **Section 3.3（Qualitative）**：Figure 3 的可视化展示所有任务输出，能看到感知/预测/规划的配合
- 相关工作：
  - VAD（arXiv:2303.12077，ICCV 2023）：向量化场景表示的端到端规划，计算效率更高
  - SparseDrive（2024）：稀疏表示进一步减少计算量
  - NAVSIM（arXiv:2406.15349，NeurIPS 2024）：重新评测 UniAD 的"真实驾驶能力"
