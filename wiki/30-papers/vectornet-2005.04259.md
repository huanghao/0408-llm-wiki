# VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation（Gao et al., CVPR 2020）

一句话总结：第一个把 HD 地图和 agent 轨迹统一表示为向量集合、用层次化图神经网络做行为预测的框架——polyline 子图提取局部几何特征，全局交互图建模跨 polyline 依赖，辅助图补全任务增强表征；比光栅化 ConvNet baseline 减少 70% 参数量和 200×+ FLOPs，同时在 Argoverse 上超越 state-of-the-art，成为后续运动预测方法的标准输入编码范式。

## 基本信息

- 论文：VectorNet: Encoding HD Maps and Agent Dynamics from Vectorized Representation
- 作者：Jiyang Gao\*、Chen Sun\*、Hang Zhao、Yi Shen、Dragomir Anguelov、Congcong Li、Cordelia Schmid（\* 等同贡献）
- 机构：¹ Waymo LLC；² Google Research
- 发表：CVPR 2020
- arXiv：2005.04259（2020 年 5 月）

---

## 核心问题

**传统行为预测方法把 HD 地图和轨迹渲染成图像再用 ConvNet 编码，引入有损渲染、感受野受限、计算成本高三个根本问题。**

HD 地图天然是结构化向量数据（车道线是 spline、路口是多边形、停车线是线段），agent 轨迹天然是时序向量序列。主流方法（IntentNet、MultiPath、PRECOG 等）把这些数据光栅化成鸟瞰图后用 ConvNet 编码，带来：

1. **有损渲染**：几何精度受像素分辨率限制，稀疏但精确的语义信息（速度限制、停车标志属性）难以编码
2. **感受野受限**：ConvNet 的有效感受野受限于 kernel 大小和图像分辨率；扩大感受野需要更大 kernel 或更高分辨率，计算代价平方增长
3. **计算冗余**：以场景为中心渲染时，对每个目标 agent 都要重新裁剪特征图；scene-centric 渲染时背景共享但 target-centric 裁剪仍有代价

**VectorNet 的核心洞察**：HD 地图和轨迹都可以表示为有序或无序的向量集合（polylines），向量天然保留了几何精度和语义属性，可以直接送入图神经网络，无需光栅化。

---

## 方法 / 核心机制

### 架构总览（Figure 2）

```
输入：HD 地图 + agent 轨迹
    ↓ 向量化
每条 polyline P_j → 若干向量节点 v_i = [d^s_i, d^e_i, a_i, j]
    ↓
【第一层：Polyline 子图网络】
每条 polyline 内部：全连接 GNN（自注意力）
    共享 MLP（node encoder） + maxpool（置换不变聚合）
    输出 polyline 级特征向量 p_j（每条车道线/轨迹一个向量）
    ↓
【第二层：全局交互图】
所有 polyline 特征 {p_1, ..., p_P} 全连接 GNN（自注意力）
    GNN(P) = softmax(P_Q P_K^T) P_V
    输出带有全场景上下文的 polyline 特征
    ↓
【预测头】
target agent 对应的 polyline 特征 → MLP → 未来轨迹坐标偏移序列

【辅助任务（训练时）】
随机 mask 部分节点特征 → MLP decoder 重建被 mask 节点
```

### 关键设计 1：向量表示

每个向量节点的特征向量：

$$\mathbf{v}_i = [\mathbf{d}^s_i, \mathbf{d}^e_i, \mathbf{a}_i, j]$$

- $\mathbf{d}^s_i$、$\mathbf{d}^e_i$：向量起止点坐标（2D 或 3D）
- $\mathbf{a}_i$：属性（地图要素的类型、速度限制；轨迹节点的时间戳、朝向等）
- $j$：所属 polyline ID（同一 polyline 的节点共享 j）

坐标以目标 agent 最后观测时刻的位置为原点归一化，使特征与绝对位置无关。

这一表示相当于把 PointNet 的思路扩展到有方向、有属性、有分组信息的向量集合。

### 关键设计 2：Polyline 子图网络

同一 polyline 内的向量节点两两全连接，用共享 MLP + maxpool 迭代更新节点特征：

$$\mathbf{v}_i^{(l+1)} = \varphi_{rel}\left(g_{enc}(\mathbf{v}_i^{(l)}),\ \varphi_{agg}\left(\{g_{enc}(\mathbf{v}_j^{(l)})\}\right)\right)$$

- $g_{enc}$：共享 MLP（single FC + LayerNorm + ReLU）
- $\varphi_{agg}$：maxpool（置换不变）
- $\varphi_{rel}$：concat 后的 relational operator

polyline 级特征 = maxpool over all node features。

**与 PointNet 的关系**：当去掉起止点区分（$\mathbf{d}^s = \mathbf{d}^e$）、去掉属性和 polyline ID 时，子图网络退化为 PointNet——VectorNet 是保留了空间结构和语义属性的 PointNet 扩展。

### 关键设计 3：全局交互图

所有 polyline 特征（地图要素 + 所有 agent 轨迹）组成一个全连接图，用单层自注意力建模跨 polyline 交互：

$$\{\mathbf{p}_i^{(l+1)}\} = \text{GNN}\left(\{\mathbf{p}_i^{(l)}\}, \mathcal{A}\right)$$

$$\text{GNN}(\mathbf{P}) = \text{softmax}(\mathbf{P}_Q \mathbf{P}_K^T)\ \mathbf{P}_V$$

全连接图让任意两条车道线或 agent 可以直接交换信息，不受 ConvNet 感受野限制。

### 关键设计 4：图补全辅助任务（Node Completion）

受 BERT masked language model 启发，训练时随机 mask 部分 polyline 节点的输入特征，要求模型从上下文重建它们：

$$\hat{\mathbf{p}}_i = \varphi_{node}(\mathbf{p}_i^{(L_t)})$$

损失 = Huber loss（重建误差）。强迫模型学习 polyline 之间的上下文关系，而非仅依赖目标 agent 自身的轨迹历史。推理时不启用。

### 总损失

$$\mathcal{L} = \mathcal{L}_{traj} + \alpha \mathcal{L}_{node}$$

$\mathcal{L}_{traj}$ 是未来轨迹的负 Gaussian log-likelihood，$\alpha=1.0$。

---

## 关键结果

### VectorNet vs ConvNet（Table 4，Argoverse，DE@3s，K=1）

| 模型 | FLOPs | #Param | DE@3s (In-house) | DE@3s (Argoverse) |
|---|---|---|---|---|
| R18-k3-c3-r400（最佳 ConvNet）| 10.56G | 246K | 1.09 | 4.81 |
| VectorNet w/o aux. | 0.041G×n | 72K | 1.05 | 3.84 |
| **VectorNet w aux.** | **0.041G×n** | **72K** | **1.00** | **3.67** |

- FLOPs：ConvNet 是固定值（场景中心化），VectorNet 随 agent 数 n 线性增长，但单 agent 仅 0.041G vs 10.56G = **200×+ 差距**
- 参数量：72K vs 246K = **70% 减少**
- Argoverse DE@3s：3.67 vs 4.81 = **12% 提升**

### Argoverse 排行榜对比（Table 5，DE@3s，K=1）

| 方法 | DE@3s | ADE |
|---|---|---|
| Constant Velocity | 7.89 | 3.53 |
| LSTM ED | 4.95 | 2.15 |
| Challenge Winner: Jean | 4.17 | 1.86 |
| **VectorNet** | **4.01** | **1.81** |

发布时达到 Argoverse leaderboard SOTA（2020-03-18 截止）。

### 消融结论（Table 2）

| 上下文输入 | 图补全 | DE@3s (Argoverse) |
|---|---|---|
| 仅目标轨迹 | — | 2.36 |
| + map | 否 | 1.75 |
| + map + agents | 否 | 1.72 |
| + map | 是 | 1.70 |
| **+ map + agents** | **是** | **1.66** |

- 加入地图信息：ADE 2.36 → 1.75，提升最大
- 加入其他 agent 轨迹：1.75 → 1.72，小幅提升
- 辅助图补全任务：一致性提升，在长时程（ADE）更明显

---

## 局限性

1. **单模态预测**：论文使用 MLP 解码器输出单条轨迹（K=1），不直接支持多模态未来预测（多种可能轨迹）；作者在结论中明确指出需要结合 MultiPath/PRECOG 等多模态解码器
2. **全连接全局图的规模问题**：当场景中 polyline 数量很大时，全连接全局图的自注意力计算代价 O(P²)；作者用单层限制了影响，但对密集场景仍有限制
3. **坐标系归一化依赖目标 agent**：特征以目标 agent 为中心归一化，预测多个 agent 时需要为每个目标单独重新归一化并重算（FLOPs 随目标数线性增长）；虽然论文提到作为 future work 共享坐标系，但原版不支持
4. **输入依赖感知结果**：轨迹来自自动感知系统（noisy），而非真实标注；论文提到这是实际部署中的现实约束，但未评估感知误差对预测的影响

---

## 现状与影响

一句话定性：**VectorNet 确立了"HD 地图 + 轨迹统一向量化 → 层次化图网络"的行为预测标准范式，被此后几乎所有主流运动预测方法（MTR、Wayformer、MotionDiffuser、UniAD 等）直接沿用或扩展，是 2020–2024 年该领域的事实编码标准。**

- **直接影响**：
  - **MTR（NeurIPS 2022）**：在 VectorNet 编码基础上引入 Motion Query Pair，把全局交互图的输出进一步用于迭代轨迹精化
  - **Wayformer（ICRA 2023）**：用 VectorNet 风格的向量化输入，系统对比了 Early/Late/Hierarchical 融合策略
  - **MotionDiffuser（CVPR 2023）**：在 VectorNet 编码的场景表征上用扩散模型生成多 agent 联合轨迹分布
  - **UniAD（CVPR 2023 Best Paper）**：端到端 AV 系统中，运动预测模块的地图/agent 编码沿用 VectorNet 层次化向量化思路
- **核心贡献持久性**：
  - 向量化表示（polyline + attribute node features）已成为 Waymo Open Motion Dataset、nuScenes、Argoverse 2 等 benchmark 的标准输入格式
  - 图补全辅助任务是 self-supervised scene representation learning 在 AV 领域的早期成功实践
- **被扩展的维度**：
  - 多模态解码：结合 MultiPath++ / Anchor-based / Diffusion-based 解码器
  - 并行多 agent 预测：改进坐标归一化，支持全场景所有 agent 同时预测（MTR/Wayformer 等）
  - Transformer 替代 GNN：后续方法把全局交互图的 GNN 替换为 Transformer（等价但工程上更高效）

---

## 和 wiki 内其他概念的关联

- [图神经网络（GNN）](../20-concepts/gnn.md)：VectorNet 是 GNN 在自动驾驶场景理解中的典型应用——polyline 子图用消息传递聚合局部几何，全局图用自注意力建模跨 polyline 交互；两层层次化结构对应 GNN 的"局部→全局"归纳偏置
- [PointNet：点云深度学习](pointnet-1612.00593.md)：VectorNet 的 polyline 子图网络是 PointNet 的直接扩展——去掉 polyline 分组和属性编码后两者等价；PointNet 处理无序点，VectorNet 处理有序有属性的向量，共享"逐元素 MLP + 置换不变聚合"的核心思路
- [Attention 直觉：Self/Cross/Local](../20-concepts/attention-intuition.md)：全局交互图的 GNN 等价于全局 self-attention（$\text{softmax}(P_Q P_K^T) P_V$），与 Transformer encoder 在数学形式上完全相同；VectorNet 是在图网络框架下重新推导出 self-attention 的案例
- [MTR: Motion Transformer](mtr-2209.13508.md)：MTR 在 VectorNet 向量化编码基础上引入 Motion Query Pair，把静态意图锚点与动态搜索查询解耦；VectorNet 是 MTR 架构的直接前驱
- [Wayformer](wayformer-2207.05844.md)：Wayformer 系统对比了不同融合策略，输入编码沿用 VectorNet 向量化范式；两者是同一编码哲学在不同融合架构下的对比研究
- [MotionDiffuser](motiondiffuser-2306.03083.md)：MotionDiffuser 的场景编码基于向量化表示，置换不变 denoiser 建立在 VectorNet 式的 polyline 特征之上
- [Argoverse Motion Forecasting](argoverse-motion-forecasting.md)：VectorNet 在 Argoverse benchmark 上发布时达到 SOTA，Argoverse 是 VectorNet 的主要公开评测基准

---

## 值得看的部分

- **Section 3.1（向量化表示）+ Figure 1**：左右对比光栅化与向量化两种输入表示，是理解整篇论文动机最直观的一页；向量节点特征 $\mathbf{v}_i = [\mathbf{d}^s, \mathbf{d}^e, \mathbf{a}, j]$ 的定义简洁但信息量大
- **Section 3.2（polyline 子图 + 与 PointNet 的关系）**：作者明确指出"当 $\mathbf{d}^s = \mathbf{d}^e$ 且去掉属性时，退化为 PointNet"——这个连接是理解两篇论文关系的关键
- **Figure 2（整体架构图）**：Input vectors → Polyline subgraphs → Global interaction graph → 双任务输出（Map Completion + Trajectory Prediction），四列清晰展示数据流
- **Table 4（FLOPs 和参数量对比）**：0.041G×n vs 10.56G，200×+ 的 FLOPs 差距说明向量化不只是表示问题，还是工程效率革命
- **Table 2（消融）**：加地图信息带来最大提升（ADE 2.36→1.75），图补全辅助任务在长时程一致有益——两个最值得引用的数字
- **Figure 4（注意力可视化）**：全局交互图的 attention 在 agent 面临多选择时（如分叉路口）聚焦到正确的候选车道，提供了定性理解模型"看哪里"的视角
