# MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion（Jiang et al., 2023）

一句话总结：MotionDiffuser 用条件扩散模型学习多 agent 轨迹的**联合分布**，天然多模态、置换不变、支持在推理时用任意可微代价函数做约束采样（attractor/repeller），在 WOMD Interactive Split 上达到 SOTA，CVPR 2023 Highlight。

## 基本信息

- 论文：MotionDiffuser: Controllable Multi-Agent Motion Prediction using Diffusion
- 作者：Chiyu "Max" Jiang\*、Andre Cornman\*、Cheolho Park、Ben Sapp、Yin Zhou、Dragomir Anguelov（\* 同等贡献）
- 机构：Waymo LLC
- 发表：CVPR 2023 Highlight
- arXiv：2306.03083

## 核心问题

运动预测的两个根本挑战：

1. **多模态性**：同一场景下未来轨迹有多种合理可能（直行、左转、右转），模型必须能表达整个分布而不是输出单条最优轨迹。现有回归方法（MultiPath/Wayformer）依赖预定义 anchor 来枚举模式，anchor 的数量和设计本身引入了归纳偏置。
2. **多 agent 联合建模**：单独预测每个 agent 的轨迹会产生不一致的联合预测（两辆车预测轨迹在同一位置碰撞），必须建模多 agent 轨迹的**联合分布**。

**核心洞察**：扩散模型（DDPM）可以同时解决这两个问题——它直接对轨迹的**联合概率分布**建模，不需要 anchor；推理时可以在去噪过程中注入约束，实现"可控生成"。

> **扩散模型（Denoising Diffusion Probabilistic Model, DDPM）**：一类生成模型，训练一个 denoiser 网络 $D_\theta$，学习从加噪数据中恢复干净数据。推理时从纯高斯噪声出发，反复调用 denoiser 逐步去噪，最终采样出符合训练数据分布的样本。与 GAN 相比，扩散模型训练更稳定（只需 L2 loss），分布覆盖更广；与 VAE 相比，不需要 encoder，直接对数据空间建模。

## 方法：三个核心设计

### 1. 条件扩散建模（Conditional Diffusion for Trajectories）

**问题形式化**：对 $N_a$ 个 agent，每个 agent $i$ 的未来轨迹 $\mathbf{s}_i \in \mathbb{R}^{N_t \times N_f}$（$N_t$ 个时间步，$N_f$ 个特征维），目标是建模联合分布 $p(\mathbf{S}; \mathbf{C})$，其中 $\mathbf{C}$ 是场景上下文（HD map、历史轨迹、交通灯）。

**训练**（Figure 2 上半部分）：
- 用 Wayformer encoder 将场景上下文编码为 condition tokens $\mathbf{C}$
- 对 GT 轨迹 $\mathbf{S}$ 加高斯噪声得到 $(\mathbf{S} + \boldsymbol{\epsilon}, \sigma)$
- 训练 denoiser $D_\theta(\mathbf{S} + \boldsymbol{\epsilon}; \mathbf{C}, \sigma)$ 用 L2 loss 预测去噪后的干净轨迹
- 整体训练目标（式 4）：$\arg\min_\theta \mathbb{E}[\|D_\theta(\mathbf{x} + \boldsymbol{\epsilon}; \mathbf{c}, \sigma) - \mathbf{x}\|_2^2]$

**推理**（Figure 2 下半部分）：
- 从纯噪声 $\mathbf{S} \sim \mathcal{N}(\mathbf{0}, \sigma_{\max}^2 \mathbf{I})$ 出发
- 用 Huen's 2阶 ODE 方法，以 32 步迭代去噪到干净轨迹
- 每次可并行采样多个独立样本，得到多条候选轨迹

### 2. 置换不变 Denoiser 架构（Figure 3）

**动机**：场景中 agent 的顺序是任意的，模型输出不应随 agent 编号改变：$D(\mathbf{S}^j; \mathbf{C}^j, \sigma) = D^j(\mathbf{S}; \mathbf{C}, \sigma)$（式 8），其中 $j$ 表示任意置换。

**实现**：
- 噪声轨迹 $\mathbf{s}_1 \cdots \mathbf{s}_{N_a}$ 拼接噪声等级 $\sigma$ 的 Fourier 编码后进入 denoiser
- **Self-Attention（跨 agent）**：在所有 agent 的 noisy 轨迹之间做 attention，让 denoiser 学到 agent 间的交互（哪辆车可能影响另一辆车的路线）
- **Cross-Attention（轨迹 → 场景上下文）**：每个 agent 的 token 与 condition tokens $\mathbf{c}_1 \cdots \mathbf{c}_{N_a}$ 做 cross-attention，把场景信息注入去噪过程
- 不对 agent 维度做位置编码 → Transformer 自然置换不变

> **置换不变（Permutation Invariant）**：函数 $f(x_1, x_2) = f(x_2, x_1)$，即输入顺序不影响输出。运动预测中场景里的车辆编号是任意的，模型不应把"第 3 辆车"和"第 7 辆车"当成本质不同的东西处理。置换不变性通过"不加位置编码 + Transformer 的 attention 天然对集合操作"实现。

### 3. PCA 压缩轨迹表示（Section 3.5）

直接在原始轨迹空间（$80 \times 2 = 160$ 维）做扩散计算量大、且轨迹的高频抖动对预测无意义。

**做法**：对 $10^5$ 条 Waymo 轨迹做 PCA，保留前 $N_p$ 个主成分。实验发现：
- **3 个主成分**已能解释 99.7% 的方差（轨迹在时间上平滑，信息高度冗余）
- 实际用 **10 个主成分**做完整模型（更高重建精度）
- PCA 重建误差（10 components）均值 **0.06 m**，远低于 SOTA 预测误差

将轨迹投影到 PCA 空间后维度从 160 降到 10，扩散 denoiser 的计算量大幅减少，同时消除了高频噪声，更利于约束采样的稳定性。

> 为什么 PCA 而不是 autoencoder？PCA 是线性变换，精确可逆，log-probability 计算（见下文）可以解析进行；autoencoder 是非线性的，精确 log-probability 更难计算。论文在 ablation（Table 3）中也证明去掉 PCA 性能显著下降（minSADE 1.03 vs 0.88）。

### 4. 约束采样框架（Section 3.4）

这是 MotionDiffuser 与以往预测模型的最大差异：**在推理时注入约束，而不需要重新训练**。

**原理**：采样目标是 $p(\mathbf{S}; \mathbf{C}) \cdot q(\mathbf{S}; \mathbf{C})$，其中 $q$ 是约束分布。联合分布的 score 为：

$$\nabla_\mathbf{S} \log(p \cdot q) = \underbrace{\nabla_\mathbf{S} \log p(\mathbf{S}; \mathbf{C}, \sigma)}_{\text{扩散 score（已知）}} + \underbrace{\nabla_\mathbf{S} \log q(\mathbf{S}; \mathbf{C}, \sigma)}_{\text{约束 score（近似）}}$$

约束 score 的近似（式 13）：任意可微代价函数 $\mathcal{L}$ 对去噪后轨迹 $D(\mathbf{S}; \mathbf{C}, \sigma)$ 求梯度，反向传播到噪声轨迹 $\mathbf{S}$：

$$\nabla_\mathbf{S} \log q(\mathbf{S}; \mathbf{C}, \sigma) \approx \lambda \frac{\partial}{\partial \mathbf{S}} \mathcal{L}(D(\mathbf{S}; \mathbf{C}, \sigma))$$

两种内置约束：
- **Attractor**（式 14）：让轨迹某时刻到达目标位置，代价 = 预测位置与目标位置的 L1 距离
- **Repeller**（式 15-16）：让 agent 之间保持距离 $r$，代价 = pairwise 距离函数

**Score Thresholding（ST）**：约束 score 的近似在高噪声等级 $\sigma$ 时不稳定，用截断（clipping）稳定训练：$\nabla_\mathbf{S} \log q := \text{clip}(\sigma \nabla_\mathbf{S} \log q, \pm 1)/\sigma$（式 18）。去掉 ST 约束满足度显著下降（Table 2 中 Ours(-ST) 的 SR2m/SR5m 分别从 0.952/0.994 降至 0.913/0.949）。

### 5. 精确 Log Probability 推断（Section 3.3）

扩散模型生成的是样本，通常无法直接计算样本的精确对数概率。MotionDiffuser 利用连续归一化流（CNF）的瞬时变量变换公式（式 9）：

$$\frac{\partial \log p(\mathbf{x}(t))}{\partial t} = -\text{Tr}\left(\frac{\partial f}{\partial \mathbf{x}}\right)$$

对 ODE 轨迹积分后得到 $\log p(\mathbf{x}(0))$（式 11）。在 PCA 压缩空间中，维度 $n$ 很小（10 维），计算 Jacobian trace 为 $O(n^2)$，可行。

这使得 MotionDiffuser 能给每条采样轨迹打概率分，用于后续排序、过滤高概率样本（Figure 4 展示了颜色编码的 log probability 热图）。

## 关键结果 / 数据

### WOMD Interactive Split（Table 1）

> **指标说明**：
> - **minSADE**：minimum joint Scene-level Average Displacement Error——6 条 joint 预测里最接近 GT 的那条，所有 agent 所有时刻预测位置与 GT 的平均距离（越小越好）
> - **minSFDE**：同上，但只看最终时刻（越小越好）
> - **SMissRate**：joint 预测全部 miss 的场景比例（越小越好）
> - **mAP**：Mean Average Precision，按 agent 行为类型（左转/右转/直行等）计算的均值精度（越大越好）
> - **Overlap**：最可能 joint 预测中发生碰撞（位置重叠）的比例（越小越好）

测试集结果（与 backbone 相同的 Wayformer 比较）：

| 方法 | minSADE ↓ | minSFDE ↓ | SMissRate ↓ | mAP ↑ |
|---|---|---|---|---|
| Wayformer | 1.00 | 2.19 | 0.49 | 0.12 |
| JFP | 0.88 | 1.99 | **0.42** | **0.21** |
| **MotionDiffuser** | **0.86** | **1.95** | 0.43 | 0.20 |

- MotionDiffuser 在 minSADE/minSFDE 上超越 JFP，在 mAP/SMissRate 上略逊于 JFP
- **Overlap = 0.091**，比 SceneTransformer(M) 的 0.046 高——扩散模型生成的多样轨迹包含更多低概率碰撞样本，但用 repeller 约束可大幅降低

### 约束采样效果（Table 2）

在 Attractor 约束（让轨迹到达 GT 终点）实验中：

| 方法 | minSADE ↓ | Overlap ↓ | SR2m ↑ | SR5m ↑ |
|---|---|---|---|---|
| Optimization（后处理优化）| 4.563 | 0.054 | 1.000 | 1.000 |
| CTG（迭代内部优化）| 1.18 | 0.057 | 0.921 | 0.957 |
| **Ours（-ST）**| 2.083 | 0.042 | 0.913 | 0.949 |
| **Ours** | **0.533** | 0.040 | **0.952** | **0.994** |

> **SR2m/SR5m**：Success Rate within 2m/5m——约束满足度，预测轨迹终点落在目标 2m/5m 范围内的比例

- Optimization 满足约束最好（SR=1.0），但偏离数据分布（minSADE=4.563，轨迹不自然）
- MotionDiffuser 在保持轨迹真实性（minSADE=0.533）的同时达到 SR2m=0.952

Repeller 约束（最小距离 5m）：Overlap 从 3.229 降到 0.008，减少一个数量级。

### 消融实验（Table 3）

| 变体 | minSADE ↓ |
|---|---|
| Ours(-PCA) | 1.03 |
| Ours(-Transformer) | 0.93 |
| Ours(-SelfAttention) | 0.91 |
| **MotionDiffuser** | **0.88** |

三个组件（PCA 压缩 / Transformer denoiser / Self-Attention 跨 agent 交互）均有独立贡献，缺一明显下降。

## 局限性

论文 Section 6（Conclusion）隐含：

- **Overlap 指标不如 JFP**：测试集上 Overlap=0.091 高于 JFP(0.061)——扩散模型的分布覆盖广，低概率的"碰撞轨迹"也会被采样到；需要配合 repeller 约束才能抑制
- **推理速度相对慢**：32 步 ODE 去噪比单次前向推理慢，实时部署需要进一步加速（蒸馏或更少步数）
- **约束 score 近似在高噪声不准**：Score Thresholding 是工程 workaround，非理论完备解
- **PCA 假设轨迹线性可表示**：在强弯曲或急刹车等非常规轨迹上 PCA 重建误差可能更大

## 现状与影响

一句话定性：**MotionDiffuser 是扩散模型进入运动预测领域的奠基工作之一——它证明了"扩散模型可以直接对多 agent 联合轨迹分布建模"并实现可控采样，在 WOMD 上达到 SOTA，此后大量自动驾驶预测工作（如 DiffusionDrive、SMITE 等）延续这条路线；Waymo 内部同期/后续工作（JFP、SceneTransformer）是直接竞争和互补关系。**

- **约束采样框架的影响最深远**：attractor/repeller 机制为 AV 仿真（scenario injection）提供了一条工程路径——可以在推理时指定"让某辆车去某个位置"或"避开某个区域"，无需重新训练，这直接影响了 Waymo 的仿真数据生成工作
- **PCA 轨迹表示**：后续多篇工作（CTG++、DiffStack 等）沿用或改进这一思路，用低维表示加速扩散推理
- **精确 log probability 推断**：为基于排名/过滤的下游任务（如从大量候选轨迹里选最可信的）提供了理论支撑，这在之前的 anchor-based 方法里是不直接可得的

## 和 wiki 内其他概念的关联

- [Wayformer](./wayformer-2207.05844.md)：MotionDiffuser 直接复用 Wayformer 的场景 encoder（transformer-based，处理 agent 历史 + road graph + 交通灯），denoiser 用 Wayformer encoder 产出的 condition tokens 做 cross-attention
- [WOMD（Waymo Open Motion Dataset）](./waymo-open-motion-dataset.md)：MotionDiffuser 的训练数据和主评测 benchmark；Interactive Split 是专为多 agent 联合预测设计的子集
- [TrafficGen](./trafficgen-2210.06609.md)：同为"生成多 agent 场景"的工作，但方向不同——TrafficGen 是从空地图生成车辆布局和轨迹（数据增强），MotionDiffuser 是给定真实场景上下文预测未来轨迹分布（运动预测）
- [MCTS](../20-concepts/mcts.md)：约束采样框架在概念上与 test-time search 类似——都是在推理时利用外部信号（约束 / 价值函数）引导采样；MotionDiffuser 通过可微代价函数的梯度引导，MCTS 通过树搜索引导
- [Neural VRP](../00-overview/neural-vrp.md)：两者都把问题建模为"从分布中采样满足约束的解"，NCO 用 REINFORCE 训练策略，MotionDiffuser 用扩散模型训练分布，约束注入的方式也类似（训练时不含约束，推理时加入）

## 值得看的部分 / 相关资料

- **Section 3.1（扩散模型基础 + 条件扩散）**：清晰推导了从 DDPM score matching 到条件 denoiser 的形式化，以及 preconditioning（式 6）解决非单位方差的技术细节
- **Figure 2（训练 vs 推理流程图）**：最直观地展示了"训练时加噪→去噪"和"推理时纯噪声→迭代去噪+可选约束"的完整流程
- **Section 3.4（约束采样）+ Table 2**：attractor/repeller 的数学推导（式 12-18）和定量验证，是理解可控生成应用场景的核心；Score Thresholding 的设计动机也在此处
- **Figure 6（定性约束效果对比）**：No Constraint / Optimization / CTG / Ours 四种方式的可视化对比，直观展示 Optimization 方法偏离数据分布的问题
- **Section 3.3（精确 log probability）**：扩散模型 + ODE + Jacobian trace 的推导，理解为什么 PCA 压缩是必要的工程前提
- 相关工作：
  - Kool et al. 2021, *Large scale interactive motion forecasting for autonomous driving: The Waymo Open Dataset*（WOMD benchmark 定义论文，[7] in 原文）
  - Luo et al. 2022, *JFP: Joint Future Prediction with Interactive Multi-Agent Modeling*（直接竞争方法，同为 Waymo 出品，mAP 略优于 MotionDiffuser）
  - Ettinger et al. 2021, *SceneTransformer*（另一个多 agent 联合预测 baseline）
