# Wayformer: Motion Forecasting via Simple & Efficient Attention Networks（Nayakanti et al., Waymo, 2022）

一句话总结：Wayformer 提出一个简洁同质化的 attention 架构家族，通过系统对比 Late/Early/Hierarchical 三种模态融合策略和 Factorized/Latent Query 两种提速手段，发现最简单的 Early Fusion 在多数配置下效果最好，在 WOMD 和 Argoverse 双榜上达到 SOTA，arXiv 2207.05844。

## 基本信息

- 论文：Wayformer: Motion Forecasting via Simple & Efficient Attention Networks
- 作者：Nigamaa Nayakanti\*、Rami Al-Rfou\*、Aurick Zhou、Kratarth Goel、Khaled S. Refaat、Benjamin Sapp（\* 同等贡献）
- 机构：Waymo
- arXiv：2207.05844（2022-07）
- 发表：ICRA 2023

---

## 核心问题

运动预测的输入是异质多模态的，来自不同信息源、维度不一致：

**Agent History** `[A, T, D_h]`
- A：场景中需要预测的 **interested agent 数量**（WOMD 每个场景指定 8 个感兴趣的 agent）。**这里的 agent 不是自车，而是场景里要被预测的他车/行人**——每个 agent 用以自己为中心的坐标系独立做一次预测，得到 A 组独立的预测结果。类比：不是"batch 是 8 辆车"，而是"这个场景里我需要预测 8 个目标，每个目标各预测一次"
- T：历史帧数，WOMD 是 10 帧（1 秒，10Hz），Argoverse 2 是 50 帧（5 秒）
- D_h：每帧的状态特征维度，包含：位置(x,y)、速度(vx,vy)、加速度(ax,ay)、朝向、尺寸(长/宽/高)、类型(车/人/自行车) 等，约 10-15 维

**Agent Interactions** `[A, T, S_i, D_i]`
- S_i：每个 agent 周围的 context agent 数量（固定取最近的 S_i 个，通常 8-16）
- D_i：context agent 的状态特征，和 D_h 类似但转换到以该 agent 为中心的相对坐标系

> **interactions 里的数据会有重复吗？** 会，但这是 ego-centric 设计的代价。如果 agent-1 和 agent-2 互相是对方的邻居，那 agent-1 的 interactions 包含 agent-2 的数据，agent-2 的 interactions 也包含 agent-1 的数据——相同的绝对状态被编码了两次，但用的是不同的相对坐标系。这是 Wayformer 局限之一（Section 7 明确指出），UniAD 的 scene-centric 设计正是为了解决这个问题。

**Road Graph** `[A, 1, S_r, D_r]`
- S_r：距离每个 agent 最近的道路段数量（通常取最近的 S_r=128 段）
- D_r：每个道路段的特征：起点(x,y)、终点(x,y)、道路类型（直道/弯道/路口）等，约 13 维
- 时间维度为 1：道路是静态的，没有时序变化

**Traffic Light State** `[A, T, S_tls, D_tls]`
- S_tls：每个 agent 附近的交通灯数量（通常取最近的 S_tls=16 个）
- D_tls：每个交通灯的状态特征：位置、信号状态（红/黄/绿/未知）、置信度

**输入规模估算**（WOMD 配置，A=8，T=10）：
```
Agent History:      [8, 10, 1, 12]  → 8 个 agent × 10 帧 × 12 维
Agent Interactions: [8, 10, 8, 12]  → 8 个 agent × 10 帧 × 8 个邻居 × 12 维
Road Graph:         [8, 1,  128, 13] → 8 个 agent × 128 条道路段 × 13 维
Traffic Light:      [8, 10, 16, 10]  → 8 个 agent × 10 帧 × 16 个信号灯 × 10 维
```

### 如何统一成 Transformer 的输入格式

Transformer 期望的输入格式是 `[batch, seq_len, d_model]`（批次 × 序列长度 × 特征维度）。

四种模态的"特征维度"不同（12、12、13、10），"序列结构"也不同（有的有时序，有的没有）。统一的步骤：

**第一步：线性投影（Projection Layer）**

每种模态有自己的线性层，把最后一维映射到公共维度 D（论文中 D=64/128/256 可配）：

```python
# 以 Agent History 为例
x_history: [A, T, S_h, D_h]  # S_h=1，去掉这维得 [A, T, D_h]
projected = Linear(D_h, D)(x_history)  # → [A, T, D]
```

**第二步：展平为序列**

把时序和空间维度都展平成一个序列维度：

```python
# Agent History: [A, T, D] → 每个 agent 得到 T 个 token
# Road Graph:    [A, S_r, D] → 每个 agent 得到 S_r 个 token
# Agent Interactions: [A, T×S_i, D] → 每个 agent 得到 T×S_i 个 token
```

**第三步：添加 Positional Embedding**

每个 token 加上位置编码，告诉模型这个 token 在时序或空间中的位置（见下方详细说明）。

**第四步：拼接（Early Fusion）或分组处理（Late/Hierarchical Fusion）**

Early Fusion：把所有模态的 token 直接拼接：
```python
tokens = concat([history_tokens, interaction_tokens, roadgraph_tokens, tls_tokens])
# 总 token 数 = T + T×S_i + S_r + T×S_tls
# 例：10 + 80 + 128 + 160 = 378 个 token，每个 D 维
# 输入 Transformer: [A, 378, D]  （A 个 agent 各自处理一次）
```

---

## 方法：三种 Fusion 策略详解

### Late Fusion（晚融合）

每种模态有自己独立的 attention encoder，分别编码后才合并。

```python
# 输入：4 种模态，各自展平为序列
history_enc    = AttentionEncoder_H(history_tokens)     # [A, T, D]
interact_enc   = AttentionEncoder_I(interact_tokens)    # [A, T*S_i, D]
roadgraph_enc  = AttentionEncoder_R(roadgraph_tokens)   # [A, S_r, D]
tls_enc        = AttentionEncoder_TLS(tls_tokens)       # [A, T*S_tls, D]

# 拼接后送给 decoder
scene_enc = concat([history_enc, interact_enc, roadgraph_enc, tls_enc])
# [A, T + T*S_i + S_r + T*S_tls, D]

# Trajectory Decoder cross-attend to scene_enc
trajectories = TrajectoryDecoder(scene_enc)  # [A, K, T_future, 4]
```

**特点**：模态间信息交流发生在 Decoder 的 cross-attention 阶段，Encoder 各自独立。计算量最低，但跨模态的联合理解最弱（比如"交通灯变红时，前方车辆可能减速"这种跨模态推理只能靠 Decoder 做）。

---

### Early Fusion（早融合）

所有模态直接拼成一个序列，只用一个 Cross-Modal Encoder。

```python
# 所有模态投影后拼接
all_tokens = concat([history_tokens, interact_tokens, roadgraph_tokens, tls_tokens])
# [A, L_total, D]，L_total = T + T*S_i + S_r + T*S_tls ≈ 300-400 个 token

# 单一 Cross-Modal Encoder，所有 token 之间都可以互相 attend
scene_enc = CrossModalEncoder(all_tokens)  # [A, L_total, D]

# Trajectory Decoder
trajectories = TrajectoryDecoder(scene_enc)  # [A, K, T_future, 4]
```

**特点**：最简单，参数最少（只有一个 encoder）。模型从第一层就能做跨模态 attention——交通灯 token 可以直接 attend 到 agent 的历史 token。缺点：token 数最多，计算量最大（self-attention 是 O(L²)）。

**为什么效果反而最好（大模型时）**：Early Fusion 给模型最大自由度去发现跨模态的关联，不需要设计"先看什么后看什么"的顺序。Late Fusion 的分开编码引入了架构偏见——模型只能在 Decoder 阶段才能做跨模态推理，而 Decoder 通常比 Encoder 浅。

---

### Hierarchical Fusion（层级融合）

先做 Late Fusion 的各独立模态编码，再用一个 Cross-Modal Encoder 融合。

```python
# Stage 1：各模态独立编码（和 Late Fusion 一样）
history_enc   = AttentionEncoder_H(history_tokens)
interact_enc  = AttentionEncoder_I(interact_tokens)
roadgraph_enc = AttentionEncoder_R(roadgraph_tokens)
tls_enc       = AttentionEncoder_TLS(tls_tokens)

# Stage 2：拼接后再过一个 Cross-Modal Encoder
combined = concat([history_enc, interact_enc, roadgraph_enc, tls_enc])
scene_enc = CrossModalEncoder(combined)  # [A, L_total, D]

# Trajectory Decoder
trajectories = TrajectoryDecoder(scene_enc)  # [A, K, T_future, 4]
```

**特点**：Encoder 深度被分配给"模态内编码"和"跨模态融合"两部分。在中等延迟预算（16-32ms）时效果最好——既有一定的模态内特征提取，又有跨模态交互。

---

## Positional Embedding 详解

### 是什么

Transformer 的 self-attention 是排列不变的（permutation equivariant）——把输入序列打乱顺序，得到的输出也只是相应打乱，不感知"谁在前面谁在后面"。

但驾驶场景里顺序有意义：agent 在时序上的轨迹（1秒前、0.5秒前、现在）是有因果关系的；道路段按行驶方向排列也有意义。Positional Embedding 把位置信息加进每个 token，让模型能利用顺序。

具体做法：给每个 token 加一个和位置相关的向量：
```
token_with_pos[t] = token[t] + PE[t]
```

### 0 初始化 vs 正弦/余弦

**正弦/余弦 PE（原始 Transformer 的做法）**：固定的数学函数，$PE[t][2i] = \sin(t / 10000^{2i/D})$，$PE[t][2i+1] = \cos(...)$。在 NLP 里效果好，因为词的位置（第几个词）是明确有意义的。

**可学习 PE（0 初始化）**：每个位置有一个可学习的向量，从 0 开始训练。Wayformer 的论文提到这样做让模型自己决定要不要使用位置信息（如果某个模态不需要顺序，PE 就学到接近 0 的值，相当于"关掉"）。

**论文的消融**：论文没有对 PE 做单独的消融（没有"去掉 PE 和加上 PE 的对比实验"），但理论依据是：Road Graph（静态道路段）对顺序不敏感，Agent History（时序轨迹）对顺序敏感——用 0 初始化让模型自适应，而不是手动决定哪些模态需要 PE。

**和 NLP 的区别**：NLP 里"第 5 个词"的位置是绝对有意义的（句子结构依赖词序）；驾驶场景里"第 3 帧历史"和"附近第 4 条车道"的"位置"是完全不同类型的信息，用统一的正弦函数不合适——可学习的 PE 更灵活。

---

## Trajectory Decoder：K 条轨迹是什么

Decoder 输出的不是一条轨迹，而是 **K 个 Gaussian 分量**（GMM，高斯混合模型），每个分量代表一种可能的未来。

**K 条轨迹的来源**：K 个可学习的初始向量（learned seeds），每个 seed 是一个 D 维向量，随机初始化，训练后学到特定含义。

**"锚点"是什么意思**：这里"锚点"只是一个直觉比喻，不是真正的空间坐标。seed 本质是一个抽象的"意图向量"——训练过程中，通过 loss 的驱动，不同的 seed 会被迫分化成不同的"专家"：某个 seed 对应直行、某个对应左转、某个对应减速停车等。这不是人为规定的，是 loss 自然驱动的结果。

**loss 是如何驱动这些 seed 分化的**：每次训练，找 K 条预测中和 GT 最近的那条，只对**那一条**计算回归 loss（让它更接近 GT）和分类 loss（提高它的概率）。其他 K-1 条不受惩罚。效果：哪个 seed 碰巧离 GT 近，它就被强化，逐渐专门化为那种运动模式。不同场景里 GT 各不相同，所以不同 seed 各自被不同场景强化，最终分化成不同的"意图专家"。

这种机制叫 **Winner-Takes-All（WTA）**：每次只更新"赢家"（最近的那条），让每个 seed 各司其职，避免所有 seed 都学到同一个平均轨迹。

```python
# K 个可学习的 seed 向量
seeds: [K, D]  # 随机初始化，训练时学习

# 每个 seed cross-attend to scene encoding
for i in range(K):
    seed_i = cross_attention(seeds[i], scene_enc)   # [D]
    
# 解码为轨迹：每条轨迹是 T_future 个高斯分布
traj_i = Linear(D, T_future * 4)(seed_i)  # 每步输出 (μ_x, μ_y, σ_x, σ_y)
prob_i = Linear(D, 1)(seed_i)             # 这条轨迹的概率（未归一化 log-prob）
```

**GMM 和高斯分量是什么**：每条预测轨迹不是一条确定的线，而是 T 个时间步的概率分布——每步输出一个 2D 高斯分布 $\mathcal{N}(\mu_x, \mu_y, \sigma_x, \sigma_y)$，表示"agent 在这一步最可能在 $(\mu_x, \mu_y)$ 附近，不确定性是 $(\sigma_x, \sigma_y)$"。整条轨迹是 T 个高斯的序列。K 条轨迹就是 K 个这样的序列，形成混合高斯模型（GMM）：

```
第 k 条轨迹 = [N(μ_x1,μ_y1,σ_x1,σ_y1), ..., N(μ_xT,μ_yT,σ_xT,σ_yT)]  （T 步）
概率 p_k        ← 这条轨迹被选中的概率
```

"轨迹点"就是每步高斯分布的均值 $(\mu_x, \mu_y)$，不确定性 $(\sigma)$ 描述模型对这步预测有多自信。

**K=6 的含义**：对于评测（WOMD/Argoverse 标准），只取 K=6 条轨迹参与计算。训练时用 K=64 个 Gaussian 分量，通过 trajectory aggregation 压缩到 6 条。

**Trajectory Aggregation 是什么**：K=64 条轨迹里有很多是冗余的——比如 10 条都在预测"直行到 30 米外"，它们终点很接近，拿 6 个名额来重复描述同一件事是浪费。Trajectory Aggregation 是一种聚类+筛选操作：

```
1. 把 K=64 条轨迹按终点位置聚类
2. 对每个聚类，选置信度最高的那条作为代表
3. 选出的代表轨迹数量超过 6 条时，继续按距离阈值合并
4. 最终保留 6 条，各自覆盖不同的方向/距离区域
```

效果：6 条轨迹尽量"分散"，每条代表不同的运动模式，而不是多条聚集在同一区域。这样 minFDE 的评测（找 6 条里最近的那条）才有意义——6 条各自覆盖不同区域，总有一条能靠近 GT，而不是 6 条都在同一个方向上堆叠。

**GT（Ground Truth）是什么**：GT 轨迹是数据集里记录的真实车辆在未来时间步的真实位置序列——传感器实际观测到的轨迹，通过后处理标注得到。GT 是"真实发生的那条轨迹"。

**GT 一定是对的吗**：不是。有几个层面的问题：
- **标注误差**：传感器精度和标注算法有误差，记录的轨迹不是真实物理位置的完美复现
- **未来的不确定性**：同一个场景再来一遍，驾驶员可能做出不同决策（选择直行或转弯），GT 只是"这次实际发生的那种"，不是"唯一正确的选择"
- **多模态性**：这也是 minADE/minFDE 用 min-over-K 而不是 mean-over-K 的根本原因——真实未来有多种合理可能，评测只能判断"K 条里最接近实际发生的那条有多近"，而不能说"其他 K-1 条都是错的"

**Loss 计算**：训练时找 K 条中和 GT 最接近的那条（最优匹配，Winner-Takes-All），计算两部分 loss：
- 分类 loss：提高最优匹配那条的概率（log-likelihood of the correct mode）
- 回归 loss：让最优匹配那条的轨迹更接近 GT（Gaussian negative log-likelihood）

---

## 训练 Loss

Wayformer 的 loss 分两部分，对应 Decoder 的两个输出：

**分类 Loss（选哪条 GMM 分量）**：

K 条轨迹各有一个预测概率 $p_1, \ldots, p_K$（softmax 归一化）。训练时找和 GT 距离最近的那条，记为第 $i^*$ 条，然后最大化它的对数概率：

```
L_cls = -log(p_{i*})
```

等价于交叉熵 loss，目标是让模型给"最好的那条"更高的概率。

**回归 Loss（最好那条的轨迹质量）**：

对第 $i^*$ 条轨迹，每步输出一个 2D 高斯分布，用 GT 轨迹计算负对数似然：

```
L_reg = -Σ_t log N(y_t | μ_t, Σ_t)
      = Σ_t [ (y_t - μ_t)^T Σ_t^{-1} (y_t - μ_t) + log|Σ_t| ]
```

其中 $y_t$ 是 GT 在第 $t$ 步的真实位置，$(\mu_t, \Sigma_t)$ 是模型预测的高斯分布参数。不确定性 $\sigma$ 越小，模型越自信，对偏差的惩罚越大。

**总 Loss**：

```
L = L_cls + λ * L_reg
```

**关键机制——Winner-Takes-All（WTA）**：只有"赢家"（最近的那条 $i^*$）参与 loss 计算，其余 K-1 条不被惩罚。这迫使不同 seed 各自专门化：在不同训练样本里，不同 seed 各自成为"赢家"，逐渐分化为不同的运动模式专家。如果对所有 K 条都计算 loss，所有 seed 会退化成预测相同的"平均轨迹"。

---

## 关键结果

### Benchmark Results（Table 1）

**WOMD 2021（Early Fusion + LQ + Multi-Axis）**：

| 指标 | Wayformer Early Fusion | 次好方法 | 含义 |
|------|----------------------|---------|------|
| minFDE ↓ | **1.126m** | 1.158m（MultiPath++） | 最近条终点误差 |
| minADE ↓ | **0.545m** | 0.556m（MultiPath++） | 最近条全程平均误差 |
| MR ↓ | **0.123** | 0.134（MultiPath++） | 12.3% 场景最近条超阈值 |
| Overlap ↓ | **0.127** | 0.131（MultiPath++） | 预测轨迹碰撞率 |
| mAP ↑ | **0.412** | 0.409（MultiPath++） | 置信度排序质量 |

**Argoverse 2021（Brier-minFDE 为主指标）**：

| 指标 | Wayformer Early Fusion | DCMS（次好） | 含义 |
|------|----------------------|------------|------|
| Brier-minFDE ↓ | **1.7451** | 1.7564 | 距离 + 置信度综合分 |
| MR ↓ | **0.0192** | 0.0194 | 未命中率 |
| minADE ↓ | 0.7672 | 0.7659 | 全程平均误差 |

**Brier-minFDE 是什么**：`(1 - p_best)² + minFDE`，其中 `p_best` 是最接近 GT 那条轨迹的预测概率。同时惩罚距离误差和置信度校准——如果你给最好条的概率很低，Brier 项会增大总分。详见 [Argoverse 文档](./argoverse-motion-forecasting.md)。

**DCMS 是什么**：Diverse Conditional Motion Set，Waymo 内部方法，排行榜匿名提交（无公开论文），被 Wayformer 超越。

### 关键消融结论

**Early Fusion 效果最好**（尤其在大模型上）：
- 低延迟（≤16ms）：Late Fusion 最优（计算省）
- 中延迟（16~32ms）：Hierarchical 有优势
- 高容量/高延迟（>32ms）：Early Fusion 追上甚至超过 Hierarchical
- 随模型容量增大，对融合策略的敏感度下降——最简单的方法在足够大的模型上也足够好

**Factorized Attention** 提速 Late Fusion 明显，对 Early/Hierarchical 提速有限。

**Latent Queries** 对全部融合策略都能 2-16× 提速，几乎无质量损失，是最值得使用的加速手段。

---

## 附录：评测指标定义

| 指标 | 计算方式 | 评判什么 | 越小/大越好 |
|------|---------|---------|-----------|
| **minADE** | K 条中最近的那条，全程每步欧氏距离均值 | 最佳条的整体轨迹准确性 | ↓ 越小越好 |
| **minFDE** | K 条中最近的那条，终点欧氏距离 | 最佳条的长期意图准确性 | ↓ 越小越好 |
| **MR（Miss Rate）** | 最近条终点误差超阈值的场景比例（WOMD: 2m，Argoverse: 速度自适应） | 真正困难场景的覆盖率 | ↓ 越小越好 |
| **mAP（WOMD）** | 各轨迹类型的 Average Precision，按置信度排序 | 置信度校准质量 | ↑ 越大越好 |
| **Brier-minFDE** | `(1-p_best)² + minFDE`，p_best=最好条的预测概率 | 距离+置信度综合 | ↓ 越小越好 |
| **Overlap** | 预测轨迹和其他 agent 真实位置发生碰撞的比例 | 轨迹的物理合理性 | ↓ 越小越好 |

详细定义见 [自动驾驶模型评测全栈概览](../00-overview/av-model-evaluation.md)。

---

## 局限性

1. **Ego-centric 建模的重复计算**：每个 agent 做一次完整编码，密集场景中 N 个 agent 需要 N 次前向传播
2. **稀疏抽象场景描述**：输入是折线段 + 状态向量，缺少视觉细节（行人姿态、车轮角度等）
3. **每个 agent 独立预测**：各 agent 的未来轨迹独立生成，无法建模多 agent 之间的因果依赖

---

## 现状与影响

**一句话定性**：Wayformer 是运动预测领域"transformer 统一化"方向的代表作，提供了系统的融合策略分析和效率优化工具箱，Early Fusion + Latent Query 的配置成为后续工作的参考基线。

- **MTR（Motion Transformer，NeurIPS 2023）**：进一步引入 motion query pair（motion intention priors），在 WOMD 2023 上超越 Wayformer
- **MotionDiffuser（2023）**：用扩散模型替换 GMM decoder，复用 Wayformer 风格的 scene encoder
- **WOMD Leaderboard**：发布时双榜 SOTA，2023 年后被 MTR 系列超越，框架思路持续被引用

---

## 和 wiki 内其他概念的关联

- [WOMD](./waymo-open-motion-dataset.md)：Wayformer 的主要训练和评测数据集
- [Argoverse Motion Forecasting](./argoverse-motion-forecasting.md)：Argoverse benchmark 详细介绍，Brier-minFDE 指标来源
- [自动驾驶模型评测全栈概览](../00-overview/av-model-evaluation.md)：运动预测指标体系，Wayformer 是核心参考点
- [TrafficGen](./trafficgen-2210.06609.md)：同样使用向量化 HD map 表示，场景编码思路相近
- [NAVSIM](./navsim-2406.15349.md)：NAVSIM 中 Wayformer 风格模型被横向比较

## 值得看的部分 / 相关资料

- **Section 3（Wayformer）全文**：Figure 2/3 是理解三种 Fusion + 两种加速的核心
- **Figure 4（效率-质量 Pareto 曲线）**：不同延迟预算下各策略的优劣，工程参考价值高
- **Table 1（Benchmark Results）**：WOMD 和 Argoverse 双榜详细对比
- MTR（arXiv:2209.13508，NeurIPS 2023）：Wayformer 的直接后继
- MultiPath++（arXiv:2111.14973）：Table 1 中主要竞争对手，同为 Waymo 工作
- VectorNet（CVPR 2020）：向量化场景表示的奠基工作
