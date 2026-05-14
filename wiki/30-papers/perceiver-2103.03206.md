# Perceiver: General Perception with Iterative Attention（Jaegle et al., 2021）

一句话总结：Perceiver 用一个小的潜在数组（latent array）通过 cross-attention 从超高维输入中反复蒸馏信息，将 Transformer 的二次方复杂度降到线性，无需任何领域专用架构即可处理图像、音频、视频、点云等任意模态，ICML 2021。

## 基本信息

- 论文：Perceiver: General Perception with Iterative Attention
- 作者：Andrew Jaegle, Felix Gimeno, Andrew Brock, Andrew Zisserman, Oriol Vinyals, Joao Carreira
- 机构：DeepMind（伦敦）
- 发表：ICML 2021（PMLR 139）
- arXiv：2103.03206

## 核心问题

**Transformer 的二次方复杂度墙**：标准 Transformer 的 self-attention 复杂度是 $O(M^2)$，$M$ 是序列长度。对图像来说 $M$ = 像素数：224×224 = 50,176；对 1 秒 48kHz 音频 $M$ ≈ 50,000；对 32 帧 224×224 视频 $M$ > 1.6M。直接做 self-attention 完全不可行。

**现有解法的代价：领域专用结构**：
- 图像用 CNN（假设 2D 局部性、平移不变性）
- 音频用 1D 卷积或 LSTM
- 点云用 PointNet
- ViT 靠先把图像切成 16×16 patch（相当于用 2D 结构把 $M$ 从 50k 压到 ≈200）

每换一种模态就要重新设计架构，"模型架构成了输入数据的函数"。

**核心洞察**：不要把所有 $M$ 个输入都互相 attend，而是引入一个小的**潜在数组**（$N$ 个 latent，$N \ll M$），让 latent 去 attend 输入（cross-attention，复杂度 $O(MN)$），然后在 latent 内部做 self-attention（$O(N^2)$，因为 $N$ 小所以便宜），整体复杂度降为 $O(MN + LN^2)$，与输入大小解耦。

---

## 方法：两组件架构

### 组件 1：Cross-Attention（输入 → Latent）

```
Byte array (M×C)  ←── 输入：M个位置，每个C维（像素/音频帧/点云等）
      ↓
  Cross-Attention
      ↓
Latent array (N×D) ──── 潜在数组：N个latent，每个D维（N << M）
```

- **Q**：来自 latent array（$N \times D$）
- **K, V**：来自 byte array（$M \times C$）
- 复杂度：$O(MN)$，线性于输入大小 $M$

这一步把 $M$ 维的输入"压缩"进 $N$ 维的 latent，类似信息瓶颈——latent 选择性地关注输入中最相关的部分。

### 组件 2：Latent Transformer（Latent 内部）

标准 Transformer self-attention，在 $N$ 个 latent 之间做：
- 复杂度：$O(N^2)$，因为 $N$（通常 512）远小于 $M$（50k+），这一步很便宜
- 使用 GPT-2 架构的 Transformer block（decoder-style，无 mask）
- 可叠加很多层（论文 ImageNet 实验用 48 层 latent Transformer block）

### 迭代 Cross-Attention（关键设计）

最简单的版本只做一次 cross-attention（输入 → latent）再做 latent 内 self-attention。但潜在数组容量有限，可能"读不够"输入。

**解法**：交替重复 cross-attention 和 latent Transformer 多次（图 1 中的虚线循环）：

```
Input → [Cross-Attn → Latent Transformer] × 重复次数 → Output
```

**权重共享**：除第一个 cross-attention 外，后续所有 cross-attention 和 latent Transformer 的权重可以**共享**。共享权重相当于把模型看成一个 RNN——在深度方向展开，但每步用同一组参数。好处：参数量减少约 10 倍，同时缓解过拟合，ImageNet 上验证性能更好。

### 位置编码：Fourier Feature

Attention 是置换不变的（天然不区分位置），所以必须显式注入位置信息。Perceiver 用**Fourier 特征**：

$$[\sin(f_k \pi x_d),\ \cos(f_k \pi x_d)]$$

其中 $x_d \in [-1, 1]$ 是第 $d$ 维的位置坐标，$f_k$ 是均匀分布的频率带。

Fourier 特征的优势：
1. **不假设具体结构**：$x_d$ 可以是图像的 (x,y)、音频的时间轴、视频的 (x,y,t)，统一处理
2. **可学习或固定**：实验中两种都可以，Fourier 固定特征在大多数实验中效果更好
3. **多模态直接拼接**：多模态输入（图像+音频）各自构建自己维度的 Fourier 特征，再用 learned modality embedding 区分来源，直接拼接即可

与 NeRF 的位置编码完全相同的设计（都用 Fourier 特征编码空间位置）。

---

## 关键结果 / 数据

### ImageNet 图像分类（Table 1 & 2）

| 模型 | 是否用 2D 卷积 | Top-1 精度（Fourier 特征输入）| Permuted Top-1 |
|---|---|---|---|
| ResNet-50 | ✓（固有假设）| 73.5% | 39.4% |
| ViT-B-16 | ✓（patch projection）| 76.7% | 61.7% |
| Transformer（64×64）| ✗ | 57.0% | 57.0% |
| **Perceiver** | ✗ | **78.0%** | **78.0%** |

关键对比：**Permuted ImageNet**——把每张图的像素随机打乱顺序后再分类。使用 2D 卷积的模型（ResNet-50 从 73.5% 跌到 39.4%，ViT 从 76.7% 跌到 61.7%）性能大幅下降，因为它们依赖空间局部性先验。**Perceiver 完全不受影响**（78.0% → 78.0%），证明它不依赖 2D 结构，位置信息完全由 Fourier 特征显式提供。

模型规模：约 **45M 参数**，与 ResNet-50 相当。

### AudioSet 音频/视频分类（Table 3）

| 模型 / 输入 | 音频 mAP | 视频 mAP | 音频+视频 mAP |
|---|---|---|---|
| CNN-14（专用架构）| 43.1 | — | — |
| Perceiver（raw audio）| 38.3 | 25.8 | 43.5 |
| Perceiver（mel spectrogram）| 38.4 | 25.8 | 43.2 |
| Perceiver（mel - tuned）| — | — | **44.2** |
| Attention AV-fusion | 38.4 | 25.7 | 46.2 |

- 单模态下 Perceiver 和专用 CNN 相当（38.3 vs 43.1，差距来自 CNN-14 用了额外 class balancing 和 AugMix）
- **同一模型无需改动**处理 raw audio、mel spectrogram、video、audio+video——展示了架构通用性

### ModelNet-40 点云分类（Table 4）

| 模型 | 精度 |
|---|---|
| PointNet++（专用，含额外几何特征）| 91.9% |
| ViT-B-2（FF）| 66.3% |
| **Perceiver** | **85.7%** |

Perceiver 将点云直接展平成 2D 坐标网格输入，85.7% vs 专用 PointNet++ 的 91.9%，差距主要来自 PointNet++ 使用了额外的几何特征工程（face normals 等）。

---

## 局限性

论文 Section 5 明确：

- **模态特定位置编码仍然需要**：虽然不需要领域专用架构，但仍需要为每种模态手动设计合适的 Fourier 特征维度（图像 2D、视频 3D、音频 1D），并非完全无先验知识
- **ImageNet 结果未超越 SOTA**：78.0% 和 ResNet-50/ViT 相当但没有超越（当时 SOTA 约 86.5%），表明通用性有代价
- **过拟合倾向**：在 ImageNet 这种中等数据集上，不加权重共享的 Perceiver 过拟合严重，需要权重共享或更大数据集缓解
- **训练效率**：使用 JAX，实验在 DeepMind JAX Ecosystem 上运行，相比 PyTorch 生态复现难度稍高（当时）

---

## 现状与影响

一句话定性：**Perceiver 是"无先验通用感知架构"路线的奠基工作——它证明了"去掉领域特定归纳偏置，用 cross-attention bottleneck 处理任意模态"这条路是可行的；直接后续 Perceiver IO（2021）把输出也泛化为任意结构，成为 DeepMind 多个后续工作（Gato、Flamingo 的架构灵感来源）的基础；但在单一模态任务上仍不如专用模型，2022 年后被更高效的方案部分取代。**

- **Perceiver IO**（Jaegle et al., 2021, NeurIPS 2021）：直接延伸——把输出也用 cross-attention 泛化（query 由任务决定），支持 optical flow、多任务 multi-modal 等任意输出结构，是 Perceiver 的完整版
- **Gato**（DeepMind 2022）：通用 agent 模型，把动作序列、图像、文本等所有模态编码成 token 序列，Perceiver 的"任意模态统一"思想直接影响了 Gato 的设计哲学
- **Flamingo**（DeepMind 2022）：视觉语言模型，用 cross-attention 把视觉特征注入冻结 LLM，cross-attention bottleneck 的设计和 Perceiver 一脉相承
- **MAE/BEiT 等自监督方向**（2022）：ImageNet 上把精度推到 87%+ 的方法基本都依赖大规模预训练，而非 Perceiver 的架构设计，Perceiver 在纯精度竞争中被甩开
- **影响总结**：Perceiver 的核心贡献是**概念层面**——"latent cross-attention bottleneck 可以打破模态壁垒"——而非某个任务的 SOTA，这个概念被后来的 Perceiver IO、Gato、Flamingo 等持续引用和扩展

---

## 和 wiki 内其他概念的关联

- [GNN](../20-concepts/gnn.md)：两者都处理"非规则结构数据"的挑战。GNN 通过图拓扑传播信息，Perceiver 通过 cross-attention bottleneck 抽象输入结构；PointNet 对点云的处理（置换不变性）和 Perceiver 的置换不变 attention 是同一思路不同实现
- [图神经网络 vs Transformer](../20-concepts/gnn.md)：Perceiver 的 latent Transformer 是在全连接 latent 图上的 self-attention，可以理解为 GNN 的特例
- [TrafficGen](./trafficgen-2210.06609.md)：TrafficGen 的 MCG 编码器（$O(I)$ 线性复杂度）和 Perceiver 的 cross-attention bottleneck 解决了同一问题——用少量全局 context 来近似 $O(I^2)$ 的全量 attention
- [Wayformer](./wayformer-2207.05844.md)：Wayformer 的多模态融合（agent 轨迹 + 地图）在思路上是 Perceiver 在 AV 领域的工程化：用统一 Transformer 处理不同类型的输入，在 latent 空间融合
- [MotionDiffuser](./motiondiffuser-2306.03083.md)：MotionDiffuser 的 denoiser 用 cross-attention 把 condition tokens 注入 noisy trajectories，是 Perceiver cross-attention 的精确实例

---

## 值得看的部分 / 相关资料

- **Section 3.1（架构总览）+ Figure 1**：cross-attention + latent Transformer 的完整流程图，架构一目了然；特别是"Uncoupling depth from input size"段落，解释了为什么 bottleneck 能让模型加深而不增加计算
- **Table 2（Permuted ImageNet）**：最有力的一组实验——对像素打乱顺序，ResNet 崩溃，Perceiver 完全不受影响，直观证明"不依赖 2D 先验"
- **Section 3.2（Fourier 位置编码）**：与 NeRF 位置编码的关系，以及为什么位置编码不破坏"通用性"宣称
- **Appendix C（位置编码细节）**、**Appendix E（AudioSet 注意力可视化）**：工程细节丰富
- 后续阅读：
  - Jaegle et al. 2021, *Perceiver IO: A General Architecture for Structured Inputs & Outputs*（arXiv:2107.14795, NeurIPS 2021）——Perceiver 的直接升级版，输出也泛化
  - Reed et al. 2022, *A Generalist Agent (Gato)*（arXiv:2205.06175）——Perceiver 通用感知理念的极致延伸
  - Alayrac et al. 2022, *Flamingo: a Visual Language Model*（arXiv:2204.14198）——cross-attention 把视觉注入 LLM
