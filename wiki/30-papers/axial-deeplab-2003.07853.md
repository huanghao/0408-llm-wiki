# Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation（Wang et al., 2020）

一句话总结：Axial-DeepLab 把 2D self-attention 分解成两个连续的 1D axial-attention（沿高和宽轴各做一次），并引入 position-sensitive 位置编码，用更低的计算复杂度实现全局感受野，在 COCO/Mapillary Vistas/Cityscapes 全景分割上超过当时 bottom-up SOTA，ECCV 2020。

## 基本信息

- 论文：Axial-DeepLab: Stand-Alone Axial-Attention for Panoptic Segmentation
- 作者：Huiyu Wang†、Yukun Zhu、Bradley Green、Hartwig Adam、Alan Yuille、Liang-Chieh Chen（†工作完成于 Google 实习期间）
- 机构：Johns Hopkins University + Google Research
- 发表：ECCV 2020
- arXiv：2003.07853
- 代码：https://github.com/csrhddlam/axial-deeplab

---

## 核心问题

**和 CLIP/ViT 的对比（先定位）**：

CLIP/ViT 把图像切成固定大小的 patch（如 16×16 像素），每个 patch 变成一个 token，然后对所有 token 做全局 self-attention。这对应于：h/16 × w/16 个 token，序列长度相对较短（224×224 的图像只有 196 个 token），可以做全局 attention。

Axial-DeepLab 工作在**更高分辨率的特征图**上（如 ResNet 内部的 56×56，每个"token"是特征图上的一个空间位置），而不是先缩小到 patch 级别。这样能保留更细粒度的空间信息，有利于分割任务，但序列长度 = 56×56 = 3136 个 token，做全局 attention 太贵。

---

**卷积的局限**：感受野受 kernel size 限制，3×3 卷积只看 3×3 的邻域，难以捕获长程依赖。

**标准 2D self-attention 的问题**：

变量说明：
- $h, w$：特征图的高和宽（ResNet 第一阶段 $h=w=56$，第二阶段 28，第三阶段 14）
- 每个空间位置是一个"token"，序列长度 = $h \times w$

复杂度 $O(h^2 w^2)$：$h=w=56$ 时序列长度 = 3136，attention 矩阵大小 = $3136^2 \approx 9.8M$，不可接受。

**Local attention 的局限**：

把每个位置的感受野限制在 $m \times m$ 的局部窗口内（$m$ 是窗口边长，如 $m=7$），复杂度降到 $O(hw \cdot m^2)$。

数字对比：$h=w=56, m=7$：
- 全局 attention：$56^4 \approx 9.8M$
- Local attention（$m=7$）：$56^2 \times 7^2 = 3136 \times 49 \approx 0.15M$，**降低 64 倍**

但代价是感受野缩回 $7 \times 7$，和 7×7 卷积一样局部——失去了使用 attention 的初衷。

**Axial-DeepLab 的解法**：把 2D attention 分解成两个 1D attention：先沿**宽轴**做一次（每行像素互相 attend，序列长 $w$），再沿**高轴**做一次（每列像素互相 attend，序列长 $h$）。

数字对比（$h=w=56$）：
- 全局 2D attention：$56^4 \approx 9.8M$
- Axial attention：$56^2 \times (56 + 56) = 3136 \times 112 \approx 0.35M$，**降低 28 倍，且保留全局感受野**

---

## 方法：Position-Sensitive Axial-Attention

### Position-Sensitive Self-Attention

**"加了 PE 还说不关心位置"的矛盾**：

标准做法是在 token embedding 上加 PE（$x_{\text{new}} = x + \text{PE}$），然后用 $x_{\text{new}}$ 计算 Q/K/V。PE 确实让每个 token 的 embedding 包含了位置信息，但 attention score 是 $Q \cdot K^T$——这个点积衡量的是两个 token 在**内容上**的相似度。

如果两个位置（比如图像左上角和右下角）的内容相似（都是蓝色的天空），它们的 Q·K 点积会很高，attention 权重会大——**不管它们距离有多远**。换句话说，PE 让模型"知道自己在哪里"，但 attention 机制本身没有"靠近的更应该 attend"的偏好，**远处内容相似的 token 和近处的 token 会得到相同的关注度**。

对图像分割来说这是个问题：边缘检测、形状感知需要位置相关的注意——"这个像素和它正上方的像素的关系"和"这个像素和它斜对角 20 个像素的关系"应该不同，但标准 attention 分不清。

论文的解法是在 **attention score 计算本身**里加入相对位置项（$q_o^T r^q_{p-o}$），让"两个位置的相对距离"直接影响 attention 权重，而不是只通过内容相似度决定。

标准 self-attention 没有利用位置信息（每个 query 只关注内容相似度，不知道"哪个位置"）。论文提出在 attention score 和 value 里都加入相对位置编码：

$$y_o = \sum_{p \in \mathcal{N}_{1 \times m}(o)} \text{softmax}_p\left(q_o^T k_p + q_o^T r^q_{p-o} + k_p^T r^k_{p-o}\right)\left(v_p + r^v_{p-o}\right)$$

- $q_o, k_p, v_p$：query、key、value（标准 attention 的三个向量）
- $r^q_{p-o}$：query 侧的相对位置编码（位置 $p$ 相对于 $o$ 的偏移）
- $r^k_{p-o}$：key 侧的相对位置编码
- $r^v_{p-o}$：value 侧的相对位置编码（让输出包含位置信息）

**三种位置编码的作用**：
- $q_o^T r^q_{p-o}$：query 感知相对位置（"我在哪里，想关注什么方向"）
- $k_p^T r^k_{p-o}$：key 感知相对位置（"我在哪里，别人找我时考虑距离"）
- $v_p + r^v_{p-o}$：输出包含位置信息（"我不只聚合内容，也记住从哪里来"）

所有位置编码都是**可学习的相对位置编码**，参数量少（在多个 head 间共享），计算开销边际。

### Axial-Attention Block

一个 axial-attention block 由两个 1D attention 层串联组成（Figure 2）：

```
输入 x: [H, W, C]
  ↓ 沿高轴（H 轴）做 1D position-sensitive attention
  x': [H, W, C]   每列的像素互相 attend，跨越整个高度 H
  ↓ 沿宽轴（W 轴）做 1D position-sensitive attention
  z: [H, W, C]   每行的像素互相 attend，跨越整个宽度 W
  ↓ 残差连接 + 1×1 conv
输出: [H, W, C']
```

两次 1D attention 的组合使得**任意两个像素之间都可以通过最多两步互相影响**（先在同行 attend，再在同列 attend），实现全局感受野。

### Axial-ResNet 和 Axial-DeepLab

**Axial-ResNet**：把 ResNet 里所有 3×3 卷积替换成 axial-attention block，其余不变。对应有 conv-stem（保留第一层 7×7 conv）和 full axial-ResNet（所有层都用 attention）两个版本。

**Axial-DeepLab**：在 Axial-ResNet 的基础上，按照 Panoptic-DeepLab 的框架加入双解码器（分别输出语义分割和实例分割），通过 majority voting 合并为全景分割输出。

---

## 关键结果

### ImageNet（表 1）

| 模型 | Top-1 Acc |
|------|----------|
| ResNet-50 | 78.8% |
| Stand-Alone Attention [68] | 77.6% |
| Conv-Stem + Attention [68] | 79.3% |
| **Axial-ResNet-L（conv-stem）** | **80.4%** |
| **Full Axial-ResNet-L** | **79.4%** |

超过同规模的所有 stand-alone attention 方法，Axial-ResNet-L 是当时 stand-alone attention 模型的 ImageNet SOTA。

### COCO 全景分割（表 2）

| 模型 | PQ | PQ^th | PQ^st |
|------|-----|--------|--------|
| Panoptic-DeepLab（conv） | 35.1 | 40.6 | 26.8 |
| **Axial-DeepLab（small）** | **37.9** | **44.0** | **28.4** |

Axial-DeepLab-S（small variant）在 COCO test-dev 上比 Panoptic-DeepLab 高 **+2.8% PQ**，是当时 bottom-up 全景分割的 SOTA，同时参数量少 3.8×，计算量少 27×。

### Mapillary Vistas 和 Cityscapes

在 Mapillary Vistas（36.0% PQ）和 Cityscapes（42.8% PQ）上也取得当时 SOTA，验证了 axial-attention 在多个数据集上的泛化性。

---

## 局限性

1. **高分辨率输入代价仍高**：虽然比 2D attention 低很多，但 $O(hw(h+w))$ 在输入分辨率极大时（如 2177×2177 的 Mapillary Vistas 图像）仍然昂贵，论文中对超大分辨率输入将 span $m$ 设为 65 而不是全局，退化为局部 attention
2. **两次 1D attention 无法完全等价于 2D attention**：某些需要对角方向或复杂 2D 空间关系的模式，需要两层的信息传播才能捕获
3. **专为视觉 2D 结构设计**：对文本序列或没有高/宽结构的输入不适用

---

## 现状与影响

**一句话定性**：Axial-DeepLab 是"把 attention 用于高分辨率视觉任务"的关键里程碑——它证明了 axial 分解可以在不牺牲全局感受野的前提下把复杂度降到实用范围，直接影响了后续视频 Transformer（TimeSformer）和驾驶场景模型（Wayformer）的设计。

**直接影响**：
- **TimeSformer（2021）**：把 axial 分解思想用到视频时序×空间维度，成为视频理解的主流架构之一
- **Wayformer（2022）**：把 axial/factorized attention 用于驾驶场景预测，时序×邻居两个维度分轴计算
- **Video Swin Transformer（2022）**：局部时空窗口 attention，受 axial 分解启发
- **已被 ViT 系列工作引用**：作为"如何处理 2D 空间结构 attention"的经典参考

**是否仍是主流**：在图像分割领域，Axial-DeepLab 被后续的 MaskFormer/Mask2Former（query-based 端到端框架）取代，不再是 SOTA。但 axial attention 的思想作为一种通用的**多维 attention 分解技术**，在视频、驾驶、医疗图像等多维结构领域持续被使用。

---

## 和 wiki 内其他概念的关联

- [Attention 优化技术](../20-concepts/attention-optimization.md)：Axial/Factorized Attention 是其中一类，这篇论文是该技术的代表性来源
- [Wayformer](./wayformer-2207.05844.md)：直接使用了 axial/factorized attention 的思想处理时序×邻居维度
- [UniAD](./uniad-2212.10156.md)：端到端 AD，场景 encoder 设计参考了类似的多维 attention 方案

## 值得看的部分 / 相关资料

- **Section 3.1（Position-Sensitive Self-Attention）**：位置编码的三项设计及其直觉
- **Section 3.2（Axial-Attention）**：axial 分解的核心公式和 block 结构（Figure 2）
- **Figure 1**：标准 non-local attention vs position-sensitive axial-attention 的计算图，直观展示差异
- 相关工作：
  - Ho et al.（arXiv:1912.12180）：另一篇更早的 axial attention 论文（图像生成，同时独立提出）
  - TimeSformer（arXiv:2102.05095）：把 axial 分解用到视频时序×空间
  - Wayformer（arXiv:2207.05844）：把 factorized attention 用到驾驶场景预测
