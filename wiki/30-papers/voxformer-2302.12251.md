# VoxFormer: Sparse Voxel Transformer for Camera-based 3D Semantic Scene Completion

**论文**: VoxFormer (arXiv:2302.12251v2)
**作者**: Yiming Li, Zhiding Yu*, Christopher Choy, Chaowei Xiao, Jose M. Alvarez, Sanja Fidler, Chen Feng, Anima Anandkumar（*corresponding）
**机构**: NYU, NVIDIA, ASU, University of Toronto, Vector Institute, Caltech
**发表**: CVPR 2023
**代码**: https://github.com/NVlabs/VoxFormer

---

## 核心问题

**MonoScene 的 2D-to-3D 直接投影会把空 voxel 和遮挡 voxel 错误地赋予视觉特征，导致训练歧义。**

MonoScene 把每个 2D 特征沿光线投影到所有可能的 3D 位置，但被车挡住的空 voxel 也会得到汽车的视觉特征。这产生大量"假正例"：网络需要从含噪的 3D 特征中区分真正有物体的 voxel，造成性能受限。

VoxFormer 的核心洞察：
1. **Reconstruction-before-hallucination**：先重建可见区域，再用可见区域作为起点推断不可见区域，比直接投影稠密特征更可靠
2. **Sparsity-in-3D-space**：3D 空间大部分是空的，用稀疏表示而非稠密表示更高效

---

## 方法 / 核心机制

### 两阶段框架概述

```mermaid
flowchart LR
    A[RGB 图像序列\nI_t, I_{t-1}, ...] --> B[ResNet-50\nFeature Extractor]
    B --> C[2D Feature Maps\nF_2D ∈ R^{b×c×d}]
    
    A --> D[Stage-1: Query Proposal\n深度估计 + 深度矫正]
    D --> E[稀疏 Voxel Queries\nQ_p ∈ R^{N_p×d}\n只保留 occupied voxels]
    
    C --> F[Stage-2: Cross-Attention\nQ_p 向 F_2D 做\nDeformable Cross-Attention]
    E --> F
    F --> G[Q_p 更新为 Q̂_p]
    
    H[Mask Token m ∈ R^d\n未 proposed voxels] --> I[合并: F_3D = Q̂_p + m\n所有 voxel]
    G --> I
    I --> J[Deformable Self-Attention\n完成全部 voxel 特征]
    J --> K[Upsample + Linear\n语义分割]
```

**关键区别**：不把 2D 特征稠密地投影到 3D，而是让 3D voxel queries **主动从 2D 图像 cross-attend**（3D-to-2D cross-attention），只有 occupied voxels 才参与 cross-attention。

---

### Stage-1：Class-Agnostic Query Proposal

目标：找到哪些 voxel 有物体（class-agnostic occupied voxel 集合）。

**Step 1：深度估计**
使用 MobileStereoNet 预测每个像素的深度 Z(u,v)，反投影为 3D 点云：
```
x = (u - cu) × z / fu
y = (v - cv) × z / fv
z = Z(u,v)
```

**Step 2：深度矫正（Occupancy Prediction）**
深度估计在远距离不准，用一个轻量 2D UNet（改自 LMSCNet）对深度点云做 binary voxel occupancy 预测：

```python
# 输入：稠密点云体素化，低分辨率
M_in ∈ {0,1}^{H×W×Z}       # 256×256×32
M_out = Θ_occ(M_in)          # 128×128×16（低分辨率，更 robust）
# M_out 是 binary occupancy map：1=occupied, 0=empty
```

**Step 3：Query Proposal**
从预定义可学习 voxel queries Q ∈ R^{h×w×z×d} 中选出 occupied 的：
```
Q_p = Reshape(Q[M_out])   # Q_p ∈ R^{N_p×d}
# N_p << N_q，大量空 voxel 被丢弃
```

---

### Stage-2：Class-Specific Segmentation

**Deformable Cross-Attention（DCA）**：Q_p 中的每个 proposed voxel query 通过 3D→2D 投影找到其参考点，从 2D 图像特征图中采样视觉特征：

```python
# 对每个 proposed voxel q_p，位置为 (x,y,z)
# 投影到各帧图像坐标系
for camera_frame t in hit_views:
    ref_2d = P(p, t)  # 3D → 2D 投影
    feat_t = deformable_attention(q_p, ref_2d, F_2D_t)

DCA(q_p) = (1/|V_t|) Σ_{t∈V_t} DA(q_p, P(p,t), F_2D_t)
```

**Deformable Self-Attention（DSA）**：将更新后的 Q̂_p 和 mask tokens（用于未 proposed 的 voxel）合并，通过 self-attention 传播信息到所有 voxel：

```python
F_3D = concat(Q̂_p, mask_token_m)  # [N_q, d]，N_q = h×w×z
F̂_3D = DSA(F_3D, F_3D)             # [h×w×z, d]
# mask token m ∈ R^d 是全局可学习参数，每个非proposed voxel共享同一 m
```

**Output Stage**：
```python
Y_t = FC(upsample(F̂_3D))   # [H×W×Z×(M+1)]，M=19 类 + 1 free
```

### 架构伪代码（维度注释）

```python
# 输入（以 VoxFormer-T 为例，输入5帧）
images: [B, T, 3, H, W]   # T=5, H=370, W=1220 (cam2 cropped SemanticKITTI)

# Feature Extractor（ResNet-50）
F_2D: [B, T, c, h_img, w_img]  # h_img=H/16, w_img=W/16, c=128（FPN 3rd stage）

# Stage-1
depth: [B, H, W]             # MobileStereoNet（stereo）or monocular depth
pts_3D = backproject(depth)  # [B, N, 3]
M_in = voxelize(pts_3D)      # [B, 256, 256, 32]，binary
M_out = Θ_occ(M_in)          # [B, 128, 128, 16]，binary

Q: [B, h, w, z, d]           # h=w=128, z=16, d=128 (predefined learnable queries)
Q_p = Q[M_out==1]            # [B, N_p, d]，稀疏

# Stage-2
# Cross-Attention: 3 个 deformable cross-attention 层
for _ in range(3):
    Q_p = DCA(Q_p, F_2D)     # [B, N_p, d]

# 合并 + Self-Attention: 2 个 deformable self-attention 层
m = learnable_mask_token     # [d]（共享）
F_3D = fill(Q, Q_p, m)       # [B, h*w*z, d] = [B, 32768, 128]
for _ in range(2):
    F_3D = DSA(F_3D, F_3D)   # [B, 32768, 128]

F_3D = F_3D.reshape(B, h, w, z, d)  # [B, 128, 128, 16, 128]

# Upsample + 分类
Y = FC(upsample(F_3D))       # [B, 256, 256, 32, 20]  (19 语义 + 1 free)
```

---

## 训练 vs 推理差异

| 方面 | 训练 | 推理 |
|------|------|------|
| **Stage-1 训练** | Binary occupancy 用 BCE loss 独立训练 | 固定 Stage-1，直接使用 |
| **Stage-2 训练** | 加权交叉熵 + scene-class affinity loss | 无 |
| **深度来源** | MobileStereoNet（stereo）or monocular depth predictor | 同上，需要 pretrained depth model |
| **Mask token** | m 是学习参数，所有非 proposed voxel 共享同一 m | 相同 |
| **多帧输入** | VoxFormer-T 用当前帧 + 过去 4 帧 | VoxFormer-S 只用当前帧 |

**重要注意**：Stage-1 和 Stage-2 分开训练（各 24 epochs），不是端到端联合训练。

---

## Loss 函数

**Stage-1**：
- Binary cross-entropy（BCE）for occupancy prediction at lower resolution

**Stage-2**：
- **加权交叉熵**：类别权重 = 逆类频率（与 LMSCNet 相同），处理类别不平衡
- **Scene-class affinity loss**（来自 MonoScene）：优化类别级别的 precision/recall/specificity

```python
L = - Σ_k Σ_c w_c * ŷ_{k,c} * log(exp(y_{k,c}) / Σ_c' exp(y_{k,c'}))
# w_c = inverse class frequency
```

---

## 消融实验

**Table 4（Query Proposal 类型，SemanticKITTI val）**：

| Query Selection | Memory (G) | IoU (%) | mIoU (%) |
|-----------------|-----------|---------|---------|
| Dense（所有 voxel） | 18.5 | 34.6 | 10.1 |
| Random 50% | 15.8 | 34.2 | 9.6 |
| Occupancy-based（Ours） | **14.6** | **44.0** | **12.4** |

结论：depth-based occupancy query proposal 相比 random 或 dense 在 IoU 上有+9.8 的提升（34.6→44.0），同时内存降低 21%。

**Table 3（深度类型消融）**：
- Stereo 深度（VoxFormer-T/S）> Monocular 深度，在 IoU 和 mIoU 上均有显著提升
- 即使用 monocular 深度，VoxFormer-S 在几何和部分语义指标上仍优于 MonoScene

**VoxFormer-T vs VoxFormer-S**（Table 1）：
- 加入过去 4 帧（VoxFormer-T）vs 单帧（VoxFormer-S）：mIoU 提升 8-22%（近距离区域更明显）

---

## 训练细节

- **Backbone**：ResNet-50（ImageNet pretrained），FPN 取第 3 stage feature
- **图像裁剪**：cam2 图像裁剪到 1220×370
- **Feature 维度**：d=128
- **Query 分辨率**：h×w×z = 128×128×16（低分辨率 queries），输出 upsample 到 256×256×32
- **Cross-attention 层**：3 层 deformable cross-attention
- **Self-attention 层**：2 层 deformable self-attention
- **采样点**：每个 attention head 8 个 sampling points
- **Stage-1 和 Stage-2 分开训练**，各训练 24 epochs
- **优化器**：lr=2×10⁻⁴
- **训练内存**：< 16GB（相比 MonoScene 显著减少）
- **参数量**：~60M（vs MonoScene ~150M）

---

## 数据

- **SemanticKITTI**：22 outdoor driving sequences，LiDAR scans voxelized 为 256×256×32（0.2m voxel，20 类：19 语义+1 free），46/0.8/6.4km；官方划分 22 sequences
- **评测体积**：51.2m × 51.2m × 6.4m ahead of car（全量 + 短距 12.8m/25.6m）

---

## 评测指标

- **IoU**：binary 场景完成（不管语义），衡量几何 completeness
- **mIoU**：19 个语义类的 mean IoU
- **距离分层评测**：12.8m / 25.6m / 51.2m 三个范围分别报告——近距离区域更安全关键

---

## 关键结果 / 数据

**SemanticKITTI hidden test set（Table 1 全量区域）**：

| 方法 | 输入 | IoU (%) | mIoU (%) |
|------|------|---------|---------|
| MonoScene | Camera | 36.80 | 11.30 |
| LMSCNet* | Stereo→LiDAR | 36.80 | 11.30 |
| **VoxFormer-S** | **Camera (Stereo depth)** | **44.02** | **12.35** |
| **VoxFormer-T** | **Camera×5 (Stereo depth)** | **44.15** | **13.35** |
| JS3CNet | LiDAR | 53.09 | 22.67 |

**短距区域（12.8m）VoxFormer-T vs MonoScene**：
- IoU: 65.38 vs 38.42（**+70%**）
- mIoU: 21.55 vs 12.25（**+76%**）

**与 LiDAR-based 方法对比（近距离）**：VoxFormer-T 在 12.8m 范围内（IoU 65.38）接近 LiDAR-based LMSCNet（74.88），说明深度估计在近距离有效。

---

## 局限性

- **依赖外部深度估计器**：Stage-1 需要 pretrained depth model（MobileStereoNet 或单目深度），引入额外误差和依赖；远距离深度不准问题仍存在
- **在 SemanticKITTI 上**：仅评测单目前视摄像头，不是 surround-view 设置
- **Stage-1 和 Stage-2 分开训练**：不是端到端，可能存在优化不一致
- **小物体仍然挑战**：虽然 VoxFormer 对小物体（bicycle, trunk, pole）显著优于 MonoScene，但绝对值仍低（mIoU < 7% 对 bicyclist/bicycle 等）

---

## 现状与影响

VoxFormer 代表了**LiDAR-assisted（或 depth-assisted）camera SSC 的重要进展**，确立了"先稀疏 proposal、再稠密完成"的两阶段范式。

- **"Reconstruction-before-hallucination"**理念：强调先处理可见结构再推断遮挡，是对 MonoScene 直接 2D-3D 投影的重要批判和改进，这一思路被后续工作采纳
- **Mask autoencoder 类比**：将 MAE（He et al. 2022）的思想引入 3D voxel completion，是早期将 MAE 范式迁移到 3D occupancy 的工作
- **模型本身**：在 SemanticKITTI 上有显著提升（vs MonoScene），但随着 Occ3D benchmark 的建立，surround-view 设置成为主流；VoxFormer 的单目前视设置不再是主流评测协议
- **在 Occ3D-nuScenes 上**：未见 VoxFormer 的官方评测，主要竞争者是 TPVFormer 和 BEVFormer 系列

**今天（2026）视角**：VoxFormer 的两阶段稀疏-稠密架构思路仍有影响，但 NVIDIA/NYU 后续工作已转向更大模型和 surround-view 设置。SemanticKITTI 上的 mIoU ~13 已被后续 LiDAR-assisted 方法超越到 20+。

**定性**：在 SSC（SemanticKITTI）任务上具有里程碑意义，LiDAR-assisted / depth-assisted occupancy 的代表作；在 surround-view occupancy 任务上影响有限。

---

## 关联概念

- [MonoScene](./monoscene-2112.00726.md) — VoxFormer 的直接前作和主要对比 baseline；VoxFormer 对其"2D-to-3D 密集投影"做了根本性改进
- [TPVFormer](./tpvformer-2302.07817.md) — 同期 CVPR 2023 工作，走不同路线（纯相机 + TPV 表示）；VoxFormer 依赖深度估计，TPVFormer 不需要
- [Occ3D](./occ3d-2304.14365.md) — Occ3D benchmark 中 VoxFormer 被引用为 SSC 方法的对比参考
- [DETR](./detr-2005.12872.md) — Deformable DETR（Zhu et al. ICLR 2021）是 VoxFormer cross-attention 和 self-attention 的底层机制

## 值得看的部分 / 相关资料

- **Section 3.4（Stage-1 Query Proposal）**：深度估计 + occupancy correction 的完整设计，包括 depth correction 的动机
- **Figure 2（总体框架）**：展示两阶段流程最清晰的一张图
- **Table 4（Query Proposal 消融）**：直接证明 depth-based proposal 相比 random/dense 的价值
- **Figure 1b（近距离对比）**：VoxFormer 在 12.8m 内相对 MonoScene 的 75%+ 提升，说明实用意义
