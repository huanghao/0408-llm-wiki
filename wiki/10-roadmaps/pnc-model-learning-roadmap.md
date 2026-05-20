# PNC 规划模型学习路线图

> 目标：理解 PNC（Planning & Control）的神经网络架构和训练体系。  
> 前置知识：Transformer 基础（attention 机制）、Python/PyTorch 基础。

PNC 的核心是 **SharedEncoder + 多个 Decoder**：SharedEncoder 以 Wayformer 风格对场景做编码，GeneralDecoder 以 DETR 风格做轨迹生成，训练方式参考 BERT 的预训练+冻结策略。读这个路线图的顺序按依赖关系排列，不按发表时间。

---

## 第一层：基础架构（必须先掌握）

**1. Transformer 原始论文** — 理解 SharedEncoder 的所有层（self-attention、FFN、positional encoding）

- 论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)（Vaswani et al., 2017）
- 相关概念：[attention intuition](../20-concepts/attention-intuition.md)、[positional encoding](../20-concepts/positional-encoding.md)

**2. DETR** — GeneralDecoder 的结构最像 DETR decoder：Q=任务 query，KV=encoder output，cross-attention 出结果

- wiki：[DETR（Carion et al., ECCV 2020）](../30-papers/detr-2005.12872.md)
- PNC 对应：GeneralDecoder 的参考线 token 充当 DETR 里的 object query；SharedEncoderFeature 充当 DETR 里的 image feature

---

## 第二层：直接原型（PNC 架构的直接来源）

**3. Wayformer** — SharedEncoder 的直接原型，读完能理解为什么用 global self-attention、token 数为什么是 634

- wiki：[Wayformer（Nayakanti et al., ICRA 2023）](../30-papers/wayformer-2207.05844.md)
- 重点：Early Fusion vs Factorized/Latent Query 三种 attention 策略的对比实验，以及为什么 joint attention 在多数场景下最优

**4. VectorNet** — road_graph_segment 的向量化地图表示方案，PNC 的路网编码思路与此高度一致

- wiki：[VectorNet（Gao et al., CVPR 2020）](../30-papers/vectornet-2005.04259.md)
- 重点：polyline segments 编码、局部 graph attention + 全局 attention 两阶段结构
- PNC 对应：v5 的 road_graph_segment `[450,1,13]` 本质上就是 VectorNet 的 polyline（13维 vs VectorNet 7维）；v6 的 lane_instance 是向结构化实例化的演进

**5. TNT（Target-driveN Trajectories）** — 理解 WTA loss 和 anchor-based 多模态预测的直接来源

- 论文：[TNT: Target-driveN Trajectory Prediction](https://arxiv.org/abs/2008.08294)（Zhao et al., CoRL 2020）
- 重点：先预测目标终点（target），再以每个 target 为条件预测完整轨迹；WTA（Winner-Takes-All）只对最接近 GT 的模态计算回归 loss
- PNC 对应：GeneralDecoder 的 9 条候选轨迹 + WTA loss 是这个设计的简化版（固定 3×3 模式，不做 target 预测）

---

## 第三层：重要对比（理解 PNC 的设计选择）

**6. MTR（Motion Transformer）** — 和 PNC 改进方向对比最密切

- wiki：[MTR（Shi et al., NeurIPS 2022）](../30-papers/mtr-2209.13508.md) · [MTR++](../30-papers/mtrpp-2306.17770.md)
- 重点：Motion Query Pair（静态意图锚点 + 动态搜索 query）、局部 polyline Transformer + 全局 SA 两阶段
- 与 PNC 的差距：MTR 的 intent anchors 是数据驱动的（K-means 聚类真实轨迹），PNC 的 9 模态是固定的 3纵×3横

**7. HiVT（Hierarchical Vector Transformer）** — 局部旋转不变特征，对应 PNC 的 PE 弱点

- 论文：[HiVT: Hierarchical Vector Transformer for Multi-Agent Motion Prediction](https://arxiv.org/abs/2206.10982)（Zhou et al., CVPR 2022）
- 重点：以每个 agent 为中心做局部坐标归一化（旋转平移不变），彻底不需要全局绝对坐标的 PE；先局部 self-attention 再全局 cross-attention 的两阶段架构
- PNC 对应：PNC 用序列位置 PE（0~633 的整数），不携带空间坐标；HiVT 的设计是一个更优解

**8. UniAD** — 端到端方案对比，理解 PNC 分离训练的代价与工程合理性

- wiki：[UniAD（Hu et al., CVPR 2023）](../30-papers/uniad-2212.10156.md)
- 重点：感知/预测/规划共享一个 BEV Transformer，联合训练；和 PNC 的分离训练（Encoder 独立预训练+冻结）形成对比

---

## 第四层：训练方法（理解 SharedEncoder 独立预训练和 DPO 微调）

**9. BERT** — SharedEncoder 独立预训练+冻结的直接参照

- 论文：[BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)（Devlin et al., 2019）
- 重点：MLM 预训练 + 下游任务 fine-tune；encoder 权重如何在不同下游任务间共享
- PNC 对应：SharedEncoder 用 GeneralDecoder+NudgeDecoder+Reconstruction 三个 head 联合预训练，然后冻结供各 Decoder 使用——和 BERT 预训练 + 下游 fine-tune 同构，但 PNC 的 Decoder 不 fine-tune Encoder

**10. DPO（Direct Preference Optimization）** — PNC 代码里已有完整实现，这是它的理论来源

- 论文：[Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)（Rafailov et al., NeurIPS 2023）
- 重点：不需要显式奖励模型，直接用人工偏好对（好轨迹 vs 坏轨迹）做对比 loss 微调
- PNC 对应：`planning_trajectory_alignment_dpo_loss.py` 实现了轨迹级别的 DPO，用人工标注的偏好排序进行偏好对齐

---

## 第五层：数据效率与规模（理解小规模代理实验的理论基础）

**11. Scaling Laws** — 数据规模、模型规模、计算量的相互关系

- wiki：[Scaling Laws for Neural Language Models（Kaplan et al., 2020）](../30-papers/scaling-laws-neural-lm-2001.08361.md)
- 对 PNC 的意义：帮助判断 2M clip 对 12-15M 参数模型是否充足；小规模代理实验的理论依据

**12. Chinchilla** — 重新标定了最优数据与模型规模比例

- wiki：[Chinchilla（Hoffmann et al., 2022）](../30-papers/chinchilla-2203.15556.md)
- 对 PNC 的意义：当前 SharedEncoder 约 12-15M 参数，按 Chinchilla 比例需要约 240-300M token 量级的数据——换算到 AV 场景的 "token" 定义有差异，但量级参考有价值

**13. DoReMi** — 多数据源混合的自动权重优化，和 PNC 的 PID 混合机制原理相近

- wiki：[DoReMi（Xie et al., NeurIPS 2023）](../30-papers/doremi-2305.10429.md)

---

## 补充：用于对比的扩展阅读

| 论文 | 状态 | 为什么读 |
|---|---|---|
| [LaneGCN（Liang et al., ECCV 2020）](https://arxiv.org/abs/2007.13732) | ⬜ | 显式 lane graph attention；PNC v6 lane_instance 改进的理论背景 |
| [DESIRE（Lee et al., CVPR 2017）](https://arxiv.org/abs/1704.04394) | ⬜ | WTA loss 与多样性约束的早期来源；Diversity Loss 改进建议的参考 |
| [GameFormer（Huang et al., ICCV 2023）](../30-papers/gameformer-2303.05760.md) | ✅ | 交互场景建模的博弈论视角 |
| [MotionDiffuser](../30-papers/motiondiffuser-2306.03083.md) | ✅ | 扩散模型 vs PNC 一步 forward 的对比 |
| [nuPlan（Caesar et al., 2021）](../30-papers/nuplan-2106.11810.md) | ✅ | 规划 benchmark，理解开环 vs 闭环评估差异 |
| [GNN 概念](../20-concepts/gnn.md) | ✅ | LaneGCN 的图神经网络基础 |
| [GMM 概念](../20-concepts/gaussian-mixture-model.md) | ✅ | GeneralDecoder 9 条候选 = 隐式 GMM 的直觉理解 |

---

## 阅读顺序建议

```
基础架构（1-2）
    ↓
直接原型（3-5）     ← 最重要的一层，读完能看懂 PNC 架构文档
    ↓
重要对比（6-8）
    ↓
训练方法（9-10）    ← 读完能理解 SharedEncoder 独立训练和 DPO 的设计逻辑
    ↓
数据效率（11-13）   ← 读完能理解小规模代理实验方法论
```

**如果只有时间读 3 篇**：Wayformer + TNT + MTR  
**如果只有时间读 1 篇**：Wayformer（和 PNC SharedEncoder 直接对应，读完能建立最强的直觉）
