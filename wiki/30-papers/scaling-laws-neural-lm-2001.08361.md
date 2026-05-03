# Scaling Laws for Neural Language Models (Kaplan et al., 2001.08361)

一句话总结：loss 与参数量 N、数据量 D、计算量 C 各自呈光滑幂律关系；固定计算预算时应优先扩大模型而非增加数据，远早于收敛就停训——这一结论被 Chinchilla（2022）修正为"两者等比例扩大"。

## 基本信息

- 论文：Scaling Laws for Neural Language Models
- 作者：Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, Dario Amodei
- 机构：Johns Hopkins University / OpenAI
- arXiv：[2001.08361](https://arxiv.org/abs/2001.08361)
- 发表：2020 年 1 月（预印本，OpenAI 技术报告）
- 本地原文：`raw/inbox/2001.08361.pdf`

## 核心问题

语言模型的性能（cross-entropy loss）如何随参数量 $N$、数据集大小 $D$、计算预算 $C$ 变化？这三个因素之间的关系是什么？给定固定计算预算，应该如何分配？

论文在 WebText2 上系统训练了从 768 到 15 亿参数的大量 Transformer，跨越多个数量级（N 跨 6 个量级，D 跨 2 个量级，C 跨 8 个量级），观察到一系列普遍的幂律规律。

## 方法 / 核心机制

### 核心发现：三条独立的幂律

论文的方法论是：**每次只让一个变量变化，其他条件设为"不构成限制"**，这样才能单独观察那个变量的影响。"不成为瓶颈"就是指这种实验设置：

- 测 N 时：给足够多的数据（WebText2 全量），让模型训到收敛——数据量和计算量不是限制，只有 N 在变。
- 测 D 时：固定一个大模型，给不同大小的数据集，早停（数据用完就停）——N 固定，只有 D 在变。
- 测 C 时：固定计算量，训练不同 N 的模型——找出在该 FLOPs 下最低 loss 的 (N, D) 组合。

这不是"限制在某个规模"，而是"控制变量"。

**变量注释（先看这里，再看公式）：**

- $N$：模型参数量（不含 embedding 层），单位：参数个数
- $D$：训练数据集大小，单位：tokens
- $C$：训练计算量，单位：FLOPs（floating point operations），约等于 $6ND$
- $L$：cross-entropy loss（越低越好）
- $\alpha_N, \alpha_D, \alpha_C$：**幂律指数**，控制 loss 随变量改善的速率——指数越大，增大该变量带来的收益越大
- $N_c, D_c, C_c$：**拟合常数**，从数据中学出来的规模参数，没有独立物理含义，只是让公式在数值上对齐

**三条幂律及直白解读：**

**1. 参数量 N（无限数据，训练到收敛）：**
$$L(N) \approx \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076$$

解读：$N$ 翻倍时，$L$ 降低的乘数因子是 $2^{-0.076} \approx 0.949$，即 loss 下降约 5%。模型越大，loss 越低，但边际收益递减（指数远小于 1）。

**2. 数据集大小 D（大模型，早停）：**
$$L(D) \approx \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095$$

解读：数据量翻倍，loss 下降约 6%。数据比参数对 loss 的边际贡献略大（指数稍高）。

**3. 计算量 $C_{min}$（最优批大小下）：**
$$L(C_{min}) \approx \left(\frac{C_c^{min}}{C_{min}}\right)^{\alpha_C^{min}}, \quad \alpha_C^{min} \approx 0.050$$

解读：计算量翻倍，loss 下降约 3.5%。计算的边际收益最小，因为它要同时受 N 和 D 的共同约束。

关于幂律本身是什么、为什么常见，参见 [幂律与 Scaling](../20-concepts/power-law-and-scaling.md)。

这些关系跨越超过 6 个量级，且几乎不依赖 Transformer 的深宽比、注意力头数等架构细节。

### 联合 L(N, D) 公式

论文提出一个同时捕捉参数量和数据量的公式：

$$L(N, D) = \left[\left(\frac{N_c}{N}\right)^{\frac{\alpha_N}{\alpha_D}} + \frac{D_c}{D}\right]^{\alpha_D}$$

当 $D \to \infty$ 时退化为 $L(N)$，当 $N \to \infty$ 时退化为 $L(D)$。这个公式还揭示了过拟合的规律：每次模型参数增大 8 倍，只需数据增加约 5 倍来避免过拟合（$D \propto N^{0.74}$）。

### 计算最优分配（Kaplan rule）

给定固定计算预算 $C$，最优分配满足：

$$N_{opt} \propto C^{0.73}, \quad D_{opt} \propto C^{0.27}$$

即：计算预算增加 10× 时，最优模型参数量增加 $\approx 5.5\times$，而训练数据只需增加 $\approx 1.8\times$。

**实践含义**：应该训练非常大的模型，并在远早于收敛的时候停训（early stop）——compute-efficient 训练在 loss 比完全收敛高约 10% 时就应停止。大模型比小模型 **sample efficient**（样本效率更高）：要达到同一个 loss 目标，大模型需要看的训练样本更少，因为它的参数容量更大，每个样本能"利用"得更充分。注意这里说的是"达到同等 loss 所需的数据更少"，不是说大模型训练数据总量少——事实上 Chinchilla 后来发现更大的模型还是需要更多数据才能充分发挥能力。

### 其他规律

- **架构无关性**：在参数量固定时，depth/width 比例、注意力头数等对性能影响很小（loss 变化仅几个百分比）。这不是说多头注意力没用——在参数量相同的条件下，多头 vs 少头的差别很小（论文发现从 1 头到 64 头，loss 只差几个百分点）。多头注意力的主要作用是让模型能并行捕捉不同类型的依赖关系，这在工程上有价值；但对 scaling 的影响可以忽略：总参数量才是关键变量。换句话说，给定 N，你怎么切分成多少头影响不大。
- **迁移泛化**：在 WebText2 训练、在其他数据分布（Books、Wikipedia 等）上测试，loss 只有一个固定的常数偏移，趋势完全一致。
- **临界批大小**：最优批大小近似为 $B_{crit} \propto L^{-1/\alpha_B}$，约为 $2 \times 10^8$ tokens。**批大小和训练步数之间存在连续的速度/效率权衡**：批大小小→更多梯度更新步数→收敛慢但内存省；批大小大→并行度高→训练快但每步信息量更冗余。临界批大小是这个权衡的最优点：超过它继续增大 batch 会带来收益递减（需要更多 FLOPs 才能达到同等 loss）。
- **训练曲线预测**：通过拟合训练曲线的早期段，可以准确外推模型最终能达到的 loss，无需跑完全程。

## 关键结果 / 数据

**幂律指数是什么：** 幂律 $L \propto X^{-\alpha}$ 意味着"$X$ 翻倍，$L$ 乘以 $2^{-\alpha}$"。$\alpha$ 越大，翻倍带来的 loss 降幅越大。以下三行的指数都很小（0.05–0.095），意味着每次翻倍只带来几个百分点的改善——这就是为什么需要多个量级的扩展才能显著提升模型能力。

| 变量 | 幂律指数 | 解读 |
|---|---|---|
| 参数量 $N$ | $\alpha_N \approx 0.076$ | 参数量翻倍，loss 降低约 5% |
| 数据量 $D$ | $\alpha_D \approx 0.095$ | 数据量翻倍，loss 降低约 6% |
| 计算量 $C$ | $\alpha_C \approx 0.050$ | 计算翻倍，loss 降低约 3.5% |

Kaplan rule 的分配建议（compute 增加 10×）：

| 变量 | 建议增幅 |
|---|---|
| 参数量 | $\times 5.5$（主要增量来源） |
| 数据量 | $\times 1.8$（增幅很小） |
| 训练步数 | $\times 3.1$ |

> **注：这里的"数据量"特指训练 token 的总数（fresh tokens，来自独立扩大训练集），而不是重复训练同一数据集。** Kaplan 论文假设所有 token 只训练一次（单 epoch），D 始终代表"见过的不同 token 数"。若通过多轮重复（multi-epoch）来凑 D，论文的幂律不适用——重复数据的边际价值会迅速递减。[Scaling Data-Constrained LMs](./scaling-data-constrained-lms-2305.16264.md) 专门研究了重复数据场景，发现重复超过 4 轮后基本没有 loss 收益。

**为什么 10× 计算对应这些增幅？不是加法，是幂律推导。** 关键在于计算量 $C \approx 6ND$（参数量 × 数据量 × 常数），所以 $N$ 和 $D$ 的增幅必须满足"乘积增加 10 倍"的约束（$5.5 \times 1.8 \approx 10$，步数增幅作为辅助结果）。在这个乘积约束下，怎么在 $N$ 和 $D$ 之间分配是一个优化问题：给 $N$ 更多还是给 $D$ 更多，哪个带来更低的 loss？Kaplan 的答案（$N^{0.73}$, $D^{0.27}$）来自对拟合出的 $L(N, D)$ 公式求偏导，找到在固定 $C$ 约束下的鞍点。Chinchilla 后来用更准确的实验数据重新推导，得到了 $N^{0.5}$, $D^{0.5}$（等比例），修正了这一结论。

Chinchilla（2022）后来发现这一建议显著低估了数据的重要性，正确比例应为参数和数据各增加约 $\times 3.2$（等比例）。

## 局限性

- **单 epoch 假设**：所有实验在大于一 epoch 的数据规模上训练，但始终保证数据不重复（fresh tokens），不适用数据有限场景——[Scaling Data-Constrained LMs](./scaling-data-constrained-lms-2305.16264.md) 填补了这一空白。
- **固定批大小偏差**：Chinchilla 指出 Kaplan 的许多模型使用固定 learning rate schedule（非随数据量调整的 cosine），导致数据量较少时 loss 被高估，从而系统性低估了数据的边际价值——这正是 Chinchilla 修正结论的根本原因。
- **小模型偏重**：Kaplan 实验中大多数运行使用不超过 1B 参数的模型，超 10B 规模的外推存在不确定性；Chinchilla 使用了更多超 1B 的运行点。
- **纯语言建模**：实验基于自回归 LM 的 cross-entropy loss，不直接覆盖 instruction following、对话、推理等后训练目标。
- **无理论推导**：作者明确指出这是经验规律，没有理论基础，因此在什么范围内成立无法从第一性原理判断。

## 现状与影响

一句话定性：**本文是 LLM scaling 研究的起点，确立了"性能遵循幂律"的实证框架；其具体分配建议（优先扩大模型）被 Chinchilla 修正，但幂律框架本身成为所有后续 scaling law 研究的基础。**

截至 2026，Kaplan 的直接影响：

- **框架奠基**：$L(N)$、$L(D)$、$L(C)$ 的幂律分解框架被 Chinchilla、[Data Mixing Laws](./data-mixing-laws-2403.16952.md)、[Scaling Data-Constrained LMs](./scaling-data-constrained-lms-2305.16264.md) 等直接继承和扩展。
- **"大模型 sample efficient"的直觉**：大模型比小模型用更少数据达到同等 loss，这一发现至今仍是理解 LLM 训练资源分配的核心直觉之一。
- **已被修正的部分**：Kaplan rule（$N \propto C^{0.73}$，$D \propto C^{0.27}$）已被 Chinchilla 的等比例规律替代，不再作为训练决策的直接参考。原因是 Kaplan 使用了固定 LR schedule，导致数据量估计偏低。
- **后续扩展方向**：多模态 scaling（视频、图像）、推理成本纳入优化目标、数据质量与 scaling 的交互，都是在 Kaplan 框架上生长出来的研究方向。

## 和 wiki 内其他概念的关联

- [Chinchilla Scaling Laws](./chinchilla-2203.15556.md)：直接修正了 Kaplan 的分配建议，将 $N \propto C^{0.73}$ 修正为 $N \propto C^{0.5}$（等比例），原因是 Kaplan 使用固定 LR schedule 导致数据价值被系统低估。
- [Scaling Data-Constrained LMs](./scaling-data-constrained-lms-2305.16264.md)：在 Kaplan/Chinchilla 框架基础上增加重复数据（多 epoch）场景，扩展了 $D$ 的建模方式。
- [Data Mixing Laws](./data-mixing-laws-2403.16952.md)：继承 Chinchilla 的参数化公式，进一步扩展到多域数据配比优化。
- [MFU](../20-concepts/mfu.md)：FLOPs 和计算效率是 Kaplan scaling law 的基础度量，MFU 是实际硬件利用率的衡量指标。
- [Perplexity](../20-concepts/perplexity.md)：本文以 cross-entropy loss 为核心指标，perplexity 是其指数变换。
- [Gopher](./gopher-2112.11446.md)：Gopher 训练决策基于 Kaplan rule，被 Chinchilla 证明"过大欠训"，是 Kaplan 错误建议的最典型案例。

## 值得看的部分 / 相关资料

- **Section 1.2（Summary of Scaling Laws）**：全文核心公式汇总，Table 4 / Table 5 / Table 6 给出所有幂律指数的精确数值。
- **Figure 1**：三条幂律（vs N、D、C）的直观展示，是全文最常被引用的图。
- **Figure 3**：给定计算预算下应如何分配到参数/批大小/步数的示意图。
- **Section 6（Optimal Allocation of the Compute Budget）**：Kaplan rule 的推导，$N_{opt} \propto C^{0.73}$。
- **Appendix B**：compute-efficient frontier 的完整推导，包括为什么 compute-efficient 训练在 loss 高于收敛约 10% 时停止。
- **Appendix C（Caveats）**：作者对自己结论的局限性的坦诚讨论，值得阅读。
- 后续修正：Hoffmann et al. 2022（[Chinchilla](./chinchilla-2203.15556.md)，arXiv:2203.15556）——直接修正了 Kaplan 的分配建议。
