# DoReMi

**论文**：*DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining*（Xie et al., 2023）
**arxiv**：2305.10429
**机构**：Google DeepMind / Stanford University
**发表**：NeurIPS 2023

## 核心问题

**预训练数据集里各来源（domain）的比例怎么定？**

The Pile 等数据集的 domain weights（各来源采样比例，如 Wikipedia 占多少、代码占多少）通常由人工启发式决定，或者根据某组下游任务调优——前者可能次优，后者会过拟合特定任务且需要大量计算。

DoReMi 的目标：**在不知道下游任务的情况下，自动找到让大模型训练更高效的 domain weights**。

## 方法：三步流程

```
Step 1: 用初始 domain weights 训练一个小的参考模型（reference model，280M）

Step 2: 用 Group DRO 训练一个同样大小的代理模型（proxy model，280M）
        优化目标：最小化各 domain 上相对于参考模型的"超额损失"的最差情况
        → 输出优化后的 domain weights ᾱ

Step 3: 用 ᾱ 定义新的采样分布，重新采样数据，训练大模型（8B）
```

**参考模型和代理模型的区别**：

| | 参考模型（Reference） | 代理模型（Proxy） |
|--|---------------------|----------------|
| 训练目标 | 普通语言模型 NLL loss | Group DRO：最小化各 domain 超额损失的最差情况 |
| 训练数据 | 初始 domain weights 采样 | 均匀采样（每个 domain 等权重） |
| 作用 | 提供"基线难度"校准 | 通过训练过程中 domain weights 的动态更新，找出哪些 domain 最值得加大采样 |
| 最终用途 | 训练完就丢弃 | 也丢弃——真正用来训练的是 Step 3 的大模型 |

直觉上：参考模型是"平均水平的学生"，代理模型是"努力追赶差距最大科目的学生"。代理模型在训练过程中不断把更多注意力放到它还没学好的 domain，这个动态调整过程产生的权重分布，就是 DoReMi 的核心输出。

代理模型训练完后本身性能不重要——它只是一个工具，用来产生 domain weights。

### 核心概念：超额损失（Excess Loss）

$$\ell_{\text{excess}}(x) = \ell_\theta(x) - \ell_{\text{ref}}(x)$$

超额损失 = 代理模型的损失 - 参考模型的损失。含义：**参考模型已经学会了但代理模型还没学会的部分**。

- 超额损失高的 domain：参考模型容易学（低损失）但代理模型还很差 → 这个 domain 有提升空间，值得加大采样
- 超额损失低的 domain：要么本身信息量少（高熵），要么代理模型已经掌握

### Group DRO 优化目标（Step 2）

$$\min_\theta \max_{\alpha \in \Delta^k} \sum_{i=1}^k \alpha_i \cdot \left[\frac{1}{\sum_{x \in D_i}|x|} \sum_{x \in D_i} (\ell_\theta(x) - \ell_{\text{ref}}(x))\right]$$

**伪代码：Step 2 的完整训练循环**

```python
# 输入
# - D[1..k]：k 个 domain 的数据（如 D[wiki], D[arxiv], D[code], ...）
# - ref_model：Step 1 训练好的参考模型（固定，不再更新）
# - T：训练步数

# 初始化
alpha = [1/k] * k          # domain weights，初始均匀分布，k 个 domain 各占 1/k
proxy_model = init_model()  # 代理模型，随机初始化
alpha_history = []          # 记录每步的 alpha，最后取平均

for t in range(T):
    # ── 1. 均匀采样一个 minibatch（不管当前 alpha）──
    batch = sample_uniform_from_all_domains(D)

    # ── 2. 对 batch 中每个样本，计算两个模型的 token 级别 loss ──
    for x in batch:
        proxy_loss[x]  = -log p_proxy(x)   # 代理模型对这个样本的 NLL loss
        ref_loss[x]    = -log p_ref(x)      # 参考模型对这个样本的 NLL loss（固定值）
        excess_loss[x] = max(proxy_loss[x] - ref_loss[x], 0)  # 超额损失，clip 到非负

    # ── 3. 按 domain 聚合：每个 domain 的平均超额损失 ──
    # excess_loss[x] 是一个标量，x 属于哪个 domain 已知（来自采样时的标记）
    for i in range(k):
        lambda[i] = mean(excess_loss[x] for x in batch if x in D[i])
        # lambda[i] 就是 domain i 当前步的"学习难度"

    # ── 4. 更新 domain weights（指数梯度上升，难度高的 domain 权重增加）──
    alpha = alpha * exp(eta * lambda)   # 逐元素乘，超额损失大的 domain 权重指数增大
    alpha = normalize(alpha)            # 归一化回概率分布（加和=1）
    alpha = (1 - c) * alpha + c * (1/k) # 平滑：防止某个 domain 权重归零（c=1e-3）

    # ── 5. 更新代理模型（用当前 alpha 加权的 loss 做梯度下降）──
    weighted_loss = sum(alpha[i] * mean(proxy_loss[x] for x in batch if x in D[i])
                        for i in range(k))
    proxy_model.update(weighted_loss)   # 标准反向传播

    alpha_history.append(alpha.copy())

# ── 6. 输出：取所有步的 alpha 平均值 ──
final_alpha = mean(alpha_history)   # shape: (k,)，就是 DoReMi 输出的 domain weights
```

**为什么最后取平均而不是最后一步的 alpha**：

训练过程中 alpha 会震荡——某个 domain 超额损失高时 alpha 上升，代理模型多学这个 domain 后超额损失降低，alpha 随之下降。这个震荡是正常的（[minimax 博弈](../20-concepts/minimax.md)的特性）。取所有步的平均，相当于在这个博弈过程中找到一个"均衡点"——数学上可以证明这个平均值是 minimax 目标的近似最优解（来自在线学习理论，Nemirovski et al. 2009）。详见 [Minimax 博弈](../20-concepts/minimax.md)。

直觉：不是某一个时刻"谁最难"就用谁的权重，而是"整个训练过程中平均来看各 domain 的相对难度"决定最终权重。

**关键设计：超额损失是 token 级别的，domain 归属是已知的**：

每个样本 x 来自哪个 domain 在采样时就已经标记好了（The Pile 里每条数据有 source 字段）。所以 Step 3 的聚合只是按 domain 标签分组取平均，不是在猜哪个 token 属于哪个 domain。

### Iterated DoReMi

将 Step 1-2 迭代多轮，每轮用上一轮输出的 $\bar{\alpha}$ 作为新的参考 domain weights。实验发现 GLaM 数据集上 **3 轮收敛**，收敛条件为任意 domain weight 变化 < 1e-3。

## 实验结果

**实验配置**：280M 代理模型优化 domain weights → 训练 8B 主模型（30× 大），在 The Pile（22 个 domain）和 GLaM（8 个 domain）上评测。

**The Pile 数据集**：

| 指标 | Baseline（默认 weights） | DoReMi（280M→8B） |
|------|------------------------|-----------------|
| 平均 one-shot 下游准确率 | ~20%（200k steps） | **+6.5pp**（2.6× 更快达到 baseline）|
| Per-domain 困惑度 | — | 所有 22 个 domain 全部下降（即使被降权的 domain 也改善）|

**DoReMi 优化后权重的变化规律**（Table 1 摘录）：
- 大幅上调：Pile-CC（+0.49）、YoutubeSubtitles（+0.46）
- 大幅下调：PubMed Central（-0.10）、ArXiv（-0.10）、StackExchange（-0.08）

PubMed Central 和 ArXiv 被降权，说明这些专业领域文本对通用语言模型的贡献低于其数据量占比。

**GLaM 数据集**：

Iterated DoReMi（2 轮后）达到与在下游任务上直接调优 domain weights 相当的性能——在完全不用下游任务信息的情况下，找到了类似的最优配比。

**"最优"怎么定义**：DoReMi 的"最优"不是针对某个特定任务的最优，而是一个鲁棒最优（minimax 意义上的最优）——**让所有 domain 上的最差表现尽量好**。形式上就是 Group DRO 的 minimax 目标：找一组权重，使得最难的那个 domain 的超额损失最小。

这和"下游任务调优"的区别是：下游任务调优的"最优"是在某几个 benchmark 上的最优，有过拟合风险；DoReMi 的"最优"是在所有 domain 的鲁棒表现上的最优，不依赖任何特定评估集。GLaM 实验发现两者结果接近，说明这个鲁棒最优和任务最优在实践中经常重合。

**计算开销**：优化 domain weights（训练两个 280M 模型）只用了训练 8B 主模型所需 FLOPs 的 **8%**。

## 为什么降权某些 domain 反而提升了这些 domain 的困惑度

论文提出的解释：

- **高熵 domain**（如 PubMed、ArXiv）的最优损失本身就高（专业词汇多，难以预测）——减少这些样本后，模型花更多算力在"可学习"的 domain 上
- **正迁移**：中等熵 domain 的提升会迁移到相关 domain，即使后者被降权也能受益
- **统计效率**：低熵 domain（如 Wikipedia）不需要太多样本就能学好；高熵 domain 多学也不会有收益

**高熵/低熵怎么计算**：这里的"熵"不是直接计算的一个数，而是通过**参考模型的损失**来间接衡量的——参考模型在某个 domain 上的平均 NLL loss 越高，就说明这个 domain 越难预测，即"熵越高"。

$$\text{entropy proxy}(D_i) \approx \frac{1}{|D_i|} \sum_{x \in D_i} \ell_{\text{ref}}(x)$$

直觉：一个高熵的文本序列，下一个词的不确定性大，任何模型都很难预测（loss 高）；一个低熵的文本（如结构化的 Wikipedia 文章），模式规律强，模型容易学到（loss 低）。

所以论文里说的"高熵 domain"（PubMed、ArXiv），就是参考模型在这些 domain 上 loss 一直居高不下的 domain——不是因为模型没学好，而是这类文本本身就难以预测（专业术语多、句式复杂）。加再多数据也提升有限，因为"困难"来自内容本身，不来自数据量不足。

## 消融实验

**代理模型规模的影响**：在同规模（X→X）设定下（280M/510M/760M/1B），DoReMi 始终比 baseline 快约 **4×** 达到相同准确率，且改善幅度不随规模增大而缩小。跨规模（280M→8B）因为需要泛化，加速比为 2.6×。

**超额损失的必要性**：直接用代理模型的绝对损失（不减参考模型损失）效果显著下降。参考模型起到了"校准基线"的作用——没有它，Domain DRO 会过度上调本身难度高的 domain。

## 现状与影响

**定性：方法论仍活跃，是数据混合配方研究的重要基准。**

DoReMi 是第一个系统性地把"数据混合配方优化"变成可计算问题的工作，发表后被广泛引用。Llama 3 的数据配方思路（用小模型实验预测大模型效果）和 DoReMi 的逻辑高度相似，虽然 Meta 没有直接说使用了 DoReMi，但论文是该方向的标准参考。

**局限性**：
- 实验在 The Pile 和 GLaM 上进行，数据集已经过时（2023 年后主流转向更大更新的数据集）
- 280M 代理模型找到的权重对 8B 模型有效，但对更大模型（70B+）的泛化性未验证
- Domain 划分是预先确定的（22 个），无法自动发现更细粒度的最优划分
- 2024 年后出现了 Data Mixing Laws 等更精确的方法，从 scaling 角度直接预测不同配比下的模型性能

**与后续工作的关系**：DoReMi 是"数据混合 = 可优化问题"这个范式的开创，Data Mixing Laws（2024）在此基础上更进一步，把配方搜索变成了 scaling law 预测问题。

## 和 wiki 内其他概念的关联

- **[Loss Functions](../20-concepts/loss-functions.md)**：DoReMi 的 Group DRO 目标函数是 NLL loss 在多个 domain 上的 minimax 变体
- **[RLHF](../20-concepts/rlhf.md)**：Group DRO 的"最差情况优化"和 RLHF 的对齐思路有相似之处——都是针对分布的鲁棒优化
- **[LLM 学习 Roadmap](../10-roadmaps/llm-learning-roadmap-20260410.md)**：数据工程章节，DoReMi 是数据混合配方方向的核心参考
- **[DCLM](../20-concepts/dclm.md)**：DCLM 固定模型只改数据做系统对比，DoReMi 自动优化配比，两者是互补的数据工程工具

## 值得看的部分

- Section 2（DoReMi 算法，含 Group DRO 的 minimax 目标和 Algorithm 1 伪代码）
- Figure 3（The Pile 和 GLaM 上的训练曲线，直观展示 2.6× 加速）
- Figure 4（per-domain 困惑度变化，展示"降权也改善"的反直觉结果）
- Table 1（The Pile 的 domain weights 变化，看哪些 domain 被上调/下调）
- Section 4（消融实验：超额损失的必要性，代理模型规模的影响）
