# Data Mixing Laws: Optimizing Data Mixtures by Predicting Language Modeling Performance

**arxiv**: 2403.16952 | **年份**: 2024 | **机构**: Fudan University, Shanghai AI Laboratory | **接收**: ICLR 2025

**作者**: Jiasheng Ye, Peiju Liu, Tianxiang Sun, Jun Zhan, Yunhua Zhou, Xipeng Qiu

**代码**: https://github.com/yegcjs/mixinglaws

---

## 核心问题

预训练数据通常由多个领域混合而成（网页、代码、学术论文等），各领域的比例（mixture proportions）对模型效果有显著影响。现有方法依赖启发式规则或定性判断来调整比例，**无法在训练前预测不同配比对应的模型性能**。

核心问题：能否在不实际训练的情况下，定量预测任意数据配比下的模型验证损失？

---

## 方法 / 核心机制

### 1. Data Mixing Laws（数据混合定律）

论文发现：在固定模型大小和训练步数的情况下，**第 i 个验证域的损失与训练域比例之间满足指数函数关系**：

$$L_i(r_{1\ldots M}) = c_i + k_i \exp\left(\sum_{j=1}^{M} t_{ij} r_j\right)$$

其中：
- $r_j$ 是第 j 个训练域的比例
- $c_i$、$k_i$、$t_{ij}$ 是可学习参数
- $t_{ij}$ 的正负反映域间关系：$t_{ij} < 0$ 表示训练域 j 有助于降低验证域 i 的损失（facilitation）；$t_{ij} > 0$ 表示 conflict

整体验证损失 = 各域损失的加权求和（显式域聚合），或通过**隐式域聚合**端到端拟合（将验证集的域比例也作为可学习参数）。

### 2. 三步嵌套预测 Pipeline

直接在大规模训练上拟合混合定律代价极高。论文嵌套使用三类 Scaling Laws：

1. **训练步数 Scaling Law**：用小步数实验预测大步数损失
2. **模型大小 Scaling Law**：用小模型预测大模型损失
3. **Data Mixing Law**：在预测的大模型大步数损失上拟合混合定律，从而预测任意新配比的损失

全流程只需小模型（70M–410M）、少量 token（30B）的实验，即可预测 1B 模型在 100B tokens 上、任意数据配比的验证损失。

### 3. 域间关系的系数矩阵

通过可视化 Pile 五个粗粒度域的 $t_{ij}$ 矩阵（图4），发现三类关系：
- **Unrelated**（无关）：大多数域对之间稀疏，各自独立
- **Facilitation**（促进）：如 Dialogue 训练数据有助于 Internet 域
- **Conflict**（冲突）：如 Symbolic 数据会损害 Prose 域

---

## 关键结果 / 数据

| 指标 | 结果 |
|------|------|
| 优化后配比 vs 默认 RedPajama | 仅需 **0.73×** 训练步数达到相同 Pile 验证困惑度 |
| 等效提升 | 优化配比等效于默认配比训练 **1.48×** tokens |
| 对比 DoGE、DoReMi | 优化配比在 Pile 验证集上困惑度最低（1B/100B tokens，图9） |
| 预测误差 | 显式域聚合 MAE ≈ 0.003–0.005；隐式聚合误差与显式持平（K≥实际域数） |

**Continual Pretraining 应用**：将混合定律应用于持续预训练（Pythia-70M，Pile + Python code），可精确预测**临界混合比例**（critical proportion）——即在不损害原始域性能的前提下，能加入目标域的最大比例，从而避免灾难性遗忘。

**优化后的 RedPajama 配比**（相比默认大幅调整）：
- CommonCrawl: 67% → 12.5%
- C4: 15% → 25%
- ArXiv: 4.5% → 75%（大幅提升）
- GitHub: 4.5% → 14%

---

## 局限性

- 拟合混合定律仍需运行多次（约40次）小规模实验，有一定成本
- 嵌套 Scaling Laws 假设小模型/小步数的排名与大模型/大步数一致；若存在排名翻转（rank reversal），预测会偏差（论文引用了 Goyal et al., 2024 等研究该问题的工作）
- 隐式域聚合的域数 K 需设为不小于实际域数，否则误差会上升
- 实验数据集限于 Pile / RedPajama，对细粒度混合（数十个域）的 scaling 待验证

---

## 现状与影响

**还在用吗？** 是的，这是数据配比优化领域的代表性定量方法，ICLR 2025 接收，实践价值明确。

**被什么取代？** 尚无直接替代；同期工作包括 DoGE（Fan et al., 2024）和 RegMix（Liu et al., arXiv:2407.01492），思路相似但方法不同。Data Mixing Laws 的优势在于显式函数形式和可解释的域间关系矩阵。

**一句话定性**：将数据配比优化从"经验调参"变成"可预测的定量科学"，是 Scaling Laws 研究从模型维度扩展到数据维度的标志性工作。

---

## 和 wiki 内其他概念的关联

- [DoReMi](doremi-2305.10429.md)：同样优化数据域配比，但用代理模型 + group DRO 而非显式函数拟合；两者互补
- [DolmIno](../20-concepts/dolmino.md)：数据混合配方研究，关注配比对下游任务的影响
- [DCLM](../20-concepts/dclm.md) / [DCLM paper](dclm-2406.11794.md)：关注数据过滤策略，与混合配比属于数据工程的不同维度
- [Perplexity](../20-concepts/perplexity.md)：混合定律拟合的核心度量指标是验证集 perplexity/NLL
- [ccNet](../20-concepts/ccnet.md) / [FineWeb](../20-concepts/fineweb.md)：数据来源 pipeline，混合定律在这些数据集上运行

---

## 值得看的部分 / 相关资料

- **图 4**（域交互矩阵）：直观展示不同域之间促进/冲突/无关关系，对理解预训练数据结构很有价值
- **图 8**（优化配比 vs 默认配比的训练曲线）：清晰展示 0.73× / 1.48× 的实际效果
- **Algorithm 1**：三步嵌套预测 pipeline 的完整伪代码，实现参考价值高
- **Section 5**（Continual Pretraining 应用）：混合定律预测临界比例、避免灾难性遗忘，是实践价值最直接的部分
- 同期对比工作：[DoGE (arXiv:2401.)?](https://arxiv.org/abs/2310.10951)、RegMix (arXiv:2407.01492)
