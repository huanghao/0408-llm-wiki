# 图神经网络（GNN）

一句话总结：GNN 通过反复聚合邻居节点的特征来更新节点表示，让每个节点的 embedding 包含其局部图结构信息，从而在节点分类、边预测、图分类等任务上学习图结构化数据。

## 核心问题

**为什么普通神经网络处理不了图？**

MLP 和 CNN 都要求输入是固定维度的向量或规则网格（图像的像素矩阵）。图数据有两个特性：

1. **节点数量和连接关系不固定**：一个图可以有 10 个节点，另一个可以有 10 万个，节点的邻居数量也各不相同
2. **顺序不变性（Permutation Invariance）**：把节点 1 和节点 2 调换编号，图结构不变，模型输出也不应该变

把图展平成向量喂给 MLP，会丢失结构信息，也无法处理不同大小的图。

**GNN 的核心思路**：不把图映射成固定结构，而是让每个节点直接在图上做"消息传递"——每轮从邻居收集信息、更新自己的表示，重复 $L$ 层，最终每个节点的 embedding 包含了 $L$ 跳范围内的结构信息。

---

## 消息传递框架（MPNN）

几乎所有 GNN 变体都可以写成统一的**消息传递**（Message Passing）形式：

$$\mathbf{h}_v^{(l+1)} = \text{UPDATE}\!\left(\mathbf{h}_v^{(l)},\ \text{AGGREGATE}\!\left(\{\mathbf{h}_u^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

- $\mathbf{h}_v^{(l)}$：节点 $v$ 在第 $l$ 层的表示（embedding）
- $\mathcal{N}(v)$：节点 $v$ 的邻居集合
- **AGGREGATE**：把所有邻居的表示汇聚成一个向量（求和/均值/最大值/attention 加权）
- **UPDATE**：用聚合结果更新自身表示（线性变换 + 激活函数）

不同 GNN 变体的区别，本质上就是 AGGREGATE 和 UPDATE 函数的设计差异。

---

## GCN：最基础的图神经网络

**论文**：Semi-Supervised Classification with Graph Convolutional Networks（Kipf & Welling, ICLR 2017）

### 核心公式

$$\mathbf{H}^{(l+1)} = \sigma\!\left(\tilde{A}_{\text{norm}} \mathbf{H}^{(l)} \mathbf{W}^{(l)}\right)$$

其中：
- $\mathbf{H}^{(l)} \in \mathbb{R}^{N \times d}$：所有节点的第 $l$ 层表示，$N$ 个节点，每个 $d$ 维
- $\mathbf{W}^{(l)}$：可学习权重矩阵（线性变换）
- $\sigma$：激活函数（ReLU）
- $\tilde{A}_{\text{norm}}$：归一化邻接矩阵

### 归一化邻接矩阵的构造

**第一步：加自环**：$\tilde{A} = A + I$

> 没有自环时，聚合操作只考虑邻居，不包含节点自身的特征。加上单位矩阵 $I$，等于把自己也算作自己的邻居。

**第二步：对称归一化**：$\tilde{A}_{\text{norm}} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$

其中 $\tilde{D}_{ii} = \sum_j \tilde{A}_{ij}$（每行之和，即加自环后的度数）。

> 为什么要归一化？若不归一化，度数大的节点（有很多邻居）聚合后的值会很大，度数小的节点值很小，数值范围不稳定，训练困难。归一化后，聚合操作变成加权平均（类似 Transformer 里的 softmax），数值稳定。

### 矩阵乘法的图含义

$\tilde{A}_{\text{norm}} \mathbf{H}$ 的第 $v$ 行 = 节点 $v$ 及其邻居的特征加权平均：

$$(\tilde{A}_{\text{norm}} \mathbf{H})_v = \sum_{u \in \mathcal{N}(v) \cup \{v\}} \frac{1}{\sqrt{\tilde{d}_v \tilde{d}_u}} \mathbf{h}_u$$

这就是"消息传递"在 GCN 里的具体实现：每个节点收到邻居发来的消息（特征向量），按度数归一化后求和。

---

## 两层 GCN 的感受野

```
第 0 层：每个节点只有自身特征
第 1 层：每个节点看到 1 跳邻居
第 2 层：每个节点看到 2 跳邻居（邻居的邻居）
第 L 层：每个节点看到 L 跳内的所有节点
```

这类似于 CNN 中的感受野——叠加多层，每个节点"看到"的范围逐渐扩大。

**过平滑（Over-Smoothing）问题**：层数太多（$L > 6$）后，所有节点的表示趋向相同（不断平均，区分度消失）。实践中 GCN 通常用 2-3 层。

---

## 主要 GNN 变体

| 变体 | AGGREGATE 方式 | 特点 |
|---|---|---|
| **GCN**（Kipf 2017）| 归一化求和 | 最简单，固定权重，无法区分邻居重要性 |
| **GraphSAGE**（Hamilton 2017）| 均值/最大值/LSTM | 归纳学习（可泛化到未见节点）|
| **GAT**（Veličković 2018）| attention 加权求和 | 学习每条边的重要性权重 |
| **GIN**（Xu 2019）| 求和 + MLP | 理论上表达能力最强（等价于 WL 图同构测试）|
| **MPNN**（Gilmer 2017）| 可学习消息函数 | 通用框架，化学分子性质预测 |

### GAT：用 Attention 替换固定权重

GCN 的邻居权重是固定的（度数的函数）。GAT 让模型学习"哪个邻居更重要"：

$$\alpha_{vu} = \frac{\exp(\text{LeakyReLU}(\mathbf{a}^T[\mathbf{W}\mathbf{h}_v \| \mathbf{W}\mathbf{h}_u]))}{\sum_{k \in \mathcal{N}(v)} \exp(\cdots)}$$

$\alpha_{vu}$ 是节点 $v$ 对邻居 $u$ 的注意力权重，由两个节点的特征共同决定。这和 Transformer 的 self-attention 在概念上完全一样——区别在于 Transformer 关注序列中的所有位置，GAT 只关注图中有边连接的邻居。

---

## 任务类型

| 任务 | 输入 | 输出 | 例子 |
|---|---|---|---|
| **节点分类** | 图 + 部分节点标签 | 所有节点的类别 | 社交网络用户分类、引用网络论文分类 |
| **边预测（链路预测）** | 图（部分边隐去） | 缺失边的概率 | 推荐系统（用户-物品图）、知识图谱补全 |
| **图分类** | 多张图 | 每张图的类别 | 分子属性预测（是否有毒）、蛋白质功能分类 |
| **图生成** | — | 新的图结构 | 药物分子设计 |

---

## 和其他模型的关系

**GNN vs Transformer**：
- Transformer 可以看作 GNN 在**全连接图**（每个节点和所有其他节点都有边）上的特例
- GNN 利用稀疏的图结构，Transformer 不假设稀疏性
- 2022 年后出现了"图 Transformer"结合两者：用 GNN 编码局部结构，用 Transformer 做全局 attention

**GNN vs CNN**：
- CNN 是 GNN 在**规则网格图**（每个像素的邻居固定为上下左右）上的特例
- GNN 处理任意图拓扑，CNN 处理固定结构

**和 wiki 里的运动预测的联系**：TrafficGen 的 MCG 编码器、Wayformer 的场景编码器，本质都是在"向量化地图 + 交通参与者"构成的图上做消息传递，是 GNN 思想在 AV 领域的工程化实现。

---

## 局限性

- **过平滑**：层数增加，节点表示趋同，2-3 层是实践上限
- **可扩展性**：大图（百万节点）全批次训练内存爆炸，需要 mini-batch 采样（GraphSAGE、ClusterGCN）
- **异质图**：节点和边有多种类型时，标准 GCN 不适用（需要 Heterogeneous GNN）
- **表达能力上限**：GCN/GAT 的表达能力不超过 1-WL 图同构测试，无法区分某些非同构图（GIN 到达理论上限但仍有此约束）

---

## Demo 代码

代码：`src/gnn_demo.py`

**任务**：Zachary's Karate Club 节点分类——34 个人，78 条友谊关系，因矛盾分裂成两派。只用节点 0（派系 A 领袖）和节点 33（派系 B 领袖）作为标签，GCN 推断其余 32 个节点的归属。

**结果**：仅用 2 个标注节点（半监督），200 轮后全部 34 个节点正确分类（100%）。

**核心 GCN 实现**（约 15 行）：

```python
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, A_norm, H):
        return A_norm @ self.W(H)  # 图聚合：A_norm · (H · W)

class GCN(nn.Module):
    def __init__(self, in_features, hidden, num_classes):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden)
        self.layer2 = GCNLayer(hidden, num_classes)

    def forward(self, A_norm, X):
        H = F.relu(self.layer1(A_norm, X))
        return self.layer2(A_norm, H)
```

训练只在有标签节点上算 loss，无标签节点的 embedding 通过图结构传播自动学习：

```python
loss = F.cross_entropy(logits[train_mask], y[train_mask])
```

运行：

```bash
python src/gnn_demo.py
```

---

## 和 wiki 内其他概念的关联

- [TrafficGen](../30-papers/trafficgen-2210.06609.md)：MCG（Multi-Context Gating）是线性复杂度的 GNN 近似，把地图向量看作图节点，做消息聚合
- [Wayformer](../30-papers/wayformer-2207.05844.md)：场景编码器本质是在 agent + 地图元素构成的图上做 attention（GAT 的思路）
- [MotionDiffuser](../30-papers/motiondiffuser-2306.03083.md)：denoiser 中跨 agent 的 self-attention 是 GNN 在全连接 agent 图上的特例
- [MCTS](mcts.md)：MCTS 的树也是图，但 GNN 的图是无向/有环的，MCTS 的树是有向无环图；AlphaGo 用 CNN 而非 GNN，但后续工作（AlphaTensor）用了 GNN
- [概率分布速查](probability-distributions.md)：GNN 的节点分类输出是 softmax 概率；图生成任务中边的存在概率常用 Bernoulli 分布建模

## 值得看的部分 / 相关资料

- **Kipf & Welling 2017**（arXiv:1609.02907）：GCN 原论文，3 页 Method，推导清晰，矩阵形式直观
- **Hamilton et al. 2017**（GraphSAGE，arXiv:1706.02216）：归纳学习，解决 GCN 不能泛化到新节点的问题
- **Veličković et al. 2018**（GAT，arXiv:1710.10903）：引入 attention，2 页核心公式
- **Xu et al. 2019**（GIN，arXiv:1810.00826）：理论分析 GNN 表达能力，"如何让 GNN 尽可能强"
- **Stanford CS224W**（课程）：图机器学习最完整的公开课，覆盖 GCN/GAT/GIN/图生成/知识图谱，有配套代码
