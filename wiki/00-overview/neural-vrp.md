# Neural Combinatorial Optimization for VRP：用神经网络解路径规划问题

⚠️ 内容基于公开论文和综述资料综合整理，非单篇原文。引用的实验数字均来自对应论文，整体叙述框架为综合分析。

一句话总结：把 VRP 建模成序列决策问题，用 encoder-decoder + attention 端到端学出启发式求解器——训练一次，推理比传统 OR 求解器快 100-1000 倍，解质量在中等规模（100 节点以内）接近最优。

## 核心问题

**VRP（Vehicle Routing Problem）的两难困境：**

传统精确求解（branch-and-bound、branch-and-price）在 50 节点以下可行，100 节点以上指数爆炸；传统启发式（LKH、OR-Tools）工程复杂、调参难、每个实例从头搜索，无法利用历史经验。

**神经组合优化（Neural Combinatorial Optimization, NCO）的核心洞察：**
VRP 实例在同一分布下共享结构，神经网络可以从大量实例中"学出通用的求解模式"——把人类设计启发式规则的工程劳动，转换成模型训练。推理时给定新实例，直接前向传播输出路线，无需逐实例搜索。

**与 RL 策略/价值的关系：** NCO 通常把路线构建（一步一步选下一个节点）建模成序列决策，encoder 扮演状态表示，decoder 的每步 attention 扮演策略（选哪个节点）。不需要显式的价值网络——reward 是最终路线总长度的负值，用 REINFORCE 直接训练策略。

## 演化路线（2015–2026）

```
Pointer Network（Vinyals 2015）
  ↓ 用 attention 替代 RNN，引入 REINFORCE 训练
Attention Model / AM（Kool 2019, ICLR）
  ↓ 利用解的对称性，多起点并行训练
POMO（Kwon 2020, NeurIPS）
  ↓ 与传统 OR 搜索结合，做大规模实例
Hybrid 方向（NLNS/LKH-3-Neural, 2021+）
  ↓ 跨分布泛化、更复杂约束
当前前沿（2023–2026）
```

---

## 核心方法：Attention Model（AM）

**论文：** Attention, Learn to Solve Routing Problems!（Kool, van Hoof, Welling; ICLR 2019, arXiv:1803.08475）

### 问题形式化

构建一条路线 = 从所有节点里自回归地逐步选下一个访问节点，直到所有节点被覆盖。策略 $\pi_\theta$ 的概率分解为：

$$p_\theta(\pi | s) = \prod_{t=1}^{N} p_\theta(\pi_t | s, \pi_{1:t-1})$$

其中 $s$ 是问题实例（节点坐标 + 需求量），$\pi_t$ 是第 $t$ 步选的节点。

### 模型架构

**Encoder（问题实例 → 节点 embedding）：**
- 输入：$N$ 个节点各自的特征（坐标 $(x_i, y_i)$，对 CVRP 还有需求量 $d_i$）
- $L$ 层 Transformer 编码器（$L=3$），输出每个节点的 embedding $h_i \in \mathbb{R}^{128}$
- 一个额外的 graph embedding（所有 $h_i$ 的均值）代表"全局状态"

**Decoder（自回归逐步选节点）：**
- 当前步的查询向量 $q_c$ 由三部分拼接：graph embedding + 最后访问节点的 embedding + 第一个节点的 embedding（对 TSP，CVRP 则额外加剩余容量）
- 单头 attention 计算 compatibility：$u_j = \frac{q_c \cdot h_j}{\sqrt{d_k}}$，再用 tanh 裁剪到 $[-C, C]$（$C=10$，防止梯度爆炸）
- mask 掉已访问节点和违反容量约束的节点，softmax 得到选择概率

### 训练：REINFORCE + Greedy Rollout Baseline

奖励信号是路线总长度的负值（越短越好）。直接用 REINFORCE 训练方差太大，AM 用 **greedy rollout baseline**：

$$\mathcal{L}(\theta) = \mathbb{E}_\pi[(L(\pi) - b(s)) \nabla \log p_\theta(\pi | s)]$$

其中 baseline $b(s)$ 是用当前最佳参数 $\theta_{BL}$ 以贪心方式构建的路线长度。每隔若干 epoch 用统计检验判断新参数是否显著优于旧参数，若是则更新 baseline 参数。这比 value network baseline 更稳定，也比 exponential moving average 方差更低。

### 关键结果

TSP100（100 节点旅行商问题）：
- AM（greedy）：4.53%（相对最优解的 gap）
- AM（sampling 1280次）：0.52%
- LKH（传统最优启发式）：0.00%（作为参考）
- 推理速度：比 LKH 快约 1000 倍（但解质量差一些）

CVRP100（容量约束 VRP）：
- AM（greedy）：6.21% gap
- AM（sampling）：2.34% gap

---

## POMO：利用解的对称性

**论文：** POMO: Policy Optimization with Multiple Optima for Reinforcement Learning（Kwon et al.; NeurIPS 2020, arXiv:2010.16011）

### 核心洞察

TSP/VRP 的最优解有 **旋转对称性**：同一最优路线，从任意节点出发都是最优的，有 $N$ 个等价起点。AM 每次只从一个起点训练，浪费了大量对称信息。

POMO 的做法：**每个训练实例同时从所有 $N$ 个节点出发，产生 $N$ 条路线，用所有 $N$ 条路线的均值作为 baseline**（而不是 greedy rollout）：

$$b(s) = \frac{1}{N} \sum_{i=1}^{N} L(\pi^{(i)})$$

其中 $\pi^{(i)}$ 是从第 $i$ 个节点出发构建的完整路线。这个 baseline 方差极低（N 条路线天然做了平均），训练信号更稳定。

### Augmentation-based Inference

推理时利用实例的几何对称性（旋转 / 翻转），把同一实例生成 8 个等价视图（4 旋转 × 2 翻转），同时求解，取最短的路线。这是一个 zero-cost 的近似改进——不需要额外的采样，只需要把输入坐标变换 8 次。

### 关键结果

TSP100：
- POMO（greedy）：0.54% gap（AM greedy 是 4.53%，降了一个数量级）
- POMO（augment×8）：**0.14% gap**
- 推理时间比 AM 快，因为 N 个起点可以并行跑

CVRP100：
- POMO（augment×8）：大幅优于 AM，接近 LKH-3

---

## 异构车队：Dual Decoder HCVRP

**论文：** Deep Reinforcement Learning for Solving the Heterogeneous Capacitated Vehicle Routing Problem（Li et al.; arXiv:2110.02629）

真实 VRP 中车辆容量往往不同（大型货车 vs 小型厢式），AM/POMO 假设同质车队。HCVRP 的挑战是每步决策变成两个耦合选择：**选哪辆车** + **这辆车去哪个节点**。

解法：**两个解码器并联**：
- Vehicle Selection Decoder：attention over 所有车辆的当前状态，输出选哪辆车
- Node Selection Decoder：给定已选车辆，attention over 所有节点，输出选哪个节点

两个解码器共享同一个 encoder 的节点 embedding，联合训练。这是你最初提到的"两个网络"的一种实现——但动机不是 bootstrap 估计，而是分解一个两阶段决策。

---

## 泛化问题：从分布内到分布外

AM/POMO 的共同弱点：**训练分布和测试分布高度绑定**。在 $n=100$ 均匀随机实例上训练的模型，遇到 $n=200$、聚类分布或真实城市地图时性能大幅下降。

### 当前主要解法方向

**1. 元学习 / 快速适应**（ECML PKDD 2022, Manchanda et al.）：把不同分布的 VRP 实例当成不同 task，用 MAML 类方法训练能快速 finetune 的初始化。推理时用少量该分布实例快速适配。

**2. 对称性/等变性建模**（BQ-NCO, arXiv:2301.03313）：把 VRP 建模成 MDP，显式利用问题的 bisimulation 对称性缩减状态空间，理论上保证某类变换下的泛化。用 imitation learning（专家路线监督）替代 RL 训练，收敛更稳定。

**3. 大模型 + 提示**（2024 年新兴方向）：用 LLM 直接做路线规划推理，或用 LLM 生成求解器代码。这条路线在小实例（$n<30$）上有竞争力，但大实例上仍远落后于专用 NCO 模型。

**4. 与传统 OR 搜索混合**：Neural Large Neighborhood Search（NLNS）——用神经网络决定"破坏"哪些路段，再用 OR 搜索修复。利用 OR 的精确性和神经网络的模式识别，互补。

---

## 当前（2026 年）研究前沿

| 方向 | 代表工作 | 核心思路 |
|---|---|---|
| 大规模实例（$n>1000$）| Divide-and-Conquer NCO | 把大实例分解成小子问题，子问题内跑 AM/POMO，再合并 |
| 实时/动态 VRP | 在线 RL，用 attention 处理新订单插入 | 每有新订单到来，重新推理而不是重新训练 |
| 多约束 VRP（时间窗/多仓库）| VRPTW 专用模型 | 在 decoder 的 mask 逻辑里编码更复杂约束 |
| 跨分布泛化 | Foundation Model for CO | 用海量多样实例预训练，少量 finetune 到目标分布 |
| LLM 辅助 | LLM 作为 heuristic 生成器或 meta-solver | 让 LLM 生成初始解 or 搜索策略，再用 local search 改善 |

**当前性能参照点（TSP/CVRP 100 节点）：**
- 精确求解（Concorde/LKH-3）：0% gap，但分钟级
- POMO + augmentation：~0.14% gap on TSP100，毫秒级
- OR-Tools（启发式）：~1-2% gap，秒级

**实际落地的判断：** 对于 $n<200$、分布相对固定的 VRP（外卖配送、同城快递），NCO 已可在毫秒级给出接近最优的解，有明确工业价值。$n>500$ 的大规模或强约束场景，混合 NCO+OR 是主流方向，纯 NCO 仍有差距。

---

## 局限性

- **大规模实例性能差距**：$n>200$ 时 POMO 等纯 NCO 方法和 LKH-3 的 gap 显著扩大
- **训练分布依赖**：真实城市路网（非均匀随机）需要专门训练或 finetune，开箱即用效果差
- **约束越复杂越难处理**：时间窗、多仓库、车辆异构叠加时，mask 逻辑复杂，模型容量和训练难度陡增
- **缺乏最优性保证**：神经网络输出是启发解，无法像 branch-and-bound 一样提供最优性证明

---

## 现状与影响

一句话定性：**NCO for VRP 已从学术玩具走向工业可用——POMO 系架构是 2020-2025 年的主流基线，当前前沿在"如何跨分布泛化"和"如何处理大规模复杂约束"两个方向推进；纯神经网络方案已在中等规模问题上超越传统启发式，大规模场景是 Neural+OR 混合的天下。**

- AM（2019）定义了"attention encoder-decoder + REINFORCE"的标准范式，此后的工作几乎都在此基础上改进
- POMO（2020）把解的对称性利用提升到核心地位，augmentation 推理成为低成本提升质量的标准技巧
- 2022-2024 年的主题是泛化：如何让模型不再被训练分布绑架
- 2025-2026 年新兴：Foundation Model for CO（大规模预训练）和 LLM-guided search

---

## 和 wiki 内其他概念的关联

- [RLHF / PPO](../20-concepts/rlhf.md)：NCO 用 REINFORCE（策略梯度），与 RLHF 的 PPO 同属同族但更简单——没有 KL 约束，reward 来自路线长度而非人类偏好
- [REINFORCE](../20-concepts/reinforce.md)：AM/POMO 的训练算法核心；Greedy Rollout Baseline 是 REINFORCE baseline 设计的典型案例
- [Monte Carlo Tree Search（MCTS）](../20-concepts/mcts.md)：另一类"在搜索空间里找好解"的框架；MCTS 适合有明确博弈树结构的问题（棋类），NCO 适合无明确树结构的连续构建问题（路线规划）
- [Minimax 博弈](../20-concepts/minimax.md)：HCVRP 的双 decoder 不是对抗关系，但两者都体现了"把复杂决策分解为可交替优化的子问题"
- [Bellman 方程](../20-concepts/bellman-equation.md)：VRP 的每步决策满足最优子结构，理论上可以写出 Bellman 方程，但状态空间指数大，NCO 用参数化策略直接近似而不走 DP

---

## 值得看的部分 / 相关资料

- **AM 论文（arXiv:1803.08475）Section 3**：完整的 encoder-decoder 架构描述，decoder 的 compatibility 计算和 masking 设计；Figure 1 是架构图
- **POMO 论文（arXiv:2010.16011）Section 3**：多起点 baseline 的数学推导，以及 augmentation 推理的实现细节
- **HCVRP dual decoder（arXiv:2110.02629）**：展示"分解两阶段决策"在异构车队问题上的直接实现
- **BQ-NCO（arXiv:2301.03313）**：2023 年跨分布泛化的代表工作，imitation learning 替代 RL 的角度值得关注
- 入门路线推荐：先读 AM 理解基础范式 → 读 POMO 理解对称性利用 → 再看某个 hybrid NCO+OR 工作理解大规模场景

关键论文列表：
- Kool et al. 2019, *Attention, Learn to Solve Routing Problems!*（arXiv:1803.08475, ICLR 2019）
- Kwon et al. 2020, *POMO: Policy Optimization with Multiple Optima*（arXiv:2010.16011, NeurIPS 2020）
- Li et al. 2021, *DRL for Heterogeneous CVRP*（arXiv:2110.02629）
- Drakulic et al. 2023, *BQ-NCO*（arXiv:2301.03313）
- Manchanda et al. 2022, *Generalization of Neural CO Heuristics*（ECML PKDD 2022）
