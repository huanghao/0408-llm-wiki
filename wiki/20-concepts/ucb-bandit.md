# UCB1 与多臂老虎机问题

## 基本信息

- 论文：Finite-time Analysis of the Multiarmed Bandit Problem
- 作者：Peter Auer, Nicolò Cesa-Bianchi, Paul Fischer
- 期刊：Machine Learning, 47(2–3), pp. 235–256
- 年份：2002
- DOI：10.1023/A:1013689704352
- 性质：非 arxiv 论文，发表于 Springer Machine Learning 期刊

## 核心问题

**多臂老虎机问题（Multi-Armed Bandit，MAB）**：面前有 $K$ 台老虎机（arms），每台有未知的奖励分布。每轮只能拉一台，拉后得到一个随机奖励。目标：在 $T$ 轮内最大化累计奖励，等价于**最小化累计遗憾（cumulative regret）**——"如果每次都拉最优的那台，能比实际多赢多少"。

核心矛盾是 **探索-利用权衡（exploration-exploitation tradeoff）**：
- **利用（exploitation）**：一直拉目前胜率最高的那台（贪心），但可能错过真正更好的选项
- **探索（exploration）**：随机尝试不常拉的机器，但浪费了拉已知好机器的机会

本文的贡献：给出 UCB1 算法，证明它是第一个**有有限时间遗憾上界**的策略，且该上界在渐近意义下是最优的。

## 方法：UCB1 算法

算法非常简单：

**初始化**：每台机器先各拉一次，得到初始观测。

**第 $t$ 轮**（$t > K$）：选择使以下 UCB 值最大的机器 $j$：

$$\text{UCB1}(j) = \bar{x}_j + \sqrt{\frac{2 \ln t}{n_j}}$$

- $\bar{x}_j$：机器 $j$ 的当前**样本均值**（历史奖励的平均值）
- $n_j$：机器 $j$ 被拉过的次数
- $t$：当前总轮数
- $\ln t$：自然对数

### 公式的直觉

两项分别对应"利用"和"探索"：

**第一项 $\bar{x}_j$（利用）**：就是已知的平均奖励。随着 $n_j$ 增加，$\bar{x}_j$ 会收敛到真实期望奖励 $\mu_j$——大数定律保证这一点。

**第二项 $\sqrt{2 \ln t / n_j}$（探索奖励）**：这是对"我们对机器 $j$ 的估计有多不确定"的量化。三个设计决策：

1. **$\ln t$ 在分子**：随总轮数缓慢增长，确保即使算法已经探索过很多次，偶尔还会回来看冷门机器——探索不会永远停止，只是越来越少。若用 $t$（线性增长），探索奖励增长太快，会过度探索；若用常数，探索奖励最终消失为 0，会过早停止探索。对数是"不快不慢"的中间选择。

2. **$n_j$ 在分母**：机器 $j$ 被拉得越多，估计越准确，不确定性越小，探索奖励越低——自然地惩罚已经充分探索的机器。

3. **整体加 $\sqrt{}$**：把量纲从"方差量级"压缩到"均值量级"，确保探索奖励和利用项（均值）在数值上可以直接相加比较。数学上，$\sqrt{2\ln t / n_j}$ 是 Hoeffding 不等式导出的置信区间宽度的上界——$\bar{x}_j + \sqrt{2\ln t/n_j}$ 以高概率是真实均值 $\mu_j$ 的**乐观上界（optimistic upper bound）**。算法总是选上界最大的机器，即"面对不确定性时保持乐观（optimism in the face of uncertainty）"。

## 关键结果：有限时间遗憾上界（Theorem 1）

论文证明：对任意 $T$ 轮，UCB1 的期望累计遗憾满足：

$$\mathbb{E}[\text{Regret}(T)] \leq \sum_{i: \mu_i < \mu^*} \frac{8 \ln T}{\Delta_i} + \left(1 + \frac{\pi^2}{3}\right) \sum_{i=1}^{K} \Delta_i$$

其中：
- $\mu^* = \max_j \mu_j$：最优机器的真实期望奖励
- $\Delta_i = \mu^* - \mu_i$：机器 $i$ 与最优机器的差距（gap）

**简化版（大 $O$ 符号）**：

$$\mathbb{E}[\text{Regret}(T)] = O\left(\sqrt{KT \ln T}\right)$$

### 如何理解这个界

- **对数增长**：遗憾随轮数 $T$ 的增长速度只有 $O(\ln T)$（精确界），意味着平均每轮的遗憾以 $O(\ln T / T) \to 0$ 的速度趋向 0——算法最终一定会收敛到最优机器
- **最优性（渐近下界）**：Lai 和 Robbins（1985）证明任何算法的遗憾都有 $\Omega(\ln T)$ 的下界。UCB1 的遗憾上界也是 $O(\ln T)$，说明 UCB1 在渐近意义下是**最优的**（匹配理论下界的常数因子）
- **gap 的影响**：$\Delta_i$ 越小（机器越难区分），遗憾越大——这符合直觉：两台胜率几乎一样的机器需要更多探索才能分辨出哪个更好

## 局限性

- **奖励必须有界**：UCB1 假设奖励在 $[0,1]$ 范围内（或等价地有界）。对无界分布需要修改
- **独立同分布（i.i.d.）假设**：每台机器的奖励分布固定不变。对非平稳（non-stationary）环境（奖励分布随时间变化）需要别的算法（如 sliding window UCB）
- **已知 $K$（机器数量）**：标准 UCB1 需要提前知道有多少台机器
- **均值是好的统计量**：如果目标不是最大化均值（如最大化 median），需要修改

## 现状与影响

一句话定性：**UCB1 是多臂老虎机领域的奠基性算法，"乐观面对不确定性"的设计原则至今是探索-利用权衡的核心思路，UCB 系列公式被直接用于 MCTS（AlphaGo）、A/B 测试、推荐系统等众多场景。**

- **MCTS 的直接来源**：UCB1 是 MCTS 中 Selection 步骤使用的 UCT（UCB applied to Trees）公式的理论基础，Kocsis & Szepesvári（2006）将 UCB1 应用到树搜索中，产生了 UCT = MCTS 的现代形式。AlphaGo 使用的 UCB 公式正是 UCT 的变体
- **工业界广泛使用**：在线广告的点击率优化（探索哪个广告）、推荐系统的新内容曝光策略、临床试验的自适应设计，都是 MAB 框架的直接应用
- **延伸方向**：Contextual Bandit（输入有上下文特征，如 LinUCB）、Adversarial Bandit（奖励由对手设置，Exp3 算法）、Best Arm Identification（纯探索，不追求遗憾最小化）

## 和 wiki 内其他概念的关联

- [MCTS](./mcts.md)：UCB1 是 MCTS Selection 步骤的理论来源。MCTS 中每个节点的 UCB 公式 $W(v)/N(v) + c\sqrt{\ln N(\text{parent})/N(v)}$ 正是 UCB1 的直接应用（Kocsis & Szepesvári 2006 将其形式化为 UCT）
- [REINFORCE / PPO](./reinforce.md)：MAB 问题是强化学习的简化版（单步、无状态转移）。UCB1 的探索策略和 PPO 的 KL 惩罚项都在做"不要离已知策略太远"的约束，来自相似的信息理论直觉
- [RLHF](./rlhf.md)：RLHF 中的 reward model 训练本质上也是一个探索问题——选择哪些 prompt 让人类标注，就是 contextual bandit 问题

## 值得看的部分 / 相关资料

- **原论文**：Auer, Cesa-Bianchi, Fischer (2002). *Finite-time Analysis of the Multiarmed Bandit Problem*. Machine Learning 47, 235–256. DOI: 10.1023/A:1013689704352
- **直觉讲解**：Jeremy Kun, *Optimism in the Face of Uncertainty: the UCB1 Algorithm*（2013）— 配有代码的清晰教程
- **MCTS 连接**：Kocsis & Szepesvári (2006). *Bandit Based Monte-Carlo Planning*. ECML 2006 — UCB1 → UCT → 现代 MCTS 的直接推导
- **理论下界**：Lai & Robbins (1985). *Asymptotically Efficient Adaptive Allocation Rules*. Advances in Applied Mathematics — 证明 $\Omega(\ln T)$ 是任何算法的遗憾下界，UCB1 匹配这个下界
