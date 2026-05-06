# Bandit Based Monte-Carlo Planning（UCT，Kocsis & Szepesvári, 2006）

一句话总结：把 UCB1 bandit 算法应用到树搜索的每个内部节点，提出 UCT 算法——这是现代 MCTS 的核心，也是 AlphaGo 之前围棋 AI 实现突破的关键。

## 基本信息

- 论文：Bandit Based Monte-Carlo Planning
- 作者：Levente Kocsis, Csaba Szepesvári
- 机构：Computer and Automation Research Institute, Hungarian Academy of Sciences, Budapest
- 发表：ECML 2006（European Conference on Machine Learning）, LNAI 4212, pp. 282–293
- 出版：Springer-Verlag Berlin Heidelberg 2006
- 本地原文：`raw/inbox/uct-kocsis-szepesvari-2006.pdf`

## 核心问题

大规模状态空间的 MDP（Markovian Decision Problem）中，如何高效找到近最优动作？

此前的 Monte-Carlo planning 算法在采样时有两种策略：
1. **Uniform sampling（均匀采样）**：每个动作等概率选择，简单但浪费计算在明显次优的动作上
2. **Heuristic biasing（启发式偏置）**：人工规则引导采样，需要领域知识且没有理论保证

本文的核心洞察：**每个内部节点的动作选择本质上是一个多臂老虎机（bandit）问题**——选哪个动作，就是"拉哪台机器"。只要把有理论保证的 bandit 算法（UCB1）应用到树的每个节点，就能得到一个既高效又有理论保证的 Monte-Carlo 规划算法。

## 方法：UCT 算法

### UCT = UCB Applied to Trees

UCT（Upper Confidence bounds applied to Trees）的核心思想：在树搜索的每个内部节点，把选择下一步动作视为一个独立的 bandit 问题，用 UCB1 公式选择：

在状态 $s$、深度 $d$、时间 $t$，选择动作 $a$ 最大化：

$$Q_t(s, a, d) + c_{N_{s,d}(t),\, N_{s,a,d}(t)}$$

其中：
- $Q_t(s, a, d)$：动作 $a$ 在状态 $s$、深度 $d$ 上的**估计价值**（已观测到的累计奖励均值）
- $N_{s,d}(t)$：到时间 $t$ 为止，状态 $s$ 在深度 $d$ 被访问的总次数（父节点访问次数）
- $N_{s,a,d}(t)$：动作 $a$ 在状态 $s$、深度 $d$ 被选择的次数（子节点访问次数）
- $c_{t,s} = 2C_p\sqrt{\ln t / s}$：UCB 偏置项（探索奖励），$C_p > 0$ 是常数

偏置项 $c_{t,s}$ 的形式来自 UCB1 的 $\sqrt{2\ln t / n}$，但因为树内节点的奖励序列是**非平稳的**（non-stationary，随着树被不断探索，payoff 分布会漂移），需要引入常数 $C_p$ 来补偿这种漂移。

### 算法框架（Figure 1 伪码）

```
function MonteCarloPlanning(state):
  repeat:
    search(state, 0)
  until Timeout
  return bestAction(state, 0)

function search(state, depth):
  if Terminal(state): return 0
  if Leaf(state, depth): return Evaluate(state)
  action := selectAction(state, depth)          ← UCB1 在这里
  (nextstate, reward) := simulateAction(state, action)
  q := reward + γ * search(nextstate, depth+1)  ← 递归展开
  UpdateValue(state, action, q, depth)           ← 反向传播
  return q
```

`selectAction` 使用 UCB 公式选择动作；`Evaluate` 是叶节点的快速估值（可以是随机 rollout 或启发式函数）；$\gamma$ 是折扣因子。

### 关键设计：非平稳 bandit 的处理

UCB1 的原始假设是每台机器的奖励分布固定。但在树中，当某个子树被更多探索后，从该节点出发的期望累计奖励会随着估计精度的提升而改变——奖励序列是非平稳的。

论文的主要理论贡献正是证明：**只要偏置项满足特定的"漂移条件"（drift conditions），UCT 仍然是一致的（consistent）**，即随着 episode 数趋向无穷，找到最优动作的失败概率趋向 0。

## 关键结果 / 数据

### 理论结果（Theorems 1–6）

**Theorem 1**（非平稳 UCB1 的上界）：在满足漂移条件时，次优臂 $i$ 被拉的次数 $T_i(n)$ 满足：

$$\mathbb{E}[T_i(n)] \leq \frac{16 C_p^2 \ln n}{(\Delta_i/2)^2} + 2N_0 + \frac{\pi^2}{3}$$

其中 $\Delta_i$ 是动作 $i$ 的值与最优动作的差距。

**Theorem 5（收敛性）**：选择次优动作的概率以多项式速率趋向 0：

$$P(\hat{I}_t \neq i^*) \leq C \cdot \left(\frac{1}{t}\right)^{\frac{\mu}{6}\left(\frac{\min_{i\neq i^*}\Delta_i}{56}\right)^2}$$

且 $\lim_{t\to\infty} P(\hat{I}_t \neq i^*) = 0$。

**Theorem 6（主定理）**：对有限水平 MDP（水平 $D$，每个状态最多 $K$ 个动作），使用 UCT 时根节点估计期望奖励 $\overline{X}_n$ 的偏差为 $O(\log(n)/n)$，且失败概率以**多项式速率**趋向 0。

### 实验结果

**P-games（人工对弈游戏）**：与 uniform MC、heuristic MC 对比，UCT 在 4×4 到 10×10 的棋盘上均显著优于竞争对手，且差距随棋盘变大而扩大——说明 UCT 的优势在复杂度高时更明显。

**Sailing domain（MDP 规划基准）**：同等计算预算下，UCT 的误差比 vanilla MC（均匀采样）低约一个数量级，比最接近的竞争对手也好 2–5 倍。

## 局限性

- **$C_p$ 需要调参**：探索常数 $C_p$ 对性能影响大，论文没有给出通用的设置方法，实践中需要针对具体问题调整
- **仅限有限水平 MDP**：主定理（Theorem 6）假设有限水平 $D$，无限水平（discounted）MDP 的理论更复杂
- **非平稳处理是近似的**：漂移条件的满足依赖于树结构和奖励分布的性质，在某些极端情况下可能不成立
- **实验规模较小**：2006 年的实验限于小规模游戏，没有直接在 19×19 围棋上测试

## 现状与影响

一句话定性：**UCT 是现代 MCTS 的奠基性算法，2006 年发表后迅速成为围棋 AI 的核心，直接推动了 2016 年 AlphaGo 的实现——AlphaGo 将 UCT 与深度神经网络结合，最终击败人类顶级棋手。**

截至 2026 的影响：

- **围棋革命（2006–2015）**：UCT 出现后，围棋 AI 的水平从"业余弱手"跳升到"业余强手"。此前基于规则的围棋 AI 几乎无法击败职业棋手，UCT 引入后大幅缩小了差距，催生了 MoGo、Fuego、Pachi 等一批实力大幅提升的围棋程序
- **AlphaGo 的直接前身**：AlphaGo（Silver et al. 2016，Nature）明确以 UCT 为基础，在 Selection 步骤中使用 UCB 变体，在 Simulation 步骤中用策略网络和价值网络替代随机 rollout。AlphaZero（2017）进一步简化，去掉随机 rollout，完全依赖神经网络
- **MCTS 研究爆发**：2006–2015 年间，MCTS/UCT 方向发表了数百篇论文，扩展到游戏（Hex、Go-moku、Checkers）、规划（机器人、物流）、组合优化等领域
- **LLM 推理的应用**：近年 MCTS 在 LLM 推理中的复兴（Tree of Thoughts 2023、o1 前身工作等）直接继承了 UCT 框架，把"节点 = token 序列前缀"、"rollout = LLM 生成到答案"、"verifier = reward model 打分"

## 和 wiki 内其他概念的关联

- [MCTS](../20-concepts/mcts.md)：UCT 就是现代 MCTS 的标准实现。wiki 里 MCTS 文档的 Selection 步骤 UCB 公式 $W(v)/N(v) + c\sqrt{\ln N(\text{parent})/N(v)}$ 直接来自本文
- [UCB1 与多臂老虎机问题](../20-concepts/ucb-bandit.md)：Auer et al. 2002 的 UCB1 是 UCT 的理论基础。Kocsis & Szepesvári 在此基础上扩展了非平稳情形的处理
- [REINFORCE / PPO](../20-concepts/reinforce.md)：UCT 生成的搜索轨迹（高质量 episode）可直接用于 RL 训练——AlphaZero 用 UCT 自我对弈产生数据，再用梯度下降（类 PPO）更新策略/价值网络

## 值得看的部分 / 相关资料

- **Section 2.3（The Proposed Algorithm）**：UCT 算法的核心描述，Figure 1 给出完整伪码
- **Section 2.4（Theoretical Analysis）**：Theorem 1–6 的串联，从非平稳 UCB1 → 树内漂移条件 → 整体收敛性，是本文的主要贡献
- **Section 3（Experiments）**：P-games 和 Sailing domain 的对比实验
- 后续关键工作：
  - Silver et al. 2016, *Mastering the game of Go with deep neural networks and tree search*（AlphaGo，Nature）——UCT + 神经网络的集大成之作
  - Silver et al. 2017, *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*（AlphaZero）——去掉 rollout 的纯网络版本
  - Browne et al. 2012, *A Survey of Monte Carlo Tree Search Methods*（IEEE TCIAG）——2012 年时 MCTS/UCT 研究的全面综述
