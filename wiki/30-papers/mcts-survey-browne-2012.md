# A Survey of Monte Carlo Tree Search Methods（Browne et al., 2012）

一句话总结：MCTS 领域第一篇系统性综述，覆盖算法推导、近 250 篇文献、主要变体与增强、围棋以外的应用，是理解 UCT 生态的标准参考文献。

## 基本信息

- 论文：A Survey of Monte Carlo Tree Search Methods
- 作者：Cameron Browne, Edward Powley, Daniel Whitehouse, Simon Lucas, Peter I. Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, Simon Colton
- 机构：Imperial College London（Browne, Tavener, Colton）；University of Essex（Lucas, Rohlfshagen, Perez, Samothrakis）；University of Bradford（Powley, Whitehouse, Cowling）
- 发表：IEEE Transactions on Computational Intelligence and AI in Games, Vol. 4, No. 1, March 2012
- DOI：10.1109/TCIAIG.2012.2186810
- 本地原文：`raw/inbox/mcts-survey-browne-2012.pdf`

## 核心问题

MCTS 在 2006 年因 UCT 的提出和围棋 AI 的突破而爆发，到 2011 年已有近 250 篇相关论文（接近每周一篇）。本文的目标是：

1. 梳理 MCTS 的理论来源（bandit 问题、UCB1、UCT 的推导链）
2. 系统分类主要变体（单人、多人、非确定性、实时、递归等）
3. 整理树策略增强（UCB1-Tuned、RAVE、Bayesian UCT 等约 40 种）
4. 汇总非游戏应用（组合优化、规划、调度、内容生成等）
5. 提炼优势、局限与开放研究方向

## 方法 / 核心机制

### MCTS 算法框架（Section 3.1）

论文给出标准算法伪码（Algorithm 1）：

```
function MCTSSEARCH(s₀):
  create root v₀ with state s₀
  while within computational budget do:
    vₗ ← TREEPOLICY(v₀)       # Selection + Expansion
    Δ ← DEFAULTPOLICY(s(vₗ))  # Simulation/Rollout
    BACKUP(vₗ, Δ)              # Backpropagation
  return a(BESTCHILD(v₀, 0))
```

四步循环（如图 2 所示）：
1. **Selection**：从根节点向下，用树策略（tree policy）选择"最紧迫的可扩展节点"
2. **Expansion**：将该节点的一个（或多个）未尝试动作加入树
3. **Simulation / Rollout**：从新节点出发，按默认策略（default policy，最简为均匀随机）模拟到终局，得到 $\Delta$
4. **Backpropagation**：沿路径更新所有节点的 $N(v)$（访问次数）和 $Q(v)$（累计奖励）

返回动作时有四种策略：Max child（最高奖励）、Robust child（最多访问）、Max-Robust child（两者兼顾）、Secure child（最大置信下界）。

### UCT 算法（Section 3.3）

**UCT = UCB1 Applied to Trees**，是 MCTS 最主流的具体实现：

$$\text{UCT}(j) = \overline{X}_j + 2C_p\sqrt{\frac{2\ln n}{n_j}}$$

- $\overline{X}_j$：子节点 $j$ 的平均奖励（利用）
- $n$：父节点访问次数，$n_j$：子节点 $j$ 的访问次数
- $C_p = 1/\sqrt{2}$ 是 Kocsis & Szepesvári 证明满足 Hoeffding 不等式的理论值，实践中可调整

UCT 的理论保证（来自 Kocsis & Szepesvári 2006，见 wiki 内对应文档）：选择次优动作的失败概率以**多项式速率**趋向 0，即给定足够时间，UCT 收敛到 minimax 最优。

### 主要变体分类（Section 4）

| 变体 | 核心思路 | 代表算法 |
|---|---|---|
| **Flat UCB** | 只在根节点用 bandit，不建树 | Flat UCB（Coquelin & Munos 2007） |
| **Single-Player MCTS** | 第三项加入方差估计 $\sqrt{\sigma^2 + D/n_i}$ | SP-MCTS（Schadd et al.）|
| **Multi-player MCTS** | 每个节点存向量奖励，各玩家独立最大化 | Paranoid UCT, Confident UCT |
| **Real-time MCTS** | 在严格时间限制下做决策，适合视频游戏 | 各种实时 UCT 变体 |
| **Nondeterministic MCTS** | 处理随机/不完全信息 | Determinization, ISUCT, Multiple MCTS |
| **Recursive MC** | 递归嵌套 MC 搜索，每层用下层 MC 估值 | Nested MC Search (NMCS), NRPA |

### 树策略增强（Section 5）

论文梳理约 40 种树策略增强，最重要的有：

**UCB1-Tuned**：用样本方差更精细地估计置信区间，将探索项从 $\sqrt{2\ln n/n_j}$ 改为 $\sqrt{(\ln n / n_j) \cdot \min(1/4, V_j(n_j))}$，其中 $V_j$ 是方差估计。实践中表现优于标准 UCB1，被围棋（MoGo）、Othello、Tron 等广泛使用，但 Auer et al. 未能为其证明遗憾界。

**RAVE（Rapid Action Value Estimation）**：假设某动作的价值在它被选择的任意上下文中都相近，用所有经过某动作的 rollout 更新该动作的统计——即使该动作不在 Selection 路径上也算。这大幅加速早期树建立。AMAF（All Moves As First）是 RAVE 的原始形式，$\alpha$-AMAF 用权重混合 UCT 和 RAVE 估计：$\alpha Q_{\text{UCT}} + (1-\alpha) Q_{\text{RAVE}}$。

**FPU（First Play Urgency）**：未访问节点赋予一个固定的乐观初始值（而非无穷大），避免每次都必须先探索所有未访问子节点。

**Progressive Bias**：在早期 $N$ 小时，UCB 公式中加入一个基于启发式的偏置项，随 $N$ 增大逐渐消失：$H(v)/N(v)$，其中 $H$ 是启发式函数。

**MCTS-Solver**（Section 5.4.1）：在树内传播已确定的胜/负证明——若某节点的所有子节点都被证明是输棋，则该节点也被标记为输棋并不再选择。这使 MCTS 在接近终局时能利用精确计算。

### 其他增强（Section 6）

- **Simulation 增强**：Rule-based、Contextual（学习 rollout 策略）、MAST（棋步统计）、PAST（模式统计）、Last Good Reply
- **Backpropagation 增强**：Score Bonus、Decay（时间衰减）、Transposition Table（合并相同局面）
- **并行化**（Section 6.3）：Leaf parallelism（叶节点并发 rollout）、Root parallelism（多棵树独立搜索后合并统计）、Tree parallelism（共享树的并发搜索）；UCT-Treesplit 是树并行的一个高效实现

## 关键结果 / 数据

### 围棋（Section 7.1）

- 2006 年 UCT 出现前，围棋 AI 与人类业余低手之间差距巨大（19×19 棋盘）
- 2007 年：CADIAPLAYER 成为通用游戏竞赛（GGP）世界冠军
- 2008 年：MOGO 在 9×9 围棋达到 *dan*（段位）级别（128 CPU 版本）
- 2009 年：FUEGO 以 9×9 击败顶级职业棋手；MOHEX 成为 Hex 世界冠军
- 2011 年综述时：MCTS 在 9×9 接近人类强手，19×19 仍是开放挑战

### 非游戏应用（Section 7.8）

- **组合优化**：TSP（Nested MC 达到 29 节点状态-of-art）；MIP（UCT 与 CPLEX 可比）；物理仿真（HOOT 优于纯 UCT）
- **规划/调度**：IPC-4 规划竞赛（MRW 表现与 Marvin/YASHP 相当）；打印调度问题（SP-MCTS 优于最优化方法）
- **约束满足**：UCTSAT 在结构化 CNF 问题上优于 CPLEX
- **内容生成（PCG）**：MCTS 的 restart 机制使其天然适合多样性目标的搜索

### Table 1：MCTS 里程碑时间线

| 年份 | 事件 |
|---|---|
| 1990 | Abramson：Monte Carlo 仿真可以评估局面价值 |
| 2002 | Auer et al.：UCB1，MAB 理论基础 |
| 2006 | Coulom：首次提出 Monte Carlo Tree Search 术语 |
| 2006 | Kocsis & Szepesvári：UCT 算法，将 UCB1 应用于树搜索 |
| 2006 | Gelly & Silver：MoGo，UCT 首次成功应用于围棋 |
| 2007 | CADIAPLAYER：GGP 世界冠军 |
| 2008 | MOGO：9×9 围棋达到 dan 级别 |
| 2009 | FUEGO：击败顶级职业棋手；MOHEX：Hex 世界冠军 |

## 局限性

论文 Section 8.3 总结了 MCTS 的已知弱点：

- **Trap states（陷阱状态）**：如果某状态在少数步内必败，MCTS 可能因探索不足而"陷入"——Ramanujan et al. 证明 UCT 在有大量陷阱状态的问题（如 Chess 的某些局面）表现不如 minimax。围棋陷阱状态少，这是 MCTS 在围棋特别成功的原因之一
- **高分支因子 + 大树深度**：纯 MCTS 会因搜索空间过大而失效，需要领域知识或剪枝来控制
- **Rollout 质量的作用被低估**：基本 MCTS 在 rollout 策略弱时表现差，需要增强
- **参数调优困难**：$C_p$、RAVE 权重 $\alpha$ 等超参数对性能影响大，但目前只能经验性调整，没有通用方法
- **理论理解不足**：MCTS 搜索动力学尚未被完全理论化；增强的叠加效果难以预测
- **仿真代价高的领域**：若每次 rollout 需要昂贵的仿真（如物理引擎），MCTS 的样本效率问题凸显

## 现状与影响

一句话定性：**Browne et al. 2012 是 MCTS 研究第一个五年（2006–2011）的"最终裁决"——它把围棋 AI 革命的方法论系统化成可复用框架，之后所有 MCTS 工作（包括 AlphaGo）都以它为参考基础，至今是该领域被引最多的综述之一。**

- **AlphaGo 的直接参考**：Silver et al. 2016（AlphaGo，Nature）明确引用了 Browne et al. 2012 对 MCTS 框架的描述，用其四步定义作为改进出发点。AlphaGo 的 Selection 步骤将 RAVE-style 的策略先验与 UCT 结合，正是本文综述的增强方向
- **标准术语来源**：Browne et al. 明确区分了 "Flat Monte Carlo"、"Flat UCB"、"MCTS"（建树）、"UCT"（UCB1 选树节点）、"Plain UCT"（最原始 Kocsis & Szepesvári 版本）——这套术语此后被该领域沿用
- **LLM 推理的间接影响**：Tree of Thoughts（2023）、AlphaZero 式推理搜索等 LLM 应用中用到的 MCTS 框架，其底层概念（四步、UCB 选节点、rollout + verifier）都有本文系统化的贡献
- **非游戏应用的号召**：2012 年时作者预测 MCTS 将在规划、调度、优化等非游戏领域大幅扩张——这一预测被 2012–2026 年的文献证实，LLM 推理是最新的延伸

## 和 wiki 内其他概念的关联

- [MCTS](../20-concepts/mcts.md)：wiki 的 MCTS 文档描述了本文综述的标准四步算法框架；UCB 公式来自 Kocsis & Szepesvári（本文第 3.3 节详述）
- [UCB1 与多臂老虎机问题](../20-concepts/ucb-bandit.md)：Auer et al. 2002 是本文第 2.4 节 Bandit-Based Methods 的核心引用，UCT 的理论基础
- [UCT（Kocsis & Szepesvári 2006）](./uct-kocsis-szepesvari-2006.md)：本文第 3.3 节对 UCT 的详细描述直接来自 Kocsis & Szepesvári，wiki 有对应文档
- [REINFORCE / PPO](../20-concepts/reinforce.md)：MCTS 与 RL 的关系（Section 4.3 Learning in MCTS）：TDL 和 MCTS 在某些条件下等价；Silver et al. 后来将 MCTS（搜索）与策略梯度（训练）组合成 AlphaZero 循环

## 值得看的部分 / 相关资料

- **Section 3（MCTS 核心算法）**：Algorithm 1–3，Figure 2，UCT 伪码，两人 backup 的 negamax 变体——最完整的标准实现描述
- **Section 5.3（AMAF/RAVE 增强）**：RAVE 是 AlphaGo 之前围棋 AI 最重要的单一增强，Section 5.3 是最系统的讲解
- **Table 1（MCTS 里程碑时间线）**：1990–2009 关键节点一览
- **Table 3–4（变体与应用速查表，原文末页）**：所有变体按游戏/非游戏分类汇总
- **Section 8（Summary）**：Impact / Strengths / Weaknesses / Research Directions，约 4 页，可作为独立阅读单元
- 后续关键工作：
  - Silver et al. 2016, *Mastering the game of Go with deep neural networks and tree search*（AlphaGo）——直接继承 UCT + RAVE 并用神经网络取代随机 rollout
  - Silver et al. 2017, *Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm*（AlphaZero）
  - Yao et al. 2023, *Tree of Thoughts*（arXiv:2305.10601）——MCTS 框架在 LLM 推理中的应用
