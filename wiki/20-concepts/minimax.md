# Minimax 博弈

## 是什么

Minimax 是博弈论里描述**两个目标相反的参与者之间对抗**的框架：

- 一方（minimizer）想让某个值**尽量小**
- 另一方（maximizer）想让同一个值**尽量大**
- 两者同时优化，最终达到一个谁都无法单方面改善的均衡

数学形式：

$$\min_x \max_y f(x, y)$$

读作：先找 y 让 f 最大，再找 x 让这个最大值最小。等价地理解为：minimizer 在 maximizer 的最坏情况下仍然找到最优策略。

---

## 期望最优 vs 鲁棒最优

这两种"最优"的区别是 minimax 的核心：

**期望最优**：假设情况是"平均水平"，在这个平均情况下表现最好。
> 例：你带一把伞，如果下雨概率 20%，期望最优的决策是"不带"（80% 的时候不需要）。

**鲁棒最优（minimax 最优）**：假设情况是"最坏情况"，在最坏情况下仍然表现最好。
> 例：你参加一个重要面试，哪怕只有 20% 下雨概率，鲁棒最优的决策是"带伞"（万一下雨了会很狼狈）。

**在 DoReMi 里**：期望最优是"加权平均损失最小"，鲁棒最优是"最差的那个 domain 损失最小"。如果只优化平均，模型可能在大 domain（如 Pile-CC）表现很好，而忽略小 domain（如 FreeLaw）——鲁棒最优要求任何 domain 都不能被牺牲太多。

---

## 直觉：两个棋手的博弈

最经典的类比是下棋（Minimax 算法起源于此）：

- 你是 minimizer，想让自己的损失最小
- 对手是 maximizer，总是走对你最不利的一步
- 你的策略：假设对手每次都走最坏的那步，然后在这个假设下找到你的最优走法

这就是"最坏情况下的最优"——不是期望最优，而是鲁棒最优。

**最小最大后悔值（Minimax Regret）**是经济学/决策理论里的相关概念，和 minimax 同族但有区别：
- **Minimax**：最小化最坏情况下的绝对损失
- **Minimax Regret**：最小化最坏情况下的"后悔值"——即"如果当时做了最好的决策，我能少损失多少"

例：你在投资，不确定明天涨还是跌。
- Minimax：选一个策略，让最坏结果（亏损）尽量小
- Minimax Regret：选一个策略，让你事后"后悔自己没有选另一个策略"的程度尽量小

两者的差别在于参照点：minimax 以零为参照（损失的绝对值），minimax regret 以"最优决策"为参照（相对损失）。

---

## 一个上手的简单例子

不用神经网络，用一个数值游戏来理解：

```python
# 两个玩家，各选一个数字
# 玩家 A（minimizer）选 x ∈ {1, 2, 3}
# 玩家 B（maximizer）选 y ∈ {1, 2, 3}
# 收益矩阵 f(x, y)（A 想最小化，B 想最大化）

f = {
    (1,1): 3,  (1,2): 1,  (1,3): 2,
    (2,1): 2,  (2,2): 4,  (2,3): 1,
    (3,1): 1,  (3,2): 2,  (3,3): 3,
}

# B 的最优响应（对每个 x，B 选让 f 最大的 y）
for x in [1, 2, 3]:
    best_y = max([1,2,3], key=lambda y: f[(x,y)])
    print(f"x={x}: B 会选 y={best_y}，f={f[(x,best_y)]}")
# x=1: B 会选 y=1，f=3
# x=2: B 会选 y=2，f=4
# x=3: B 会选 y=3，f=3

# A 的 minimax 决策：在 B 最优响应下，选让 f 最小的 x
worst_case = {x: max(f[(x,y)] for y in [1,2,3]) for x in [1,2,3]}
# worst_case = {1: 3, 2: 4, 3: 3}
best_x = min(worst_case, key=worst_case.get)
print(f"A 的 minimax 选择：x={best_x}，最坏情况 f={worst_case[best_x]}")
# A 的 minimax 选择：x=1（或 x=3），最坏情况 f=3
```

可以直接运行，输出告诉你 A 应该选 x=1 或 x=3，这样即使 B 做出最坏的响应，损失也只有 3 而不是 4。

---

## 在机器学习里的三个典型应用

### 1. GAN（生成对抗网络）

生成器 G 和判别器 D 之间的对抗：

$$\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1 - D(G(z)))]$$

- D（maximizer）：想把真假样本分得越清楚越好
- G（minimizer）：想骗过 D，让 D 以为生成的样本是真的

训练时两者交替更新：先更新 D 几步，再更新 G 一步，循环往复。理论上均衡时 G 生成的分布等于真实数据分布，D 对任何样本的判断都是 0.5（无法区分真假）。

**问题**：实际训练极不稳定，两者容易失衡——D 太强或太弱都会导致 G 无法有效学习。

### 2. DoReMi（数据混合配方优化）

domain weights α（maximizer）和代理模型 θ（minimizer）之间的对抗：

$$\min_\theta \max_\alpha \sum_i \alpha_i \cdot \text{excess\_loss}_i(\theta)$$

- α（maximizer）：把权重集中在超额损失最高的 domain，迫使模型关注最弱的地方
- θ（minimizer）：努力降低所有 domain 的超额损失

训练时同样交替更新：每步先用当前 θ 计算各 domain 的超额损失 → 更新 α（指数梯度上升）→ 用新 α 更新 θ（梯度下降）。

均衡时：没有任何一个 domain 的超额损失显著高于其他 domain，所有 domain 都被"照顾到"了。

### 3. RLHF / PPO 中的 KL 约束

PPO 的目标也有 minimax 结构（对偶形式）：

$$\min_\pi \max_\lambda \left[ -\mathbb{E}[r(x)] + \lambda \cdot D_{\text{KL}}(\pi \| \pi_{\text{ref}}) \right]$$

- λ（Lagrange 乘子，隐式 maximizer）：约束 KL 不超过上限
- π（minimizer）：最大化奖励同时不偏离参考策略太远

实践中通常直接加 KL 惩罚项 $\beta \cdot D_{\text{KL}}$，而不显式求解 minimax，但背后的数学结构相同。

---

## 训练时的震荡问题

Minimax 优化**天然会震荡**，这不是 bug，而是博弈均衡的特性：

```
alpha 上升（maximizer 发现 domain A 超额损失高）
    → proxy_model 多学 domain A（minimizer 响应）
    → domain A 的超额损失下降
    → alpha 下降
    → proxy_model 减少对 domain A 的关注
    → 超额损失又上升
    → alpha 又上升……
```

这个循环永远不会"停下来收敛到一个点"，而是围绕均衡点振荡。

**解决方案：取训练轨迹的平均值**

在线学习理论（Online Learning Theory）证明：对于凸-凹的 minimax 问题，参数轨迹的时间平均值会收敛到均衡点（Nemirovski et al. 2009）。

直觉：虽然每一步 alpha 都在震荡，但振荡是围绕真正均衡值上下波动的，平均后噪声抵消，剩下的就是均衡值。

```python
# 不用最后一步（不稳定）
final_alpha = alpha_history[-1]        # ❌

# 用所有步的平均（收敛到均衡）
final_alpha = mean(alpha_history)      # ✓
```

这就是为什么 DoReMi 的输出是 $\bar{\alpha} = \frac{1}{T}\sum_{t=1}^T \alpha_t$，而不是最后一步的 $\alpha_T$。

---

## 凸-凹条件

取平均收敛的前提是问题是**凸-凹**的（convex-concave）：

- f(x, y) 对 x 是凸的（minimizer 的问题是凸优化）
- f(x, y) 对 y 是凹的（maximizer 的问题是凹优化）

DoReMi 满足这个条件：
- 对 α（最大化方）：线性函数，既凸又凹
- 对 θ（最小化方）：NLL loss 对神经网络参数是非凸的，但实践中近似成立

GAN **不满足**这个条件（判别器和生成器都是非凸非凹的神经网络），这就是 GAN 训练不稳定的根本原因。

---

## 和 Nash 均衡的关系

**Nash 均衡**：在一个多人博弈里，如果每个参与者在其他人策略不变的情况下，都没有理由改变自己的策略——这个状态就叫 Nash 均衡。

用上面数值例子理解：当 A 选了 x=1，B 选了 y=1（f=3）。
- A 想改吗？改成 x=2，B 还是选 y=2，f=4，更差。改成 x=3，f=3，一样。所以 A 不想改（或改了也没有更好）。
- B 想改吗？B 已经在 x=1 下选了让 f 最大的 y=1，没有更好的选择。

双方都不想单方面改变 → 这就是 Nash 均衡。

Minimax 均衡在**零和博弈**（一方的收益 = 另一方的损失，收益之和为零）里等价于 Nash 均衡。

| | GAN | DoReMi |
|--|-----|--------|
| 理论均衡 | G 生成真实分布，D 判断概率 = 0.5（无法区分真假） | 所有 domain 超额损失相等（没有特别难的 domain）|
| 实践中能达到吗 | 很难（非凸凹，GAN 经常崩） | 近似能（接近凸凹，振荡平均后足够好）|

均衡是一个"稳定状态"的描述，不是说训练会停在那里——minimax 训练天然震荡，但震荡的中心就是均衡点。

---

## 相关概念

- **[Loss Functions](./loss-functions.md)**：DoReMi 的 Group DRO 目标是 minimax loss 的一个实例
- **[RLHF](./rlhf.md)**：PPO 的 KL 约束有 minimax 的对偶结构
- **[DoReMi](../30-papers/doremi-2305.10429.md)**：minimax 在数据混合配方优化里的具体应用
