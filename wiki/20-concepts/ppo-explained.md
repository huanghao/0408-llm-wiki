# PPO 核心概念讲解

对应代码：`tools/lunarlander_sb3.py`（SB3 快速版）、`tools/ppo_demo.py`（手写教学版）

贯穿本文的例子：**LunarLander-v3**——训练一个飞船，让它学会在月球上安全降落。

---

## 核心概念对照表

| RL 概念 | 飞船着陆里的对应 | 真实 LLM RLHF 里的对应 |
|---------|----------------|----------------------|
| **agent** | 飞船控制器 | 语言模型 |
| **environment** | 月球物理引擎 | 奖励模型（RM） |
| **state / observation** | 8 维向量：位置(x,y)、速度(vx,vy)、角度、角速度、左腿/右腿是否着地 | 已生成的所有 token |
| **action** | 4 个离散动作：0=不喷火、1=左引擎、2=主引擎、3=右引擎 | 从词表里选下一个词 |
| **reward** | 每步 -0.3（鼓励快降落），着陆 +100~+140，坠毁 -100，引擎消耗 -0.03/帧 | RM 给完整回答的打分 |
| **episode / trajectory** | 一次从出生到着陆/坠毁的完整飞行 | 一次完整的回答生成 |
| **policy** | 给定当前飞船状态，输出 4 个动作的概率分布 | 语言模型的 softmax 输出 |
| **value** | 给定当前飞船状态，预测这局最终能得多少分 | 价值模型的预测 |

---

## policy 是什么

**policy（策略）** = 在当前状态下，选择每个动作的概率分布。

飞船的 policy：
```
输入：[x=0.1, y=0.5, vx=-0.2, vy=-0.8, angle=0.05, ω=0.0, 左腿=0, 右腿=0]
输出：[不喷=0.1, 左引擎=0.2, 主引擎=0.65, 右引擎=0.05]
```

policy 不是直接输出"选哪个动作"，而是输出一个概率分布，然后从中**采样**。训练初期 policy 是随机的（四个动作概率各 25%），训练结束后 policy 学会了"飞船下降太快时以高概率选主引擎"。

**action 可以同时是多个吗**：可以，取决于 action space 的设计：

- **Discrete**（离散单选）：每步只选一个动作，LunarLander 就是这样——4 个引擎里只能开一个
- **MultiDiscrete**（多个离散维度）：每步同时选多个独立动作。比如飞船同时控制"主引擎开/关"和"方向喷嘴左/中/右"，输出两个独立的概率分布，各自采样
- **Box**（连续动作）：每步输出一个连续值向量，比如机械臂同时控制 6 个关节的角度，policy 输出 6 维的均值和方差，从高斯分布采样
- **Tuple / Dict**：混合类型，比如同时控制离散的"模式选择"和连续的"力度"

GPU 调度的例子里，如果想同时调度多个任务，可以用 MultiDiscrete——每个队列槽位独立决定"调度/不调度"，policy 同时输出 8 个二值决策。

---

## rollout 是什么

**rollout（轨迹采样）** = 让 policy 跑一遍完整的飞行，记录每步的状态、动作和概率。

```
步骤 0：obs=[0.0, 1.0, 0.0, -0.5, ...]  → 选主引擎(2)  log_prob=-0.43
步骤 1：obs=[0.0, 0.9, 0.0, -0.4, ...]  → 选主引擎(2)  log_prob=-0.43
步骤 2：obs=[0.1, 0.8, 0.1, -0.3, ...]  → 选不喷(0)    log_prob=-1.61
...
步骤 N：飞船着陆 → episode 结束
```

rollout 的产物：一条完整轨迹，包含每步的 obs、action、log_prob，以及每步的 reward。

---

## reward 是什么，为什么是稀疏的

飞船的 reward 每步都有，但**主要信号在最后**：

```
步骤 0~200：每步约 -0.3（时间惩罚）+ 引擎消耗
步骤 201：成功着陆 → +120 分（大正奖励）
                 或 坠毁   → -100 分（大负奖励）
```

虽然每步都有小奖励，但决定这局好坏的关键是最后能不能降落——这个信号在 episode 结束时才给出。模型需要搞清楚"前面哪些动作导致了最后的成功/失败"，这就是**稀疏奖励问题**的本质。

---

## value 模型是什么，为什么需要它

**value（价值）** = 从当前状态出发，预期这局最终能得多少分。

```
飞船在高空、速度正常、姿态稳定 → V ≈ +150（大概率能成功降落）
飞船在低空、下降太快、倾斜严重 → V ≈ -80（大概率要坠毁了）
```

**为什么需要它**：

假设某步飞船选了"不喷火"，这局最终得了 +80 分。这个动作是好的还是坏的？

- 没有 value 模型：不知道，只能用 +80 来更新这步，但也许不喷火是个错误，只是后来侥幸着陆了
- 有 value 模型：V(这一步) 预测"从这步出发预期能得 +120"，实际只得了 +80。实际 < 预期，说明"不喷火"让情况变差了，应该惩罚这个动作

**value 模型初始是随机的，怎么能学好**：

value 模型确实从随机初始化开始，但它通过 `value_loss` 持续被监督训练——每次 rollout 结束后，用实际的 returns 来纠正 value 的预测。

关键是两个模型**协同提升**：
1. 初期：value 很差 → advantages 是噪声 → policy 学得慢但还是在学
2. 随着 policy 稍微好一点，rollout 的质量提高，value 能看到更多"成功着陆"的样本，开始学到一些规律
3. value 好了 → advantages 更准 → policy 学得更快 → 产生更好的 rollout → value 继续改进

这是一个**自举（bootstrapping）**过程，两个模型互相拉动。

**有没有更好的初始化方式**：有，常见做法：
- **预训练 value 模型**：先用随机策略跑几千局，收集 (state, return) 对，用监督学习预训练 value，再开始 PPO。这样 value 一开始就有基本的方向感
- **共享底层网络**（LunarLander 里就是这样）：policy 和 value 共享同一个特征提取层，policy 学到的状态表示可以帮助 value 更快学习
- **SB3 的做法**：observation 归一化（把 8 维状态缩放到均值 0、方差 1），让 value 网络的输入更稳定，收敛更快

---

## returns 和 advantages 是什么

### returns（实际回报）

从每一步出发，**往后**累计的折扣奖励之和：

```
飞船飞行了 200 步，每步 reward 约 -0.3，最后着陆得 +120
GAMMA = 0.99

最后一步（t=199）：return = +120（只有着陆奖励）
倒数第2步（t=198）：return = -0.3 + 0.99 × 120 = 118.5
倒数第3步（t=197）：return = -0.3 + 0.99 × 118.5 = 117.1
...
第0步（t=0）：return ≈ -0.3×(1+0.99+0.99²+...) + 0.99¹⁹⁹×120 ≈ +90
```

**越早的步，return 越低**——不是因为"越近越确定"，而是因为折扣：离最终奖励越远，折扣累积越多，return 自然越低。第 0 步的 return ≈ 90，最后一步的 return ≈ 120，差距来自 GAMMA 的累积折扣。

**折扣的真正用途**：让模型重视"近期能得到的奖励"，而不是无限追求遥远未来的奖励。GAMMA=0.99 意味着 100 步之后的 1 分只相当于现在的 0.99¹⁰⁰ ≈ 0.37 分。

### advantages（优势）

每步的"实际回报 - value 预期"：

```python
advantages[t] = returns[t] - V(obs_t)
```

飞船例子：
```
t=50，飞船姿态良好：
  V(obs_50) = 130（value 预测这局能得 130 分）
  returns[50] = 110（实际从 t=50 出发只得了 110 分）
  advantages[50] = 110 - 130 = -20  ← 比预期差，这步的动作可能不好

t=100，飞船刚做了一个完美的减速：
  V(obs_100) = 80（value 预测还能得 80 分）
  returns[100] = 115（实际得了 115 分）
  advantages[100] = 115 - 80 = +35  ← 比预期好，这步的动作值得鼓励
```

advantages 告诉 policy："这步的动作比平均水平好了多少"，而不是简单地用最终得分来更新所有步骤。

---

## 两个 loss 是什么意思

### policy_loss

```python
ratio = exp(new_log_prob - old_log_prob)   # 新旧策略在同一动作上的概率比
surr1 = ratio * advantages
surr2 = clamp(ratio, 0.8, 1.2) * advantages
policy_loss = -min(surr1, surr2).mean()
```

飞船例子——t=100 那步选了主引擎，advantages=+35：
- `ratio > 1`：新 policy 比旧 policy 更倾向于在这种状态下选主引擎 → 鼓励，surr1 增大
- clip 把 ratio 限制在 [0.8, 1.2]：防止"这步太好了，一下子把主引擎概率推到 99%"，保持稳定

t=50 那步选了不喷火，advantages=-20：
- ratio 越大（越倾向于不喷火）→ surr1 越负 → loss 越大 → 梯度推动减少"不喷火"的概率

### value_loss

```python
value_loss = (V(obs_t) - returns[t]) ** 2
```

让 value 模型的预测接近实际 returns。t=100 那步 V=80 但实际 return=115，value_loss 推动 V 往 115 靠近，下次遇到类似状态时预测会更准。

---

## 每个 epoch 在干什么

```
Epoch 1:
  rollout：飞船飞了 2048 步（可能包含多个 episode）
  计算 GAE：把每步的 reward 折算成 returns 和 advantages
  PPO 更新 10 次：
    - 每次取 64 步的 mini-batch
    - 计算 policy_loss 和 value_loss
    - 反向传播，更新参数
    - clip 保证每次更新不超过 20%

Epoch 2:
  用更新后的 policy 重新 rollout...
```

**为什么每次 rollout 2048 步而不是 1 步**：收集足够多的样本后再更新，advantages 的估计更稳定（单步噪声很大，2048 步的平均更可靠）。

---

## 为什么这个训练是收敛的

1. **初期**：policy 随机，飞船乱飞，大多数 episode 坠毁（reward ≈ -100）
2. **偶尔**：随机动作碰巧让飞船降落（reward ≈ +100），advantages 为正，policy 学到"这种状态下这个动作是好的"
3. **逐渐**：policy 开始偏向有效的动作，成功率提高，value 模型也越来越准
4. **最终**：policy 学会了完整的降落策略（先减速，再对齐，最后轻触地面）

**clip 保证稳定**：没有 clip，某次特别好的 rollout 可能让 policy 剧烈更新，下次 rollout 崩掉。clip 让每次更新都是小步前进。

**value 和 policy 的协同**：value 越准，advantages 越能精确指出"哪步动作导致了成功/失败"，policy 的梯度方向越清晰，收敛越快。两个模型互相拉动，这是 Actor-Critic 方法的核心。

---

## 训练完之后：推理只用 policy

**结论：推理时只需要 policy，value 网络可以扔掉。**

训练阶段两个模型各有分工：

| | 训练时 | 推理时 |
|--|--------|--------|
| **policy** | 生成动作 + 被 PPO 更新 | ✓ 需要，给定状态输出动作概率 |
| **value** | 估计期望回报，用于计算 advantages | ✗ 不需要，只服务于训练 |

value 模型的唯一作用是帮 policy 算梯度方向，一旦训练结束，梯度就不再需要了。

**推理代码极简**：

```python
# 训练完后，推理只做一件事：给定 obs，选动作
model = PPO.load("lunarlander_model.zip")  # 只加载 policy 部分

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)  # 只用 policy
    obs, reward, done, _, _ = env.step(action)
    if done:
        break
```

`deterministic=True` 表示直接选概率最高的动作（不采样），训练时用采样是为了探索，推理时要稳定输出。

**LLM 里的对应**：RLHF 训练完后，发布出去的是语言模型本身（policy），奖励模型（reward model）和价值模型都留在训练流程里，用户调用的 API 只有 policy。ChatGPT 背后就是一个 policy，没有 value 网络在线推理。

**Actor-Critic 共享网络的情况**：LunarLander 里 policy 和 value 共享底层特征提取层，存成同一个文件。推理时加载整个文件，但只调用 `actor` head，`critic` head 的权重带着但不用——内存多占一点，功能没影响。如果真的要省内存，可以只导出 actor 部分的权重。

## 和真实 LLM RLHF 的对比

| | LunarLander | 真实 RLHF |
|--|-------------|-----------|
| action 空间 | 4 个离散动作 | 几万个词 |
| episode 长度 | 约 200~1000 步 | 几百~几千 token |
| reward | 物理引擎实时计算 | 奖励模型（RM）打分 |
| state | 8 维物理状态 | 完整的历史 token 序列 |
| value 模型大小 | 极小（和 policy 共享底层） | 和 policy 一样大 |
| 额外约束 | 无 | KL 惩罚（防止偏离 SFT） |

核心机制完全相同：rollout → 计算 advantages → clip 更新 policy → 训练 value 模型。

---

## stable-baselines3（SB3）：生产级 PPO 实现

手写 PPO 能帮你理解原理，但要让它真正收敛到好效果，有很多容易漏掉的"魔鬼细节"：

- `Adam eps=1e-5`（默认 `1e-8` 会导致收敛慢）
- value loss clipping（防止 value 模型更新过激）
- observation 归一化（让不同量纲的状态特征可比）
- 并行多个环境同时收集数据（提高采样效率）

**stable-baselines3（SB3）** 是一个 Python RL 库，提供了把这些细节都处理好的 PPO 实现，相当于 RL 领域的 scikit-learn。由学术界研究者维护，代码经过严格测试。

```python
from stable_baselines3 import PPO
import gymnasium as gym

env = gym.make("LunarLander-v3")
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=500_000)   # 约 5 分钟，均分达到 200+
```

**手写版 vs SB3 的实际差距**：两者算法逻辑完全相同，但在 LunarLander 上：
- 手写版（漏掉 eps 等细节）：300k steps 后均分约 -60
- SB3：300k steps 后均分约 100，500k steps 后稳定 200+

差距不来自算法，来自这些工程细节的组合效果。

**什么时候用哪个**：
- 理解原理、改算法内部逻辑 → 手写（`tools/ppo_demo.py`）
- 验证一个环境能不能被解决、快速出结果 → SB3（`tools/lunarlander_sb3.py`）
- 做研究 → 通常用 SB3 做 baseline，再手写自己的改进
