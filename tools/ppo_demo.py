"""
PPO 核心机制 Demo
================
用最简单的设置演示 PPO 的三个关键组件：
  1. 策略（policy）：生成动作（token）
  2. 奖励（reward）：评估完整序列
  3. 优势（advantage）：把稀疏奖励分摊到每步，指导梯度方向

任务：生成一个 5 步序列，奖励 = 序列里 "好" token 的数量

运行：python tools/ppo_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── 超参数 ──────────────────────────────────────────
VOCAB_SIZE  = 4      # token 空间：0=坏, 1=好, 2=中, 3=中  Q：为什么有两个中呢，这里想表达什么？
SEQ_LEN     = 5      # 每次生成 5 个 token
HIDDEN      = 16
EPSILON     = 0.2    # PPO clip 范围
GAMMA       = 0.99   # 折扣因子
LR          = 3e-3
EPOCHS      = 300

# ── 策略模型（policy）：给定当前步，输出下一个 token 的概率 ──
class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, VOCAB_SIZE)
        )
    def forward(self, step):
        # step: 当前是第几步（归一化到 0-1）
        return F.softmax(self.net(step), dim=-1)

# ── 价值模型（value/critic）：估计从当前步出发能得到多少奖励 ──
class Value(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, HIDDEN), nn.ReLU(),
            nn.Linear(HIDDEN, 1)
        )
    def forward(self, step):
        return self.net(step).squeeze(-1)

# Q：这里的策略模型是从step -》动作的分类；value是step到一个奖励数值的预测？

# ── 奖励函数：生成完整序列后打分 ──
def reward_fn(tokens):
    # 奖励 = 序列里 token==1（"好"）的数量
    return (tokens == 1).float().sum()

# ── 生成一条轨迹 ──
def rollout(policy):
    tokens, log_probs = [], []
    for t in range(SEQ_LEN):
        step = torch.tensor([[t / SEQ_LEN]])
        probs = policy(step)                    # (1, VOCAB_SIZE)
        dist  = torch.distributions.Categorical(probs)   # Q：这是什么意思？
        token = dist.sample()                   # 采样一个 token
        tokens.append(token.item())
        log_probs.append(dist.log_prob(token))  # Q：这个概率是怎么计算的？代表什么意思？
    return torch.tensor(tokens), torch.stack(log_probs)

# ── 计算优势函数（把最终奖励分摊到每步）──
def compute_advantages(reward, value_model):
    """
    用 GAE（Generalized Advantage Estimation）把稀疏的最终奖励分摊：
    - 最后一步：advantage = reward - V(最后一步)
    - 其他步：advantage = r_t + gamma * V(t+1) - V(t)
    """
    advantages, returns = [], []
    R = reward  # 从最终奖励开始反向传播
    for t in reversed(range(SEQ_LEN)):
        step = torch.tensor([[t / SEQ_LEN]])
        v_t  = value_model(step).detach()
        advantages.insert(0, R - v_t.item())
        returns.insert(0, R)
        R = R * GAMMA  # 越早的步折扣越多
    # Q：sum(returns) 不等于R，怎么是「分摊」呢？不理解
    # Q：advantages：是递增的，returns也是底层的，这俩是什么作用，表达什么？
    return torch.tensor(advantages), torch.tensor(returns)

# ── 训练 ──────────────────────────────────────────────
policy = Policy()
value  = Value()
opt_p  = torch.optim.Adam(policy.parameters(), lr=LR)
opt_v  = torch.optim.Adam(value.parameters(),  lr=LR)

for epoch in range(EPOCHS):
    # 1. 用当前策略采样轨迹（旧策略）
    with torch.no_grad():
        tokens, old_log_probs = rollout(policy)
    reward = reward_fn(tokens)
    advantages, returns = compute_advantages(reward, value)

    # 2. PPO 更新（多步梯度，复用同一批轨迹）
    # Q：看不懂，为什么做3次？
    for _ in range(3):
        # 用同一批 tokens 重新计算新策略的 log prob（不重新采样）
        new_log_probs = []
        for t in range(SEQ_LEN):
            step  = torch.tensor([[t / SEQ_LEN]])
            probs = policy(step)
            dist  = torch.distributions.Categorical(probs)
            new_log_probs.append(dist.log_prob(tokens[t]))
        new_log_probs = torch.stack(new_log_probs)

        # 重要性采样比：新策略 / 旧策略
        ratio = torch.exp(new_log_probs - old_log_probs.detach())
        # Q：两个分布相减，再用e算指数，这是什么意思？这和softmax有点像吗？

        # PPO clip：限制更新幅度，防止策略突变
        surr1 = ratio * advantages  # Q：这里维度对吗？ratio是动作空间的维度，adv是seqlen？
        surr2 = torch.clamp(ratio, 1 - EPSILON, 1 + EPSILON) * advantages  # Q：这里也看不懂，怎么就clamp了？
        policy_loss = -torch.min(surr1, surr2).mean()

        # 价值模型损失：让 V(t) 尽量准确预测实际回报
        value_loss = sum(
            (value(torch.tensor([[t / SEQ_LEN]])) - returns[t]) ** 2
            for t in range(SEQ_LEN)
        ) / SEQ_LEN
        # Q：整个value loss在算什么？完全看不懂

        opt_p.zero_grad(); policy_loss.backward(); opt_p.step()
        opt_v.zero_grad(); value_loss.backward();  opt_v.step()

    if (epoch + 1) % 75 == 0:  # Q：为什么是75，这是什么意思
        avg_r = sum(reward_fn(rollout(policy)[0]) for _ in range(50)) / 50
        print(f"Epoch {epoch+1:3d}  avg_reward={avg_r:.2f}/5  "
              f"last_tokens={tokens.tolist()}")

print("\n训练完成，策略在每步选 token==1 的概率：")
for t in range(SEQ_LEN):
    probs = policy(torch.tensor([[t / SEQ_LEN]])).detach()
    print(f"  步骤 {t}: P(好 token) = {probs[0, 1]:.3f}")
