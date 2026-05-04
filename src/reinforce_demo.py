"""
REINFORCE 极简 demo

任务：训练一个策略，从 {0, 1, 2, 3, 4} 里选数字，目标是尽量选 1。
奖励：选到 1 得 +1，其他得 -1。

不用神经网络，用一个可学习的 logits 向量代表策略，
展示 REINFORCE 梯度更新的核心机制。

运行：python tools/reinforce_demo.py
"""

import numpy as np

np.random.seed(42)

# --- 超参数 ---
N_ACTIONS = 5       # 动作空间：{0, 1, 2, 3, 4}
TARGET = 1          # 目标动作
LR = 0.1            # 学习率
N_EPISODES = 300    # 训练轮数
N_SAMPLES = 8       # 每轮采样次数（用于计算基线）

# --- 策略：softmax(logits) ---
logits = np.zeros(N_ACTIONS)  # 初始均匀分布


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def get_reward(action):
    return 1.0 if action == TARGET else -1.0


# --- 训练循环 ---
print(f"{'Episode':>8}  {'P(选1)':>8}  {'平均奖励':>10}")
print("-" * 35)

for ep in range(N_EPISODES):
    probs = softmax(logits)

    # 采样 N_SAMPLES 次
    actions = np.random.choice(N_ACTIONS, size=N_SAMPLES, p=probs)
    rewards = np.array([get_reward(a) for a in actions])

    # 基线：本轮奖励均值（降低方差）
    baseline = rewards.mean()

    # REINFORCE 梯度更新
    # ∇logits ∝ (R - baseline) * ∇log p(a)
    # 对 softmax logits，∇log p(a) 的更新等价于：
    #   对被选中的动作：logits[a] += lr * advantage（如果好于平均，提高概率）
    #   其他动作通过 softmax 归一化隐式降低
    grad = np.zeros(N_ACTIONS)
    for a, r in zip(actions, rewards):
        advantage = r - baseline
        # one-hot gradient of log p(a) w.r.t. logits[a]: (1 - p(a))
        # simplified: just accumulate advantage at the chosen action
        grad[a] += advantage

    logits += LR * grad / N_SAMPLES

    if (ep + 1) % 30 == 0:
        probs = softmax(logits)
        avg_r = rewards.mean()
        print(f"{ep+1:>8}  {probs[TARGET]:>8.3f}  {avg_r:>10.3f}")

# --- 最终结果 ---
probs = softmax(logits)
print("\n最终策略分布：")
for i, p in enumerate(probs):
    marker = " ← 目标" if i == TARGET else ""
    print(f"  动作 {i}: {p:.4f}{marker}")

print(f"\n收敛后 P(选1) = {probs[TARGET]:.4f}，期望奖励 ≈ {2*probs[TARGET]-1:.4f}")
