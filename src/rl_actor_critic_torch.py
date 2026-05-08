"""
Actor-Critic：PyTorch 版

和 rl_actor_critic.py 完全等价，区别只在于：
  - 手写版：直接操作 numpy 数组，梯度手动推导写进去
  - PyTorch 版：定义 loss，调用 .backward()，框架自动计算梯度

环境、超参数、训练逻辑完全相同，方便对比。

运行：python src/rl_actor_critic_torch.py
"""

import numpy as np
import torch
import torch.nn.functional as F

torch.manual_seed(0)
np.random.seed(0)

# ── 环境（和 rl_actor_critic.py 完全一样） ────────────────────────────
ROWS, COLS = 4, 4
GOAL   = (3, 3)
TRAPS  = {(1, 1), (2, 3)}
N_STATES  = ROWS * COLS
N_ACTIONS = 4
GAMMA = 0.9
ACTIONS = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
ACTION_NAMES = ["↑","↓","←","→"]

LR_ACTOR  = 0.05
LR_CRITIC = 0.1
N_EPISODES = 2000


def state_idx(s):
    return s[0] * COLS + s[1]


def is_terminal(s):
    return s == GOAL or s in TRAPS


def step(state, action):
    if is_terminal(state):
        return state, 0.0
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    ns = (nr, nc)
    reward = 10.0 if ns == GOAL else (-10.0 if ns in TRAPS else -1.0)
    return ns, reward


# ── 可学习参数：和手写版对应 ──────────────────────────────────────────
#
# 手写版：
#   actor_logits = np.zeros((N_STATES, N_ACTIONS))   # numpy 数组
#   critic_V     = np.zeros(N_STATES)                # numpy 数组
#
# PyTorch 版：同样的形状，用 requires_grad=True 告诉框架"这是要求导的变量"

actor_logits = torch.zeros(N_STATES, N_ACTIONS, requires_grad=True)
critic_V     = torch.zeros(N_STATES,            requires_grad=True)

# 分别给两个参数配优化器（SGD = 最简单的梯度下降）
optimizer_actor  = torch.optim.SGD([actor_logits], lr=LR_ACTOR)
optimizer_critic = torch.optim.SGD([critic_V],     lr=LR_CRITIC)


# ── 训练 ─────────────────────────────────────────────────────────────
def train():
    episode_returns = []

    for ep in range(N_EPISODES):
        state = (0, 0)
        total_reward = 0.0
        steps = 0

        while not is_terminal(state) and steps < 50:
            si = state_idx(state)

            # Actor：从 logits 得到概率分布，采样动作
            # 手写版：probs = softmax(actor_logits[si])
            probs = F.softmax(actor_logits[si], dim=0)          # shape: [4]
            action = torch.multinomial(probs.detach(), 1).item() # 采样，不需要梯度

            next_state, reward = step(state, action)
            ni = state_idx(next_state)
            total_reward += reward
            steps += 1

            # ── Critic 更新 ───────────────────────────────────────────
            #
            # 手写版：
            #   v_next   = 0.0 if is_terminal(next_state) else critic_V[ni]
            #   td_error = reward + GAMMA * v_next - critic_V[si]
            #   critic_V[si] += LR_CRITIC * td_error
            #
            # PyTorch 版：
            #   定义同样的 td_error，但用 tensor 运算
            #   loss_critic = 0.5 * td_error²  （MSE 损失）
            #   .backward() 自动算出 d(loss)/d(critic_V[si]) = -td_error
            #   optimizer.step() 执行 critic_V[si] -= lr * (-td_error)
            #                                      = critic_V[si] += lr * td_error
            #   和手写版完全等价

            v_curr = critic_V[si]
            v_next = torch.tensor(0.0) if is_terminal(next_state) else critic_V[ni].detach()
            td_error = reward + GAMMA * v_next - v_curr          # 标量 tensor

            loss_critic = 0.5 * td_error ** 2

            optimizer_critic.zero_grad()
            loss_critic.backward()
            optimizer_critic.step()

            # ── Actor 更新 ────────────────────────────────────────────
            #
            # 手写版：
            #   grad_log_pi = -probs.copy()
            #   grad_log_pi[action] += 1.0          # = one_hot(a) - probs
            #   actor_logits[si] += LR_ACTOR * td_error * grad_log_pi
            #
            # PyTorch 版：
            #   log_prob = log π(a|s)，即被选动作的对数概率
            #   loss_actor = -td_error * log_prob   （负号：最大化变最小化）
            #   .backward() 自动算出 d(loss)/d(logits) = -td_error * (one_hot(a) - probs)
            #   optimizer.step() 执行 logits -= lr * (-td_error * grad_log_pi)
            #                             = logits += lr * td_error * grad_log_pi
            #   和手写版完全等价

            log_prob = F.log_softmax(actor_logits[si], dim=0)[action]  # 标量 tensor
            loss_actor = -td_error.detach() * log_prob  # detach td_error：不回传到 critic

            optimizer_actor.zero_grad()
            loss_actor.backward()
            optimizer_actor.step()

            state = next_state

        episode_returns.append(total_reward)

        if (ep + 1) % 200 == 0:
            recent = np.mean(episode_returns[-200:])
            print(f"  Episode {ep+1:5d}  最近200局平均回报: {recent:+.2f}")

    return episode_returns


# ── 展示学到的策略 ────────────────────────────────────────────────────
def show_policy():
    print("\n学到的策略（每格最优动作）：")
    print("  " + "  ".join([f" {c}" for c in range(COLS)]))
    for r in range(ROWS):
        row = f"{r} "
        for c in range(COLS):
            s = (r, c)
            if s == GOAL:
                row += " G  "
            elif s in TRAPS:
                row += " T  "
            else:
                si = state_idx(s)
                best = int(actor_logits[si].argmax().item())
                row += f" {ACTION_NAMES[best]}  "
        print(row)

    print("\n学到的 V 值：")
    print("  " + "  ".join([f"   {c}" for c in range(COLS)]))
    for r in range(ROWS):
        row = f"{r} "
        for c in range(COLS):
            s = (r, c)
            if s == GOAL:
                row += "  G   "
            elif s in TRAPS:
                row += "  T   "
            else:
                row += f"{critic_V[state_idx(s)].item():+5.1f} "
        print(row)


if __name__ == "__main__":
    print("=" * 50)
    print("Actor-Critic 训练格子世界（PyTorch 版）")
    print(f"环境：{ROWS}×{COLS}，终点 G={GOAL}，陷阱 T={TRAPS}")
    print("=" * 50)
    print()
    print("和手写版的唯一区别：")
    print("  手写版  critic_V[si] += LR * td_error")
    print("          actor_logits[si] += LR * td_error * (one_hot(a) - probs)")
    print()
    print("  torch版  loss_critic = 0.5 * td_error²  → .backward() → .step()")
    print("           loss_actor  = -td_error * log_prob → .backward() → .step()")
    print()

    returns = train()
    show_policy()

    print("\n两个版本应收敛到相同策略，V 值也应相近。")
