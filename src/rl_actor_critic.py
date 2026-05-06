"""
Actor-Critic 极简实现

环境：格子世界（同 rl_gridworld.py）
方法：Actor-Critic，两个独立的参数表：
  - actor：每个状态下各动作的 logits → 策略（policy）
  - critic：每个状态的 V 值估计

和 REINFORCE 的区别：
  - REINFORCE：等完整一局结束，用 return 更新
  - Actor-Critic：每一步就更新，用 TD error 作为 advantage 的近似

TD error（时序差分误差）：
  δ = R + γ * V(s') - V(s)
  含义："实际得到的比 critic 预测的好多少"
       δ > 0 → 这步比预期好，提高这个动作的概率
       δ < 0 → 这步比预期差，降低这个动作的概率

运行：python src/rl_actor_critic.py
"""

import numpy as np

np.random.seed(0)

# ── 环境（复用格子世界） ──────────────────────────────────────────────
ROWS, COLS = 4, 4
GOAL   = (3, 3)
TRAPS  = {(1, 1), (2, 3)}
N_STATES  = ROWS * COLS
N_ACTIONS = 4
GAMMA = 0.9
ACTIONS = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}
ACTION_NAMES = ["↑","↓","←","→"]


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


def softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


# ── Actor-Critic 参数 ─────────────────────────────────────────────────
# actor_logits[s, a]：状态 s 下动作 a 的 logit（未归一化对数概率）
# critic_V[s]：状态 s 的 V 值估计
actor_logits = np.zeros((N_STATES, N_ACTIONS))
critic_V     = np.zeros(N_STATES)

LR_ACTOR  = 0.05
LR_CRITIC = 0.1
N_EPISODES = 2000


# ── 训练 ─────────────────────────────────────────────────────────────
def train():
    episode_returns = []

    for ep in range(N_EPISODES):
        state = (0, 0)   # 每局从左上角出发
        total_reward = 0.0
        steps = 0

        while not is_terminal(state) and steps < 50:
            si = state_idx(state)
            probs = softmax(actor_logits[si])
            action = np.random.choice(N_ACTIONS, p=probs)

            next_state, reward = step(state, action)
            ni = state_idx(next_state)
            total_reward += reward
            steps += 1

            # ── Critic 更新：最小化 TD error ──────────────────────────
            v_next = 0.0 if is_terminal(next_state) else critic_V[ni]
            td_error = reward + GAMMA * v_next - critic_V[si]
            critic_V[si] += LR_CRITIC * td_error

            # ── Actor 更新：用 TD error 作为 advantage 近似 ────────────
            # ∇log π(a|s) w.r.t. logits：one-hot(a) - π(·|s)
            grad_log_pi = -probs.copy()
            grad_log_pi[action] += 1.0
            actor_logits[si] += LR_ACTOR * td_error * grad_log_pi

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
                best = int(np.argmax(actor_logits[si]))
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
                row += f"{critic_V[state_idx(s)]:+5.1f} "
        print(row)


if __name__ == "__main__":
    print("=" * 50)
    print("Actor-Critic 训练格子世界")
    print(f"环境：{ROWS}×{COLS}，终点 G={GOAL}，陷阱 T={TRAPS}")
    print("=" * 50)

    returns = train()

    show_policy()

    print("\n关键概念回顾：")
    print("  Actor  = policy 模型，决定走哪个方向")
    print("  Critic = V 值估计器，预测当前状态值多少分")
    print("  TD error δ = R + γV(s') - V(s)")
    print("    δ > 0：实际比预期好 → 提高这个动作的概率，同时提高 V(s)")
    print("    δ < 0：实际比预期差 → 降低这个动作的概率，同时降低 V(s)")
    print("  两个模型同时更新，互相依赖，比 REINFORCE 收敛快")
    print("  PPO = Actor-Critic + clip 约束（防止更新步子太大）")
