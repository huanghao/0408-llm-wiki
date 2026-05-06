"""
格子世界：V 值、Q 值、Advantage 的可视化

环境：4×4 格子，从左上角出发，到右下角是终点。
     每步 -1（鼓励走最短路），到终点 +10，掉入陷阱 -10。

演示概念：
  - 状态（state）：agent 所在格子的坐标
  - 动作（action）：上/下/左/右
  - V 值：当前格子"值多少分"
  - Q 值：在当前格子"往某个方向走值多少分"
  - Advantage：某个方向比平均好多少
  - 策略迭代：从随机策略出发，用 V 值改进策略

运行：python src/rl_gridworld.py
"""

import numpy as np

# ── 环境定义 ──────────────────────────────────────────────────────────
ROWS, COLS = 4, 4
GOAL   = (3, 3)   # 终点：+10
TRAPS  = {(1, 1), (2, 3)}  # 陷阱：-10
GAMMA  = 0.9      # 折扣因子
ACTIONS = {0: (-1,0), 1: (1,0), 2: (0,-1), 3: (0,1)}   # 上下左右
ACTION_NAMES = {0: "↑", 1: "↓", 2: "←", 3: "→"}


def is_terminal(s):
    return s == GOAL or s in TRAPS


def step(state, action):
    """执行动作，返回 (next_state, reward)。"""
    if is_terminal(state):
        return state, 0.0
    r, c = state
    dr, dc = ACTIONS[action]
    nr, nc = r + dr, c + dc
    # 撞墙：留在原地
    if not (0 <= nr < ROWS and 0 <= nc < COLS):
        nr, nc = r, c
    next_state = (nr, nc)
    if next_state == GOAL:
        reward = 10.0
    elif next_state in TRAPS:
        reward = -10.0
    else:
        reward = -1.0
    return next_state, reward


# ── 策略评估：给定策略，计算每个格子的 V 值 ─────────────────────────

def policy_evaluation(policy, theta=1e-4):
    """
    V(s) = R(s,a) + γ * V(s')，其中 a 由 policy 决定。
    反复迭代直到收敛。
    """
    V = np.zeros((ROWS, COLS))
    while True:
        delta = 0
        for r in range(ROWS):
            for c in range(COLS):
                s = (r, c)
                if is_terminal(s):
                    continue
                a = policy[r, c]
                s_next, reward = step(s, a)
                v_new = reward + GAMMA * V[s_next]
                delta = max(delta, abs(V[s] - v_new))
                V[r, c] = v_new
        if delta < theta:
            break
    return V


# ── Q 值：在状态 s 执行动作 a 的期望回报 ────────────────────────────

def compute_q(V):
    """Q(s,a) = R(s,a) + γ * V(s')"""
    Q = np.zeros((ROWS, COLS, len(ACTIONS)))
    for r in range(ROWS):
        for c in range(COLS):
            s = (r, c)
            if is_terminal(s):
                continue
            for a in ACTIONS:
                s_next, reward = step(s, a)
                Q[r, c, a] = reward + GAMMA * V[s_next]
    return Q


# ── 策略改进：选每个格子 Q 值最大的动作 ─────────────────────────────

def policy_improvement(Q):
    return np.argmax(Q, axis=2).astype(int)


# ── 可视化工具 ───────────────────────────────────────────────────────

def print_grid(title, data, fmt="{:.1f}"):
    print(f"\n{title}")
    print("  " + "  ".join([f" {c} " for c in range(COLS)]))
    for r in range(ROWS):
        row_str = f"{r} "
        for c in range(COLS):
            s = (r, c)
            if s == GOAL:
                cell = " G  "
            elif s in TRAPS:
                cell = " T  "
            elif isinstance(data[r][c], str):
                cell = f" {data[r][c]}  "
            else:
                cell = fmt.format(data[r][c]).rjust(4) + " "
            row_str += cell
        print(row_str)


def policy_to_arrows(policy):
    arrows = []
    for r in range(ROWS):
        row = []
        for c in range(COLS):
            s = (r, c)
            if s == GOAL:
                row.append("G")
            elif s in TRAPS:
                row.append("T")
            else:
                row.append(ACTION_NAMES[policy[r, c]])
        arrows.append(row)
    return arrows


# ── 主流程 ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 50)
    print("格子世界：策略迭代演示")
    print(f"环境：{ROWS}×{COLS}，终点 G={GOAL}，陷阱 T={TRAPS}")
    print("=" * 50)

    # 初始策略：所有格子都往右走
    policy = np.ones((ROWS, COLS), dtype=int) * 3   # 3 = 右

    for iteration in range(6):
        V = policy_evaluation(policy)
        Q = compute_q(V)

        print_grid(f"\n── 第 {iteration+1} 轮：V 值（每个格子值多少分）──",
                   V.tolist())
        print_grid(f"── 第 {iteration+1} 轮：当前策略 ──",
                   policy_to_arrows(policy))

        new_policy = policy_improvement(Q)

        # 策略没有改变则收敛
        if np.array_equal(new_policy, policy):
            print(f"\n✓ 策略在第 {iteration+1} 轮收敛！")
            break
        policy = new_policy

    # 展示最终 Q 值和 Advantage
    print("\n── 最终 Q 值（以起点 (0,0) 为例）──")
    r0, c0 = 0, 0
    v_s = V[r0, c0]
    print(f"  V(起点) = {v_s:.2f}  ← 按最优策略，从这里出发期望得 {v_s:.2f} 分")
    print(f"  {'动作':4}  {'Q值':>8}  {'Advantage':>10}  {'说明'}")
    print("  " + "-" * 45)
    for a, name in ACTION_NAMES.items():
        q = Q[r0, c0, a]
        adv = q - v_s
        note = "← 最优" if a == np.argmax(Q[r0, c0]) else ""
        print(f"  {name:4}  {q:>8.2f}  {adv:>+10.2f}  {note}")

    print("\n关键观察：")
    print("  V(s)   = 这个格子按最优策略走，期望总得分")
    print("  Q(s,a) = 在这个格子往某方向走，期望总得分")
    print("  A(s,a) = Q(s,a) - V(s)  = 这个动作比平均好多少")
    print("  策略改进 = 每步选 Q 值最大的动作 = 选 Advantage 最大的动作")
