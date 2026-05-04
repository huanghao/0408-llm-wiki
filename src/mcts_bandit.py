"""
三臂老虎机上的 MCTS 最小实现
对应 wiki/20-concepts/mcts.md 中的例子

三个选项 L / M / R，真实胜率分别为 0.3 / 0.6 / 0.4（算法不知道）。
运行：python3 scripts/mcts_bandit.py
"""

import math
import random

# 三个选项的真实胜率（MCTS 不知道）
TRUE_WIN_RATES = {"L": 0.3, "M": 0.6, "R": 0.4}
ACTIONS = list(TRUE_WIN_RATES.keys())
C = math.sqrt(2)  # 探索系数

# 每个节点存：累计得分 W，访问次数 N
stats = {a: {"W": 0, "N": 0} for a in ACTIONS}


def ucb(action, total_n):
    s = stats[action]
    if s["N"] == 0:
        return float("inf")  # 未访问的节点优先级无穷大
    return s["W"] / s["N"] + C * math.sqrt(math.log(total_n) / s["N"])


def rollout(action):
    return 1 if random.random() < TRUE_WIN_RATES[action] else 0


def mcts(n_iter=200):
    for i in range(1, n_iter + 1):
        total_n = sum(s["N"] for s in stats.values())
        # Selection：选 UCB 最高的动作
        chosen = max(ACTIONS, key=lambda a: ucb(a, max(total_n, 1)))
        # Simulation：rollout
        result = rollout(chosen)
        # Backpropagation：更新统计
        stats[chosen]["W"] += result
        stats[chosen]["N"] += 1

        if i in (10, 50, 100, 200):
            print(f"\n迭代 {i:3d} 次后：")
            for a in ACTIONS:
                s = stats[a]
                rate = s["W"] / s["N"] if s["N"] > 0 else 0
                print(
                    f"  {a}: 访问 {s['N']:3d} 次，估计胜率 {rate:.2f}"
                    f"（真实 {TRUE_WIN_RATES[a]}）"
                )
    best = max(ACTIONS, key=lambda a: stats[a]["W"] / stats[a]["N"])
    print(f"\n最终选择：{best}（真实最优：M）")


if __name__ == "__main__":
    mcts()
