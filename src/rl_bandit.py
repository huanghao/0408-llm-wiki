"""
多臂老虎机：RL 概念入门

任务：面前有 5 台老虎机，每台有未知的赢率。
     每次只能拉一台，目标是 500 局内尽量多赢。

演示概念：
  - agent / 环境 / 动作 / 奖励
  - 随机策略 vs 贪心策略 vs epsilon-greedy vs epsilon-decay
  - 探索与利用的权衡
  - 收益/风险量化（均值、标准差、夏普比率、最差情况）
  - 固定成本对净收益的影响

运行：python src/rl_bandit.py
"""

import numpy as np

# ── 环境 ──────────────────────────────────────────────────────────────
TRUE_WIN_RATES = [0.1, 0.4, 0.6, 0.3, 0.2]   # 真实赢率（agent 不知道）
N_ARMS = len(TRUE_WIN_RATES)
N_ROUNDS = 500
N_TRIALS = 500          # 蒙特卡洛模拟次数，用于统计收益/风险
COST_PER_ROUND = 0.0    # 每局固定成本（0 = 无成本，可改为 0.2 / 0.4 等观察影响）

BEST_ARM = int(np.argmax(TRUE_WIN_RATES))
BEST_POSSIBLE = TRUE_WIN_RATES[BEST_ARM] * N_ROUNDS   # 理论上限（每次都选最优）
RANDOM_BASELINE = np.mean(TRUE_WIN_RATES)             # 随机策略期望赢率


# ── 策略定义 ──────────────────────────────────────────────────────────

def random_policy(counts, values, rng):
    """随机策略：每次等概率选一台机器。完全不利用已有信息。"""
    return rng.randint(N_ARMS)


def greedy_policy(counts, values, rng):
    """贪心策略：先把每台机器各试一次，之后永远选估计赢率最高的。
    问题：初始化时运气好的机器会被一直选，可能锁在次优机器上。"""
    untried = np.where(counts == 0)[0]
    if len(untried) > 0:
        return int(untried[0])
    return int(np.argmax(values))


def eg_policy(counts, values, rng, epsilon=0.1):
    """ε-greedy：以 ε 概率随机探索，以 1-ε 概率贪心利用。"""
    if rng.rand() < epsilon:
        return rng.randint(N_ARMS)
    return int(np.argmax(values))


def decay_policy(counts, values, rng, epsilon0=0.5, decay=0.01):
    """ε-decay：探索率随总拉取次数指数衰减。
    ε(t) = ε₀ · exp(−decay · t)"""
    t = counts.sum()
    epsilon = epsilon0 * np.exp(-decay * t)
    if rng.rand() < epsilon:
        return rng.randint(N_ARMS)
    return int(np.argmax(values))


# ── 单次运行 ──────────────────────────────────────────────────────────

def run_once(policy_fn, seed):
    rng = np.random.RandomState(seed)
    counts = np.zeros(N_ARMS)
    values = np.zeros(N_ARMS)
    gross_reward = 0.0

    for _ in range(N_ROUNDS):
        arm = policy_fn(counts, values, rng)
        reward = 1.0 if rng.rand() < TRUE_WIN_RATES[arm] else 0.0
        counts[arm] += 1
        values[arm] += (reward - values[arm]) / counts[arm]
        gross_reward += reward

    net_reward = gross_reward - COST_PER_ROUND * N_ROUNDS
    return net_reward


# ── 蒙特卡洛统计 ─────────────────────────────────────────────────────

def evaluate(policy_fn, label):
    results = np.array([run_once(policy_fn, seed) for seed in range(N_TRIALS)])
    mean   = results.mean()
    std    = results.std()
    p10    = np.percentile(results, 10)   # 最差 10% 情况
    p90    = np.percentile(results, 90)   # 最好 10% 情况
    # 超额收益 = 超过随机基准的部分；夏普 = 超额收益 / 风险
    baseline_net = (RANDOM_BASELINE - COST_PER_ROUND) * N_ROUNDS
    sharpe = (mean - baseline_net) / std if std > 0 else 0.0
    return dict(label=label, mean=mean, std=std, p10=p10, p90=p90, sharpe=sharpe)


# ── 输出 ──────────────────────────────────────────────────────────────

def print_results(stats_list):
    best_net = BEST_POSSIBLE - COST_PER_ROUND * N_ROUNDS
    baseline_net = (RANDOM_BASELINE - COST_PER_ROUND) * N_ROUNDS

    print("=" * 75)
    print(f"多臂老虎机：{N_TRIALS} 次模拟的收益/风险分析")
    print(f"赢率分布（隐藏）: {TRUE_WIN_RATES}   最优机器: #{BEST_ARM}（赢率 {TRUE_WIN_RATES[BEST_ARM]:.0%}）")
    if COST_PER_ROUND > 0:
        print(f"每局成本: {COST_PER_ROUND}  →  总成本: {COST_PER_ROUND * N_ROUNDS:.0f}")
    print(f"理论上限（每次选最优）: {best_net:.0f}  随机基准: {baseline_net:.0f}")
    print("=" * 75)
    print(f"{'策略':<20} {'均值':>6} {'标准差':>6} {'最差10%':>8} {'最好10%':>8} {'夏普':>6}  {'风险等级'}")
    print("-" * 75)

    for s in stats_list:
        # 风险等级：按标准差占均值的比例
        cv = s['std'] / abs(s['mean']) if s['mean'] != 0 else 999
        if cv < 0.1:
            risk = "低"
        elif cv < 0.3:
            risk = "中"
        else:
            risk = "高 ⚠"
        print(f"  {s['label']:<18} {s['mean']:>6.1f} {s['std']:>6.1f} "
              f"{s['p10']:>8.1f} {s['p90']:>8.1f} {s['sharpe']:>6.2f}  {risk}")

    print()
    print("  均值   = 期望总收益（500 局）")
    print("  标准差 = 风险（波动率）—— 越大越不稳定")
    print("  最差10% = 倒霉时的收益下限（类似 Value at Risk）")
    print("  夏普   = 超额收益 / 风险，越高越好（风险调整后回报）")

    if COST_PER_ROUND > 0:
        print()
        print(f"  注：加入每局成本 {COST_PER_ROUND} 后，均值整体平移 −{COST_PER_ROUND*N_ROUNDS:.0f}，")
        print(f"  策略排名和标准差不变——成本是常数，只影响盈亏平衡点。")
        loss_strategies = [s for s in stats_list if s['mean'] < 0]
        if loss_strategies:
            print(f"  当前成本下净亏损的策略: {[s['label'] for s in loss_strategies]}")


if __name__ == "__main__":
    strategies = [
        (lambda c, v, r: random_policy(c, v, r),           "随机策略"),
        (lambda c, v, r: greedy_policy(c, v, r),            "纯贪心"),
        (lambda c, v, r: eg_policy(c, v, r, epsilon=0.1),   "ε=0.1"),
        (lambda c, v, r: eg_policy(c, v, r, epsilon=0.3),   "ε=0.3"),
        (lambda c, v, r: decay_policy(c, v, r),             "ε-decay"),
    ]

    stats = [evaluate(fn, label) for fn, label in strategies]
    print_results(stats)

    # ── 演示成本影响 ──────────────────────────────────────────────────
    print()
    print("=" * 75)
    print("成本敏感性分析：不同成本下各策略的均值净收益")
    print("=" * 75)
    costs = [0.0, 0.2, 0.4, 0.55, 0.6]
    header = f"{'成本/局':<10}" + "".join(f"  {s['label']:>10}" for s in stats)
    print(header)
    print("-" * 75)
    for cost in costs:
        row = f"{cost:<10.2f}"
        for s in stats:
            net = s['mean'] - cost * N_ROUNDS
            marker = " ✗" if net < 0 else "  "
            row += f"  {net:>9.1f}{marker}"
        print(row)
    print()
    print("  ✗ = 净亏损（总收益 < 总成本）")
    print("  成本只平移所有策略的净收益，不改变策略间的相对差距和标准差。")
    print("  但它改变了「哪些策略值得做」——成本越高，越需要高效的探索策略。")
