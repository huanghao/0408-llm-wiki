"""
参数调度曲线可视化

演示 ML 训练中常见的参数衰减形式：
  - 指数衰减
  - 线性衰减
  - 余弦退火（含 warm restart）
  - 阶梯衰减
  - Warmup + 余弦（LLM 预训练标准做法）

以及两种不同性质的"clip"：
  - 硬截断（ε-greedy 的 [0,1] 范围约束）
  - PPO 的比值 clip（不是衰减，是每步更新幅度约束）

运行：python src/decay_schedules.py
如果没有 matplotlib，只输出文本对比表格。
"""

import numpy as np

T = 1000   # 总步数


# ── 各种调度函数 ──────────────────────────────────────────────────────

def exponential(t, alpha0=1.0, lam=0.005):
    """指数衰减：α(t) = α₀ · e^(−λt)
    特点：前期衰减快，后期趋近 0 但永远不到 0。
    适合：ε-greedy 探索率，早期 RL。
    """
    return alpha0 * np.exp(-lam * t)


def linear(t, alpha0=1.0, T=T):
    """线性衰减：α(t) = α₀ · (1 − t/T)
    特点：匀速减小到 0，在 t=T 时恰好归零。
    适合：PPO 的 clip 参数（从 0.2 线性衰减到 0）。
    """
    return alpha0 * np.maximum(0, 1 - t / T)


def cosine(t, alpha_min=0.0, alpha_max=1.0, T=T):
    """余弦退火：α(t) = α_min + ½(α_max − α_min)(1 + cos(πt/T))
    特点：平滑的 S 形曲线，初末慢、中间快。
    适合：LLM 预训练学习率（Llama、GPT 系列标准做法）。
    """
    return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t / T))


def cosine_with_restarts(t, alpha_min=0.1, alpha_max=1.0, T_cycle=250):
    """余弦退火 + 周期重启（SGDR）
    每 T_cycle 步重置到 alpha_max，重新退火。
    效果：帮助逃离局部最优，每次重启后可能找到更好的区域。
    """
    t_mod = t % T_cycle
    return alpha_min + 0.5 * (alpha_max - alpha_min) * (1 + np.cos(np.pi * t_mod / T_cycle))


def step_decay(t, alpha0=1.0, drop=0.5, every=200):
    """阶梯衰减：每隔 every 步乘以 drop
    特点：简单，有明确的训练阶段感。
    适合：图像分类（ResNet 等经典做法）。
    """
    return alpha0 * (drop ** (t // every))


def warmup_cosine(t, alpha_max=1.0, warmup_steps=100, T=T):
    """Warmup + 余弦退火：LLM 预训练最常用的组合
    - 前 warmup_steps 步：从 0 线性升到 alpha_max（避免初期梯度爆炸）
    - 之后：余弦衰减到接近 0
    适合：Llama 3、GPT-4 等大模型预训练。
    支持标量和 numpy 数组输入。
    """
    t = np.asarray(t, dtype=float)
    T_after = T - warmup_steps
    warmup_val = alpha_max * t / warmup_steps
    t_after = t - warmup_steps
    cosine_val = cosine(t_after, alpha_min=0.0, alpha_max=alpha_max, T=T_after)
    return np.where(t < warmup_steps, warmup_val, cosine_val)


def print_schedule_comparison():
    ts = np.arange(T + 1)

    schedules = [
        ("指数衰减",         exponential(ts)),
        ("线性衰减",         linear(ts)),
        ("余弦退火",         cosine(ts)),
        ("余弦+重启",        cosine_with_restarts(ts)),
        ("阶梯衰减",         step_decay(ts)),
        ("Warmup+余弦",      warmup_cosine(ts)),
    ]

    checkpoints = [0, 100, 250, 500, 750, 1000]

    print("=" * 70)
    print("参数调度曲线对比（起始值均归一化为 1.0）")
    print("=" * 70)

    header = f"{'调度方式':<12}" + "".join(f"  t={c:>4}" for c in checkpoints)
    print(header)
    print("-" * 70)

    for name, vals in schedules:
        row = f"{name:<12}"
        for c in checkpoints:
            row += f"  {vals[c]:>6.3f}"
        print(row)

    print()
    print("=" * 70)
    print("关键差异总结：")
    print()
    print("  指数衰减：永远不到 0，后期几乎停止更新——探索率常用，学习率少用")
    print("  线性衰减：t=T 时精确归零，PPO 的 clip ε 常用此形式")
    print("  余弦退火：中间衰减快，两端慢，最后一段特别平滑——LLM 预训练首选")
    print("  余弦+重启：周期性「重新探索」，有助于逃离局部最优")
    print("  阶梯衰减：简单直观，适合有明确训练阶段的任务")
    print("  Warmup+余弦：LLM 标准做法，warmup 避免初期梯度爆炸")


# ── PPO clip 的特殊说明（它不是衰减）────────────────────────────────

def print_ppo_clip_explanation():
    print()
    print("=" * 70)
    print("PPO 的 clip：和上面的衰减是两件事")
    print("=" * 70)
    print()
    print("  上面的调度：参数随「时间步」单调变化，控制整体训练进度")
    print()
    print("  PPO clip：在「每一次参数更新」时截断比值，控制单步更新幅度")
    print()
    print("  PPO 的操作：")
    print("    ratio = π_new(a|s) / π_old(a|s)   # 新旧策略的概率比")
    print("    clipped = clip(ratio, 1-ε, 1+ε)    # 截断到 [0.8, 1.2]")
    print("    loss = -min(ratio * A, clipped * A) # 取保守的那个")
    print()

    # 演示 clip 的效果
    ratios = np.array([0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0])
    eps = 0.2
    clipped = np.clip(ratios, 1 - eps, 1 + eps)

    print(f"  {'原始 ratio':>12}  {'clip 后':>8}  {'是否被截断':>10}")
    print("  " + "-" * 35)
    for r, c in zip(ratios, clipped):
        truncated = "✗ 截断" if abs(r - c) > 1e-9 else "  通过"
        print(f"  {r:>12.2f}  {c:>8.2f}  {truncated:>10}")

    print()
    print("  直觉：ratio=2.0 意味着新策略把某动作的概率翻倍了。")
    print("  clip 强制：「不管 advantage 多高，单步最多把概率改变 20%」。")
    print("  这和学习率衰减无关——它是每步都生效的硬约束，不随时间变化。")
    print()
    print("  类比：")
    print("    学习率衰减 = 随着经验积累，下注越来越保守（整体策略）")
    print("    PPO clip   = 每一笔交易，最多押注 ±20%（单步约束）")


if __name__ == "__main__":
    print_schedule_comparison()
    print_ppo_clip_explanation()

    # 如果有 matplotlib，画图
    try:
        import matplotlib.pyplot as plt
        import matplotlib

        matplotlib.rcParams['font.family'] = ['Arial Unicode MS', 'sans-serif']

        ts = np.arange(T + 1)
        fig, axes = plt.subplots(2, 3, figsize=(14, 8))
        fig.suptitle("参数调度曲线对比", fontsize=14)

        plots = [
            ("指数衰减\nε-greedy 探索率", exponential(ts), "steelblue"),
            ("线性衰减\nPPO clip 参数", linear(ts), "darkorange"),
            ("余弦退火\nLLM 学习率", cosine(ts), "green"),
            ("余弦退火 + 重启\nSGDR", cosine_with_restarts(ts), "red"),
            ("阶梯衰减\n图像分类", step_decay(ts), "purple"),
            ("Warmup + 余弦\nLlama/GPT 预训练", warmup_cosine(ts), "brown"),
        ]

        for ax, (title, vals, color) in zip(axes.flat, plots):
            ax.plot(ts, vals, color=color, linewidth=2)
            ax.set_title(title, fontsize=10)
            ax.set_xlabel("训练步数 t")
            ax.set_ylabel("参数值")
            ax.set_ylim(-0.05, 1.1)
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        plt.savefig("docs/decay_schedules.png", dpi=150, bbox_inches='tight')
        print("\n图表已保存到 docs/decay_schedules.png")
        plt.show()

    except ImportError:
        print("\n（未安装 matplotlib，跳过绘图。pip install matplotlib 后可生成图表）")
