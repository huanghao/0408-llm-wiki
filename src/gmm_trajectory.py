"""
GMM 轨迹预测演示：模拟 Wayformer/MTR 的输出格式

演示内容：
1. 模型输出 K=6 条 GMM 轨迹（每条是 T 步 2D 高斯序列）
2. 计算 minADE / minFDE / Brier-minFDE
3. 选取最优轨迹，提取均值点作为控制模块用的轨迹
4. 可视化所有轨迹 + 不确定性范围

运行：python src/gmm_trajectory.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams['font.family'] = ['Hiragino Sans GB', 'STHeiti', 'PingFang HK', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(0)

# ── 参数 ──────────────────────────────────────────────────────────────
T = 8        # 预测时间步数（WOMD：8 秒，10Hz = 80 步；这里简化为 8 步，每步 1 秒）
K = 6        # 模态数量（评测标准）
DT = 1.0     # 时间步长（秒）


# ── 1. 模拟模型输出（K 条 GMM 轨迹） ──────────────────────────────────
#
# 真实模型输出：K 条轨迹，每条 T 步，每步 (μ_x, μ_y, σ_x, σ_y, ρ, 概率 π)
# 这里手动构造 6 种意图：直行、直行变道左、直行变道右、左转、右转、减速停车

def make_trajectory(intent, T=T):
    """构造一条 GMM 轨迹（T 步的均值序列 + 不确定性）"""
    t = np.arange(1, T + 1)  # 1..T 秒

    if intent == "straight":
        mu_x = np.zeros(T)
        mu_y = t * 8.0          # 直行，每秒 8m
        sigma = np.ones(T) * 0.5
        pi = 0.40

    elif intent == "straight_left":
        mu_x = -t * 1.0          # 缓慢向左偏
        mu_y = t * 7.5
        sigma = np.ones(T) * 0.8
        pi = 0.15

    elif intent == "straight_right":
        mu_x = t * 1.0
        mu_y = t * 7.5
        sigma = np.ones(T) * 0.8
        pi = 0.10

    elif intent == "left_turn":
        angle = t * (np.pi / 2 / T)   # 在 T 秒内完成 90° 左转
        radius = 15.0
        mu_x = -radius * np.sin(angle)
        mu_y = radius * (1 - np.cos(angle))
        sigma = np.ones(T) * 1.2
        pi = 0.20

    elif intent == "right_turn":
        angle = t * (np.pi / 2 / T)
        radius = 12.0
        mu_x = radius * np.sin(angle)
        mu_y = radius * (1 - np.cos(angle))
        sigma = np.ones(T) * 1.2
        pi = 0.10

    elif intent == "slow_stop":
        speed = np.maximum(0, 8.0 - t * 2.0)    # 每秒减速 2m/s，直到停
        mu_x = np.zeros(T)
        mu_y = np.cumsum(speed)
        sigma = np.ones(T) * 0.3
        pi = 0.05

    return {
        "intent": intent,
        "mu": np.column_stack([mu_x, mu_y]),   # [T, 2]
        "sigma": sigma,                          # [T]，简化为各向同性
        "pi": pi
    }

intents = ["straight", "straight_left", "straight_right", "left_turn", "right_turn", "slow_stop"]
trajs = [make_trajectory(i) for i in intents]

# 归一化概率（确保和为 1）
total_pi = sum(t["pi"] for t in trajs)
for t in trajs:
    t["pi"] /= total_pi


# ── 2. 模拟 GT 轨迹（假设真实车辆直行） ───────────────────────────────

gt_t = np.arange(1, T + 1)
gt = np.column_stack([
    np.zeros(T) + np.random.normal(0, 0.3, T),    # 真实有小幅随机偏差
    gt_t * 8.0 + np.random.normal(0, 0.5, T)
])


# ── 3. 计算评测指标 ────────────────────────────────────────────────────

def ade(traj_mu, gt):
    """Average Displacement Error：每步欧氏距离的平均"""
    return np.mean(np.linalg.norm(traj_mu - gt, axis=1))

def fde(traj_mu, gt):
    """Final Displacement Error：终点欧氏距离"""
    return np.linalg.norm(traj_mu[-1] - gt[-1])

print("=" * 60)
print("各轨迹的评测指标")
print("=" * 60)
print(f"{'意图':<20} {'ADE':>6} {'FDE':>6} {'π':>6}")
print("-" * 45)

ade_list, fde_list, pi_list = [], [], []
for t in trajs:
    a = ade(t["mu"], gt)
    f = fde(t["mu"], gt)
    ade_list.append(a)
    fde_list.append(f)
    pi_list.append(t["pi"])
    print(f"  {t['intent']:<18} {a:>6.2f} {f:>6.2f} {t['pi']:>6.3f}")

# minADE / minFDE
best_k_ade = np.argmin(ade_list)
best_k_fde = np.argmin(fde_list)
min_ade = min(ade_list)
min_fde = min(fde_list)

print(f"\nminADE (K=6) = {min_ade:.2f}m  (最好条: {trajs[best_k_ade]['intent']})")
print(f"minFDE (K=6) = {min_fde:.2f}m  (最好条: {trajs[best_k_fde]['intent']})")

# Brier-minFDE：(1 - p_best)² + minFDE，p_best 是最好条的概率
p_best = pi_list[best_k_fde]
brier_minfde = (1 - p_best)**2 + min_fde
print(f"Brier-minFDE = (1-{p_best:.3f})² + {min_fde:.2f} = {brier_minfde:.2f}")
print(f"  → 如果 p_best=1.0：Brier-minFDE = {0 + min_fde:.2f}（和 minFDE 一样，置信度完美）")
print(f"  → 如果 p_best=1/{K}={1/K:.3f}：Brier-minFDE = {(1-1/K)**2 + min_fde:.2f}（均匀分配概率时的惩罚）")


# ── 4. 选取最优轨迹 → 控制模块可用的轨迹 ───────────────────────────────
#
# 运动预测给下游（规划/控制）传的通常是：
#   - 均值轨迹点序列（离散时间步的 x, y）
#   - 对应的概率（权重）
#   - 可选：不确定性 σ（用于安全边距计算）
#
# 「选哪条」的策略：
#   - 最高置信度（π 最大）：概率最高的意图
#   - 最近 GT（评测用，实际无法用，因为不知道 GT）
#   - 最保守/安全（FDE 最小 + 碰撞检测）

best_k_confident = np.argmax(pi_list)
best_traj = trajs[best_k_confident]

print(f"\n选取策略：最高置信度")
print(f"  选中轨迹：{best_traj['intent']}（π={best_traj['pi']:.3f}）")
print(f"  控制模块使用的轨迹点（均值，每秒一个位置）：")
for step, (x, y) in enumerate(best_traj["mu"]):
    print(f"    t={step+1}s: ({x:+.1f}m, {y:.1f}m)")

print(f"\n注意：")
print(f"  - 均值点是离散的，相邻两点间的路径通常用样条曲线插值")
print(f"  - 实际控制频率更高（10-100Hz），需要在离散点之间做插值")
print(f"  - σ 用来给规划模块提供安全边距：μ ± 2σ 是 95% 置信区间")


# ── 5. 可视化 ─────────────────────────────────────────────────────────

try:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    colors = ['steelblue', 'darkorange', 'green', 'red', 'purple', 'brown']

    # 子图1：所有 K 条轨迹 + GT
    ax = axes[0]
    ax.set_title(f"K={K} 条 GMM 轨迹预测", fontsize=12)

    for i, (t, color) in enumerate(zip(trajs, colors)):
        mu = t["mu"]
        sigma = t["sigma"]
        pi = t["pi"]

        # 画均值轨迹
        ax.plot(mu[:, 0], mu[:, 1], '-o', color=color, linewidth=2,
                markersize=4, alpha=0.8,
                label=f"{t['intent']} (π={pi:.2f})")

        # 画 1σ 不确定性圆圈（每步）
        for step in range(T):
            circle = plt.Circle((mu[step, 0], mu[step, 1]), sigma[step],
                                 color=color, alpha=0.08, fill=True)
            ax.add_patch(circle)

    # 画 GT
    ax.plot(gt[:, 0], gt[:, 1], 'k--o', linewidth=2.5, markersize=6,
            label='GT（真实轨迹）', zorder=5)

    # 标记起点
    ax.plot(0, 0, 'ks', markersize=12, zorder=6, label='起点')

    ax.set_xlabel("横向位置 x (m)")
    ax.set_ylabel("纵向位置 y (m)")
    ax.legend(fontsize=8, loc='upper left')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-30, 30)
    ax.set_ylim(-5, 70)

    # 子图2：选取最优轨迹，附不确定性范围
    ax = axes[1]
    ax.set_title(f"选取最优轨迹 ({best_traj['intent']}) → 控制输入", fontsize=12)

    best_mu = best_traj["mu"]
    best_sigma = best_traj["sigma"]

    # 不确定性带（2σ = 95% 置信区间）
    ax.fill_between(range(T + 1),
                    [0] + list(best_mu[:, 1] - 2 * best_sigma),
                    [0] + list(best_mu[:, 1] + 2 * best_sigma),
                    alpha=0.2, color='steelblue', label='±2σ（95% 置信区间）')

    # 均值轨迹（控制模块实际使用的点）
    all_x = [0] + list(best_mu[:, 0])
    all_y = [0] + list(best_mu[:, 1])
    ax.plot(all_x, all_y, 'b-o', linewidth=2.5, markersize=8,
            label=f'均值轨迹（控制输入）', zorder=5)

    # GT
    ax.plot([0] + list(gt[:, 0]), [0] + list(gt[:, 1]),
            'k--o', linewidth=2, markersize=6, label='GT', zorder=4)

    # 标注时间步
    for step, (x, y) in enumerate(best_mu):
        ax.annotate(f"t={step+1}s\n({x:+.1f},{y:.0f})",
                    (x, y), textcoords="offset points",
                    xytext=(8, 0), fontsize=7, color='blue')

    ax.set_xlabel("时间步 t")
    ax.set_ylabel("位置 (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 在 x 轴放时间刻度（不是空间）
    ax.set_xticks(range(T + 1))
    ax.set_xticklabels([f"t={i}s" for i in range(T + 1)], rotation=45)

    plt.tight_layout()
    plt.savefig("docs/gmm_trajectory.png", dpi=120, bbox_inches='tight')
    print("\n图表已保存到 docs/gmm_trajectory.png")
    plt.show()

except ImportError:
    print("\n（未安装 matplotlib，跳过绘图）")
