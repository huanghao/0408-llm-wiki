"""
GMM 通用演示：用身高体重数据理解高斯混合模型

场景：一组人的身高分布——男女身高各自接近正态，合在一起是双峰分布。
单个高斯拟合不好，GMM 可以自动找出两个峰。

运行：python src/gmm_demo.py
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

plt.rcParams['font.family'] = ['Hiragino Sans GB', 'STHeiti', 'PingFang HK', 'Arial Unicode MS', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

np.random.seed(42)


# ── 1. 生成双峰数据（模拟男女身高） ────────────────────────────────────

n_female, n_male = 300, 270
heights_female = np.random.normal(loc=163, scale=6, size=n_female)   # 女性平均 163cm
heights_male   = np.random.normal(loc=175, scale=7, size=n_male)      # 男性平均 175cm
heights_all    = np.concatenate([heights_female, heights_male])
np.random.shuffle(heights_all)


# ── 2. EM 算法手动实现（帮助理解 GMM 是怎么"学"的） ────────────────────

def gmm_em_1d(data, K=2, n_iter=50):
    """1D GMM 的 EM 算法：从数据中学出 K 个高斯分量的参数"""
    n = len(data)

    # 初始化：随机选 K 个均值，方差和权重均匀
    mu = np.random.choice(data, K)
    sigma = np.array([data.std()] * K)
    pi = np.ones(K) / K

    for _ in range(n_iter):
        # E 步：计算每个样本属于每个分量的概率（responsibility）
        resp = np.zeros((n, K))
        for k in range(K):
            resp[:, k] = pi[k] * np.exp(-0.5 * ((data - mu[k]) / sigma[k])**2) / (sigma[k] * np.sqrt(2 * np.pi))
        resp /= resp.sum(axis=1, keepdims=True)  # 归一化

        # M 步：用责任值加权更新参数
        Nk = resp.sum(axis=0)
        mu = (resp * data[:, None]).sum(axis=0) / Nk
        sigma = np.sqrt((resp * (data[:, None] - mu)**2).sum(axis=0) / Nk)
        pi = Nk / n

    return mu, sigma, pi

# 2D GMM EM 算法
def gmm_em_2d(X, K=2, n_iter=60):
    n, d = X.shape
    # 初始化：用 k-means 风格，随机选 K 个点作为均值
    idx = np.random.choice(n, K, replace=False)
    mu = X[idx].astype(float)
    cov = [np.cov(X.T) * 0.5 for _ in range(K)]
    pi = np.ones(K) / K

    for _ in range(n_iter):
        # E 步
        resp = np.zeros((n, K))
        for k in range(K):
            diff = X - mu[k]
            cov_inv = np.linalg.inv(cov[k])
            det = np.linalg.det(cov[k])
            exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
            resp[:, k] = pi[k] / (2 * np.pi * np.sqrt(det + 1e-10)) * np.exp(exponent)
        resp /= resp.sum(axis=1, keepdims=True) + 1e-10

        # M 步
        Nk = resp.sum(axis=0)
        mu = (resp.T @ X) / Nk[:, None]
        for k in range(K):
            diff = X - mu[k]
            cov[k] = (resp[:, k:k+1] * diff).T @ diff / Nk[k]
            cov[k] += np.eye(d) * 1e-6  # 防止奇异
        pi = Nk / n

    return mu, cov, pi

print("=" * 60)
print("1D GMM：拟合双峰身高分布")
print("=" * 60)
mu_fit, sigma_fit, pi_fit = gmm_em_1d(heights_all, K=2)

# 按均值排序，让 k=0 是小均值（女性）
order = np.argsort(mu_fit)
mu_fit, sigma_fit, pi_fit = mu_fit[order], sigma_fit[order], pi_fit[order]

print(f"\n真实参数：")
print(f"  女性：μ=163, σ=6, π={n_female/(n_female+n_male):.2f}")
print(f"  男性：μ=175, σ=7, π={n_male/(n_female+n_male):.2f}")
print(f"\nGMM 学出的参数：")
for k in range(2):
    label = "分量0（女性?）" if k == 0 else "分量1（男性?）"
    print(f"  {label}：μ={mu_fit[k]:.1f}cm, σ={sigma_fit[k]:.1f}cm, π={pi_fit[k]:.2f}")


# ── 3. 1D vs 2D 高斯的区别演示 ────────────────────────────────────────

print("\n" + "=" * 60)
print("1D vs 2D 高斯的区别")
print("=" * 60)
print("""
1D 高斯（单变量）：描述一个数轴上的分布
  - 均值 μ：数轴上的中心位置
  - 方差 σ²：沿数轴的扩散程度
  - 例：身高的分布，只有一个维度

2D 高斯（双变量）：描述平面上的分布
  - 均值 (μ_x, μ_y)：平面上的中心点
  - 协方差矩阵 Σ：x 方向扩散多大、y 方向扩散多大、两个方向是否相关
  - 例：车辆终点的 (x, y) 坐标分布

关键区别：
  - 1D 多峰 GMM：同一个变量（如身高）有多个聚集区域
  - 2D 高斯：一个事件发生在二维空间里的某个位置
  - 2D GMM：在 2D 空间里有多个聚集区域

驾驶里为什么用 2D 高斯：
  车辆的「意图」最终体现在一个 2D 位置（终点的 x,y）。
  左转的终点可能是 (-20m, +15m)，但由于执行不确定性，
  实际停在 (-18m, +16m) 或 (-22m, +14m) 都是合理的。
  2D 高斯的 (μ_x, μ_y) 是最可能的终点，
  (σ_x, σ_y) 描述 x/y 方向各自有多大的不确定性。
""")


# ── 4. 用 matplotlib 画图 ─────────────────────────────────────────────

try:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("GMM 演示：身高双峰分布", fontsize=14)

    x = np.linspace(140, 205, 500)

    # 子图1：原始数据直方图
    ax = axes[0]
    ax.hist(heights_all, bins=40, density=True, alpha=0.6, color='steelblue', label='观测数据')
    ax.set_xlabel("身高 (cm)")
    ax.set_ylabel("概率密度")
    ax.set_title("原始数据：双峰分布")
    ax.legend()

    # 子图2：单高斯 vs GMM 拟合
    ax = axes[1]
    ax.hist(heights_all, bins=40, density=True, alpha=0.4, color='steelblue')

    # 单高斯拟合（平均数、总方差）
    single_mu, single_sigma = heights_all.mean(), heights_all.std()
    single_pdf = np.exp(-0.5 * ((x - single_mu) / single_sigma)**2) / (single_sigma * np.sqrt(2 * np.pi))
    ax.plot(x, single_pdf, 'r-', lw=2, label=f'单高斯 μ={single_mu:.0f}')

    # GMM 拟合
    gmm_pdf = sum(pi_fit[k] * np.exp(-0.5 * ((x - mu_fit[k]) / sigma_fit[k])**2) /
                  (sigma_fit[k] * np.sqrt(2 * np.pi)) for k in range(2))
    ax.plot(x, gmm_pdf, 'g-', lw=2, label='GMM (K=2)')

    # 各分量
    for k in range(2):
        comp = pi_fit[k] * np.exp(-0.5 * ((x - mu_fit[k]) / sigma_fit[k])**2) / (sigma_fit[k] * np.sqrt(2 * np.pi))
        ax.fill_between(x, comp, alpha=0.2, label=f'分量{k}: μ={mu_fit[k]:.0f}cm')

    ax.set_xlabel("身高 (cm)")
    ax.set_title("单高斯 vs GMM")
    ax.legend(fontsize=8)

    # 子图3：2D GMM 拟合（EM 算法，身高 vs 体重）
    ax = axes[2]

    # 生成 2D 数据（混合在一起，标签对 EM 不可见）
    w_female = np.random.normal(loc=57, scale=8, size=n_female)
    w_male   = np.random.normal(loc=72, scale=10, size=n_male)
    X2d = np.column_stack([
        np.concatenate([heights_female, heights_male]),
        np.concatenate([w_female, w_male])
    ])

    mu2d, cov2d, pi2d = gmm_em_2d(X2d, K=2)

    # 散点（颜色只用于参考，EM 不知道标签）
    ax.scatter(X2d[:n_female, 0], X2d[:n_female, 1], alpha=0.2, s=5, c='pink')
    ax.scatter(X2d[n_female:, 0], X2d[n_female:, 1], alpha=0.2, s=5, c='lightblue')

    # 画 EM 学出的 2D 高斯等高线椭圆（1σ 和 2σ）
    colors_2d = ['red', 'blue']
    for k, color in enumerate(colors_2d):
        # 从协方差矩阵计算椭圆参数
        eigvals, eigvecs = np.linalg.eigh(cov2d[k])
        eigvals = np.maximum(eigvals, 0)
        angle = np.degrees(np.arctan2(eigvecs[1, -1], eigvecs[0, -1]))
        for n_sigma, alpha in [(1, 0.6), (2, 0.3)]:
            ell = Ellipse(
                xy=mu2d[k],
                width=2 * n_sigma * np.sqrt(eigvals[-1]),
                height=2 * n_sigma * np.sqrt(eigvals[0]),
                angle=angle,
                edgecolor=color, facecolor='none',
                linewidth=2, linestyle='-' if n_sigma == 1 else '--', alpha=alpha
            )
            ax.add_patch(ell)
        ax.plot(*mu2d[k], '+', color=color, markersize=14, markeredgewidth=2.5,
                label=f'分量{k} μ=({mu2d[k,0]:.0f},{mu2d[k,1]:.0f}) π={pi2d[k]:.2f}')

    # 画 GMM 合并后的联合密度等高线
    h_grid = np.linspace(140, 210, 200)
    w_grid = np.linspace(30, 110, 200)
    HH, WW = np.meshgrid(h_grid, w_grid)
    grid = np.column_stack([HH.ravel(), WW.ravel()])
    density = np.zeros(len(grid))
    for k in range(2):
        diff = grid - mu2d[k]
        cov_inv = np.linalg.inv(cov2d[k])
        det = np.linalg.det(cov2d[k])
        exponent = -0.5 * np.sum(diff @ cov_inv * diff, axis=1)
        density += pi2d[k] / (2 * np.pi * np.sqrt(det)) * np.exp(exponent)
    density = density.reshape(HH.shape)
    ax.contour(HH, WW, density, levels=6, colors='gray', alpha=0.5, linewidths=1)

    ax.set_xlabel("身高 (cm)")
    ax.set_ylabel("体重 (kg)")
    ax.set_title("2D GMM（EM 拟合，K=2）\n灰色=联合密度等高线，实/虚=1σ/2σ")
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("docs/gmm_demo.png", dpi=120, bbox_inches='tight')
    print("\n图表已保存到 docs/gmm_demo.png")
    plt.show()

except ImportError:
    print("\n（未安装 matplotlib，跳过绘图）")
