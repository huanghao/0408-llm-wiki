"""
生成 MinHash 文档用的两张图：
1. SE 收敛曲线（标准误差 vs k）
2. LSH S 曲线（候选对概率 vs Jaccard 相似度）

输出到 wiki/assets/minhash_se_convergence.png 和 wiki/assets/minhash_lsh_scurve.png
运行：python tools/generate_minhash_plots.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['axes.unicode_minus'] = False

os.makedirs('wiki/assets', exist_ok=True)

# ── 图 1：SE 收敛曲线 ────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

k_vals = np.arange(1, 513)
se_vals = 0.5 / np.sqrt(k_vals)  # 最坏情况 J=0.5

ax.plot(k_vals, se_vals, color='#2563eb', linewidth=2)
ax.fill_between(k_vals, se_vals, alpha=0.08, color='#2563eb')

# 标注关键点
for k, label in [(64, 'k=64\n±0.063'), (128, 'k=128\n±0.044'),
                  (256, 'k=256\n±0.031'), (512, 'k=512\n±0.022')]:
    se = 0.5 / np.sqrt(k)
    ax.plot(k, se, 'o', color='#2563eb', markersize=7, zorder=5)
    ax.annotate(label, xy=(k, se), xytext=(k + 15, se + 0.005),
                fontsize=8.5, color='#1e40af',
                arrowprops=dict(arrowstyle='-', color='#93c5fd', lw=0.8))

ax.set_xlabel('k (number of hash functions)', fontsize=11)
ax.set_ylabel('Standard Error (SE)', fontsize=11)
ax.set_title('MinHash Estimation Accuracy vs k  (worst case J=0.5)', fontsize=12)
ax.set_xlim(0, 520)
ax.set_ylim(0, 0.12)
ax.grid(True, alpha=0.3)

ax.axvspan(0, 128, alpha=0.04, color='green', label='High-gain zone')
ax.axvspan(128, 512, alpha=0.04, color='orange', label='Diminishing returns')
ax.legend(fontsize=9, loc='upper right')

ax.annotate('Doubling k only\nreduces SE by ~30%',
            xy=(256, 0.031), xytext=(300, 0.065),
            fontsize=8.5, color='#92400e',
            arrowprops=dict(arrowstyle='->', color='#d97706', lw=1))

plt.tight_layout()
plt.savefig('wiki/assets/minhash_se_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 生成 wiki/assets/minhash_se_convergence.png")


# ── 图 2：LSH S 曲线 ─────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4.5))

s = np.linspace(0, 1, 500)

# All configs use k=128 (b*r=128)
configs = [
    (32, 4,  '#ef4444', 'b=32, r=4   k=128  threshold≈0.59  (aggressive)'),
    (16, 8,  '#2563eb', 'b=16, r=8   k=128  threshold≈0.78  (typical)'),
    (8,  16, '#16a34a', 'b=8,  r=16  k=128  threshold≈0.91  (conservative)'),
]

for b, r, color, label in configs:
    p = 1 - (1 - s**r)**b
    threshold = (1/b)**(1/r)
    ax.plot(s, p, color=color, linewidth=2, label=label)
    # Mark the threshold point on the curve
    p_at_threshold = 1 - (1 - threshold**r)**b
    ax.plot(threshold, p_at_threshold, 'o', color=color, markersize=8, zorder=5)
    ax.axvline(threshold, color=color, linewidth=0.8, linestyle='--', alpha=0.4)

ax.axhline(0.5, color='gray', linewidth=0.6, linestyle=':', alpha=0.6,
           label='P=0.5 (inflection reference)')
ax.set_xlabel('Jaccard similarity s', fontsize=11)
ax.set_ylabel('P(become candidate pair)', fontsize=11)
ax.set_title('LSH S-curve  (all configs: k=128)\nP = 1 - (1 - s^r)^b,  dots mark the threshold s*=(1/b)^(1/r)', fontsize=11)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=8.5, loc='upper left')

ax.annotate('False Negatives\n(similar, not selected)',
            xy=(0.68, 0.12), fontsize=8.5, color='#7c3aed',
            ha='center')
ax.annotate('False Positives\n(dissimilar, selected)',
            xy=(0.35, 0.35), fontsize=8.5, color='#b45309',
            ha='center')

plt.tight_layout()
plt.savefig('wiki/assets/minhash_lsh_scurve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ 生成 wiki/assets/minhash_lsh_scurve.png")
