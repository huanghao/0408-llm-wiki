"""
GNN Demo: 图神经网络从零实现
用纯 PyTorch 实现 GCN（Graph Convolutional Network），
任务：节点分类（Zachary's Karate Club，34 个节点，分 2 类）

运行: python gnn_demo.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ─────────────────────────────────────────
# 1. 数据：Zachary's Karate Club
#    34 个成员，78 条边，两派（0/1）
# ─────────────────────────────────────────

# 边列表（无向图，每条边存两个方向）
edges = [
    (0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8),(0,10),(0,11),(0,12),
    (0,13),(0,17),(0,19),(0,21),(0,31),
    (1,2),(1,3),(1,7),(1,13),(1,17),(1,19),(1,21),(1,30),
    (2,3),(2,7),(2,8),(2,9),(2,13),(2,27),(2,28),(2,32),
    (3,7),(3,12),(3,13),
    (4,6),(4,10),
    (5,6),(5,10),(5,16),
    (6,16),
    (8,30),(8,32),(8,33),
    (9,33),
    (13,33),
    (14,32),(14,33),
    (15,32),(15,33),
    (18,32),(18,33),
    (19,33),
    (20,32),(20,33),
    (22,32),(22,33),
    (23,25),(23,27),(23,29),(23,32),(23,33),
    (24,25),(24,27),(24,31),
    (25,31),
    (26,29),(26,33),
    (27,33),
    (28,31),(28,33),
    (29,32),(29,33),
    (30,32),(30,33),
    (31,32),(31,33),
    (32,33),
]

N = 34  # 节点数

# 构建邻接矩阵（对称）
A = torch.zeros(N, N)
for u, v in edges:
    A[u, v] = 1.0
    A[v, u] = 1.0

# 节点特征：用 one-hot 编码节点 ID（34 维），真实场景会用更有意义的特征
X = torch.eye(N)  # shape: (34, 34)

# 标签：0 = 节点 0 所在派系，1 = 节点 33 所在派系
# 真实标签基于 Karate Club 历史记录
labels_list = [
    0,0,0,0,0,0,0,0,1,1,
    0,0,0,0,1,1,0,0,1,0,
    1,0,1,1,1,1,1,1,1,1,
    1,1,1,1
]
y = torch.tensor(labels_list, dtype=torch.long)

# 训练时只用节点 0 和节点 33 作为监督信号（半监督，其余节点无标签）
train_mask = torch.zeros(N, dtype=torch.bool)
train_mask[0] = True
train_mask[33] = True

# ─────────────────────────────────────────
# 2. 归一化邻接矩阵
#    GCN 用 Ã = D̃^{-1/2} Ã D̃^{-1/2}
#    其中 Ã = A + I（加自环），D̃ 是 Ã 的度矩阵
# ─────────────────────────────────────────

def normalize_adjacency(A):
    """
    GCN 标准归一化：Ã_norm = D̃^{-1/2} (A+I) D̃^{-1/2}

    作用：
    - +I（加自环）：聚合时包含节点自身特征
    - D^{-1/2}...D^{-1/2}：对称归一化，防止度数大的节点主导聚合
    """
    A_hat = A + torch.eye(N)                    # 加自环
    D = A_hat.sum(dim=1)                        # 每行之和 = 度数（含自环）
    D_inv_sqrt = torch.diag(D.pow(-0.5))        # D^{-1/2}
    return D_inv_sqrt @ A_hat @ D_inv_sqrt      # 对称归一化

A_norm = normalize_adjacency(A)

# ─────────────────────────────────────────
# 3. GCN 模型定义
#
# 核心公式（一层 GCN）：
#   H^{(l+1)} = σ( Ã_norm · H^{(l)} · W^{(l)} )
#
# - Ã_norm · H：消息传递，把邻居特征聚合进来（加权平均）
# - · W：线性变换（学习如何利用聚合结果）
# - σ：激活函数（ReLU）
# ─────────────────────────────────────────

class GCNLayer(nn.Module):
    """单层 GCN：H_out = σ(A_norm · H_in · W)"""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W = nn.Linear(in_features, out_features, bias=False)

    def forward(self, A_norm, H):
        # 先线性变换，再做图聚合（等价于先聚合再变换，但更节省内存）
        return A_norm @ self.W(H)


class GCN(nn.Module):
    """两层 GCN：输入特征 → 隐层 → 分类 logits"""
    def __init__(self, in_features, hidden, num_classes, dropout=0.5):
        super().__init__()
        self.layer1 = GCNLayer(in_features, hidden)
        self.layer2 = GCNLayer(hidden, num_classes)
        self.dropout = dropout

    def forward(self, A_norm, X):
        H = F.relu(self.layer1(A_norm, X))
        H = F.dropout(H, p=self.dropout, training=self.training)
        H = self.layer2(A_norm, H)      # 最后一层不加激活，直接输出 logits
        return H                        # shape: (N, num_classes)

# ─────────────────────────────────────────
# 4. 训练
# ─────────────────────────────────────────

torch.manual_seed(42)
model = GCN(in_features=N, hidden=16, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

def train(epochs=200):
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()

        logits = model(A_norm, X)                            # 所有节点的 logits
        loss = F.cross_entropy(logits[train_mask], y[train_mask])  # 只在有标签节点上算损失
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            with torch.no_grad():
                logits = model(A_norm, X)
                pred = logits.argmax(dim=1)
                acc_train = (pred[train_mask] == y[train_mask]).float().mean()
                acc_all   = (pred == y).float().mean()
            print(f"Epoch {epoch:3d} | loss {loss:.4f} | "
                  f"train acc {acc_train:.2f} | all-node acc {acc_all:.2f}")

train()

# ─────────────────────────────────────────
# 5. 最终评估 + 可视化节点嵌入
# ─────────────────────────────────────────

model.eval()
with torch.no_grad():
    logits = model(A_norm, X)
    pred = logits.argmax(dim=1)
    acc = (pred == y).float().mean()

print(f"\n最终准确率（全部 34 节点）: {acc:.2%}")
print(f"预测: {pred.tolist()}")
print(f"真实: {y.tolist()}")

# 打印哪些节点预测错了
wrong = (pred != y).nonzero(as_tuple=True)[0].tolist()
print(f"预测错误的节点: {wrong if wrong else '无'}")

# 可视化中层嵌入（用第一层输出的 16 维 embedding 做 PCA 降到 2 维）
with torch.no_grad():
    H1 = F.relu(model.layer1(A_norm, X))   # shape: (34, 16)

try:
    from sklearn.decomposition import PCA
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pca = PCA(n_components=2)
    coords = pca.fit_transform(H1.numpy())

    colors = ['steelblue' if yi == 0 else 'tomato' for yi in y.tolist()]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(coords[:, 0], coords[:, 1], c=colors, s=80, edgecolors='k', linewidths=0.5)
    for i, (cx, cy) in enumerate(coords):
        ax.annotate(str(i), (cx, cy), fontsize=7, ha='center', va='center', color='white', fontweight='bold')
    ax.set_title('GCN 节点嵌入（第一层，PCA 降维）\n蓝=派系0  红=派系1')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    plt.tight_layout()
    plt.savefig('gnn_embeddings.png', dpi=150)
    print("\n嵌入可视化已保存到 gnn_embeddings.png")
except ImportError:
    print("\n（sklearn/matplotlib 未安装，跳过可视化）")
