"""
Word Embedding 从零手搓（PyTorch 版）
=====================================
实现 Skip-gram + 负采样，用 torch。

目标：
  1. 理解嵌入矩阵是怎么从随机数训练出来的
  2. 理解负采样在做什么
  3. 验证训练后语义相近的词向量确实更近
  4. 演示推广到非词特征（任意离散 ID）

运行：python tools/word_embedding_scratch.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter
import time

# ─────────────────────────────────────────────
# 1. 准备语料
# ─────────────────────────────────────────────

CORPUS = """
猫 喜欢 吃 鱼 猫 喜欢 喝 牛奶 猫 是 动物
狗 喜欢 吃 骨头 狗 喜欢 玩 球 狗 是 动物
猫 和 狗 都 是 宠物 宠物 需要 照顾
苹果 是 水果 香蕉 是 水果 橙子 是 水果
苹果 和 香蕉 都 很 甜 水果 对 身体 好
汽车 是 交通工具 自行车 是 交通工具
汽车 需要 汽油 自行车 需要 人力
猫 喜欢 晒太阳 狗 喜欢 跑步
苹果 香蕉 橙子 都 是 健康 食品
猫 狗 宠物 动物 都 需要 食物 和 水
""".strip().split()

CORPUS = CORPUS * 50  # 重复语料增加样本量

# ─────────────────────────────────────────────
# 2. 建词表
# ─────────────────────────────────────────────

word_counts = Counter(CORPUS)
vocab = sorted(word_counts.keys())
word2id = {w: i for i, w in enumerate(vocab)}
id2word = {i: w for w, i in word2id.items()}
V = len(vocab)

print(f"词表大小：{V} 个词")
print(f"语料长度：{len(CORPUS)} 个词\n")

# ─────────────────────────────────────────────
# 3. 生成训练样本（Skip-gram）
# ─────────────────────────────────────────────
# Skip-gram：给定中心词，预测上下文词。
# 比如句子 "猫 喜欢 吃 鱼"，窗口=2，以"吃"为中心，
# 生成样本对 (吃, 猫), (吃, 喜欢), (吃, 鱼)

WINDOW = 2

def build_skipgram_pairs(corpus, word2id, window):
    ids = [word2id[w] for w in corpus]
    pairs = []
    for i, center in enumerate(ids):
        for j in range(max(0, i - window), min(len(ids), i + window + 1)):
            if i != j:
                pairs.append((center, ids[j]))
    return torch.tensor(pairs, dtype=torch.long)  # (N, 2)

pairs = build_skipgram_pairs(CORPUS, word2id, WINDOW)
print(f"Skip-gram 正样本对数：{len(pairs)}")
print(f"示例（前5对）：{[(id2word[c.item()], id2word[ctx.item()]) for c, ctx in pairs[:5]]}\n")

# ─────────────────────────────────────────────
# 4. 负采样权重（按词频^0.75 平滑）
# ─────────────────────────────────────────────
# 高频词被压低，低频词被提升，避免负例全是"的/了/是"这类高频词

counts = torch.tensor([word_counts[id2word[i]] for i in range(V)], dtype=torch.float)
neg_weights = counts ** 0.75
neg_weights /= neg_weights.sum()  # 归一化为概率分布

# ─────────────────────────────────────────────
# 5. 模型定义
# ─────────────────────────────────────────────
# 两个嵌入矩阵：
#   W_in：中心词用，shape (V, DIM)
#   W_out：上下文词用，shape (V, DIM)
#
# 为什么要两个？训练时中心词和上下文词扮演不同角色，
# 用两个矩阵效果更好。训练完后通常只用 W_in。

DIM = 16

class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.W_in  = nn.Embedding(vocab_size, dim)   # 中心词嵌入
        self.W_out = nn.Embedding(vocab_size, dim)   # 上下文词嵌入

        # 初始化：W_in 均匀随机，W_out 全零（标准做法）
        nn.init.uniform_(self.W_in.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.W_out.weight)

    def forward(self, center_ids, context_ids):
        """
        输入：
          center_ids:  (B,)    中心词 ID
          context_ids: (B, 1+N_NEG)  第0列是正例，后N_NEG列是负例

        输出：
          logits: (B, 1+N_NEG)  每对的内积得分（未经 sigmoid）
        """
        center_vecs  = self.W_in(center_ids)           # (B, DIM)
        context_vecs = self.W_out(context_ids)         # (B, 1+N_NEG, DIM)

        # 每个中心词向量和对应的上下文向量做内积
        # einsum 'bd,bnd->bn'：对每个 batch，中心词向量 dot 每个上下文向量
        logits = torch.einsum('bd,bnd->bn', center_vecs, context_vecs)
        return logits  # (B, 1+N_NEG)


model = SkipGram(V, DIM)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print(f"模型参数：两个嵌入矩阵，各 {V}×{DIM}，共 {2*V*DIM} 个参数\n")

# ─────────────────────────────────────────────
# 6. 训练
# ─────────────────────────────────────────────
# 损失函数：BCEWithLogitsLoss（二元交叉熵）
#   正例目标 = 1（这对词确实共现过）
#   负例目标 = 0（这对词是随机凑的）
#
# 直觉：训练完后，真实共现的词对内积大（sigmoid → 接近1），
#       随机词对内积小（sigmoid → 接近0）

N_NEG  = 5     # 每个正例配几个负例
EPOCHS = 50
BATCH  = 512

loss_fn = nn.BCEWithLogitsLoss()

print("开始训练...")
t0 = time.perf_counter()

for epoch in range(EPOCHS):
    # 每轮打乱顺序
    perm = torch.randperm(len(pairs))
    total_loss = 0.0
    n_batches = 0

    for i in range(0, len(pairs), BATCH):
        batch = pairs[perm[i:i+BATCH]]          # (B, 2)
        center_ids = batch[:, 0]                # (B,)
        pos_ids    = batch[:, 1]                # (B,)  正例

        # 按词频权重采样负例
        B = len(center_ids)
        neg_ids = torch.multinomial(
            neg_weights.expand(B, -1),          # (B, V)
            num_samples=N_NEG,
            replacement=True,
        )                                       # (B, N_NEG)

        # 把正例和负例拼在一起：(B, 1+N_NEG)
        context_ids = torch.cat([pos_ids.unsqueeze(1), neg_ids], dim=1)

        # 标签：正例=1，负例=0
        labels = torch.zeros(B, 1 + N_NEG)
        labels[:, 0] = 1.0

        # 前向 + 反向
        logits = model(center_ids, context_ids)  # (B, 1+N_NEG)
        loss = loss_fn(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  loss={total_loss/n_batches:.4f}")

train_time = time.perf_counter() - t0
print(f"\n训练完成，耗时 {train_time:.2f} 秒\n")

# ─────────────────────────────────────────────
# 7. 验证：语义相近的词向量是否更近
# ─────────────────────────────────────────────

# 取出训练好的嵌入矩阵
embeddings = model.W_in.weight.detach()  # (V, DIM)

def most_similar(word, top_k=4):
    if word not in word2id:
        return []
    wid = word2id[word]
    vec = embeddings[wid]                  # (DIM,)
    # 和所有词计算余弦相似度
    sims = F.cosine_similarity(vec.unsqueeze(0), embeddings)  # (V,)
    sims[wid] = -1  # 排除自身
    top_ids = sims.topk(top_k).indices.tolist()
    return [(id2word[i], sims[i].item()) for i in top_ids]

def cosine(w1, w2):
    v1 = embeddings[word2id[w1]]
    v2 = embeddings[word2id[w2]]
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()

print("=" * 50)
print("验证：训练后语义相近的词向量余弦相似度")
print("=" * 50)

for word in ["猫", "苹果", "汽车"]:
    neighbors = most_similar(word)
    print(f"\n'{word}' 最近的词：")
    for neighbor, sim in neighbors:
        print(f"  {neighbor:6s}  {sim:.3f}")

print("\n" + "─" * 40)
print("直接对比：")
for w1, w2 in [("猫","狗"), ("猫","苹果"), ("苹果","香蕉"), ("苹果","汽车")]:
    mark = "✓近" if cosine(w1,w2) > 0.3 else "✗远"
    print(f"  {w1} ↔ {w2}：{cosine(w1,w2):.3f}  {mark}")

# ─────────────────────────────────────────────
# 8. 推广：任意离散特征的嵌入（用户-商品共现）
# ─────────────────────────────────────────────
# 核心思路完全一样，只需把"词"换成任意离散 ID，
# 把"上下文共现"换成你关心的"共现"定义。

print("\n" + "=" * 50)
print("推广：「用户-商品」共现 embedding（推荐系统场景）")
print("=" * 50)

purchase_logs = [
    ["user_A", "item_手机", "item_耳机", "item_充电器"],
    ["user_A", "item_手机", "item_手机壳"],
    ["user_B", "item_书籍", "item_笔记本", "item_钢笔"],
    ["user_B", "item_书籍", "item_书签"],
    ["user_C", "item_手机", "item_书籍", "item_耳机"],
    ["user_D", "item_充电器", "item_耳机", "item_手机壳"],
] * 30

item_corpus = [token for log in purchase_logs for token in log]
item_counts = Counter(item_corpus)
item_vocab  = sorted(item_counts.keys())
item2id     = {w: i for i, w in enumerate(item_vocab)}
id2item     = {i: w for w, i in item2id.items()}
VI = len(item_vocab)

item_pairs   = build_skipgram_pairs(item_corpus, item2id, window=2)
item_weights = torch.tensor(
    [item_counts[id2item[i]] for i in range(VI)], dtype=torch.float
) ** 0.75
item_weights /= item_weights.sum()

# ── 完全相同的模型结构，只换了词表大小 ──
item_model = SkipGram(VI, DIM)
item_opt   = torch.optim.Adam(item_model.parameters(), lr=0.01)

for epoch in range(50):
    perm = torch.randperm(len(item_pairs))
    for i in range(0, len(item_pairs), BATCH):
        batch      = item_pairs[perm[i:i+BATCH]]
        center_ids = batch[:, 0]
        pos_ids    = batch[:, 1]
        B          = len(center_ids)
        neg_ids    = torch.multinomial(item_weights.expand(B, -1), N_NEG, replacement=True)
        context_ids = torch.cat([pos_ids.unsqueeze(1), neg_ids], dim=1)
        labels     = torch.zeros(B, 1 + N_NEG)
        labels[:, 0] = 1.0
        loss = loss_fn(item_model(center_ids, context_ids), labels)
        item_opt.zero_grad(); loss.backward(); item_opt.step()

item_emb = item_model.W_in.weight.detach()

def item_most_similar(word, top_k=5):
    wid  = item2id[word]
    vec  = item_emb[wid]
    sims = F.cosine_similarity(vec.unsqueeze(0), item_emb)
    sims[wid] = -1
    top_ids = sims.topk(top_k).indices.tolist()
    return [(id2item[i], sims[i].item()) for i in top_ids]

print(f"\n'{item_vocab}'")
print("\n'item_手机' 最近的 item/user：")
for neighbor, sim in item_most_similar("item_手机"):
    print(f"  {neighbor:20s}  {sim:.3f}")

print("\n关键点：")
print("  代码结构和词向量完全一样")
print("  只需把「词」换成「任意离散 ID」")
print("  「共现」的定义可以自己设计（购买、点击、同一文档出现…）")
