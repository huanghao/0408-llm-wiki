"""
fastText Demo
=============
Demo 1: 用 PyTorch 手写 fastText，分推理和训练两部分
Demo 2: 用真实 fastText 库训练语言识别分类器，测量速度

运行：python tools/fasttext_demo.py
"""

import time
import random
import tempfile
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# ═══════════════════════════════════════════════════════
# 公共工具：特征提取（推理和训练都用）
# ═══════════════════════════════════════════════════════

VOCAB_SIZE = 300   # 词表槽数
BUCKET_SIZE = 1000 # n-gram 哈希桶大小（实际是 200 万）
EMBED_SIZE = VOCAB_SIZE + BUCKET_SIZE  # 嵌入矩阵总行数
DIM = 16           # 向量维度

def fnv1a_hash(s: str) -> int:
    """FNV-1a 哈希，和 fastText 源码一致"""
    h = 2166136261
    for c in s.encode("utf-8"):
        h ^= c
        h = (h * 16777619) & 0xFFFFFFFF
    return h

def get_char_ngrams(word: str, minn=3, maxn=4) -> list[str]:
    """提取字符 n-gram，加 <> 边界符"""
    word = f"<{word}>"
    return [word[i:i+n]
            for n in range(minn, maxn + 1)
            for i in range(len(word) - n + 1)]

def text_to_ids(text: str) -> list[int]:
    """
    文本 → 特征 ID 列表
    每个词贡献：词本身的 ID + 所有字符 n-gram 的 ID
    词 ID  = hash(词) % VOCAB_SIZE
    n-gram ID = VOCAB_SIZE + (hash(n-gram) % BUCKET_SIZE)
    """
    ids = []
    for word in text.lower().split():
        ids.append(fnv1a_hash(word) % VOCAB_SIZE)
        for ng in get_char_ngrams(word):
            ids.append(VOCAB_SIZE + fnv1a_hash(ng) % BUCKET_SIZE)
    return ids

# ═══════════════════════════════════════════════════════
# Demo 1a：推理——给一段文本打分（用随机权重演示流程）
# ═══════════════════════════════════════════════════════

def demo1a_inference():
    print("=" * 60)
    print("Demo 1a：推理流程（用随机权重，演示每一步在做什么）")
    print("=" * 60)

    # 1. 字符 n-gram 提取
    print("\n── 步骤1：字符 n-gram 提取 ──")
    word = "running"
    ngrams = get_char_ngrams(word)
    print(f"  '{word}' 的 n-gram（minn=3, maxn=4）：{ngrams}")

    oov = "runing"  # 拼写错误的词
    overlap = set(ngrams) & set(get_char_ngrams(oov))
    print(f"  拼写错误 '{oov}' 与 '{word}' 共享的 n-gram：{sorted(overlap)}")
    print(f"  → OOV 词通过共享 n-gram 仍能得到合理向量")

    # 2. 哈希分桶
    print("\n── 步骤2：哈希分桶（n-gram → embedding ID）──")
    print(f"  词表槽 0~{VOCAB_SIZE-1}，n-gram 桶 {VOCAB_SIZE}~{EMBED_SIZE-1}")
    for ng in ngrams[:3]:
        nid = VOCAB_SIZE + fnv1a_hash(ng) % BUCKET_SIZE
        print(f"  '{ng}' → hash={fnv1a_hash(ng)} → ID={nid}")
    print(f"  不同 n-gram 可能映射到同一 ID（哈希碰撞），桶足够大时影响不大")

    # 3. 查嵌入矩阵 + 取平均
    print("\n── 步骤3：查嵌入矩阵 + 取平均 ──")
    torch.manual_seed(42)
    embedding = nn.Embedding(EMBED_SIZE, DIM)  # 嵌入矩阵，此时是随机初始化

    text = "machine learning is great"
    ids = torch.tensor(text_to_ids(text))
    vecs = embedding(ids)          # (N_features, DIM)：查表，每个特征取对应行
    avg  = vecs.mean(dim=0)        # (DIM,)：所有向量取平均
    print(f"  文本：'{text}'")
    print(f"  特征数：{len(ids)}（词 + n-gram）")
    print(f"  平均向量（前4维）：{avg[:4].tolist()}")

    # 4. 线性层 + softmax
    print("\n── 步骤4：线性层 + softmax → 类别概率 ──")
    classifier = nn.Linear(DIM, 2)  # 2 类：正面/负面
    logits = classifier(avg.unsqueeze(0))          # (1, 2)
    probs  = F.softmax(logits, dim=-1)
    print(f"  logits：{logits.tolist()}")
    print(f"  概率（随机权重，无意义）：正面={probs[0,0].item():.2f}，负面={probs[0,1].item():.2f}")
    print(f"  → 训练后权重会变成有意义的值")

# ═══════════════════════════════════════════════════════
# Demo 1b：训练——让模型从数据中学习
# ═══════════════════════════════════════════════════════

class FastTextClassifier(nn.Module):
    """
    fastText 分类器的 PyTorch 实现
    参数：
      embedding (V+B, DIM)：输入嵌入矩阵，词 + n-gram 共用
      classifier (DIM, C)：线性分类层
    """
    def __init__(self, embed_size, dim, n_classes):
        super().__init__()
        self.embedding  = nn.Embedding(embed_size, dim)
        self.classifier = nn.Linear(dim, n_classes)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """
        ids: (N,) 特征 ID 列表
        返回: (n_classes,) logits
        """
        vecs   = self.embedding(ids)       # (N, DIM)  查嵌入矩阵
        avg    = vecs.mean(dim=0)          # (DIM,)    取平均
        logits = self.classifier(avg)      # (n_classes,) 线性分类
        return logits

def demo1b_training():
    print("\n" + "=" * 60)
    print("Demo 1b：训练流程（情绪分类，含反向传播）")
    print("=" * 60)

    # 训练数据：(文本, 标签)，0=负面，1=正面
    train_data = [
        ("great excellent amazing wonderful fantastic", 1),
        ("good nice happy enjoy love brilliant",        1),
        ("awesome superb perfect outstanding",          1),
        ("terrible awful horrible bad disgusting",      0),
        ("hate dislike ugly worst poor",                0),
        ("disappointing dreadful nasty inferior",       0),
    ]

    model     = FastTextClassifier(EMBED_SIZE, DIM, n_classes=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn   = nn.CrossEntropyLoss()

    print(f"\n模型参数：嵌入矩阵 {EMBED_SIZE}×{DIM} + 分类层 {DIM}×2")
    print(f"训练 {len(train_data)} 条样本，100 轮\n")

    for epoch in range(100):
        total_loss = 0.0
        random.shuffle(train_data)
        for text, label in train_data:
            ids    = torch.tensor(text_to_ids(text))
            logits = model(ids).unsqueeze(0)             # (1, 2)
            target = torch.tensor([label])
            loss   = loss_fn(logits, target)

            optimizer.zero_grad()
            loss.backward()   # 反向传播：自动计算所有参数的梯度
            optimizer.step()  # 更新参数（包括 embedding 里用到的行）
            total_loss += loss.item()

        if (epoch + 1) % 25 == 0:
            print(f"  Epoch {epoch+1:3d}  loss={total_loss/len(train_data):.4f}")

    # 测试
    print("\n训练后预测：")
    test_cases = [
        ("great wonderful amazing", 1),
        ("terrible horrible bad",   0),
        ("good nice",               1),
        ("awful ugly",              0),
    ]
    model.eval()
    with torch.no_grad():
        for text, label in test_cases:
            ids   = torch.tensor(text_to_ids(text))
            probs = F.softmax(model(ids), dim=0)
            pred  = probs.argmax().item()
            ok    = "✓" if pred == label else "✗"
            print(f"  {ok} '{text}' → {'正面' if pred==1 else '负面'}"
                  f"（正面概率={probs[1].item():.2f}）")

    print("\n反向传播做了什么：")
    print("  loss.backward() 自动计算梯度，沿路径反向传播：")
    print("  分类层权重 ← 平均向量 ← 每个用到的 embedding 行")
    print("  只有这次 forward 用到的 embedding 行被更新（稀疏更新）")


def demo1_manual():
    demo1a_inference()
    demo1b_training()


# ─────────────────────────────────────────────
# Demo 2: 用真实 fastText 库训练语言识别，测量速度
# ─────────────────────────────────────────────

def demo2_real_fasttext():
    print("\n" + "=" * 60)
    print("Demo 2: 真实 fastText 训练语言识别分类器 + 速度测量")
    print("=" * 60)

    import fasttext

    # ── 构造训练数据：中文、英文、法文各 500 句 ──
    train_data = []

    en_sentences = [
        "the quick brown fox jumps over the lazy dog",
        "machine learning is transforming the world",
        "natural language processing enables computers to understand text",
        "deep learning models require large amounts of training data",
        "the weather is nice today and I went for a walk",
        "she sells seashells by the seashore every morning",
        "artificial intelligence will change how we work and live",
        "the stock market reached a new high yesterday afternoon",
        "scientists discovered a new species of dinosaur in argentina",
        "the library has thousands of books on various subjects",
    ]
    zh_sentences = [
        "今天天气很好，我出去散步了",
        "机器学习正在改变世界",
        "自然语言处理让计算机能够理解文本",
        "深度学习模型需要大量训练数据",
        "人工智能将改变我们的工作和生活方式",
        "股市昨天下午创下新高",
        "科学家在阿根廷发现了新的恐龙物种",
        "图书馆里有数千本各类书籍",
        "这家餐厅的食物非常美味可口",
        "他每天早上六点起床去跑步锻炼身体",
    ]
    fr_sentences = [
        "le renard brun rapide saute par dessus le chien paresseux",
        "l apprentissage automatique transforme le monde entier",
        "le traitement du langage naturel permet aux ordinateurs de comprendre",
        "les modèles d apprentissage profond nécessitent beaucoup de données",
        "l intelligence artificielle va changer notre façon de travailler",
        "la bourse a atteint un nouveau record hier après midi",
        "des scientifiques ont découvert une nouvelle espèce de dinosaure",
        "la bibliothèque possède des milliers de livres sur divers sujets",
        "ce restaurant propose une cuisine délicieuse et raffinée",
        "il se lève à six heures chaque matin pour faire du jogging",
    ]

    # 每种语言重复扩充到 300 句
    for i in range(300):
        train_data.append(f"__label__en {en_sentences[i % len(en_sentences)]}")
        train_data.append(f"__label__zh {zh_sentences[i % len(zh_sentences)]}")
        train_data.append(f"__label__fr {fr_sentences[i % len(fr_sentences)]}")

    random.shuffle(train_data)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("\n".join(train_data))
        train_file = f.name

    print(f"\n训练数据：{len(train_data)} 条（英/中/法各 300 句）")

    # ── 训练 ──
    print("\n训练中...")
    t0 = time.perf_counter()
    model = fasttext.train_supervised(
        input=train_file,
        epoch=10,
        lr=0.5,
        wordNgrams=2,
        dim=16,
        minCount=1,
        verbose=0,
    )
    train_time = time.perf_counter() - t0
    print(f"训练耗时：{train_time:.3f} 秒（{len(train_data)} 条数据，10 epoch）")

    # ── 推理速度测量 ──
    test_texts = [
        "this is a test sentence in english language",
        "这是一句中文测试句子",
        "voici une phrase de test en français",
        "hello world how are you doing today",
        "我喜欢吃火锅和烤鸭",
        "bonjour comment allez vous aujourd hui",
    ] * 1000  # 6000 条

    print(f"\n推理速度测试（{len(test_texts)} 条文本）...")
    t0 = time.perf_counter()
    labels, probs = model.predict(test_texts)  # 批量预测接口
    infer_time = time.perf_counter() - t0

    throughput = len(test_texts) / infer_time
    print(f"推理耗时：{infer_time:.3f} 秒")
    print(f"吞吐量：{throughput:,.0f} 条/秒")
    print(f"单条延迟：{infer_time / len(test_texts) * 1000:.4f} 毫秒")

    # ── 准确率 ──
    test_cases = [
        ("this is english text", "__label__en"),
        ("这是中文文本", "__label__zh"),
        ("ceci est du texte français", "__label__fr"),
        ("machine learning deep learning", "__label__en"),
        ("自然语言处理技术", "__label__zh"),
        ("apprentissage automatique", "__label__fr"),
    ]
    print("\n分类结果：")
    correct = 0
    for text, expected in test_cases:
        labels_single, probs_single = model.predict([text])
        pred = labels_single[0][0]
        confidence = probs_single[0]
        ok = "✓" if pred == expected else "✗"
        print(f"  {ok} '{text}' → {pred} ({confidence[0]:.2f})")
        if pred == expected:
            correct += 1
    print(f"\n准确率：{correct}/{len(test_cases)}")

    # ── 关键数字对比 ──
    print("\n" + "-" * 40)
    print("速度对比（量级估算）：")
    print(f"  fastText（本次）：{throughput:,.0f} 条/秒（CPU，单线程）")
    print(f"  BERT-base：      ~50–200 条/秒（GPU）")
    print(f"  差距：           约 {throughput / 100:.0f}x ~ {throughput / 50:.0f}x")
    print("\n→ 处理 Common Crawl 的 300 亿文档：")
    print(f"  fastText：约 {300e9 / throughput / 3600:.0f} 小时（单线程）")
    print(f"  BERT：    约 {300e9 / 100 / 3600:.0f} 小时（单 GPU）")

    os.unlink(train_file)
    print("\n✓ Demo 2 完成")


if __name__ == "__main__":
    demo1_manual()
    demo2_real_fasttext()
