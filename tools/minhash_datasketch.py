"""
MinHash 去重 - 使用 datasketch 库
和 minhash_from_scratch.py 用相同的文档，对比结果。

安装：pip install datasketch
运行：python tools/minhash_datasketch.py
"""

from datasketch import MinHash, MinHashLSH


def get_ngrams(text: str, n: int = 5) -> set[bytes]:
    """词级 n-gram，返回 bytes（datasketch 要求 hashable + bytes-like）"""
    import re
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return {' '.join(words[i:i+n]).encode('utf-8') for i in range(len(words) - n + 1)}


def make_minhash(ngrams: set[bytes], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for gram in ngrams:
        m.update(gram)
    return m


docs = {
    "doc_A": "The quick brown fox jumps over the lazy dog near the river bank",
    "doc_B": "The quick brown fox jumps over the lazy dog near the river bank today",
    "doc_C": "Machine learning models require large amounts of training data to perform well",
    "doc_D": "Deep learning models need massive datasets for training to achieve good results",
    "doc_E": "Completely unrelated content about cooking recipes and kitchen techniques",
}

num_perm = 128
threshold = 0.5  # LSH 的相似度阈值，低于此值不会被分到同一个桶

ngrams = {doc_id: get_ngrams(text) for doc_id, text in docs.items()}
minhashes = {doc_id: make_minhash(ng, num_perm) for doc_id, ng in ngrams.items()}

# ── 两两估计 Jaccard ──
pairs = [("doc_A", "doc_B"), ("doc_C", "doc_D"), ("doc_A", "doc_C"), ("doc_A", "doc_E")]
print("── datasketch 估计的 Jaccard ──")
for x, y in pairs:
    est = minhashes[x].jaccard(minhashes[y])
    print(f"  {x} vs {y}: {est:.3f}")

# ── LSH 查询 ──
lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
for doc_id, m in minhashes.items():
    lsh.insert(doc_id, m)

print(f"\n── LSH 查询（threshold={threshold}）──")
for doc_id, m in minhashes.items():
    results = lsh.query(m)
    neighbors = [r for r in results if r != doc_id]
    if neighbors:
        print(f"  {doc_id} 的近邻: {neighbors}")
