"""
MinHash 去重 - 从零实现（不依赖第三方库）
演示完整流程：特征化 → MinHash 签名 → LSH 分桶 → 候选对验证

运行：python tools/minhash_from_scratch.py
"""

import hashlib
import re
from collections import defaultdict


# ── 1. 特征化：文档 → n-gram 集合 ──────────────────────────────

def get_ngrams(text: str, n: int = 5) -> set[str]:
    """把文本转成词级 n-gram 集合，预处理：lowercase + 去标点"""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    return {' '.join(words[i:i+n]) for i in range(len(words) - n + 1)}


# ── 2. MinHash 签名 ────────────────────────────────────────────

def minhash_signature(ngrams: set[str], k: int = 128) -> list[int]:
    """
    用 k 个不同 seed 的哈希函数计算签名向量。
    用 hashlib.md5 + seed 模拟 k 个独立哈希函数（演示用，生产环境用 mmh3）。
    """
    sig = []
    for seed in range(k):
        min_val = float('inf')
        for gram in ngrams:
            h = int(hashlib.md5(f"{seed}:{gram}".encode()).hexdigest(), 16)
            min_val = min(min_val, h)
        sig.append(min_val)
    return sig


def jaccard_from_signatures(sig_a: list[int], sig_b: list[int]) -> float:
    """从签名向量估计 Jaccard 相似度"""
    matches = sum(a == b for a, b in zip(sig_a, sig_b))
    return matches / len(sig_a)


def true_jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ── 3. LSH 分桶 ────────────────────────────────────────────────

def lsh_candidates(signatures: dict[str, list[int]], b: int = 16, r: int = 8) -> set[tuple]:
    """
    把签名向量切成 b 个 band，每个 band 有 r 行（b * r = k）。
    同一个 band 里落入同一个桶的文档对成为候选近重复对。
    """
    candidates = set()
    for band_idx in range(b):
        buckets: dict[tuple, list] = defaultdict(list)
        start, end = band_idx * r, (band_idx + 1) * r
        for doc_id, sig in signatures.items():
            band_key = tuple(sig[start:end])
            buckets[band_key].append(doc_id)
        for docs in buckets.values():
            if len(docs) > 1:
                for i in range(len(docs)):
                    for j in range(i + 1, len(docs)):
                        candidates.add((min(docs[i], docs[j]), max(docs[i], docs[j])))
    return candidates


# ── 4. 演示 ────────────────────────────────────────────────────

docs = {
    "doc_A": "The quick brown fox jumps over the lazy dog near the river bank",
    "doc_B": "The quick brown fox jumps over the lazy dog near the river bank today",  # A 的轻微改写
    "doc_C": "Machine learning models require large amounts of training data to perform well",
    "doc_D": "Deep learning models need massive datasets for training to achieve good results",  # C 的语义改写
    "doc_E": "Completely unrelated content about cooking recipes and kitchen techniques",
}

ngrams = {doc_id: get_ngrams(text) for doc_id, text in docs.items()}

pairs = [("doc_A", "doc_B"), ("doc_C", "doc_D"), ("doc_A", "doc_C"), ("doc_A", "doc_E")]

print("── 真实 Jaccard 相似度 ──")
for x, y in pairs:
    print(f"  {x} vs {y}: {true_jaccard(ngrams[x], ngrams[y]):.3f}")

k, b, r = 128, 16, 8
print(f"\n── MinHash 签名（k={k}）计算中... ──")
signatures = {doc_id: minhash_signature(ng, k=k) for doc_id, ng in ngrams.items()}

print("\n── MinHash 估计的 Jaccard ──")
for x, y in pairs:
    est = jaccard_from_signatures(signatures[x], signatures[y])
    print(f"  {x} vs {y}: {est:.3f}")

threshold = (1 / b) ** (1 / r)
candidates = lsh_candidates(signatures, b=b, r=r)
print(f"\n── LSH 候选对（b={b}, r={r}，转折点阈值≈{threshold:.2f}）──")
if candidates:
    for x, y in sorted(candidates):
        est = jaccard_from_signatures(signatures[x], signatures[y])
        print(f"  {x} vs {y}  估计 Jaccard={est:.3f}")
else:
    print("  （无候选对，相似度均低于阈值）")
