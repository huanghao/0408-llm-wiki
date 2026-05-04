"""
Perplexity 计算过程逐步演示
============================
目标：看清楚 perplexity 到底在计算什么，每一步数字从哪里来。

分三个阶段：
  1. 训练：从语料统计 n-gram 频率，建一张"查找表"
  2. 打分：给一段新文本，逐词查表得到条件概率，计算 log P
  3. 困惑度：把 log P 转换成 perplexity，理解这个数字的含义

运行：python tools/perplexity_explained.py
"""

import math
from collections import defaultdict, Counter

# ══════════════════════════════════════════════════════════
# 阶段 1：训练——从语料建 n-gram 频率表
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("阶段 1：训练——统计 n-gram 频率")
print("=" * 60)

# 用 bigram（n=2）演示，方便手算验证
# bigram 的意思：看前 1 个词预测下一个词
N = 2

CORPUS = [
    "机器 学习 很 有 趣",
    "机器 学习 很 有 用",
    "机器 学习 需要 数据",
    "深度 学习 很 强大",
    "深度 学习 需要 算力",
    "学习 需要 耐心",
]

# 统计所有 bigram 的出现次数
# ngram_counts[context][word] = 出现次数
ngram_counts = defaultdict(Counter)
vocab = set()

for sentence in CORPUS:
    tokens = ["<s>"] + sentence.split() + ["</s>"]
    vocab.update(tokens)
    for i in range(1, len(tokens)):
        context = (tokens[i - 1],)   # bigram：context 是前 1 个词
        word    = tokens[i]
        ngram_counts[context][word] += 1

print(f"\n语料：{len(CORPUS)} 句话，词表 {len(vocab)} 个词")
print("\n「学习」后面跟了哪些词（bigram 频率表）：")
ctx = ("学习",)
for word, cnt in sorted(ngram_counts[ctx].items(), key=lambda x: -x[1]):
    total = sum(ngram_counts[ctx].values())
    prob  = cnt / total
    print(f"  P({word!r:8} | '学习') = {cnt}/{total} = {prob:.3f}   log₁₀={math.log10(prob):.3f}")

print("\n「机器」后面跟了哪些词：")
ctx2 = ("机器",)
for word, cnt in sorted(ngram_counts[ctx2].items(), key=lambda x: -x[1]):
    total = sum(ngram_counts[ctx2].values())
    prob  = cnt / total
    print(f"  P({word!r:8} | '机器') = {cnt}/{total} = {prob:.3f}   log₁₀={math.log10(prob):.3f}")

# ══════════════════════════════════════════════════════════
# 阶段 2：打分——逐词查表，计算一段文本的 log P
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("阶段 2：打分——逐词查条件概率，累加 log P")
print("=" * 60)

OOV_LOG_PROB = math.log10(1e-5)   # 没见过的词给一个很低的概率
SMOOTH_K     = 0.1                 # add-k 平滑

def log_prob(word, context):
    """查表得到 log₁₀ P(word | context)，带平滑"""
    if word not in vocab:
        return OOV_LOG_PROB
    ctx_counts = ngram_counts[context]
    count = ctx_counts.get(word, 0) + SMOOTH_K
    total = sum(ctx_counts.values()) + SMOOTH_K * len(vocab)
    return math.log10(count / total)

def score_with_trace(sentence):
    """计算 log P，并打印每一步的细节"""
    tokens     = ["<s>"] + sentence.split() + ["</s>"]
    total_logp = 0.0
    print(f"\n  句子：「{sentence}」")
    print(f"  {'词':<8} {'context':<10} {'log₁₀ P':>10}  说明")
    print(f"  {'-'*50}")
    for i in range(1, len(tokens)):
        ctx  = (tokens[i - 1],)
        word = tokens[i]
        lp   = log_prob(word, ctx)
        total_logp += lp
        ctx_cnt  = sum(ngram_counts[ctx].values())
        word_cnt = ngram_counts[ctx].get(word, 0)
        note = f"{word_cnt}/{ctx_cnt} 次" if ctx_cnt > 0 and word in vocab else "平滑/OOV"
        print(f"  {word:<8} | {tokens[i-1]:<8} | {lp:>10.3f}  ({note})")
    print(f"  {'─'*50}")
    print(f"  log₁₀ P(整句) = {total_logp:.3f}   →   P ≈ 10^{total_logp:.1f}")
    return total_logp, len(tokens) - 1   # 返回 log P 和词数（不含 <s>）

# 对三种不同质量的句子打分
test_cases = [
    ("机器 学习 很 有 趣",   "训练语料里见过"),
    ("深度 学习 需要 数据",  "部分 bigram 见过，部分没见过"),
    ("asdf qwer zxcv 1234", "完全没见过（乱码）"),
]

score_results = []
for sentence, note in test_cases:
    print(f"\n【{note}】")
    logp, length = score_with_trace(sentence)
    score_results.append((sentence, logp, length, note))

# ══════════════════════════════════════════════════════════
# 阶段 3：困惑度——把 log P 转成一个直觉上可读的数字
# ══════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("阶段 3：困惑度——log P 转换成「平均每步多少种选择」")
print("=" * 60)

print(f"\n{'句子':<25} {'log P':>8}  {'词数':>4}  {'每词平均 log P':>14}  {'perplexity':>12}  说明")
print("-" * 90)

for sentence, logp, length, note in score_results:
    avg_logp = logp / length
    pp       = 10 ** (-avg_logp)
    print(f"  {sentence:<23} {logp:>8.2f}  {length:>4}  {avg_logp:>14.3f}  {pp:>12.1f}  {note}")

print("""
推导过程（以第一句为例）：
  log P(整句) / 词数  =  每个词的平均 log₁₀ P（负数）
  取负号              =  绝对值（正数）
  10^(该值)           =  perplexity

直觉：perplexity = X  意味着模型平均每步面临 X 种等可能的选择
  perplexity ≈ 10   → 模型很确定，文本符合训练分布
  perplexity ≈ 1000 → 模型很困惑，文本不像训练语料
""")

# ══════════════════════════════════════════════════════════
# 附：为什么不能直接用概率，必须用对数
# ══════════════════════════════════════════════════════════

print("=" * 60)
print("附：浮点下溢演示——为什么必须用对数")
print("=" * 60)

print("\n假设每个词的概率都是 0.1（10 种等可能选择）：")
for n_words in [5, 10, 20, 50, 100, 300]:
    prob = 0.1 ** n_words
    logp = n_words * math.log10(0.1)
    print(f"  {n_words:>3} 个词：P = 0.1^{n_words} = {prob:.2e}  "
          f"{'← 下溢为 0.0！' if prob == 0.0 else ''}  log P = {logp:.1f}")

print("""
→ 词数超过约 323 时，float64 就下溢到 0.0，无法继续计算。
→ 对数把乘法变成加法，把 10^-300 变成 -300，数值始终稳定。
→ 这就是为什么语言模型 API 返回 log probability 而不是 probability。
""")
