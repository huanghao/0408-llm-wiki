"""
KenLM 困惑度过滤 Demo
=====================
演示 ccNet 的核心质量过滤思路：用语言模型困惑度区分高质量和低质量文本。

因为 KenLM 需要预训练模型文件（.arpa.bin），这个 demo 用纯 Python 实现
一个简化的 n-gram 语言模型，逻辑和 KenLM 完全一样，只是规模小。

运行：python tools/kenlm_perplexity_demo.py
"""

import math
import re
from collections import defaultdict, Counter

# ─────────────────────────────────────────────
# 1. 简化版 n-gram 语言模型
# ─────────────────────────────────────────────

class NgramLM:
    """
    简化版 n-gram 语言模型，实现和 KenLM 相同的核心逻辑：
    - 统计训练语料里 n-gram 的频率
    - 用 add-k 平滑处理未见过的 n-gram
    - 返回 log₁₀ 概率（和 kenlm.model.score() 一致）
    """

    def __init__(self, n=3, k=0.1):
        self.n = n        # n-gram 阶数（ccNet 用 5-gram）
        self.k = k        # 平滑参数（add-k smoothing）
        self.ngram_counts = defaultdict(Counter)  # {context: {word: count}}
        self.vocab = set()

    def normalize(self, text):
        """和 ccNet 一样的文本归一化"""
        text = text.lower()
        text = re.sub(r'\d', '0', text)           # 数字替换为 0
        text = re.sub(r'[^\w\s]', '', text)       # 去掉标点
        return text.split()

    def train(self, texts):
        """在文本列表上训练"""
        for text in texts:
            tokens = ['<s>'] * (self.n - 1) + self.normalize(text) + ['</s>']
            self.vocab.update(tokens)
            for i in range(self.n - 1, len(tokens)):
                context = tuple(tokens[i - self.n + 1:i])
                word = tokens[i]
                self.ngram_counts[context][word] += 1
        print(f"训练完成：词表 {len(self.vocab)} 个词，"
              f"{sum(len(v) for v in self.ngram_counts.values())} 个唯一 n-gram")

    def log_prob(self, word, context):
        """
        计算 log₁₀ P(word | context)
        - 词表外的词（OOV）：给一个很低的固定概率（模拟 KenLM 的 OOV 惩罚）
        - 词表内但 n-gram 未见过：add-k 平滑
        """
        # OOV 惩罚：词表外的词概率极低
        if word not in self.vocab:
            return math.log10(1e-6)

        ctx_counts = self.ngram_counts[context]
        count = ctx_counts.get(word, 0) + self.k
        total = sum(ctx_counts.values()) + self.k * len(self.vocab)
        if total == 0:
            return math.log10(1.0 / len(self.vocab))
        return math.log10(count / total)

    def score(self, text):
        """
        计算一段文本的 log₁₀ 概率之和（对应 kenlm.model.score()）
        返回值是负数，越接近 0 说明文本越"像训练语料"
        """
        tokens = ['<s>'] * (self.n - 1) + self.normalize(text) + ['</s>']
        total_log_prob = 0.0
        for i in range(self.n - 1, len(tokens)):
            context = tuple(tokens[i - self.n + 1:i])
            word = tokens[i]
            total_log_prob += self.log_prob(word, context)
        return total_log_prob

    def perplexity(self, text):
        """
        计算文本的困惑度，和 ccNet 公式一致：
        perplexity = 10 ^ (-log_score / length)
        """
        tokens = self.normalize(text)
        if not tokens:
            return float('inf')
        length = len(tokens) + 1  # +1 for </s>，和 ccNet 一致
        log_score = self.score(text)
        return 10.0 ** (-log_score / length)


# ─────────────────────────────────────────────
# 2. 准备训练数据（模拟"高质量语料"，比如 Wikipedia）
# ─────────────────────────────────────────────

# 模拟 Wikipedia 风格的高质量文本（足够大的语料让词表有意义）
TRAIN_CORPUS = [
    "机器学习是人工智能的一个分支 通过算法让计算机从数据中自动学习规律",
    "深度学习使用多层神经网络来学习数据的层次化表示",
    "自然语言处理是计算机科学与语言学的交叉领域",
    "卷积神经网络在图像识别任务上取得了突破性进展",
    "transformer 模型通过注意力机制捕捉序列中的长距离依赖关系",
    "预训练语言模型在大规模文本上训练后可以迁移到各种下游任务",
    "数据预处理是机器学习流程中至关重要的一个环节",
    "模型评估通常使用准确率 精确率 召回率等指标",
    "过拟合是指模型在训练集上表现好但在测试集上表现差的现象",
    "正则化技术如 dropout 和 weight decay 可以缓解过拟合问题",
    "梯度下降是训练神经网络最常用的优化算法",
    "批归一化可以加速训练并提高模型的稳定性",
    "词嵌入将词语映射到低维稠密向量空间中",
    "注意力机制允许模型在处理序列时动态关注不同位置",
    "迁移学习利用在一个任务上学到的知识来改善另一个任务的性能",
    "数据增强通过对训练数据进行变换来增加数据多样性",
    "集成学习通过组合多个模型来提高预测性能",
    "强化学习通过与环境交互来学习最优策略",
    "生成对抗网络由生成器和判别器两个网络组成",
    "语义分割是将图像中每个像素分配到对应类别的任务",
    "神经网络的训练需要大量标注数据和计算资源",
    "模型的泛化能力决定了它在未见过数据上的表现",
    "特征工程是传统机器学习中非常重要的步骤",
    "随机森林是一种基于决策树的集成学习方法",
    "支持向量机通过找到最优超平面来进行分类",
    "聚类算法可以在没有标签的情况下发现数据结构",
    "降维技术如 PCA 可以减少数据的维度同时保留主要信息",
    "交叉验证是评估模型性能的常用方法",
    "超参数调优对模型最终性能有重要影响",
    "模型压缩技术可以减小模型大小同时保持较好性能",
    "知识蒸馏将大模型的知识迁移到小模型中",
    "联邦学习允许多个参与方在不共享原始数据的情况下训练模型",
    "对抗样本是经过微小扰动后能让模型产生错误预测的输入",
    "可解释性是评估机器学习模型的重要维度之一",
    "偏差和方差是影响模型误差的两个主要来源",
    "激活函数引入非线性使神经网络能够学习复杂模式",
    "反向传播算法通过链式法则计算各层的梯度",
    "学习率是控制参数更新步长的重要超参数",
    "批量大小影响训练的稳定性和收敛速度",
    "dropout 在训练时随机丢弃神经元以防止过拟合",
] * 8  # 重复 8 次增加样本量

# ─────────────────────────────────────────────
# 3. 训练语言模型
# ─────────────────────────────────────────────

print("=" * 60)
print("KenLM 困惑度过滤 Demo")
print("（用简化版 n-gram LM 演示，逻辑和 ccNet 完全一致）")
print("=" * 60)
print()

lm = NgramLM(n=3, k=0.01)
print(f"在 {len(TRAIN_CORPUS)} 条百科风格文本上训练 {lm.n}-gram 语言模型...")
lm.train(TRAIN_CORPUS)
print()

# ─────────────────────────────────────────────
# 4. 对不同质量的文本计算困惑度
# ─────────────────────────────────────────────

TEST_TEXTS = [
    # 高质量：和训练语料风格相近
    ("高质量-百科", "深度学习模型通过大规模数据训练来学习特征表示"),
    ("高质量-百科", "注意力机制是 transformer 模型的核心组成部分"),
    ("高质量-百科", "梯度下降算法通过计算损失函数的梯度来更新模型参数"),

    # 中等质量：口语化，但语法正确
    ("中等-口语", "这个模型效果还不错，跑起来也挺快的"),
    ("中等-口语", "今天学了一下机器学习，感觉挺有意思的"),

    # 低质量：boilerplate 模板文字
    ("低质量-版权", "版权所有 保留所有权利 未经授权禁止转载"),
    ("低质量-导航", "首页 关于我们 联系方式 隐私政策 使用条款"),
    ("低质量-广告", "点击这里订阅我们的newsletter 获取最新优惠信息"),

    # 极低质量：乱码/随机字符
    ("极低-乱码", "asdf jkl qwer zxcv 1234 5678 abcd efgh"),
    ("极低-重复", "的的的的的的的的的的的的的的的的的的的的的"),
]

print(f"{'类型':<12} {'困惑度':>10}  文本")
print("-" * 70)

results = []
for label, text in TEST_TEXTS:
    pp = lm.perplexity(text)
    results.append((label, pp, text))
    # 困惑度太大时显示为 ∞
    pp_str = f"{pp:>10.1f}" if pp < 1e6 else f"{'∞':>10}"
    print(f"{label:<12} {pp_str}  {text[:35]}...")

# ─────────────────────────────────────────────
# 5. 模拟 ccNet 的分段过滤
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("模拟 ccNet 分段策略（head/middle/tail）")
print("=" * 60)

# 计算百分位阈值（ccNet 用第 30 和第 60 百分位）
import statistics
valid_pps = [pp for _, pp, _ in results if pp < 1e6]
valid_pps_sorted = sorted(valid_pps)

if len(valid_pps_sorted) >= 3:
    p30_idx = int(len(valid_pps_sorted) * 0.3)
    p60_idx = int(len(valid_pps_sorted) * 0.6)
    threshold_30 = valid_pps_sorted[p30_idx]
    threshold_60 = valid_pps_sorted[p60_idx]
else:
    threshold_30 = 200
    threshold_60 = 500

print(f"\n第 30 百分位阈值（head/middle 边界）：{threshold_30:.1f}")
print(f"第 60 百分位阈值（middle/tail 边界）：{threshold_60:.1f}")
print()

for label, pp, text in results:
    if pp <= threshold_30:
        bucket = "HEAD  ✓ 保留（最高质量）"
    elif pp <= threshold_60:
        bucket = "MIDDLE  ~ 可选保留"
    else:
        bucket = "TAIL  ✗ 过滤掉"
    pp_str = f"{pp:.1f}" if pp < 1e6 else "∞"
    print(f"  [{bucket}]  pp={pp_str:>8}  {label}")

# ─────────────────────────────────────────────
# 6. 直觉演示：log probability 是什么
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("直觉演示：log probability 是什么")
print("=" * 60)

demo_texts = [
    "深度学习使用神经网络来学习数据",
    "asdf qwer zxcv 1234",
]
for text in demo_texts:
    tokens = lm.normalize(text)
    log_score = lm.score(text)
    length = len(tokens) + 1
    pp = lm.perplexity(text)
    prob_approx = 10 ** log_score  # 整段文本的概率（极小数）

    print(f"\n文本：'{text}'")
    print(f"  词数（length）：{length}")
    print(f"  log₁₀ P（整段）：{log_score:.2f}  →  P ≈ 10^{log_score:.0f}（极小数，用对数避免下溢）")
    print(f"  每词平均 log₁₀ P：{log_score/length:.2f}")
    print(f"  困惑度 = 10^(-{log_score/length:.2f}) = {pp:.1f}")

print()
print("结论：")
print("  - log probability 是负数，越接近 0 说明文本越像训练语料")
print("  - 困惑度是 log probability 的指数还原，值越低越好")
print("  - ccNet 用 Wikipedia 文本训练 LM，困惑度低 = 像 Wikipedia = 高质量")
