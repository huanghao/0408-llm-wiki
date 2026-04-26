"""
KenLM 困惑度过滤 Demo（使用真实 kenlm 库）
==========================================
演示 ccNet 的核心质量过滤思路：用语言模型困惑度区分高质量和低质量文本。

用 KenLM 的 lmplz 工具训练语言模型，再用 kenlm Python 库加载模型并打分。
这里不在 Python 里手写 n-gram 统计、平滑或 ARPA 文件。

依赖：
  pip install kenlm
  # lmplz 编译路径：/Users/huanghao/workspace/sources/kenlm/build/bin/lmplz
  # 编译方法：git clone https://github.com/kpu/kenlm && cd kenlm
  #           mkdir build && cd build && cmake .. && make -j

运行：python tools/kenlm_perplexity_demo.py
"""

import os
import shutil
import subprocess
import re

import kenlm

# 工作目录：训练语料和 ARPA 文件写到这里
DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "kenlm_demo"
)
os.makedirs(DATA_DIR, exist_ok=True)

# lmplz 编译路径，优先用这里；其次找 PATH
LMPLZ_PATH = "/Users/huanghao/workspace/sources/kenlm/build/bin/lmplz"

# ─────────────────────────────────────────────
# 1. 文本归一化 + 调用 KenLM 训练工具
# ─────────────────────────────────────────────


def normalize(text):
    """和 ccNet 一样的文本归一化"""
    text = text.lower()
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.split()

def require_lmplz():
    """返回可用的 lmplz 路径，找不到时报清晰错误。"""
    if os.path.isfile(LMPLZ_PATH):
        return LMPLZ_PATH
    found = shutil.which("lmplz")
    if found:
        return found
    raise RuntimeError(
        f"未找到 lmplz（试过 {LMPLZ_PATH} 和 PATH）。\n"
        "编译方法：\n"
        "  git clone https://github.com/kpu/kenlm && cd kenlm\n"
        "  mkdir build && cd build && cmake .. && make -j\n"
    )

def train_with_kenlm(corpus, work_dir, order=2):
    """
    用 KenLM lmplz 从训练语料生成 ARPA 模型。
    文件写入 work_dir（默认 data/kenlm_demo/）。
    """
    corpus_path = os.path.join(work_dir, "train.txt")
    arpa_path = os.path.join(work_dir, "model.arpa")

    with open(corpus_path, "w", encoding="utf-8") as f:
        for text in corpus:
            tokens = normalize(text)
            if tokens:
                f.write(" ".join(tokens) + "\n")

    subprocess.run(
        [
            require_lmplz(),
            "-o",
            str(order),
            "--text",
            corpus_path,
            "--arpa",
            arpa_path,
            "--discount_fallback",
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return arpa_path

# ─────────────────────────────────────────────
# 2. 训练语料（模拟 Wikipedia 风格文本）
# ─────────────────────────────────────────────

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
    "降维技术如 pca 可以减少数据的维度同时保留主要信息",
    "交叉验证是评估模型性能的常用方法",
    "超参数调优对模型最终性能有重要影响",
    "模型压缩技术可以减小模型大小同时保持较好性能",
] * 8

# ─────────────────────────────────────────────
# 3. 训练模型（写 ARPA → kenlm 加载）
# ─────────────────────────────────────────────

print("=" * 60)
print("KenLM 困惑度过滤 Demo（使用真实 kenlm 库）")
print("=" * 60)
print()

print(f"从 {len(TRAIN_CORPUS)} 条语料训练 bigram 模型...")
print(f"工作目录：{DATA_DIR}")
arpa_path = train_with_kenlm(TRAIN_CORPUS, DATA_DIR, order=3)

# 用 kenlm.Model 加载 KenLM 生成的 ARPA 文件
model = kenlm.Model(arpa_path)
print(f"模型加载完成，阶数：{model.order}-gram\n")

# ─────────────────────────────────────────────
# 4. 用 kenlm API 打分
# ─────────────────────────────────────────────


def score_text(text):
    """用 kenlm 计算 log P，返回 (log_score, length)"""
    tokens = normalize(text)
    if not tokens:
        return 0.0, 1
    # model.score() 接受空格分隔的字符串，bos/eos 控制是否加句首句尾符
    log_score = model.score(" ".join(tokens), bos=True, eos=True)
    length = len(tokens) + 1  # +1 for </s>，和 ccNet 一致
    return log_score, length


def perplexity(text):
    log_score, length = score_text(text)
    return 10.0 ** (-log_score / length)


# ─────────────────────────────────────────────
# 5. 逐词打分（展示 kenlm 的 full_scores API）
# ─────────────────────────────────────────────

print("── 逐词打分（kenlm.full_scores）──")
demo_sent = "深度学习使用神经网络"
tokens = normalize(demo_sent)
print(f"\n句子：「{demo_sent}」 → tokens: {tokens}")
print(f"{'词':<10} {'log₁₀ P':>10}  {'ngram_length':>12}")
print("-" * 40)
total_logp = 0.0
for (logp, ngram_len, oov), word in zip(
    model.full_scores(" ".join(tokens), bos=True, eos=True), tokens + ["</s>"]
):
    total_logp += logp
    oov_mark = " ← OOV" if oov else ""
    print(f"  {word:<8} {logp:>10.4f}  {ngram_len:>12}{oov_mark}")
print(f"  {'合计':<8} {total_logp:>10.4f}")
print(
    f"  perplexity = 10^({-total_logp:.2f} / {len(tokens) + 1}) = {perplexity(demo_sent):.1f}\n"
)

# ─────────────────────────────────────────────
# 6. 不同质量文本的困惑度对比
# ─────────────────────────────────────────────

TEST_TEXTS = [
    ("高质量-百科", "深度学习模型通过大规模数据训练来学习特征表示"),
    ("高质量-百科", "注意力机制是 transformer 模型的核心组成部分"),
    ("中等-口语", "这个模型效果还不错跑起来也挺快的"),
    ("低质量-版权", "版权所有 保留所有权利 未经授权禁止转载"),
    ("低质量-导航", "首页 关于我们 联系方式 隐私政策 使用条款"),
    ("极低-乱码", "asdf jkl qwer zxcv 0000 5678 efgh"),
    ("极低-重复", "的的的的的的的的的的的的的的的的的的的的的"),
]

print("── 困惑度对比 ──\n")
print(f"{'类型':<12} {'困惑度':>10}  文本")
print("-" * 65)

results = []
for label, text in TEST_TEXTS:
    pp = perplexity(text)
    results.append((label, pp, text))
    pp_str = f"{pp:>10.1f}" if pp < 1e6 else f"{'∞':>10}"
    print(f"{label:<12} {pp_str}  {text[:35]}")

# ─────────────────────────────────────────────
# 7. 模拟 ccNet 分段策略
# ─────────────────────────────────────────────

print()
print("── ccNet 分段策略（head/middle/tail）──\n")

valid_pps = sorted(pp for _, pp, _ in results if pp < 1e6)
p30 = valid_pps[int(len(valid_pps) * 0.3)]
p60 = valid_pps[int(len(valid_pps) * 0.6)]
print(f"第 30 百分位（head/middle 边界）：{p30:.1f}")
print(f"第 60 百分位（middle/tail 边界）：{p60:.1f}\n")

for label, pp, text in results:
    if pp <= p30:
        bucket = "HEAD   ✓"
    elif pp <= p60:
        bucket = "MIDDLE ~"
    else:
        bucket = "TAIL   ✗"
    pp_str = f"{pp:.1f}" if pp < 1e6 else "∞"
    print(f"  [{bucket}]  pp={pp_str:>8}  {label}")
