"""
MinHash 去重 - 使用 datasketch 库跑真实 Markdown 文档。

安装：
  pip install datasketch

运行：
  python tools/minhash_datasketch.py wiki
  python tools/minhash_datasketch.py . --threshold 0.4 --top-k 15
"""

from __future__ import annotations

import argparse
import heapq
import re
import statistics
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from datasketch import MinHash, MinHashLSH


@dataclass
class DocumentStats:
    path: Path
    word_count: int
    char_count: int
    ngram_count: int


class ProgressBar:
    def __init__(self, total: int, label: str, width: int = 28) -> None:
        self.total = max(total, 1)
        self.label = label
        self.width = width
        self.current = 0
        self.start_time = time.perf_counter()
        self.last_render = 0.0

    def update(self, step: int = 1) -> None:
        self.current += step
        now = time.perf_counter()
        if self.current >= self.total or now - self.last_render >= 0.1:
            self.last_render = now
            self.render()

    def render(self) -> None:
        ratio = min(self.current / self.total, 1.0)
        filled = int(self.width * ratio)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.perf_counter() - self.start_time
        rate = self.current / elapsed if elapsed > 0 else 0.0
        sys.stderr.write(
            f"\r{self.label:<18} [{bar}] {self.current:>7}/{self.total:<7} "
            f"{ratio * 100:>6.2f}%  {rate:>8.1f}/s"
        )
        if self.current >= self.total:
            sys.stderr.write("\n")
        sys.stderr.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="对目录中的 Markdown 文档运行 MinHash + LSH。")
    parser.add_argument("directory", help="要扫描的目录，会递归读取其中所有 .md 文件")
    parser.add_argument("--ngram-size", type=int, default=5, help="词级 n-gram 大小，默认 5")
    parser.add_argument("--num-perm", type=int, default=128, help="MinHash 签名长度，默认 128")
    parser.add_argument("--threshold", type=float, default=0.5, help="LSH 相似度阈值，默认 0.5")
    parser.add_argument("--top-k", type=int, default=10, help="输出最相似文档对的数量，默认 10")
    parser.add_argument(
        "--report-file",
        default="minhash_report.md",
        help="报告输出文件路径，默认 minhash_report.md",
    )
    parser.add_argument(
        "--report-top-k",
        type=int,
        default=200,
        help="报告中保留多少组非完全重复的高相似文档，默认 200",
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.001,
        help="只保留不低于该相似度的非完全重复文档对，默认 0.001",
    )
    return parser.parse_args()


def normalize_words(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def get_ngrams(text: str, n: int = 5) -> set[bytes]:
    """词级 n-gram，返回 bytes（datasketch 适合直接 update bytes）。"""
    words = normalize_words(text)
    if len(words) < n:
        if not words:
            return set()
        return {" ".join(words).encode("utf-8")}
    return {" ".join(words[i : i + n]).encode("utf-8") for i in range(len(words) - n + 1)}


def make_minhash(ngrams: set[bytes], num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for gram in ngrams:
        m.update(gram)
    return m


def collect_markdown_files(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.rglob("*.md")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(directory).parts)
    )


def read_documents(paths: list[Path], ngram_size: int) -> tuple[list[DocumentStats], dict[str, set[bytes]]]:
    docs: list[DocumentStats] = []
    ngrams_by_doc: dict[str, set[bytes]] = {}
    progress = ProgressBar(len(paths), "读取 Markdown")

    for path in paths:
        text = path.read_text(encoding="utf-8")
        words = normalize_words(text)
        ngrams = get_ngrams(text, n=ngram_size)
        doc_id = str(path)
        docs.append(
            DocumentStats(
                path=path,
                word_count=len(words),
                char_count=len(text),
                ngram_count=len(ngrams),
            )
        )
        ngrams_by_doc[doc_id] = ngrams
        progress.update()

    return docs, ngrams_by_doc


def summarize(values: list[int | float]) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0
    return min(values), statistics.mean(values), max(values)


def print_doc_stats(docs: list[DocumentStats], root: Path) -> None:
    word_stats = summarize([doc.word_count for doc in docs])
    char_stats = summarize([doc.char_count for doc in docs])
    ngram_stats = summarize([doc.ngram_count for doc in docs])

    print("── 文档统计 ──")
    print(f"目录: {root}")
    print(f"Markdown 文件数: {len(docs)}")
    print(f"词数  min/avg/max: {word_stats[0]:.0f} / {word_stats[1]:.1f} / {word_stats[2]:.0f}")
    print(f"字符数 min/avg/max: {char_stats[0]:.0f} / {char_stats[1]:.1f} / {char_stats[2]:.0f}")
    print(f"n-gram 数 min/avg/max: {ngram_stats[0]:.0f} / {ngram_stats[1]:.1f} / {ngram_stats[2]:.0f}")

    shortest = min(docs, key=lambda doc: doc.word_count)
    longest = max(docs, key=lambda doc: doc.word_count)
    print(f"最短文档: {shortest.path} ({shortest.word_count} 词)")
    print(f"最长文档: {longest.path} ({longest.word_count} 词)")


def build_minhashes(ngrams_by_doc: dict[str, set[bytes]], num_perm: int) -> dict[str, MinHash]:
    progress = ProgressBar(len(ngrams_by_doc), "计算 MinHash")
    minhashes: dict[str, MinHash] = {}
    for doc_id, ngrams in ngrams_by_doc.items():
        minhashes[doc_id] = make_minhash(ngrams, num_perm=num_perm)
        progress.update()
    return minhashes


def analyze_pairs(
    minhashes: dict[str, MinHash],
    console_top_k: int,
    report_top_k: int,
    min_score: float,
) -> tuple[list[tuple[float, str, str]], list[tuple[float, str, str]], list[tuple[float, str, str]], int]:
    all_doc_ids = sorted(minhashes)
    total_pairs = len(all_doc_ids) * (len(all_doc_ids) - 1) // 2
    progress = ProgressBar(total_pairs, "两两相似度")
    non_exact_heap: list[tuple[float, str, str]] = []
    exact_pairs: list[tuple[float, str, str]] = []

    for left, right in combinations(all_doc_ids, 2):
        score = minhashes[left].jaccard(minhashes[right])
        if score >= 0.999999:
            exact_pairs.append((score, left, right))
        elif score >= min_score:
            item = (score, left, right)
            if len(non_exact_heap) < report_top_k:
                heapq.heappush(non_exact_heap, item)
            elif item > non_exact_heap[0]:
                heapq.heapreplace(non_exact_heap, item)
        progress.update()

    non_exact_pairs = sorted(non_exact_heap, reverse=True)
    return (
        non_exact_pairs[:console_top_k],
        non_exact_pairs,
        sorted(exact_pairs, reverse=True),
        total_pairs,
    )


def write_report(
    report_path: Path,
    root: Path,
    docs: list[DocumentStats],
    args: argparse.Namespace,
    timings: dict[str, float],
    total_pairs: int,
    non_exact_pairs: list[tuple[float, str, str]],
    exact_pairs: list[tuple[float, str, str]],
    neighbor_map: dict[str, list[str]],
) -> None:
    word_stats = summarize([doc.word_count for doc in docs])
    char_stats = summarize([doc.char_count for doc in docs])
    ngram_stats = summarize([doc.ngram_count for doc in docs])
    shortest = min(docs, key=lambda doc: doc.word_count)
    longest = max(docs, key=lambda doc: doc.word_count)

    lines = [
        "# MinHash Report",
        "",
        f"- 目录: `{root}`",
        f"- Markdown 文件数: `{len(docs)}`",
        f"- 文档对总数: `{total_pairs}`",
        f"- 生成时间: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
        "## 参数",
        "",
        f"- n-gram size: `{args.ngram_size}`",
        f"- num_perm: `{args.num_perm}`",
        f"- LSH threshold: `{args.threshold}`",
        f"- report top-k: `{args.report_top_k}`",
        f"- min score: `{args.min_score}`",
        "",
        "## 文档统计",
        "",
        f"- 词数 min/avg/max: `{word_stats[0]:.0f} / {word_stats[1]:.1f} / {word_stats[2]:.0f}`",
        f"- 字符数 min/avg/max: `{char_stats[0]:.0f} / {char_stats[1]:.1f} / {char_stats[2]:.0f}`",
        f"- n-gram 数 min/avg/max: `{ngram_stats[0]:.0f} / {ngram_stats[1]:.1f} / {ngram_stats[2]:.0f}`",
        f"- 最短文档: `{shortest.path.relative_to(root)}` ({shortest.word_count} 词)",
        f"- 最长文档: `{longest.path.relative_to(root)}` ({longest.word_count} 词)",
        "",
        "## 性能统计",
        "",
        f"- 扫描文件耗时: `{timings['scan']:.4f}s`",
        f"- 读取并切分文本耗时: `{timings['read']:.4f}s`",
        f"- MinHash 计算耗时: `{timings['minhash']:.4f}s`",
        f"- LSH 建索引耗时: `{timings['lsh_build']:.4f}s`",
        f"- LSH 查询耗时: `{timings['query']:.4f}s`",
        f"- 两两相似度排序耗时: `{timings['pair']:.4f}s`",
        "",
        "## 高相似但不完全相同的文档对",
        "",
        "| score | left | right |",
        "| --- | --- | --- |",
    ]

    if non_exact_pairs:
        for score, left, right in non_exact_pairs:
            lines.append(
                f"| {score:.4f} | `{Path(left).relative_to(root)}` | `{Path(right).relative_to(root)}` |"
            )
    else:
        lines.append("| - | 没有找到非完全重复的相似文档对 | - |")

    lines.extend(
        [
            "",
            "## 完全重复文档对",
            "",
            "| score | left | right |",
            "| --- | --- | --- |",
        ]
    )
    if exact_pairs:
        for score, left, right in exact_pairs:
            lines.append(
                f"| {score:.4f} | `{Path(left).relative_to(root)}` | `{Path(right).relative_to(root)}` |"
            )
    else:
        lines.append("| - | 没有完全重复文档对 | - |")

    lines.extend(
        [
            "",
            f"## LSH 近邻结果（threshold={args.threshold}）",
            "",
        ]
    )
    if neighbor_map:
        for doc_id in sorted(neighbor_map):
            neighbors = ", ".join(f"`{Path(path).relative_to(root)}`" for path in neighbor_map[doc_id])
            lines.append(f"- `{Path(doc_id).relative_to(root)}` -> {neighbors}")
    else:
        lines.append("- 没有找到超过阈值的近邻文档。")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"目录不存在或不是目录: {root}")

    scan_start = time.perf_counter()
    markdown_files = collect_markdown_files(root)
    scan_time = time.perf_counter() - scan_start
    if not markdown_files:
        raise SystemExit(f"目录中没有找到 Markdown 文件: {root}")

    read_start = time.perf_counter()
    docs, ngrams_by_doc = read_documents(markdown_files, args.ngram_size)
    read_time = time.perf_counter() - read_start

    minhash_start = time.perf_counter()
    minhashes = build_minhashes(ngrams_by_doc, num_perm=args.num_perm)
    minhash_time = time.perf_counter() - minhash_start

    lsh_build_start = time.perf_counter()
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    lsh_progress = ProgressBar(len(minhashes), "构建 LSH")
    for doc_id, m in minhashes.items():
        lsh.insert(doc_id, m)
        lsh_progress.update()
    lsh_build_time = time.perf_counter() - lsh_build_start

    query_start = time.perf_counter()
    neighbor_map: dict[str, list[str]] = {}
    query_progress = ProgressBar(len(minhashes), "查询近邻")
    for doc_id, m in minhashes.items():
        results = sorted(r for r in lsh.query(m) if r != doc_id)
        if results:
            neighbor_map[doc_id] = results
        query_progress.update()
    query_time = time.perf_counter() - query_start

    pair_start = time.perf_counter()
    top_pairs, report_pairs, exact_pairs, total_pairs = analyze_pairs(
        minhashes=minhashes,
        console_top_k=args.top_k,
        report_top_k=args.report_top_k,
        min_score=args.min_score,
    )
    pair_time = time.perf_counter() - pair_start
    report_path = Path(args.report_file).expanduser().resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    timings = {
        "scan": scan_time,
        "read": read_time,
        "minhash": minhash_time,
        "lsh_build": lsh_build_time,
        "query": query_time,
        "pair": pair_time,
    }
    write_report(
        report_path=report_path,
        root=root,
        docs=docs,
        args=args,
        timings=timings,
        total_pairs=total_pairs,
        non_exact_pairs=report_pairs,
        exact_pairs=exact_pairs,
        neighbor_map=neighbor_map,
    )

    print_doc_stats(docs, root)

    print("\n── 运行参数 ──")
    print(f"n-gram size: {args.ngram_size}")
    print(f"num_perm: {args.num_perm}")
    print(f"LSH threshold: {args.threshold}")
    print(f"报告文件: {report_path}")

    print("\n── 性能统计 ──")
    print(f"扫描文件耗时: {scan_time:.4f}s")
    print(f"读取并切分文本耗时: {read_time:.4f}s")
    print(f"MinHash 计算耗时: {minhash_time:.4f}s")
    print(f"LSH 建索引耗时: {lsh_build_time:.4f}s")
    print(f"LSH 查询耗时: {query_time:.4f}s")
    print(f"两两相似度排序耗时: {pair_time:.4f}s")

    print(f"\n── 高相似但不完全相同的前 {len(top_pairs)} 组文档 ──")
    if top_pairs:
        for score, left, right in top_pairs:
            print(f"{score:.3f}  {Path(left).relative_to(root)}  <->  {Path(right).relative_to(root)}")
    else:
        print("没有找到非完全重复的相似文档对。")

    if exact_pairs:
        print(f"\n── 完全重复文档对 ──")
        for score, left, right in exact_pairs[: min(len(exact_pairs), args.top_k)]:
            print(f"{score:.3f}  {Path(left).relative_to(root)}  <->  {Path(right).relative_to(root)}")

    print(f"\n── LSH 近邻结果（threshold={args.threshold}）──")
    if not neighbor_map:
        print("没有找到超过阈值的近邻文档。")
        return

    for doc_id in sorted(neighbor_map):
        neighbors = ", ".join(str(Path(path).relative_to(root)) for path in neighbor_map[doc_id])
        print(f"{Path(doc_id).relative_to(root)} -> {neighbors}")


if __name__ == "__main__":
    main()
