#!/usr/bin/env python3
"""
TOC-anchored cleanup for markitdown output.

Pipeline (in order):
  1. toc       — PDF TOC + page numbers: identify headings, drop page numbers
  2. rule      — digit/symbol heuristics: char noise, figure/table data
  3. kenlm     — perplexity filter: word-based table fragments
  4. sandwich  — context: short paragraphs between two artifact anchors

Usage:
    python tools/md-cleanup/toc_anchored_cleanup.py <input.md> --pdf <file.pdf>
           [-o output.md] [--log logfile.jsonl] [-v]
"""

import re
import sys
import json
import argparse
from pathlib import Path

import fitz  # pymupdf

sys.path.insert(0, str(Path(__file__).parent))
import cleanup as rule_mod
import kenlm_cleanup as lm_mod
import toc_match as toc_mod


# ── helpers ───────────────────────────────────────────────────────────────────

def load_toc(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    return [{"level": item[0], "title": item[1].strip(), "page": item[2]}
            for item in doc.get_toc(simple=False)]


def is_page_number(para: rule_mod.Paragraph) -> bool:
    return (para.num_lines == 1 and
            bool(re.fullmatch(r"\d{1,4}", para.lines[0].strip())))


_SENTENCE_END = re.compile(r'[.!?]["\']?$')


def is_small(para: rule_mod.Paragraph) -> bool:
    """Short, non-sentence paragraph — sandwich candidate."""
    words = para.text.split()
    if len(words) > 20:
        return False
    if para.num_lines > 1:          # multi-line = likely real content (e.g. paper title)
        return False
    if _SENTENCE_END.search(para.lines[-1].strip()) and len(words) > 6:
        return False
    return True


# ── main pipeline ─────────────────────────────────────────────────────────────

def process(text: str, toc: list[dict], model,
            threshold: float = lm_mod.DEFAULT_THRESHOLD) -> tuple[str, list[dict]]:
    """
    Returns (cleaned_text, log_records).

    Each log record:
      {lineno, label, pp, preview}
      label: 'readable' | 'toc:heading:N' | 'toc:page_number' |
             'rule:<type>' | 'kenlm' | 'sandwich'
    """
    paragraphs = rule_mod.split_paragraphs(text)
    n = len(paragraphs)
    labels = [None] * n
    pp_values = [0.0] * n

    # ── 1. TOC + page numbers ──
    heading_matches = toc_mod.match_headings(toc, paragraphs)
    for i, para in enumerate(paragraphs):
        if is_page_number(para):
            labels[i] = "toc:page_number"
        elif i in heading_matches:
            labels[i] = f"toc:heading:{heading_matches[i].depth}"

    # ── 2. rule classifier ──
    for i, para in enumerate(paragraphs):
        if labels[i] is not None:
            continue
        art = rule_mod.classify(para)
        if art:
            labels[i] = f"rule:{art}"

    # ── 3. KenLM perplexity ──
    for i, para in enumerate(paragraphs):
        pp = lm_mod.perplexity(model, para.text)
        pp_values[i] = pp
        if labels[i] is None and pp > threshold:
            labels[i] = "kenlm"

    # ── 4. sandwich detection ──
    def is_anchor(lbl):
        return lbl is not None

    for i in range(n):
        if labels[i] is not None:
            continue
        if not is_small(paragraphs[i]):
            continue
        prev_anchor = next(
            (j for j in range(i - 1, -1, -1)
             if is_anchor(labels[j]) or not is_small(paragraphs[j])),
            None)
        next_anchor = next(
            (j for j in range(i + 1, n)
             if is_anchor(labels[j]) or not is_small(paragraphs[j])),
            None)
        if (prev_anchor is not None and is_anchor(labels[prev_anchor]) and
                next_anchor is not None and is_anchor(labels[next_anchor])):
            labels[i] = "sandwich"

    # ── build output + log ──
    output_parts = []
    log_records = []
    stats: dict[str, int] = {}

    for i, (para, label, pp) in enumerate(zip(paragraphs, labels, pp_values)):
        preview = para.text[:120].replace("\n", "↵")

        if label is None:
            output_parts.append(para.text)
            final_label = "readable"

        elif label.startswith("toc:heading:"):
            depth = int(label.split(":")[-1])
            match = heading_matches.get(i)
            heading_text = (f"{match.number} {match.title}" if match
                            else para.lines[0].strip())
            output_parts.append("#" * depth + " " + heading_text)
            final_label = label

        elif label == "toc:page_number":
            final_label = label  # silently dropped

        else:
            tag = label.replace("rule:", "")
            output_parts.append(f"<!-- [LAYOUT_ARTIFACT: {tag}] -->")
            final_label = label

        stats[final_label] = stats.get(final_label, 0) + 1
        log_records.append({
            "lineno": para.start_lineno,
            "label": final_label,
            "pp": round(pp) if pp else None,
            "preview": preview,
        })

    log_records.append({"stats": stats})
    return "\n\n".join(output_parts) + "\n", log_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("--log", help="Write JSONL decision log to this file")
    parser.add_argument("-t", "--threshold", type=float, default=lm_mod.DEFAULT_THRESHOLD)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print non-readable decisions to stderr")
    args = parser.parse_args()

    toc = load_toc(args.pdf)
    model = lm_mod.load_model()
    text = Path(args.input).read_text(encoding="utf-8")
    result, log_records = process(text, toc, model, threshold=args.threshold)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")

    if args.log:
        with open(args.log, "w") as f:
            for rec in log_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.verbose:
        for rec in log_records[:-1]:  # skip stats line
            if rec["label"] not in ("readable", "toc:page_number"):
                print(f"  [{rec['label']:24s}] line {rec['lineno']:4d}"
                      f" pp={rec['pp'] or 0:8.0f}: {rec['preview']!r}",
                      file=sys.stderr)
        print(f"\nStats: {log_records[-1]['stats']}", file=sys.stderr)

    if not args.output:
        print(result, end="")


if __name__ == "__main__":
    main()
