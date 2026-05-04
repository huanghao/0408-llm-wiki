#!/usr/bin/env python3
"""
TOC-anchored cleanup for markitdown output.

Pipeline:
  1. toc      — PDF TOC + page numbers: exact structural match, no scoring
  2. scoring  — three signals combined; paragraph deleted only if total ≥ threshold
       A. rule_score    : max weight of triggered heuristic rules (0–1)
       B. pp_score      : perplexity rank within document (0–1, adaptive)
       C. sandwich_score: 0.5 if single-line short para between two anchors
     final = max(rule_score, pp_score) * 0.7 + sandwich_score * 0.3
     delete if final >= 0.5

Adaptive threshold: KenLM score uses within-document perplexity rank,
not a fixed value. Avoids miscalibration across different paper styles.

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

DELETE_THRESHOLD = 0.55
SANDWICH_WEIGHT  = 0.3   # sandwich is auxiliary; alone it scores 0.5*0.3 = 0.15 < threshold
MAIN_WEIGHT      = 0.7   # rule or kenlm (whichever is stronger)


# ── helpers ───────────────────────────────────────────────────────────────────

def load_toc(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    return [{"level": item[0], "title": item[1].strip(), "page": item[2]}
            for item in doc.get_toc(simple=False)]


def is_page_number(para: rule_mod.Paragraph) -> bool:
    return (para.num_lines == 1 and
            bool(re.fullmatch(r"\d{1,4}", para.lines[0].strip())))


_SENTENCE_END = re.compile(r'[.!?]["\']?$')


_FIGURE_TABLE_RE = re.compile(r"^(Figure|Table|Fig\.?)\s+\d+", re.IGNORECASE)


def is_caption(para: rule_mod.Paragraph) -> bool:
    """Figure/Table captions are readable content, never artifacts."""
    return para.num_lines >= 1 and bool(_FIGURE_TABLE_RE.match(para.lines[0].strip()))


def is_sandwich_candidate(para: rule_mod.Paragraph) -> bool:
    """Single-line, short, non-sentence paragraph."""
    if para.num_lines > 1:
        return False
    words = para.text.split()
    if len(words) > 20:
        return False
    if _SENTENCE_END.search(para.lines[-1].strip()) and len(words) > 6:
        return False
    return True


# ── main pipeline ─────────────────────────────────────────────────────────────

def process(text: str, toc: list[dict], model) -> tuple[str, list[dict]]:
    """
    Returns (cleaned_text, log_records).

    Log record fields:
      lineno   : line number in original file (1-based)
      label    : decision — 'readable' | 'toc:heading:N' | 'toc:page_number' | 'artifact'
      score    : final combined score (None for toc decisions)
      rule     : rule_score component
      pp_rank  : pp_score component (perplexity rank 0-1)
      sandwich : sandwich_score component
      pp       : raw perplexity value
      preview  : first 120 chars of paragraph text
    Last record: {"stats": {...}}
    """
    paragraphs = rule_mod.split_paragraphs(text)
    n = len(paragraphs)

    # ── Step 1: TOC + page numbers (exact, no scoring) ──
    heading_matches = toc_mod.match_headings(toc, paragraphs)
    toc_labels = {}   # para index → 'toc:heading:N' or 'toc:page_number'
    for i, para in enumerate(paragraphs):
        if is_page_number(para):
            toc_labels[i] = "toc:page_number"
        elif i in heading_matches:
            toc_labels[i] = f"toc:heading:{heading_matches[i].depth}"

    # ── Compute raw signals for all non-toc paragraphs ──
    rule_scores = []
    raw_pp      = []
    for para in paragraphs:
        rule_scores.append(rule_mod.rule_confidence(para))
        raw_pp.append(lm_mod.perplexity(model, para.text))

    # Adaptive pp_score: rank within document
    sorted_pp = sorted(raw_pp)
    def pp_rank(pp_val: float) -> float:
        # fraction of paragraphs with lower perplexity
        lo = 0
        for v in sorted_pp:
            if v < pp_val:
                lo += 1
        return lo / n

    pp_scores = [pp_rank(pp) for pp in raw_pp]

    # ── Sandwich pass: needs anchor map ──
    # Anchors = toc labels + paragraphs whose combined score is already high
    # We do a quick pre-pass with rule+pp only to seed anchors for sandwich
    pre_scores = [
        max(rule_scores[i], pp_scores[i]) * MAIN_WEIGHT
        for i in range(n)
    ]

    def is_anchor(i: int) -> bool:
        return i in toc_labels or pre_scores[i] >= DELETE_THRESHOLD

    sandwich_scores = [0.0] * n
    for i in range(n):
        if i in toc_labels:
            continue
        if not is_sandwich_candidate(paragraphs[i]):
            continue
        prev = next((j for j in range(i-1, -1, -1)
                     if is_anchor(j) or not is_sandwich_candidate(paragraphs[j])), None)
        nxt  = next((j for j in range(i+1, n)
                     if is_anchor(j) or not is_sandwich_candidate(paragraphs[j])), None)
        if prev is not None and is_anchor(prev) and nxt is not None and is_anchor(nxt):
            sandwich_scores[i] = 0.5

    # ── Final scores + decisions ──
    output_parts = []
    log_records  = []
    stats: dict[str, int] = {}

    for i, para in enumerate(paragraphs):
        preview = para.text[:120].replace("\n", "↵")
        pp      = raw_pp[i]
        rs      = rule_scores[i]
        ps      = pp_scores[i]
        ss      = sandwich_scores[i]
        final   = max(rs, ps) * MAIN_WEIGHT + ss * SANDWICH_WEIGHT

        if i in toc_labels:
            label = toc_labels[i]
            if label.startswith("toc:heading:"):
                depth = int(label.split(":")[-1])
                match = heading_matches.get(i)
                heading_text = (f"{match.number} {match.title}" if match
                                else para.lines[0].strip())
                output_parts.append("#" * depth + " " + heading_text)
            # page_number: silently dropped
            log_records.append({
                "lineno": para.start_lineno, "label": label,
                "score": None, "rule": None, "pp_rank": None,
                "sandwich": None, "pp": round(pp), "preview": preview,
            })
            stats[label] = stats.get(label, 0) + 1

        elif final >= DELETE_THRESHOLD and not is_caption(para):
            output_parts.append("<!-- [LAYOUT_ARTIFACT] -->")
            log_records.append({
                "lineno": para.start_lineno, "label": "artifact",
                "score": round(final, 3), "rule": round(rs, 3),
                "pp_rank": round(ps, 3), "sandwich": round(ss, 3),
                "pp": round(pp), "preview": preview,
            })
            stats["artifact"] = stats.get("artifact", 0) + 1

        else:
            output_parts.append(para.text)
            log_records.append({
                "lineno": para.start_lineno, "label": "readable",
                "score": round(final, 3), "rule": round(rs, 3),
                "pp_rank": round(ps, 3), "sandwich": round(ss, 3),
                "pp": round(pp), "preview": preview,
            })
            stats["readable"] = stats.get("readable", 0) + 1

    log_records.append({"stats": stats})
    return "\n\n".join(output_parts) + "\n", log_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--pdf", required=True)
    parser.add_argument("-o", "--output")
    parser.add_argument("--log", help="Write JSONL decision log")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    toc   = load_toc(args.pdf)
    model = lm_mod.load_model()
    text  = Path(args.input).read_text(encoding="utf-8")
    result, log_records = process(text, toc, model)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")

    if args.log:
        with open(args.log, "w") as f:
            for rec in log_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.verbose:
        for rec in log_records[:-1]:
            if rec["label"] not in ("readable", "toc:page_number"):
                print(
                    f"  [{rec['label']:20s}] line {rec['lineno']:4d}"
                    f" score={rec['score'] or '—':>5}  "
                    f"rule={rec['rule'] or '—':>4}  "
                    f"pp_rank={rec['pp_rank'] or '—':>4}  "
                    f"sw={rec['sandwich'] or '—':>3}  "
                    f"{rec['preview'][:60]!r}",
                    file=sys.stderr,
                )
        print(f"\nStats: {log_records[-1]['stats']}", file=sys.stderr)

    if not args.output:
        print(result, end="")


if __name__ == "__main__":
    main()
