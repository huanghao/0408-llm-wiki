#!/usr/bin/env python3
"""
Combined cleanup: rule-based OR KenLM (union).
A paragraph is marked as artifact if either method flags it.

Usage:
    python tools/md-cleanup/combined_cleanup.py <input.md> [-o output.md] [-v]
"""

import re
import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cleanup as rule_mod
import kenlm_cleanup as lm_mod


def process(text: str, model, threshold: float = lm_mod.DEFAULT_THRESHOLD,
            verbose: bool = False) -> str:
    rule_paras = rule_mod.split_paragraphs(text)
    lm_paras = lm_mod.split_paragraphs(text)

    # build lineno → pp map
    pp_map = {}
    for lines, lineno in lm_paras:
        pp_map[lineno] = lm_mod.perplexity(model, "\n".join(lines))

    output_parts = []
    stats = {"readable": 0, "rule": 0, "kenlm": 0, "both": 0}

    for para in rule_paras:
        rule_hit = rule_mod.classify(para)
        lm_hit = pp_map.get(para.start_lineno, 0) > threshold

        if rule_hit and lm_hit:
            stats["both"] += 1
            output_parts.append("<!-- [LAYOUT_ARTIFACT: both] -->")
            tag = "both"
        elif rule_hit:
            stats["rule"] += 1
            output_parts.append(f"<!-- [LAYOUT_ARTIFACT: {rule_hit}] -->")
            tag = rule_hit
        elif lm_hit:
            stats["kenlm"] += 1
            output_parts.append("<!-- [LAYOUT_ARTIFACT: kenlm] -->")
            tag = "kenlm"
        else:
            stats["readable"] += 1
            output_parts.append(para.text)
            tag = None

        if verbose and tag:
            pp = pp_map.get(para.start_lineno, 0)
            preview = para.text[:70].replace("\n", "↵")
            print(f"  [{tag:12s}] line {para.start_lineno:4d} pp={pp:8.0f}: {preview!r}",
                  file=sys.stderr)

    if verbose:
        total = sum(stats.values())
        print(f"\nSummary: {stats['readable']} readable | "
              f"{stats['rule']} rule-only | {stats['kenlm']} kenlm-only | "
              f"{stats['both']} both | total={total}", file=sys.stderr)

    return "\n\n".join(output_parts) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-t", "--threshold", type=float, default=lm_mod.DEFAULT_THRESHOLD)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    model = lm_mod.load_model()
    text = Path(args.input).read_text(encoding="utf-8")
    result = process(text, model, threshold=args.threshold, verbose=args.verbose)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
