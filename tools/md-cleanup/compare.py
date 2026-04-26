#!/usr/bin/env python3
"""
Compare rule-based cleanup vs KenLM-based cleanup on the same input file.

Usage:
    python tools/md-cleanup/compare.py <input.md>
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import cleanup as rule_mod
import kenlm_cleanup as lm_mod

THRESHOLD = lm_mod.DEFAULT_THRESHOLD


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.md>")
        sys.exit(1)

    text = Path(sys.argv[1]).read_text(encoding="utf-8")
    paragraphs = rule_mod.split_paragraphs(text)

    # --- rule-based results ---
    rule_artifacts = set()
    for para in paragraphs:
        art = rule_mod.classify(para)
        if art:
            rule_artifacts.add(para.start_lineno)

    # --- kenlm results ---
    model = lm_mod.load_model()
    lm_scored = lm_mod.split_paragraphs(text)  # returns (lines, lineno) tuples
    lm_artifacts = set()
    lm_pp = {}
    for lines, lineno in lm_scored:
        pp = lm_mod.perplexity(model, "\n".join(lines))
        lm_pp[lineno] = pp
        if pp > THRESHOLD:
            lm_artifacts.add(lineno)

    # --- comparison ---
    only_rule = rule_artifacts - lm_artifacts
    only_lm   = lm_artifacts - rule_artifacts
    both      = rule_artifacts & lm_artifacts
    neither   = set(p.start_lineno for p in paragraphs) - rule_artifacts - lm_artifacts

    print(f"{'='*70}")
    print(f"Comparison: rule-based vs KenLM (threshold={THRESHOLD:,})")
    print(f"{'='*70}")
    print(f"  Total paragraphs : {len(paragraphs)}")
    print(f"  Both agree artifact : {len(both)}")
    print(f"  Both agree readable : {len(neither)}")
    print(f"  Only RULE flags     : {len(only_rule)}")
    print(f"  Only KENLM flags    : {len(only_lm)}")
    print()

    if only_rule:
        print("── Flagged by RULE only (KenLM missed) ──")
        for para in paragraphs:
            if para.start_lineno in only_rule:
                pp = lm_pp.get(para.start_lineno, 0)
                preview = para.text[:100].replace("\n", "↵")
                print(f"  line {para.start_lineno:4d}  pp={pp:8.0f}  {preview!r}")
        print()

    if only_lm:
        print("── Flagged by KENLM only (rule missed) ──")
        for para in paragraphs:
            if para.start_lineno in only_lm:
                pp = lm_pp.get(para.start_lineno, 0)
                preview = para.text[:100].replace("\n", "↵")
                print(f"  line {para.start_lineno:4d}  pp={pp:8.0f}  {preview!r}")
        print()

    print("── Agreed artifacts (both methods) ──")
    for para in paragraphs:
        if para.start_lineno in both:
            pp = lm_pp.get(para.start_lineno, 0)
            preview = para.text[:80].replace("\n", "↵")
            print(f"  line {para.start_lineno:4d}  pp={pp:8.0f}  {preview!r}")


if __name__ == "__main__":
    main()
