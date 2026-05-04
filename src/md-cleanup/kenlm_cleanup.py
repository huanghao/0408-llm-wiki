#!/usr/bin/env python3
"""
KenLM-based layout artifact detection for markitdown output.

Replaces the hand-coded rule classifier in cleanup.py with perplexity scoring.
Paragraphs with perplexity above a threshold are marked as layout artifacts.

Usage:
    python tools/md-cleanup/kenlm_cleanup.py <input.md> [-o output.md] [-v]
    python tools/md-cleanup/kenlm_cleanup.py <input.md> --hist   # show distribution plot
"""

import re
import sys
import argparse
import math
from pathlib import Path

import kenlm

BINARY_MODEL = Path(__file__).parent.parent.parent / "data" / "kenlm_academic" / "en_academic.binary"
ARPA_MODEL   = Path(__file__).parent.parent.parent / "data" / "kenlm_academic" / "en_academic.arpa"

# Perplexity threshold: paragraphs above this are artifacts.
# Calibrated on the dclm paper: normal prose ~200-800, noise spikes to 5000+
DEFAULT_THRESHOLD = 50000


def load_model() -> kenlm.Model:
    path = BINARY_MODEL if BINARY_MODEL.exists() else ARPA_MODEL
    if not path.exists():
        raise FileNotFoundError(f"No KenLM model found. Run train_lm.py first.\nExpected: {BINARY_MODEL}")
    return kenlm.Model(str(path))


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d", "0", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def perplexity(model: kenlm.Model, text: str) -> float:
    tokens = normalize(text).split()
    if not tokens:
        return float("inf")
    log_score = model.score(" ".join(tokens), bos=True, eos=True)
    length = len(tokens) + 1  # +1 for </s>
    return 10.0 ** (-log_score / length)


# ---------- segmentation (same as cleanup.py) ----------

def split_paragraphs(text: str) -> list[tuple[list[str], int]]:
    """Returns list of (lines, start_lineno)."""
    paragraphs = []
    current: list[str] = []
    current_start = 1
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            if current:
                paragraphs.append((current, current_start))
                current = []
            current_start = lineno + 1
        else:
            if not current:
                current_start = lineno
            current.append(line)
    if current:
        paragraphs.append((current, current_start))
    return paragraphs


# ---------- main ----------

def process(text: str, model: kenlm.Model, threshold: float = DEFAULT_THRESHOLD,
            verbose: bool = False) -> tuple[str, list[tuple]]:
    """
    Returns (cleaned_text, scored_paragraphs).
    scored_paragraphs: list of (lineno, pp, is_artifact, preview)
    """
    paragraphs = split_paragraphs(text)
    output_parts = []
    scored = []

    for lines, lineno in paragraphs:
        para_text = "\n".join(lines)
        pp = perplexity(model, para_text)
        is_artifact = pp > threshold

        scored.append((lineno, pp, is_artifact, para_text[:80].replace("\n", "↵")))

        if is_artifact:
            output_parts.append("<!-- [LAYOUT_ARTIFACT] -->")
            if verbose:
                print(f"  [pp={pp:8.0f}] line {lineno:4d}: {para_text[:70].replace(chr(10), '↵')!r}", file=sys.stderr)
        else:
            output_parts.append(para_text)

    if verbose:
        n_artifact = sum(1 for _, _, a, _ in scored if a)
        print(f"\nSummary: {len(scored)-n_artifact} readable, {n_artifact} artifacts "
              f"(threshold={threshold})", file=sys.stderr)

    return "\n\n".join(output_parts) + "\n", scored


def show_histogram(scored: list[tuple]):
    import matplotlib.pyplot as plt
    import numpy as np

    pps = [pp for _, pp, _, _ in scored if pp < 1e6]
    artifacts = [pp for _, pp, a, _ in scored if a and pp < 1e6]
    readable = [pp for _, pp, a, _ in scored if not a and pp < 1e6]

    log_pps = np.log10(pps)
    bins = np.linspace(0, max(log_pps) + 0.5, 40)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(np.log10(readable), bins=bins, alpha=0.7, label="readable", color="steelblue")
    ax.hist(np.log10(artifacts), bins=bins, alpha=0.7, label="artifact", color="salmon")
    ax.axvline(math.log10(DEFAULT_THRESHOLD), color="red", linestyle="--",
               label=f"threshold={DEFAULT_THRESHOLD}")
    ax.set_xlabel("log₁₀(perplexity)")
    ax.set_ylabel("paragraph count")
    ax.set_title("Paragraph perplexity distribution (KenLM)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("/tmp/perplexity_dist.png", dpi=150)
    print("Histogram saved → /tmp/perplexity_dist.png", file=sys.stderr)
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("-o", "--output")
    parser.add_argument("-t", "--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--hist", action="store_true", help="Show perplexity histogram")
    args = parser.parse_args()

    model = load_model()
    text = Path(args.input).read_text(encoding="utf-8")
    result, scored = process(text, model, threshold=args.threshold, verbose=args.verbose)

    if args.hist:
        show_histogram(scored)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
