#!/usr/bin/env python3
"""
MD Cleanup: Rule-based layout artifact detection for markitdown output.

Splits markdown into paragraphs, classifies each as READABLE or LAYOUT_ARTIFACT,
and replaces artifacts with placeholder comments.
"""

import re
import sys
import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Paragraph:
    lines: list[str]
    start_lineno: int  # 1-based, for debugging

    @property
    def text(self) -> str:
        return "\n".join(self.lines)

    @property
    def all_chars(self) -> str:
        return self.text

    @property
    def num_lines(self) -> int:
        return len(self.lines)


# ---------- feature extraction ----------

def digit_ratio(para: Paragraph) -> float:
    chars = para.all_chars
    if not chars.strip():
        return 0.0
    return sum(c.isdigit() for c in chars) / len(chars.replace(" ", "").replace("\n", "") or "x")


def alpha_ratio(para: Paragraph) -> float:
    chars = para.all_chars.replace(" ", "").replace("\n", "")
    if not chars:
        return 0.0
    return sum(c.isalpha() for c in chars) / len(chars)


def avg_line_len(para: Paragraph) -> float:
    lens = [len(l) for l in para.lines]
    return sum(lens) / len(lens) if lens else 0.0


def line_len_cv(para: Paragraph) -> float:
    """Coefficient of variation of line lengths."""
    if len(para.lines) < 2:
        return 0.0
    lens = [len(l) for l in para.lines]
    mean = sum(lens) / len(lens)
    if mean == 0:
        return 0.0
    variance = sum((l - mean) ** 2 for l in lens) / len(lens)
    return (variance ** 0.5) / mean


def single_char_lines_ratio(para: Paragraph) -> float:
    """Fraction of lines that are 0–1 characters (after strip)."""
    single = sum(1 for l in para.lines if len(l.strip()) <= 1)
    return single / para.num_lines if para.num_lines else 0.0


def has_no_space_long_line(para: Paragraph, min_len: int = 40) -> bool:
    """True if any line has no spaces and is longer than min_len (figure/table text smashed together)."""
    for line in para.lines:
        stripped = line.strip()
        if len(stripped) >= min_len and " " not in stripped:
            return True
    return False


def has_digit_led_long_line(para: Paragraph, min_len: int = 60) -> bool:
    """True if any long line starts with a dense run of digits immediately followed by more digits
    (graph axis values concatenated: '10152025...'). Excludes lines where digits are followed by a
    letter or comma+space, which indicates affiliation numbers or citation lists."""
    for line in para.lines:
        stripped = line.strip()
        # Must be long enough to be a concatenated axis/legend line
        if len(stripped) < min_len:
            continue
        # Match: starts with 3+ digits immediately followed by another digit group (no separator)
        # e.g. '10152025303540Core evals...' — the digit runs have no space/comma between them
        if re.match(r"^\d{2,}(?:\d{2,}){2,}", stripped):
            return True
    return False


def is_isolated_number(para: Paragraph) -> bool:
    """Single line with just a 1-4 digit number (page number)."""
    if para.num_lines != 1:
        return False
    return bool(re.fullmatch(r"\d{1,4}", para.lines[0].strip()))


def is_isolated_punctuation(para: Paragraph) -> bool:
    """Single line of only punctuation/brackets — leftover from arXiv metadata."""
    if para.num_lines != 1:
        return False
    return bool(re.fullmatch(r"[^\w\s]+", para.lines[0].strip()))


# ---------- classifier ----------

ArtifactType = str  # 'page_number' | 'char_noise' | 'figure' | 'table' | None


def classify(para: Paragraph) -> ArtifactType:
    text = para.text.strip()
    if not text:
        return None

    # Rule 1: isolated page number
    if is_isolated_number(para):
        return "page_number"

    # Rule 2: isolated punctuation (arXiv metadata bracket residue, etc.)
    if is_isolated_punctuation(para):
        return "char_noise"

    # Rule 3: single-character-per-line noise (arXiv metadata, etc.)
    if para.num_lines >= 4 and single_char_lines_ratio(para) > 0.6:
        return "char_noise"

    # Rule 4: no-space long line (figure/graph text smashed together)
    if has_no_space_long_line(para):
        return "figure"

    # Rule 5: digit-led long line (graph axis values concatenated with labels)
    if has_digit_led_long_line(para):
        return "figure"

    # Rule 4: many short lines with high digit ratio → table column data
    avg_len = avg_line_len(para)
    if avg_len < 14 and para.num_lines >= 4 and digit_ratio(para) > 0.3:
        return "table"

    # Rule 6: very low alpha ratio across whole paragraph
    if alpha_ratio(para) < 0.3 and para.num_lines >= 3:
        return "table"

    # Rule 7: extreme line-length variance + high digit ratio (mixed table data)
    if line_len_cv(para) > 2.0 and digit_ratio(para) > 0.2 and para.num_lines >= 5:
        return "table"

    return None


# Rule weights for confidence scoring (used by toc_anchored_cleanup)
_RULE_WEIGHTS: list[tuple[str, float]] = [
    ("isolated_punctuation", 1.0),
    ("char_noise_lines",     1.0),
    ("no_space_long_line",   0.8),
    ("digit_led_long_line",  0.8),
    ("short_digit_lines",    0.6),
    ("low_alpha",            0.6),
    ("high_cv_digit",        0.6),
]


def rule_confidence(para: Paragraph) -> float:
    """Return max weight among all triggered rules (0 if none)."""
    text = para.text.strip()
    if not text:
        return 0.0
    hits = []
    if is_isolated_punctuation(para):
        hits.append(1.0)
    if para.num_lines >= 4 and single_char_lines_ratio(para) > 0.6:
        hits.append(1.0)
    if has_no_space_long_line(para):
        hits.append(0.8)
    if has_digit_led_long_line(para):
        hits.append(0.8)
    avg_len = avg_line_len(para)
    if avg_len < 14 and para.num_lines >= 4 and digit_ratio(para) > 0.3:
        hits.append(0.6)
    if alpha_ratio(para) < 0.3 and para.num_lines >= 3:
        hits.append(0.6)
    if line_len_cv(para) > 2.0 and digit_ratio(para) > 0.2 and para.num_lines >= 5:
        hits.append(0.6)
    return max(hits) if hits else 0.0


# ---------- segmentation ----------

def split_paragraphs(text: str) -> list[Paragraph]:
    """Split on blank lines, preserving paragraph boundaries."""
    paragraphs = []
    current_lines: list[str] = []
    start_lineno = 1
    current_start = 1

    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.strip() == "":
            if current_lines:
                paragraphs.append(Paragraph(lines=current_lines, start_lineno=current_start))
                current_lines = []
            current_start = lineno + 1
        else:
            if not current_lines:
                current_start = lineno
            current_lines.append(line)

    if current_lines:
        paragraphs.append(Paragraph(lines=current_lines, start_lineno=current_start))

    return paragraphs


# ---------- main ----------

def process(text: str, verbose: bool = False) -> str:
    paragraphs = split_paragraphs(text)
    output_parts = []

    stats = {"readable": 0, "artifact": 0}

    for para in paragraphs:
        artifact_type = classify(para)
        if artifact_type:
            placeholder = f"<!-- [LAYOUT_ARTIFACT: {artifact_type}] -->"
            output_parts.append(placeholder)
            stats["artifact"] += 1
            if verbose:
                preview = para.text[:80].replace("\n", "↵")
                print(f"  [ARTIFACT:{artifact_type:12s}] line {para.start_lineno:4d}: {preview!r}", file=sys.stderr)
        else:
            output_parts.append(para.text)
            stats["readable"] += 1

    if verbose:
        total = stats["readable"] + stats["artifact"]
        print(f"\nSummary: {stats['readable']} readable, {stats['artifact']} artifacts out of {total} paragraphs", file=sys.stderr)

    # Join with double newlines (preserving paragraph spacing)
    return "\n\n".join(output_parts) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Clean layout artifacts from markitdown output")
    parser.add_argument("input", help="Input markdown file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print artifact details to stderr")
    args = parser.parse_args()

    text = Path(args.input).read_text(encoding="utf-8")
    result = process(text, verbose=args.verbose)

    if args.output:
        Path(args.output).write_text(result, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(result, end="")


if __name__ == "__main__":
    main()
