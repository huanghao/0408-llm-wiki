#!/usr/bin/env python3
"""
TOC → numbered heading matcher.

Given a TOC (list of {level, title, page}) and a list of markdown paragraphs,
produces a mapping: paragraph index → (heading_depth, title).

Strategy:
  1. Infer section numbers from TOC level structure  (e.g. level=2 → "3.2")
  2. Build candidate text = number + title  (e.g. "3.2 Competition scales...")
  3. For each TOC entry, find the page-number anchor in the markdown
     (single-line paragraph whose text equals the expected page number)
  4. Within a search window around that page anchor, find the paragraph
     whose text best matches the candidate heading text
  5. Accept if similarity ≥ threshold

Page anchor reliability:
  - We look for a paragraph that is a pure integer AND matches the expected
    page number from the TOC.  We also tolerate ±1 page offset to handle
    cases where the page number line appears slightly before/after the heading.
  - If no page anchor is found for a given entry, we fall back to a wider
    search window (full document), still requiring high text similarity.
"""

import re
from difflib import SequenceMatcher
from dataclasses import dataclass

from cleanup import Paragraph


# ── number inference ──────────────────────────────────────────────────────────

def infer_numbers(toc: list[dict]) -> list[str]:
    """Return a section-number string for each TOC entry, e.g. '3.2'."""
    counters: list[int] = []
    numbers = []
    for entry in toc:
        depth = entry["level"]
        # extend counters list if needed
        while len(counters) < depth:
            counters.append(0)
        # truncate deeper levels when we go back up
        counters = counters[:depth]
        counters[depth - 1] += 1
        numbers.append(".".join(str(c) for c in counters))
    return numbers


# ── similarity ────────────────────────────────────────────────────────────────

def _sim(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower().strip(), b.lower().strip()).ratio()


# ── page anchor index ─────────────────────────────────────────────────────────

def build_page_index(paragraphs: list[Paragraph]) -> dict[int, list[int]]:
    """
    Returns {page_number: [paragraph_indices]} for all single-line pure-integer paragraphs.
    Multiple occurrences of the same number are kept (some pdfs repeat page numbers).
    """
    index: dict[int, list[int]] = {}
    for i, para in enumerate(paragraphs):
        if para.num_lines == 1:
            t = para.lines[0].strip()
            if re.fullmatch(r"\d{1,4}", t):
                n = int(t)
                index.setdefault(n, []).append(i)
    return index


# ── main matcher ──────────────────────────────────────────────────────────────

WINDOW = 15          # paragraphs to search around the page anchor
FALLBACK_SIM = 0.88  # stricter threshold when no page anchor found
ANCHOR_SIM = 0.78    # looser threshold when page anchor confirms location


@dataclass
class HeadingMatch:
    para_idx: int
    depth: int
    number: str        # e.g. "3.2"
    title: str         # clean title text from TOC
    similarity: float
    via_page_anchor: bool


def match_headings(toc: list[dict],
                   paragraphs: list[Paragraph]) -> dict[int, HeadingMatch]:
    """
    Returns {paragraph_index: HeadingMatch} for each successfully matched heading.
    """
    numbers = infer_numbers(toc)
    page_index = build_page_index(paragraphs)
    n = len(paragraphs)
    results: dict[int, HeadingMatch] = {}
    used_para_indices: set[int] = set()

    for entry, number in zip(toc, numbers):
        depth = entry["level"]
        title = entry["title"]
        page = entry["page"]

        # candidate text: what we expect to see in the markdown
        candidate = f"{number} {title}"

        # find page anchor paragraphs (accept ±1 page)
        anchor_indices: list[int] = []
        for p in (page - 1, page, page + 1):
            anchor_indices.extend(page_index.get(p, []))

        if anchor_indices:
            # search within WINDOW paragraphs of any anchor
            search_set = set()
            for ai in anchor_indices:
                for j in range(max(0, ai - WINDOW), min(n, ai + WINDOW + 1)):
                    search_set.add(j)
            threshold = ANCHOR_SIM
        else:
            # no page anchor: search whole document, stricter threshold
            search_set = set(range(n))
            threshold = FALLBACK_SIM

        best_idx, best_sim = -1, 0.0
        for j in sorted(search_set):
            if j in used_para_indices:
                continue
            para = paragraphs[j]
            if para.num_lines != 1:
                continue
            text = para.lines[0].strip()
            s = _sim(candidate, text)
            if s > best_sim:
                best_sim, best_idx = s, j

        if best_idx >= 0 and best_sim >= threshold:
            results[best_idx] = HeadingMatch(
                para_idx=best_idx,
                depth=depth,
                number=number,
                title=title,
                similarity=best_sim,
                via_page_anchor=bool(anchor_indices),
            )
            used_para_indices.add(best_idx)

    return results


if __name__ == "__main__":
    # quick smoke test
    import json, sys
    sys.path.insert(0, ".")
    from cleanup import split_paragraphs
    from pathlib import Path

    if len(sys.argv) < 3:
        print("Usage: toc_match.py <toc.json> <input.md>")
        sys.exit(1)

    toc = json.loads(Path(sys.argv[1]).read_text())
    text = Path(sys.argv[2]).read_text()
    paras = split_paragraphs(text)
    matches = match_headings(toc, paras)

    for idx, m in sorted(matches.items(), key=lambda x: x[0]):
        anchor = "page✓" if m.via_page_anchor else "fallback"
        print(f"  para {idx:3d}  [{anchor}] sim={m.similarity:.2f}  "
              f"{'#'*m.depth} {m.number} {m.title}")
