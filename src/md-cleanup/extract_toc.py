#!/usr/bin/env python3
"""
Extract TOC (headings + page numbers) from a PDF using PyMuPDF.

Usage:
    python tools/md-cleanup/extract_toc.py <file.pdf>

Output: JSON to stdout
    [{"level": 1, "title": "Introduction", "page": 2}, ...]
"""

import json
import sys
import fitz  # pymupdf


def extract_toc(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    toc = doc.get_toc(simple=False)  # [[level, title, page, extra], ...]

    entries = []
    for item in toc:
        level, title, page = item[0], item[1], item[2]
        entries.append({
            "level": level,
            "title": title.strip(),
            "page": page,
        })
    return entries


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <file.pdf>", file=sys.stderr)
        sys.exit(1)

    entries = extract_toc(sys.argv[1])
    print(json.dumps(entries, ensure_ascii=False, indent=2))
