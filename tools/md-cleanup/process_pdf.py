#!/usr/bin/env python3
"""
Full PDF → structured output directory pipeline.

Steps:
  1. markitdown PDF  → <name>/foo.original.md
  2. extract TOC     → <name>/foo.toc.json
  3. 4-layer cleanup → <name>/foo.cleaned.md
  4. split by H1     → <name>/<slug>.md  (one file per top-level section)

Usage:
    python tools/md-cleanup/process_pdf.py <file.pdf> [-o output_dir] [-v]

Output directory defaults to data/processed/<stem>/ relative to repo root.
"""

import re
import sys
import json
import argparse
import subprocess
from pathlib import Path

# allow sibling imports
sys.path.insert(0, str(Path(__file__).parent))
import kenlm_cleanup as lm_mod
import toc_anchored_cleanup as pipeline


# ── helpers ───────────────────────────────────────────────────────────────────

def slugify(text: str) -> str:
    """Convert heading text to a safe filename slug."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    text = text.strip("-")
    return text or "section"


def split_by_h1(cleaned_text: str) -> list[tuple[str, str]]:
    """
    Split markdown by H1 headings (lines starting with a single '#').

    Returns list of (heading_text, section_content) tuples.
    Content before the first H1 is returned as ("_preamble", ...) if non-empty.
    """
    sections = []
    current_heading = "_preamble"
    current_lines: list[str] = []

    for line in cleaned_text.splitlines():
        if re.match(r"^# ", line):
            if current_lines and any(l.strip() for l in current_lines):
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = line[2:].strip()
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines and any(l.strip() for l in current_lines):
        sections.append((current_heading, "\n".join(current_lines).strip()))

    return sections


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", help="Input PDF file")
    parser.add_argument("-o", "--output-dir", help="Output directory (default: data/processed/<stem>)")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).resolve()
    if not pdf_path.exists():
        print(f"Error: {pdf_path} not found", file=sys.stderr)
        sys.exit(1)

    stem = pdf_path.stem
    # strip trailing arXiv ID patterns like -2407.21783
    stem = re.sub(r"-\d{4}\.\d+$", "", stem)

    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        out_dir = repo_root / "data" / "processed" / stem

    out_dir.mkdir(parents=True, exist_ok=True)

    original_md = out_dir / f"{stem}.original.md"
    toc_json    = out_dir / f"{stem}.toc.json"
    cleaned_md  = out_dir / f"{stem}.cleaned.md"
    log_jsonl   = out_dir / f"{stem}.cleanup.log.jsonl"

    # ── Step 1: PDF → markdown ───────────────────────────────────────────────
    print(f"[1/4] Converting PDF → markdown ...", file=sys.stderr)
    result = subprocess.run(
        ["markitdown", str(pdf_path)],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"markitdown failed:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)
    original_md.write_text(result.stdout, encoding="utf-8")
    if args.verbose:
        print(f"  wrote {original_md}", file=sys.stderr)

    # ── Step 2: extract TOC ──────────────────────────────────────────────────
    print(f"[2/4] Extracting TOC ...", file=sys.stderr)
    toc = pipeline.load_toc(str(pdf_path))
    toc_json.write_text(json.dumps(toc, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.verbose:
        print(f"  {len(toc)} TOC entries → {toc_json}", file=sys.stderr)

    # ── Step 3: 4-layer cleanup ──────────────────────────────────────────────
    print(f"[3/4] Running cleanup pipeline ...", file=sys.stderr)
    model = lm_mod.load_model()
    text = original_md.read_text(encoding="utf-8")
    cleaned_text, log_records = pipeline.process(text, toc, model)
    cleaned_md.write_text(cleaned_text, encoding="utf-8")
    with open(log_jsonl, "w", encoding="utf-8") as f:
        for rec in log_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    stats = log_records[-1].get("stats", {})
    if args.verbose:
        print(f"  stats: {stats}", file=sys.stderr)
        print(f"  wrote {cleaned_md}", file=sys.stderr)

    # ── Step 4: split by H1 ──────────────────────────────────────────────────
    print(f"[4/4] Splitting into chapter files ...", file=sys.stderr)
    sections = split_by_h1(cleaned_text)

    # deduplicate slugs
    seen: dict[str, int] = {}
    chapter_files = []
    for heading, content in sections:
        base = slugify(heading)
        count = seen.get(base, 0)
        seen[base] = count + 1
        slug = base if count == 0 else f"{base}-{count}"
        chapter_path = out_dir / f"{slug}.md"
        chapter_path.write_text(content + "\n", encoding="utf-8")
        chapter_files.append(chapter_path.name)
        if args.verbose:
            print(f"  {chapter_path.name}  ({len(content)} chars)", file=sys.stderr)

    print(f"\nDone. Output directory: {out_dir}", file=sys.stderr)
    print(f"  {original_md.name}", file=sys.stderr)
    print(f"  {toc_json.name}", file=sys.stderr)
    print(f"  {cleaned_md.name}", file=sys.stderr)
    for name in chapter_files:
        print(f"  {name}", file=sys.stderr)


if __name__ == "__main__":
    main()
