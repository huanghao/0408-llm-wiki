#!/usr/bin/env python3
"""
Train a KenLM trigram model on arXiv abstracts.
Output: data/kenlm_academic/en_academic.arpa

Usage: python tools/md-cleanup/train_lm.py
"""

import os
import re
import shutil
import subprocess
from pathlib import Path

LMPLZ = "/Users/huanghao/workspace/sources/kenlm/build/bin/lmplz"
BUILD_BINARY = "/Users/huanghao/workspace/sources/kenlm/build/bin/build_binary"
OUT_DIR = Path(__file__).parent.parent.parent / "data" / "kenlm_academic"
ARPA_PATH = OUT_DIR / "en_academic.arpa"
BINARY_PATH = OUT_DIR / "en_academic.binary"
CORPUS_PATH = OUT_DIR / "train.txt"
N_DOCS = 50000
ORDER = 3


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\d", "0", text)          # digits → 0  (CCNet convention)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_corpus():
    print(f"Loading {N_DOCS} arXiv abstracts...")
    from datasets import load_dataset
    ds = load_dataset("ccdv/arxiv-summarization", split=f"train[:{N_DOCS}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CORPUS_PATH, "w") as f:
        for row in ds:
            text = normalize(row["abstract"])
            if len(text.split()) >= 10:
                f.write(text + "\n")

    n_lines = sum(1 for _ in open(CORPUS_PATH))
    print(f"Corpus: {n_lines} lines → {CORPUS_PATH}")
    return n_lines


def train():
    lmplz = LMPLZ if Path(LMPLZ).exists() else shutil.which("lmplz")
    if not lmplz:
        raise RuntimeError("lmplz not found")

    print(f"Training {ORDER}-gram KenLM model...")
    with open(ARPA_PATH, "w") as arpa_out:
        subprocess.run(
            [lmplz, "-o", str(ORDER), "--text", str(CORPUS_PATH), "--discount_fallback"],
            stdout=arpa_out,
            check=True,
        )
    print(f"ARPA written → {ARPA_PATH}")


def binarize():
    bb = BUILD_BINARY if Path(BUILD_BINARY).exists() else shutil.which("build_binary")
    if not bb:
        print("build_binary not found, skipping binarization (ARPA will be used directly)")
        return False
    print("Binarizing...")
    subprocess.run([bb, str(ARPA_PATH), str(BINARY_PATH)], check=True)
    print(f"Binary model → {BINARY_PATH}")
    return True


if __name__ == "__main__":
    build_corpus()
    train()
    binarize()
    print("Done.")
