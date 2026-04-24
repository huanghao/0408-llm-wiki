"""
build_bilingual.py

把 main-body-clean.md + sidecar JSON 合并成双语对照 md。

格式：
  标题保持原样（不翻译）
  每个有译文的段落：原文 + 空行 + > 译文
  没有译文的段落：原文（保持原样）
  占位符、公式：保持原样
"""
import re
import json
import argparse
from pathlib import Path

MD_FILE = Path(__file__).parent.parent / "20250421-llama3-exp/main-body-clean.md"
SIDECAR_FILE = MD_FILE.with_suffix(".translation.json")
OUT_FILE = MD_FILE.parent / "main-body-bilingual.md"


def extract_paragraphs_with_positions(md_text: str):
    """
    返回 [(para_id, start_line, end_line), ...]
    以及 section_stack 的变化记录
    """
    lines = md_text.split('\n')
    section_stack = []
    para_counts = {}
    result = []
    para_start = None
    current_para_lines = []

    def get_section_path():
        if not section_stack:
            return "0"
        return ".".join(n for _, n in section_stack)

    def flush_para(end_line):
        nonlocal para_start
        if para_start is None or not current_para_lines:
            current_para_lines.clear()
            para_start = None
            return
        text = '\n'.join(current_para_lines).strip()
        if not text or len(text) < 20:
            current_para_lines.clear()
            para_start = None
            return
        if text.startswith('> **[Figure') or text.startswith('> **[Table'):
            current_para_lines.clear()
            para_start = None
            return
        if text.startswith('$$'):
            current_para_lines.clear()
            para_start = None
            return
        section_path = get_section_path()
        idx = para_counts.get(section_path, 0)
        para_id = f"{section_path}-p{idx}"
        para_counts[section_path] = idx + 1
        result.append((para_id, para_start, end_line - 1))
        current_para_lines.clear()
        para_start = None

    for i, line in enumerate(lines):
        heading_match = re.match(r'^(#{1,3})\s+(\d+(?:\.\d+)*)\.\s+(.*)', line)
        if heading_match:
            flush_para(i)
            level = len(heading_match.group(1))
            num_str = heading_match.group(2)
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, num_str))
            continue
        if not line.strip():
            flush_para(i)
            continue
        if para_start is None:
            para_start = i
        current_para_lines.append(line)

    flush_para(len(lines))
    return result


def build_bilingual(md_text: str, translations: dict, only_translated: bool = False) -> str:
    """
    生成双语对照 md。
    only_translated=True 时只输出有译文的段落（用于 demo）。
    """
    lines = md_text.split('\n')
    para_positions = extract_paragraphs_with_positions(md_text)
    
    # 建立 line -> para_id 映射
    line_to_para = {}
    for para_id, start, end in para_positions:
        for ln in range(start, end + 1):
            line_to_para[ln] = para_id
    
    # 哪些段落有译文
    translated_paras = set(translations.keys())
    
    out_lines = []
    i = 0
    while i < len(lines):
        line = lines[i]
        para_id = line_to_para.get(i)
        
        if para_id and para_id in translated_paras:
            # 找这个段落的所有行
            para_info = next((p for p in para_positions if p[0] == para_id), None)
            if para_info:
                _, start, end = para_info
                # 输出原文段落
                para_lines = lines[start:end+1]
                out_lines.extend(para_lines)
                out_lines.append('')
                # 输出译文（blockquote 样式）
                translation = translations[para_id]
                out_lines.append(f'> 🌐 {translation}')
                out_lines.append('')
                i = end + 1
                continue
        elif para_id and only_translated:
            # only_translated 模式：跳过没有译文的段落
            para_info = next((p for p in para_positions if p[0] == para_id), None)
            if para_info:
                i = para_info[2] + 1
                continue
        
        out_lines.append(line)
        i += 1
    
    return '\n'.join(out_lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only-translated", action="store_true",
                        help="只输出有译文的段落（demo 模式）")
    args = parser.parse_args()

    md_text = MD_FILE.read_text()
    translations = json.loads(SIDECAR_FILE.read_text()) if SIDECAR_FILE.exists() else {}
    
    print(f"原文段落总数：{len(extract_paragraphs_with_positions(md_text))}")
    print(f"已有译文：{len(translations)} 条")
    
    result = build_bilingual(md_text, translations, only_translated=args.only_translated)
    
    OUT_FILE.write_text(result)
    print(f"输出：{OUT_FILE} ({len(result.splitlines())} 行)")


if __name__ == "__main__":
    main()
