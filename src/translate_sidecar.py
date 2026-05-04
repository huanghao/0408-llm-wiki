"""
translate_sidecar.py

从 main-body-clean.md 提取段落，调用 mdv 翻译 API，
生成 sidecar 文件 main-body-clean.translation.json。

段落 ID 格式：{section_path}-p{index}
例如：3.1.1-p0, 3.1.1-p1

用法：
  python translate_sidecar.py [--limit N] [--section "3.1"]
  --limit N       只翻译前 N 个段落（用于验证）
  --section TEXT  只翻译包含此路径的章节
"""
import re
import json
import time
import argparse
import urllib.request
import urllib.error
from pathlib import Path

TRANSLATE_API = "http://localhost:3000/api/translate"
MD_FILE = Path(__file__).parent.parent / "20250421-llama3-exp/main-body-clean.md"
OUT_FILE = MD_FILE.with_suffix(".translation.json")


def translate(text: str) -> str:
    """调用本地翻译 API"""
    payload = json.dumps({"text": text}).encode()
    req = urllib.request.Request(
        TRANSLATE_API,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
        return data["translatedText"]


def extract_paragraphs(md_text: str):
    """
    从 md 文本提取段落，返回 [(para_id, text), ...]
    只提取正文段落，跳过：标题、占位符（> **[Figure/Table]**）、空行、公式
    """
    lines = md_text.split('\n')
    
    # 当前章节路径
    section_stack = []  # [(level, number_str)]
    para_counts = {}    # section_path -> count
    
    paragraphs = []
    current_para_lines = []
    
    def flush_para():
        if not current_para_lines:
            return
        text = '\n'.join(current_para_lines).strip()
        if not text:
            current_para_lines.clear()
            return
        # 跳过占位符行
        if text.startswith('> **[Figure') or text.startswith('> **[Table'):
            current_para_lines.clear()
            return
        # 跳过公式
        if text.startswith('$$'):
            current_para_lines.clear()
            return
        # 跳过很短的行（可能是噪音）
        if len(text) < 20:
            current_para_lines.clear()
            return
        
        section_path = get_section_path()
        idx = para_counts.get(section_path, 0)
        para_id = f"{section_path}-p{idx}"
        para_counts[section_path] = idx + 1
        
        paragraphs.append((para_id, text))
        current_para_lines.clear()
    
    def get_section_path():
        if not section_stack:
            return "0"
        return ".".join(n for _, n in section_stack)
    
    for line in lines:
        # 检测标题
        heading_match = re.match(r'^(#{1,3})\s+(\d+(?:\.\d+)*)\.\s+(.*)', line)
        if heading_match:
            flush_para()
            level = len(heading_match.group(1))
            num_str = heading_match.group(2)
            # 更新章节栈
            # 弹出同级或更深的章节
            while section_stack and section_stack[-1][0] >= level:
                section_stack.pop()
            section_stack.append((level, num_str))
            continue
        
        # 空行 → 段落分隔
        if not line.strip():
            flush_para()
            continue
        
        # 普通行 → 加入当前段落
        current_para_lines.append(line)
    
    flush_para()
    return paragraphs


def load_existing(out_file: Path) -> dict:
    if out_file.exists():
        return json.loads(out_file.read_text())
    return {}


def save(out_file: Path, data: dict):
    out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="只翻译前 N 个段落")
    parser.add_argument("--section", type=str, default=None, help="只翻译包含此路径的章节")
    args = parser.parse_args()

    md_text = MD_FILE.read_text()
    paragraphs = extract_paragraphs(md_text)
    
    print(f"共提取 {len(paragraphs)} 个段落")
    
    # 过滤
    if args.section:
        paragraphs = [(pid, text) for pid, text in paragraphs if pid.startswith(args.section)]
        print(f"过滤后 {len(paragraphs)} 个段落（section={args.section}）")
    
    if args.limit:
        paragraphs = paragraphs[:args.limit]
        print(f"限制前 {args.limit} 个")
    
    # 加载已有翻译
    existing = load_existing(OUT_FILE)
    print(f"已有翻译 {len(existing)} 条")
    
    # 翻译
    new_count = 0
    for i, (para_id, text) in enumerate(paragraphs):
        if para_id in existing:
            print(f"  [{i+1}/{len(paragraphs)}] {para_id} 已有，跳过")
            continue
        
        # 截断过长文本（模型有输入限制）
        if len(text) > 500:
            text_to_translate = text[:500] + "..."
        else:
            text_to_translate = text
        
        try:
            result = translate(text_to_translate)
            existing[para_id] = result
            new_count += 1
            print(f"  [{i+1}/{len(paragraphs)}] {para_id}: {text[:50]!r} → {result[:40]!r}")
            # 保存（增量）
            if new_count % 5 == 0:
                save(OUT_FILE, existing)
        except Exception as e:
            print(f"  [{i+1}/{len(paragraphs)}] {para_id} 翻译失败: {e}")
        
        time.sleep(0.05)  # 避免过快
    
    save(OUT_FILE, existing)
    print(f"\n完成！新增 {new_count} 条，总计 {len(existing)} 条 → {OUT_FILE}")


if __name__ == "__main__":
    main()
