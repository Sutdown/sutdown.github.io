#!/usr/bin/env python3
"""
删除 content/post 文件夹下每篇文章 front matter 中的 image: 行
"""
import re
from pathlib import Path

CONTENT_DIR = 'content/post'


def process_post(md_file: Path, pattern: str = None) -> bool:
    """处理单个文件，删除 image: 行"""
    
    # 关键词过滤
    if pattern:
        if pattern.lower() not in md_file.stem.lower():
            return False
    
    with open(md_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 检查是否有 image: 行
    if 'image:' not in content:
        return False
    
    # 删除包含 image: 的行
    new_content = re.sub(r'^.*image:.*\n?', '', content, flags=re.MULTILINE)
    
    with open(md_file, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    return True


def main():
    import sys
    
    # 检查是否传入搜索关键词
    pattern = sys.argv[1] if len(sys.argv) > 1 else None
    
    processed = []
    for md_file in Path(CONTENT_DIR).rglob('*.md'):
        if md_file.name == '_index.md':
            continue
        if process_post(md_file, pattern):
            processed.append(str(md_file))
    
    # 显示结果
    print(f"已处理 {len(processed)} 个文件：")
    for f in processed:
        print(f"  - {f}")


if __name__ == '__main__':
    main()
