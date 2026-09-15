#!/usr/bin/env python3
"""为 SutdownBlog 新建文章骨架（md + 图片目录）。

用法:
  python new_post.py --category ai --slug mcp-deep-dive \
      --title "MCP详解指南" --description "一句话摘要" \
      --categories "AI,MCP" [--date 2026-09-16] [--imgdir img_mcp] [--dry-run]

行为:
  1. 创建 content/post/<category>/<YYYY-M-D>-<slug>.md   （文件名日期不带前导零）
  2. front matter 的 date 用 YYYY-MM-DD（带前导零）
  3. 创建 static/img/<imgdir>/ 目录（默认 img_<slug>）
  4. 打印后续需要填的内容清单

已存在同名文件时直接报错退出，不覆盖。
"""

import argparse
import re
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
if not (ROOT / "content" / "post").is_dir():
    ROOT = Path.cwd()

CATEGORIES = ["ai", "C++", "leveldb", "algorithm", "cs-code", "go", "book", "os",
              "log", "recreation"]


def build_front_matter(title: str, description: str, day: date, categories: list) -> str:
    cats = "\n".join(f"    - {c.strip()}" for c in categories if c.strip())
    return (
        "---\n"
        f'title:        "{title}"\n'
        f'description:  "{description}"\n'
        f"date:         {day.isoformat()}\n"
        "toc: true\n"
        "categories:\n"
        f"{cats}\n"
        "---\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Scaffold a new SutdownBlog post.")
    ap.add_argument("--category", required=True, help=f"分类目录，可选: {', '.join(CATEGORIES)}")
    ap.add_argument("--slug", required=True, help="英文短名，用于文件名与图片目录")
    ap.add_argument("--title", required=True, help="中文标题")
    ap.add_argument("--description", required=True, help="一句话摘要")
    ap.add_argument("--categories", default="", help="逗号分隔的分类标签，默认取 --category")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD，默认今天")
    ap.add_argument("--imgdir", default=None, help="图片目录名，默认 img_<slug>")
    ap.add_argument("--root", default=str(ROOT), help="仓库根目录，默认自动推断")
    ap.add_argument("--dry-run", action="store_true", help="只打印将创建的内容")
    args = ap.parse_args()

    if not re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", args.slug):
        print(f"[error] slug 必须是小写英文连字符形式，如 mcp-deep-dive：{args.slug}")
        return 1

    root = Path(args.root).resolve()
    day = date.fromisoformat(args.date) if args.date else date.today()
    cats = [c for c in args.categories.split(",") if c.strip()] or [args.category]
    imgdir = args.imgdir or f"img_{args.slug.replace('-', '_')}"

    post_dir = root / "content" / "post" / args.category
    post_path = post_dir / f"{day.year}-{day.month}-{day.day}-{args.slug}.md"
    img_path = root / "static" / "img" / imgdir

    if post_path.exists():
        print(f"[error] 文章已存在，不覆盖：{post_path}")
        return 1

    fm = build_front_matter(args.title, args.description, day, cats)

    if args.dry_run:
        print(fm)
        print(f"[dry-run] 将创建 {post_path}")
        print(f"[dry-run] 将创建目录 {img_path}")
        return 0

    post_dir.mkdir(parents=True, exist_ok=True)
    post_path.write_text(fm, encoding="utf-8")
    img_path.mkdir(parents=True, exist_ok=True)

    print(f"[ok] 文章: {post_path}")
    print(f"[ok] 图片目录: {img_path}")
    print("\n接下来要填：")
    print("  1. 开头一句话：本文从哪几个角度讲清楚什么")
    print("  2. ## / ### 分节，至少一处表格或代码块")
    print(f"  3. 配图放 static/img/{imgdir}/，正文用 /img/{imgdir}/xx.png")
    print("  4. 结尾 ## 参考链接： + 编号列表")
    return 0


if __name__ == "__main__":
    sys.exit(main())
