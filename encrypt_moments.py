#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把「仅我可见」的随心记加密成 data/moments-private.json

为什么需要这一步：
    博客是纯静态站，没有后端，任何写进 HTML 的内容都等于公开。
    所以私密条目不能出现在 content/ 里 —— 它们只以密文形式进仓库，
    浏览器拿到密文后，要靠访客输入的密码才能解开。

用法：
    python encrypt_moments.py                 # 交互式输入密码（输入时不回显）
    python encrypt_moments.py -p 你的密码      # 直接指定（注意会留在 shell 历史里）

源文件：private/moments.md（已加进 .gitignore，明文永不入库）
        格式与公开随心记一致：

            ::: 2026-09-22 23:10
            正文，支持 Markdown 与内联 HTML（比如 <span class="emoji-plain">🌙</span>）
            :::

输出：data/moments-private.json —— 只有密文、salt 和校验值，可以放心提交。
      改内容或换密码，重跑一次即可（旧密文会被覆盖）。

密码不需要写进仓库：文件里只留 salt 与明文的 SHA-256 校验值，
前端拿这两个东西验证「密码对不对」，但反推不出密码本身。
"""

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "private" / "moments.md"
OUT = ROOT / "data" / "moments-private.json"

ITERATIONS = 120000
SALT_BYTES = 16
FORMAT_VERSION = 1


# ---------------------------------------------------------------- 加密

def derive_key(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, ITERATIONS, dklen=32)


def keystream(key: bytes, length: int) -> bytes:
    """用 HMAC-SHA256(key, counter) 逐块拼出流密钥，与前端 WebCrypto 实现保持一致。"""
    out = bytearray()
    counter = 0
    while len(out) < length:
        out += hmac.new(key, counter.to_bytes(4, "big"), hashlib.sha256).digest()
        counter += 1
    return bytes(out[:length])


def xor_bytes(data: bytes, key: bytes) -> bytes:
    ks = keystream(key, len(data))
    return bytes(a ^ b for a, b in zip(data, ks))


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


# ---------------------------------------------------------------- 极简 Markdown

def render_markdown(text: str) -> str:
    """够用就好的渲染器：段落 / 行内代码 / 粗体 / 斜体 / 链接，其余原样保留。

    刻意不装第三方库 —— 私密条目大多是短句，内联 HTML（emoji-plain、sticker）
    本来就允许直接写，所以这里只补最基础的 Markdown 语法。
    """
    blocks = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = para.strip()
        if not para:
            continue
        # 已经是块级 HTML 就原样输出，不再包 <p>
        if re.match(r"^<(p|div|ul|ol|blockquote|figure|h[1-6]|pre)\b", para, re.I):
            blocks.append(para)
            continue
        inline = markdown_inline(para)
        blocks.append("<p>%s</p>" % inline)
    return "\n".join(blocks)


def markdown_inline(text: str) -> str:
    # 行内代码先占位，避免里面的 * _ 被当成强调
    codes = []

    def stash(match):
        codes.append(match.group(1))
        return "\x00%d\x00" % (len(codes) - 1)

    text = re.sub(r"`([^`]+)`", stash, text)

    text = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", r'<a href="\2" target="_blank" rel="noopener">\1</a>', text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", text)

    def restore(match):
        idx = int(match.group(1))
        return "<code>%s</code>" % codes[idx] if idx < len(codes) else ""

    return re.sub(r"\x00(\d+)\x00", restore, text)


# ---------------------------------------------------------------- 解析源文件

def parse_source(text: str) -> list:
    entries = []
    for raw in text.split(":::"):
        block = raw.strip("\r\n")
        if not block:
            continue
        lines = block.split("\n")
        head = lines[0].strip()
        body = "\n".join(lines[1:]).strip() if len(lines) > 1 else ""

        dt = parse_time(head)
        if not dt:
            print("  ! 跳过（首行不是可解析的时间）：%s" % head[:30])
            continue

        entries.append({
            "time": dt,
            "html": render_markdown(body),
        })

    entries.sort(key=lambda e: e["time"], reverse=True)
    return entries


def parse_time(head: str):
    """支持 2026-09-22 23:10 / 2026-09-22 / 2026年9月22日 三种写法。"""
    import datetime

    head = head.strip()
    m = re.match(r"^(\d{4})-(\d{1,2})-(\d{1,2})(?:\s+(\d{1,2}):(\d{2}))?", head)
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
        hh = int(m.group(4)) if m.group(4) else 12
        mm = int(m.group(5)) if m.group(5) else 0
        try:
            return datetime.datetime(y, mo, d, hh, mm)
        except ValueError:
            return None

    m = re.match(r"^(\d{4})年(\d{1,2})月(\d{1,2})日", head)
    if m:
        try:
            return datetime.datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 12, 0)
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------- 主流程

def main() -> int:
    ap = argparse.ArgumentParser(description="加密私密随心记")
    ap.add_argument("-p", "--password", help="密码（不填则交互式输入）")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过二次确认（脚本/非交互环境用）")
    ap.add_argument("--src", default=str(SRC), help="源文件，默认 private/moments.md")
    ap.add_argument("--out", default=str(OUT), help="输出文件，默认 data/moments-private.json")
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print("找不到源文件：%s" % src)
        print("先在 private/moments.md 里写好条目，格式：")
        print("    ::: 2026-09-22 23:10")
        print("    正文")
        print("    :::")
        return 1

    entries = parse_source(src.read_text(encoding="utf-8"))
    if not entries:
        print("没解析出任何条目，检查一下 ::: 分块与时间行。")
        return 1

    password = args.password or os.environ.get("MOMENTS_PASSWORD") or getpass.getpass("设置/输入密码（不回显）：")
    if not password:
        print("密码不能为空。")
        return 1
    if args.password and not args.yes and sys.stdin.isatty():
        confirm = getpass.getpass("再输一次确认：")
        if confirm != password:
            print("两次不一致，放弃。")
            return 1

    plaintext = json.dumps(
        [{"time": e["time"].strftime("%Y-%m-%d %H:%M"), "html": e["html"]} for e in entries],
        ensure_ascii=False,
    ).encode("utf-8")

    salt = os.urandom(SALT_BYTES)
    key = derive_key(password, salt)
    payload = {
        "v": FORMAT_VERSION,
        "iter": ITERATIONS,
        "salt": b64(salt),
        "check": b64(hashlib.sha256(plaintext).digest()),
        "data": b64(xor_bytes(plaintext, key)),
        "count": len(entries),
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("已加密 %d 条 → %s" % (len(entries), out.relative_to(ROOT)))
    print("密文可以提交；明文 %s 已被 .gitignore 排除，不会进仓库。" % src.relative_to(ROOT))
    print("改内容或换密码，重跑本脚本即可。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
