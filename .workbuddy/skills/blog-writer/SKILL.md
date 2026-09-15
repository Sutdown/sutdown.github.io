---
name: blog-writer
description: 为 SutdownBlog（Hugo + Stack 主题，站点 sutdown.github.io）按江舟本人的写作风格生成技术文章并落盘预览。当用户说「写篇博客 / 用 blog-writer 写 X / 把刚讨论的这个话题整理成文章」，或给出文章关键词、要求把某个讨论过的话题写成文章时使用。流程覆盖主题捕获 → 多方检索 → 风格对齐 → 成稿（front matter / 配图 / 参考链接）→ 本地 hugo server 预览 → 用户明确确认后才提交推送。
agent_created: true
---

## Overview

这个 skill 只服务 `E:/project/SutdownBlog` 这一个仓库。目标不是"写一篇通顺的技术文"，而是
**写一篇读起来像江舟自己写的文章，并且严格贴合你们刚刚讨论过的主题**。

两件事最重要，任何一步都不能省：

1. **主题要严丝合缝**——写的是"我们讨论的那个话题"，不是"这个关键词的通用科普"。
2. **风格要像本人**——动笔前必须读同类旧文，套风格卡，反 AI 味。

## 硬门禁（跳过即视为违规）

| 门禁 | 内容 |
| --- | --- |
| G1 主题 | 落笔前必须有一份主题笔记（`.workbuddy/tmp/topics/<slug>.md`）或明确记录的核心论点列表 |
| G2 风格 | 动笔前必须读过同分类旧文 ≥3 篇，并读完 `references/writing-style.md` |
| G3 来源 | 事实性断言（数据 / 论文结论 / 版本号 / 性能指标）必须有出处 URL，无出处就改成"个人理解"或删掉 |
| G4 配图 | 文章至少 1 张图（架构 / 流程 / 界面 / 对比），放在 `static/img/<dir>/`，正文用 `/img/...` 引用 |
| G5 参考 | 结尾必须有 `## 参考链接：` 编号列表 |
| G6 预览 | 必须起本地服务并把 URL 交给用户，**不得**在用户确认前自动 push |

## Workflow

### Step 0 · 主题捕获（最重要）

**触发方式（用户已确认：显式指令 + 自动兜底）**

- 用户说「开始主题：xxx」→ 立刻创建 `.workbuddy/tmp/topics/<slug>.md`，并开始边聊边追加。
- 用户没说「开始」→ 先扫当前会话上下文；再扫 `.workbuddy/tmp/topics/` 已有笔记；仍找不到就用
  `conversation_search` 检索历史会话。找到后向用户复述一遍核心论点再动笔。
- 只有检索不到任何相关讨论时，才询问用户"这个主题的核心观点是什么"。

**主题笔记格式**（边讨论边更新，不要等到写稿才回忆）：

```markdown
# <主题名>

- 状态：讨论中 / 待写 / 已发布
- 目标分类：ai
- 核心论点：
  - （写"江舟的观点"，不是百科定义）
- 存疑 / 待确认：
- 我的口径要求：（比如"不要写成教程，要写踩坑"）
- 素材线索：
  - [论文/视频/博客] 标题 — URL — 可用点
```

### Step 1 · 多方检索

至少 3 类来源交叉，避免只抄一家：

- 一手：论文（arXiv / 官方 paper）、官方文档、RFC、源码
- 讲解：课程视频、技术演讲、官方文档的 tutorial
- 工程：高质量博客、知乎专栏、GitHub issue / 源码注释

每条素材当场记录 URL 与"它支撑文章哪一段"，直接进参考链接池。
参考 `references/sourcing.md`。

### Step 2 · 风格对齐

必读：`references/writing-style.md`（硬规则 + 反 AI 味黑名单）。
再读同分类旧文 ≥3 篇（用 Glob 按分类目录找），重点看：开头怎么切入、标题怎么断、图表怎么用、
结尾怎么收。写完后**对照黑名单自检一遍**再落盘。

### Step 3 · 成稿落盘

用脚手架建骨架（也可手写，但 front matter 必须一致）：

```bash
C:/Users/xiangfei/.workbuddy/binaries/python/versions/3.13.12/python.exe \
  .workbuddy/skills/blog-writer/scripts/new_post.py \
  --category ai --slug <english-slug> --title "中文标题" \
  --description "一句话摘要" --categories "AI,子主题"
```

规则详见 `references/post-conventions.md`。要点：

- 路径 `content/post/<分类>/YYYY-M-D-<slug>.md`（文件名日期**不带**前导零，front matter 的 date **带**前导零）
- **不写 `image` 字段**：2026-09-02 起站点已移除封面图机制（`static/img/anime/` 已删，`header.html` 只剩 details partial），写了也不渲染。配图只放正文内
- 目录用 `toc: true` + `##` / `###` 两级，不要四级以上
- 结尾 `## 参考链接：` + `1 [标题](URL)` 编号列表

### Step 4 · 配图

用户已确认：**按内容需要自选来源**（架构图/原理图自己画或截官方文档，真实界面自己截图，
实在没有再搜网络合规图）。每张图在上下文一句说明，图注写在图的上方或下方。
详细规范见 `references/sourcing.md`。

### Step 5 · 本地预览

```bash
cd E:/project/SutdownBlog && hugo server --bind 0.0.0.0 --port 1313 --disableFastRender
```

把 `http://localhost:1313` 给用户，**让他自己浏览器打开**（他看不到内置预览面板，不要用 present_files 代替）。
汇报：文件路径、分类、字数、配图数量、参考链接条数。
然后停在这里等反馈——改稿循环可以反复进行。

### Step 6 · 提交（仅当用户明确说"提交 / 推送"）

```bash
git add -A && git commit -m "..." && git push origin master
```

- 走 SSH（`git@github.com:Sutdown/SutdownBlog.git`），**不要**用 HTTPS + 代理，会被 reset
- push 到 `master` 后由 GitHub Actions 构建发布到 `gh-pages`，站点 https://sutdown.github.io
- 用户没说提交就绝不提交；用户说"先放着"就保持工作区改动不动

## 参考文件

- `references/writing-style.md` — 风格卡 + 反 AI 味黑名单（Step 2 必读）
- `references/post-conventions.md` — front matter / 路径 / 图片 / 环境坑（Step 3 必读）
- `references/sourcing.md` — 检索渠道与配图获取（Step 1、Step 4）
- `scripts/new_post.py` — 建文章骨架 + 图片目录

## 环境速查

- Hugo：`E:/app/hugp/hugo_extended_0.154.3_windows-amd64/hugo`（v0.154.3 extended）
- Python：`C:/Users/xiangfei/.workbuddy/binaries/python/versions/3.13.12/python.exe`（自带 Pillow）
- 构建参考：152 页 / 311 静态文件 / 约 6 秒（2026-09-16 实测；页数会随文章增加，只作量级参考）
- 文章 URL 是**标题**不是文件名：`/p/<URL编码后的标题>/`，详见 `references/post-conventions.md`
