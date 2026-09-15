# 文章落盘规范（SutdownBlog 约定 + 历史坑）

## 一、front matter 模板（照抄这个对齐方式）

```toml
---
title:        "中文标题"
description:  "一句话摘要，说清这篇文章讲什么"
date:         2026-09-16
toc: true
categories:
    - AI
    - 子主题
---
```

- `title` / `description` 后面跟空格对齐到同一列（旧文风格），不是硬性要求但保持一致
- `date`：**带前导零** `YYYY-MM-DD`；而**文件名不带前导零** `YYYY-M-D-slug.md`。两者不一致是正常的，别改
- `toc: true` 必写，目录靠它生成
- `categories` 缩进 4 空格，可以多个（旧文通常 1–3 个）
- **不写 `image` 字段**——站点已无封面图机制，写了也不渲染（见文末"封面现状"）
- 早期文章有 `author: "Sutdown"`，**新文章一律不加**

## 二、路径与命名

```
content/post/<分类目录>/<YYYY-M-D>-<英文slug>.md
static/img/<图片目录>/<xxx>.png|jpg
```

分类目录（大小写敏感，照抄）：`ai` `C++` `leveldb` `algorithm` `cs-code` `go` `book` `os` `log` `recreation`

slug 用小写英文连字符：`mcp-deep-dive`、`week10`、`raft-consensus`

## 三、图片

- 存到 `static/img/<目录>/`，目录名惯例：`img_cs146s`、`images`、`images_c++`、`images_stl`、`images_go`…
  新文章建议 `img_<slug>`（用 `new_post.py` 自动生成）
- 正文引用路径是 **`/img/<目录>/<文件>`**，**不是** `/static/img/...`（项目根目录的 `fix_paths.py` 就是修这个的）
- alt 文字可以很短（旧文里大量是 `![1](/img/...)`），但**描述性 alt 更好**：`![hash table](/img/images/11.jpg)`
- 图的大小：宽度压到 800–1200px 即可，别塞 4K 原图

## 四、参考链接（结尾必须有）

```markdown
## 参考链接：

1 [MCP (Model Context Protocol)，一篇就够了。 - 知乎](https://zhuanlan.zhihu.com/p/29001189476)

2 [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol)
```

- 标题就是 `## 参考链接：`（带冒号），编号后一个空格再写链接
- 条目之间空一行
- 一般 3–8 条，太少说明检索不足，太多说明没筛选

## 五、站点机制速查（别踩）

- **封面现状（2026-09-16 确认）：全站没有封面图**。`header.html` 只剩
  `partialCached "article/components/details" . .RelPermalink`，`static/img/anime/`（80 张）与
  `about.jpg` 已于 2026-09-02 删除（commit `3c02650`）。列表页与详情页的文章卡片只有
  `.article-details`（分类角标 + 标题 + subtitle + 元信息），全站唯一的 img 是侧边栏头像。
  → **文章的视觉表达只能靠正文内的配图**，所以 G4「至少 1 张图」更重要
- **菜单**：菜单项定义在各页面 front matter 的 `menu.main` 里，**不要动 `config/_default/menu.toml`**（会重复）
- **文章的 URL 用「标题」，不是文件名**：产出目录形如 `public/p/<标题>/`。
  例：`title: "MCP详解指南"` → `public/p/mcp详解指南/`；
  `title: "SutdownBlog 视觉重构记录"` → `public/p/sutdownblog-视觉重构记录/`。
  **不是** `public/post/log/<文件名日期-slug>/`。
  标题里的空格会变成连字符、大写转小写，中文原样保留（Hugo 直接输出中文目录名，
  浏览器自己编码，不需要手动 `urlencode`）。
  → 预览给用户的 URL 就是 `http://localhost:1313/p/<标题小写化后的形式>/`；
  不确定时直接 `ls public/p/ | grep <关键词>` 看实际生成的目录名。
  这是**全站既有行为**（81 篇一致），别误判成路径 bug。
- **验证文章渲染别只看字节数**：`hugo server` 对新建文章有重建延迟，
  刚生成时请求可能返回一个 13KB 左右的空壳页。等 rebuild 完再请求，
  或直接看 `public/p/<标题>/index.html` 的大小（正常在 40KB+）。
- **自定义 CSS** 只改 `assets/scss/custom.scss`；**自定义 JS 只改 `assets/ts/custom.ts`**
  （`assets/js/custom.js` 会 404，别走这条路）
- **正文排版不要动**：中文排版方案（justify + 首行缩进 / 改行宽行高）已被否决两次，不要再改
- 静态资源 `static/` 下改动即时生效；改了 scss/ts 需要 hugo server 重新编译
- **hugo server 有缓存**：删掉的 partial、错误状态会残留，验证前重启服务

## 六、预览与发布

```bash
# 本地预览（给用户 URL 让他自己开浏览器）
cd E:/project/SutdownBlog && hugo server --bind 0.0.0.0 --port 1313 --disableFastRender

# 提交（仅当用户明确要求）
git add -A && git commit -m "add post: <标题>" && git push origin master
```

- 推送走 **SSH**；HTTPS + 代理会被 reset
- origin 里写的 `Sutdown/SutdownBlog` 是旧名，实际仓库 `Sutdown/sutdown.github.io`，GitHub 自动重定向，push 正常
- push `master` → GitHub Actions 构建 → 发布到同仓库 `gh-pages` → https://sutdown.github.io
- 本地一般不需要手动 `hugo` 构建，`public/` 由 CI 覆盖
