---
title:        "给博客配一个会写文章的 Skill"
description:  "把写作风格、格式规范和踩过的坑沉淀成一个项目级 Skill，让 AI 按我的方式给博客写文章。"
date:         2026-09-16
toc: true
categories:
    - Blog
    - Hugo
---

这篇不是教程，是记录：我给这个博客做了个写文章的 Skill，把「写之前要交代什么」和「写成什么样算过关」都写死进去，顺手还撞上一个挺有意思的 bug。

## 起因

每次想让 AI 帮我写篇博客，开场白都长这样：文章放 `content/post/ai/` 还是 `log/`；front matter 里 `date` 要带前导零、文件名里的日期**不带**；图片路径是 `/img/...` 不是 `/static/img/...`；结尾要有 `## 参考链接：`；写完不许自己 push。

这些机械要求说完，真正的问题才浮出来——写出来的东西一眼 AI。句子都很对，也都很难看：开头铺垫三行，中间四个平均分配的小节，结尾再来个「总而言之」。我不想每次都从头吵一遍，所以干脆把它做成一个 Skill。

## Skill 就是一个目录

没别的，一个文件夹：

```
.workbuddy/skills/blog-writer/
├── SKILL.md                      主流程 + 六条硬门禁
├── references/
│   ├── writing-style.md          风格卡 + 反 AI 味黑名单
│   ├── post-conventions.md       front matter / 路径 / 图片 / 站点坑
│   └── sourcing.md               检索渠道与配图规范
└── scripts/
    └── new_post.py               建 md 骨架 + 图片目录
```

它按**三层渐进式披露**加载，这点挺聪明：平时只有 SKILL.md 的 `name` 和 `description` 挂在系统提示里（约 100 token），只有判断「这活儿跟它有关」才读正文，用到风格卡了才去读 `references/`，脚本只把结果带回上下文、代码本身不进窗口。

![Skill 的三层加载结构](/img/img_blog_writer_skill/skill-layers.svg)

## 六条门禁

流程反而是最省事的部分，真正让它「像我干的活」的是这几条不许跳过的门禁：

| 门禁 | 内容 |
| --- | --- |
| G1 主题 | 落笔前必须有主题笔记，写的是「我们讨论的那个话题」，不是关键词的通用科普 |
| G2 风格 | 动笔前读同分类旧文 ≥3 篇，套风格卡 |
| G3 来源 | 数据、版本号、论文结论必须有出处 URL，没有就降级成「个人理解」或删掉 |
| G4 配图 | 至少 1 张图，放 `static/img/<dir>/`，正文用 `/img/...` 引用 |
| G5 参考 | 结尾 `## 参考链接：` 编号列表 |
| G6 提交 | 必须起本地服务交给我预览，**我明说才 push** |

G6 是我特意加的。自动发布这种事，一次都不想试。

## 风格卡才是重头戏

风格卡我写了两页，一半是「该怎么写」，一半是「不许怎么写」。前者大致是：

- 开头一句话直给：本文从 X、Y、Z 几个角度讲清楚什么，**不写铺垫段**
- 中英术语混排，关键概念加粗，允许口语插入和自问自答（`那么这部分代码是怎么实现的？`）
- 并列对比优先用表格，技术文必须有代码实践
- 结尾不写总结段，直接接参考链接

后者是黑名单，命中即重写：「在当今…的时代」「值得注意的是」「综上所述」「不仅仅是 A 更是 B」，以及那些形态上的毛病——全篇没有一张图一个表格、每个小节篇幅完全均等、通篇零第一人称零口语的「无菌体」。

外加自检三问：有没有一句「只有我会这么写」的话？去掉加粗和表格还看得出结构吗？有没有哪段是任何人都能写出来的正确废话？

## 一次翻车：int("08")

给博客加「随心记」分区时我写了个时间解析的 helper，把 `2026-09-16 01:40` 这种字符串拆成年月日。本地跑得好好的，某次预览时页面突然变成 `Hugo Server: Error`，报的是：

```
error calling int: unable to cast "08" of type string to int:
strconv.ParseInt: parsing "08": invalid syntax
```

原因在 Go 的 `strconv.ParseInt`：base 为 0 时，前缀 `0` 会被当成**八进制**，`08`、`09` 这种合法的十进制写法就炸了。`01` 到 `07` 恰好都是合法八进制，所以这个 bug 只在 8 月和 9 月、以及每月 8 号 9 号才冒出来——藏得相当好。

修法是在转 int 前先把前导零去掉：

```go-html-template
{{- $ms := replaceRE "^0+" "" (index $dp 1) -}}
{{- $m  := cond (eq $ms "") 0 (int $ms) -}}
```

更有意思的是后半段：我打开 `parse-time.html` 一看，前导零明明早就处理掉了，可错误还在。折腾半天才反应过来——**hugo server 拿的是旧模板的缓存**，等它 rebuild 完再请求就 200 了。所以「代码明明修好了还报旧错」的时候，别急着怀疑人生，先重启服务。

## 顺带：封面这件事

这次预览还确认了一件事，博客现在**没有文章封面图**了。列表页七张卡片、文章详情页，全站唯一的 `<img>` 是侧边栏头像。

是之前主动删的（commit `3c02650`）：`header.html` 被简化成只剩 `partialCached "article/components/details"`，`static/img/anime/` 那 80 张图也一并清了。只是清得不够干净——列表卡片上还挂着 `has-cover`：

```go-html-template
<article class="{{ if $image.exists }}has-image{{ else }}has-cover{{ end }}">
```

封面没了以后 `$image.exists` 恒为 false，于是每张卡片都挂着一个空 class。我把编译后的 CSS 拉下来 grep 了一遍，`has-cover` 出现 0 次，说明它早就没有任何样式对应了，直接删掉 `else` 分支了事。

现在 front matter 里的 `image` 字段完全无效，所以新文章一律不写，视觉表达只能靠正文里的图——这也是 G4 那条门禁现在更硬的原因。

## 参考链接：

1 [Equipping agents for the real world with Agent Skills](https://anthropic.com/engineering/equipping-agents-for-the-real-world-with-agent-skills)

2 [The Complete Guide to Building Skills for Claude](https://resources.anthropic.com/hubfs/The-Complete-Guide-to-Building-Skill-for-Claude.pdf)

3 [Removing leading 0 from number in string, int function error - Hugo Discourse](https://discourse.gohugo.io/t/creating-a-tpl-templates-test-case-that-uses-variables/7915)

4 [Golang Quirk: Number-strings starting with "0" are Octals](https://scripter.co/golang-quirk-number-strings-starting-with-0-are-octals/)
