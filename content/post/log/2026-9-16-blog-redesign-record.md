---
title:        "SutdownBlog 视觉重构记录"
description:  "从 Stack 主题的默认样子到藕粉手绘风，记录这一年里三个阶段的改造：换了什么、为什么换、踩了哪些坑。"
date:         2026-09-16
toc: true
categories:
    - Blog
    - Hugo
---

这个博客用的是 Hugo + Stack 主题，2026 年 1 月建站，到现在改过三轮。这篇把三个阶段的改动整理成一条线：换了什么、为什么换、踩了哪些坑。内容偏记录性质，不做教程。

三个阶段大致是这样分的：

| 阶段 | 时间 | 重点 |
| --- | --- | --- |
| 搭建期 | 2026-01 | 框架落地、内容迁移、页面骨架 |
| 内容期 | 2026-05 ~ 07 | 学习笔记批量入库，前端顺手调整 |
| 重构期 | 2026-09 | 视觉体系重建：配色、手绘风、独立页、随心记 |

![三个阶段与配色演变](/img/img_page_redesign/redesign-phases.svg)

## 一、搭建期：框架落地

### 1.1 从零到能跑

2026 年 1 月 11 日那天的提交密度很高（十几条），基本是一天之内把站点跑起来：`Initial commit` 之后紧跟着 `index`、`comment`、`post image`、`remover 标签 and add toc`，然后是内容批量入库。

技术选型上没有纠结太久。之前用宝塔面板 + Typecho 搭过一个动态博客（2023 年那篇《宝塔面板+typecho》记的就是那个），这次改用 Hugo 是因为它编译成纯静态，扔到 GitHub Pages 就完事，不需要维护服务器。

### 1.2 内容结构化

建站初期最集中的工作是给文章补 front matter。1 月 13 日有一条提交叫 `catrgory`（拼错了，无伤大雅），一次性给 7 篇 AI 笔记补上分类字段。这种批量整理是必要的——**分类、标签、目录这些结构信息如果不从第一天就规范化，后面会越积越乱**。

同期还删了主题自带的示例分类 `content/categories/example-category/`，以及做了两处页面调整：

- `content/page/about/index.md` 重写（49 行）
- `content/page/travelling/index.md` 新建，配 `assets/icons/train-subway.svg` 图标

到这一步，站点骨架成型：`content/post/` 放文章，`content/page/` 放独立页，四个栏目 Home / Archives / About / Travelling 挂在导航上。

### 1.3 评论与部署

`comment` 这条提交接入了评论系统，最终用的是 Waline（serverURL 指向一个 Vercel 实例）。部署走 GitHub Actions，push 到 `master` 触发构建，产物推到同仓库的 `gh-pages` 分支。

这里有个细节值得记一下：**源码和站点产物放在同一个仓库的两个分支里**（`master` 存源码、`gh-pages` 存构建结果），配置自洽，不需要额外的部署仓库。

## 二、内容期：笔记入库与布局微调

5 月主要是往仓库里搬学习笔记，CS146S 的周记从 week1 一直补到 week9，中间夹着 LangChain、MCP 这些 AI 主题的整理。前端改动零散但有几个有意义：

### 2.1 分类挂件与页脚

`8359bff` 这条「前端调整」给 `custom.scss` 加了 70 行，同时动了两处布局：`layouts/partials/footer/footer.html` 补了 26 行，`layouts/partials/widget/categories.html` 微调。这个阶段还是零敲碎打，没有形成体系——**改一个地方就写一段 CSS，样式之间没有统一的设计变量**。

### 2.2 归档页与索引调整

5 月 14 日连续几条 `adjust`、`archives`、`前端调整` 提交，主要在调归档页的展示。同期还处理过一次 git 冲突（`1cff054` fix leftover git conflict markers in posts），说明当时有过一次不太顺利的合并。这些痕迹后来都清掉了。

### 2.3 关于主题目录

`52428df` 这条提交叫 `ignore themes directory`，把 `themes/` 排除出版本控制。原因是主题通过 Hugo Modules 引入（`config/_default/module.toml` 里声明 `github.com/CaiJimmy/hugo-theme-stack/v3`），`themes/Stack/` 是个空目录，真正的内容在模块缓存里。**这意味着改主题只有两条路：覆盖 layouts/partials 里的同名文件，或者用 CSS 覆盖。** 后面重构期的所有改动都遵循这个原则。

7 月只有一条 `add`，基本是间歇期。

## 三、重构期：视觉体系重建

9 月是改动最密集的时段，40 条提交。这不是零散的调整，是一次完整的视觉体系重建。

### 3.1 配色：藕粉 + 米白纸张

起点是 `c9ca501`「藕粉配色 + 渐变封面 + 微交互，整体视觉焕新」。核心决策是**换掉主题默认的灰蓝 `#34495e`，改用藕粉 `#d49ba8`**，页面背景从纯白改成微暖的 `#fdf8f9`。

选藕粉不是随手挑的。之前的头像是浅灰紫 / 浅粉紫调（主色约 `#D0D0E0`、`#E0D0E0`），饱和度极低、明度偏高，属于莫兰迪浅调。配色如果跟着这个走，就得保持低饱和 + 高明度，避免霓虹色和高饱和撞色。藕粉正好落在这个区间。

同期修了一个隐蔽问题：主题没有定义 `--accent-color-rgb`，而 `custom.scss` 里有几处 `rgba(var(--accent-color-rgb), .1)` 依赖它。变量缺失时不会报错，只是那些背景色**静默失效**——侧边栏 hover 高亮就属于这种情况。补上定义之后才正常。

### 3.2 手绘春日风

`ea195f1`「手绘春日风格改造」把配色推进到风格层，有四个动作：

1. 背景从 `#fdf8f9` 再调成米白纸张色 `#fdf9f2`，增加薄荷绿 / 淡黄 / 浅棕三个点缀色
2. 标题字体换成手写体「霞鹜文楷」，通过 CDN 加载
3. 卡片和挂件改用不规则边框（wobbly border），模拟手绘的不规则感
4. 樱花元素统一加浅棕墨水描边，分隔线改成手绘波浪

这里踩了一个值得记的坑：**字体规则不能写在 SCSS 里**。Hugo 压缩 CSS 时会把带空格的字体名 `"LXGW WenKai"` 处理成小写无引号的形式，字体就失效了。最后把字体规则内联写进 `layouts/partials/head/custom.html` 的 `<style>` 标签里，绕过压缩器，问题解决。

### 3.3 装饰体系

`1c48adc` 到 `fdf1422` 之间是一连串装饰相关的提交，最终保留下来的是：

| 装饰 | 状态 |
| --- | --- |
| 樱花花瓣飘落 | 保留 |
| 标题花饰 | 保留 |
| 花枝横幅 + 底部小树 + 花枝分隔线 | 保留 |
| 导航栏花枝、云朵飘过 | 保留 |
| 页面底部左右溜达的小猫咪 | 做了 → 重画 → 移除 |
| 鼠标移动花瓣轨迹 | 做了 → 移除 |
| 卡片鼠标跟随光斑 | 保留 |
| 深色模式樱花 | 保留 |

有两条是做了又拆的。猫咪前后改了三次（新增 → 重画更可爱 + 放慢速度 → 移除蹲坐小猫），最后只保留云朵和导航栏花枝。鼠标花瓣轨迹则是被主动删掉的——它跟着光标撒花瓣，刚做出来觉得挺有意思，实际用起来干扰阅读，尤其影响文章页。

**这里有个判断：装饰的价值不在数量，在于它是否干扰内容。** 背景元素（花瓣飘落、云朵）跟内容是分层的，不抢注意力；跟随鼠标的元素（花瓣轨迹、光斑）跟内容同层，就会让人分心。重构期最后的取舍基本是按照这条线来的。

还有一处实现细节：花瓣轨迹的 CSS 里原本写的是 `animation: petal-trail var(--dur-slow) var(--ease-out)-out forwards` —— `var(--ease-out)-out` 是拼接错误，这条动画从写下去就没生效过。删的时候才发现。

### 3.4 设计 token 落地

`727c272`「建立设计 token 并统一列表与正文排版」是个分水岭。在此之前所有样式都是散着写的（比如各处硬编码 `border-radius: 12px`），这条提交之后统一收口成变量：

```scss
--radius-wobbly      手绘不规则圆角
--radius-wobbly-sm   小尺寸版本
--shadow-lift        悬浮阴影
--dur-base           基础动画时长
--ease-out           缓动曲线
```

这一步的实际收益不是「看起来整齐」，而是**后面改视觉时只需要动变量值**。比如从米白调到纸张色、从灰蓝调到藕粉，都是改几个变量的量级，不用去各处找硬编码。

### 3.5 功能补全

同期还补了几个功能性组件：

- **图片灯箱**：点击正文图片放大查看
- **返回顶部进度环**：按钮边缘的圆环跟随滚动进度
- **移动端底部导航**：手机端底部固定栏
- **代码块美化**：语言角标 + 复制按钮
- **阅读进度条**：顶部横向进度（仅文章页）

这些的实现都遵循了「尊重 `prefers-reduced-motion`」和「`IntersectionObserver` 不可用时直接显示」两条兜底原则，避免在低端设备或特殊设置下出现内容不可见。

### 3.6 封面：从有到无

封面这块改了很多轮，值得单独说，因为它反映了一个反复的决策过程：

| 版本 | 做法 | 结果 |
| --- | --- | --- |
| 1 | 主题默认，无封面 | 列表页略显单调 |
| 2 | 客户端随机动漫图 API | 每次刷新都变，列表和详情不同图，否决 |
| 3 | 本地 16 张图随机 | 重复太多 |
| 4 | 本地 80 张图按哈希固定映射 | 每篇永久对应同一张，可用 |
| 5 | 「完整图 + 同图模糊铺底」双图层 | 解决竖图横图比例不一导致的裁切/留白 |
| 6 | 全部删除 | 最终方案 |

第 4 版的技术点是 `mod (hash.FNV32a .RelPermalink) 80` —— 用页面路径的哈希取模，保证每篇文章永久对应同一张图，列表页和详情页天然一致，不会随机变化。

第 5 版解决的是**图片比例问题**。那 80 张图宽高比从 0.71（竖）到 1.78（横）都有，单一 `object-fit` 无法兼顾：用 `cover` 会裁掉画面，用 `contain` 会留白。最后的方案是双层——底层用 `cover` + `blur(14px)` 铺满，上层用 `contain` 完整显示，两层叠加看不出留白。

第 6 版全部删除（`3c02650`）。**理由其实很简单：这些封面跟文章内容没有任何关系**，纯粹是为了让列表页不空。而且 `slug` 哈希这个机制本身不透明——你没法通过修改文章来控制封面。删掉之后列表页靠排版和分类角标撑住，反而更干净。

清得不太干净，`header.html` 简化后列表卡片上还挂着 `has-cover` class。因为 `$image.exists` 恒为 false，每张卡片都挂着一个空 class。后来 grep 编译后的 CSS 确认这个选择器出现 0 次，才删掉 `else` 分支。

## 四、独立页：三个页面的改造

About / Links / Travelling 三个独立页一开始跟整体风格不搭。这个问题的诊断过程比修复过程更有意思。

### 4.1 诊断：找到共同的病根

最初的判断是「字体和间距不对」，但动手前先做了一件事——把所有背景色变量拉出来对比：

| 变量 | 值 |
| --- | --- |
| 页面背景 `--body-background` | `#fdf9f2` 米白 |
| 卡片背景 `--card-background` | `#ffffff` 纯白 |

问题出在这。文章页是一张纯白卡片浮在米白页面上，但**文章内容足够密**（正文、代码块、配图把卡片填满了），白底看起来像「纸张」；独立页内容稀疏，同样一张纯白卡片铺在那里就是一大片空白。

所以「风格不搭」的表象，底层是**内容密度差异把同一个颜色衬出了不同观感**。判断的时候要分清是「颜色本身不对」还是「颜色在这个场景下不对」，这次是后者。

### 4.2 `--card-padding`：理解横向对齐的钥匙

Stack 主题横向排版全靠一个变量 `--card-padding`（响应式取 20 / 25 / 30px）：

```scss
// 内容靠 padding 内缩
.article-content { padding: 0 var(--card-padding); }

// 但某些块要顶到满宽，用等量负 margin 反向拉出去
.article-content blockquote,
.article-content figure,
.article-content .highlight,
.article-content pre {
  margin-left: calc(var(--card-padding) * -1);
  margin-right: calc(var(--card-padding) * -1);
  width: calc(100% + var(--card-padding) * 2);
}
```

这是**故意的设计**：代码块、引用块、图片比正文宽出两侧 30px，形成通栏出血，把阅读节奏切成几段。

理解这一条，所有横向对齐问题都有了解法：**要顶满宽就用它的负值，要收在容器里就把负 margin 归零。**

### 4.3 三次返工：友链卡底色

这是整个改造里返工最多的部分，因为「不好看」是模糊反馈，只能靠排除法逼近。

| 版本 | 值 | 结果 |
| --- | --- | --- |
| 第一版 | `#ffffff` 纯白 | 白得发空，卡片像贴上去的补丁 |
| 第二版 | `transparent` 全透明 | 和页面糊成一片，favicon 失去承托 |
| 定稿 | `#fffdf9` 暖白 | 只比页面亮一档，卡片立住了 |

中间那版是判断失误。用户说「不需要白色背景」，我理解成「去掉背景色」，设成了 `transparent`。结果卡片和页面同为米白，只有一圈描边在撑场面，看着像没画完。

![卡片底色的三段折中：三档之间只差 1~2 个色阶，观感差别很大](/img/img_page_redesign/card-background-tiers.svg)

三档之间色值差 1~2 个色阶，观感差得很远。**底色这件事没有「随便挑一个」的余地。**

### 4.4 favicon 那圈白的真相

还有个反馈是「头像框没底色发虚」。第一反应是去找卡片背景的锅，改了半天没用，最后决定去下载 favicon 二进制看看到底长什么样。结果是：

`favicon.im` 返回的是**透明背景的 SVG**。GitHub 的图标就是一个 `fill="#24292E"` 的纯深色形状，**本身没有任何白色底衬**。

所以那圈白根本不是装饰，是**功能性承托**——深色图标直接压在米白页面上会糊。搞清楚之后改法就明确了：保留白底 `background: #fff`，深色模式下给 `rgba(255,255,255,0.94)`，既承托又不刺眼。

**别猜，去取证。** 一个 4KB 的 SVG 比盯着样式改半天有用。

### 4.5 About 页：从分块到扁平

About 页试过两个方向。第一版是按 `h3` 把小节切成卡片（`custom.ts` 里写了切块逻辑，配 `.about-card` 样式），做完发现**卡片框把本来就不多的内容切得更碎了**，用户直接否掉。第二版改成扁平展示，去掉所有分块容器，小节标题用手写体 + 左侧藕粉竖线标识。

这跟前面「装饰是否干扰内容」是同一个判断：**内容稀疏的时候，容器越多越显空。**

### 4.6 一个幽灵字段

清理 front matter 时发现 `content/page/about/index.md` 里写着：

```yaml
layout: "about"
```

但 `layouts/` 下根本没有 `about.html`（`travelling.html` 倒是有）。查 Hugo 的单页模板查找顺序：

```
/layouts/TYPE/LAYOUT.html        ← page/about.html（不存在）
/layouts/SECTION/LAYOUT.html     ← page/about.html（不存在）
/layouts/TYPE/single.html        ← page/single.html（不存在）
/layouts/SECTION/single.html     ← page/single.html（不存在）
/layouts/_default/single.html    ← 命中
```

一路回落到底，用的是默认单页模板。更关键的是：**About 页的样式从来不是靠这个字段挂钩的**，而是靠 `single.html` 里输出的 `page-{{ .ContentBaseName | urlize }}` 生成的 class（即 `.page-about`）。所以这个 `layout` 字段从头到尾没起过作用，删掉即可。

如果遇到「写了但好像没生效」的配置，值得花几分钟去查找顺序表对一遍。

### 4.7 Travelling 页

Travelling 最后没有做成内容页，而是改成了一个**跳转过渡页**（`layouts/page/travelling.html`），配转圈樱花动画。这个决定挺务实的——既然没有内容可写，与其留一个空页面，不如说明清楚它是干什么的。

## 五、随心记（Moments）

`8a851cb` 新建了「随心记」分区，用来放那些不成篇的碎片想法。

### 5.1 数据格式

按月分文件，条目用 `:::` 分隔：

```markdown
::: 2026-09-16 01:40

正文，支持多段 / 图片 / 行内代码 / 链接

:::
```

模板用 `split .RawContent ":::"` 切块，首行解析时间，**解析失败则跳过该块**（防止内容里出现 `:::` 把文件吞掉）。

### 5.2 三个坑

1. **`int "08"` 报错**：Go 的 `strconv.ParseInt` 在 base 为 0 时把前缀 `0` 当**八进制**，`08`、`09` 这种合法的十进制写法会炸。`01` 到 `07` 恰好都是合法八进制，所以这个 bug 只在 8 月、9 月和每月 8 号 9 号才冒出来。修法是转 int 前先去掉前导零。

2. **`trim " "` 在管道里把整串吃成空**：`{{ .body | plainify | replaceRE ... | trim " " }}` 结果为空，换成 `strings.TrimSpace` 才正常。

3. **月文件 date 落在今天会被 Hugo 判为 future**，整个月不渲染。用 `hugo list future` 可以查出来，已在配置里开 `buildFuture = true`。

### 5.3 阅读页净化

`574033a` 这条提交做了件有意思的事：**独立页恢复装饰，阅读页去掉装饰**。

原本逻辑是判断有没有 `.article-page`，有就认为是阅读页，关掉花瓣、小树、云朵。但独立页（About / Links / Travelling）走的也是 `single.html`，同样输出 `.article-page`，于是被误判成阅读页，装饰全没了。

修法是在 `single.html` 的 body-class 块里加标记：

```go-html-template
{{ if ne .Section "post" }}page-standalone{{ end }}
```

`post` 下面才是真阅读页。JS 改成：

```ts
var isReading = !!document.querySelector('.article-page');
var isStandalone = !!document.querySelector('.page-standalone');
var decorEnabled = !isReading || isStandalone;
```

这个改动的意义是**把「读文章时该安静」和「逛页面时可以活泼」两种状态分开**。读文章需要专注，装饰会干扰；独立页是展示性的，装饰恰好补充了内容密度上的不足。

## 六、两次栽在 CSS 权重上

重构期有两次印象比较深的翻车，都属于「代码写对了但没生效」这一类。

### 6.1 选择器权重输给主题

说明区的代码块顶穿了卡片描边。查出来的原因是主题给 `.article-content .highlight` 加了通栏出血（就是 4.2 节那套负 margin），在说明卡里溢出 30px。

我一开始写的是 `.links-note-body .highlight`，改完**完全没生效**。算权重就明白了：

| 选择器 | 权重 (a,b,c) |
| --- | --- |
| `.article-content .highlight`（主题） | (0, 2, 1) |
| `.links-note-body .highlight`（我的） | (0, 2, 0) |

差在最后一位。CSS 权重是**从左往右逐列比，前一位赢了后面就不用看**，所以 (0,2,0) 无论写多少条都赢不了 (0,2,1)。最后上了 `!important`。

顺带纠正一个我原先的误解。我以为「两个 `!important` 相撞就看源码顺序」，查 MDN 才发现不准确：

> When two important declarations from the same origin and layer apply to the same element, browsers select and use the declaration with the highest specificity. **Only if the selectors had the same specificity would source order matter.**

也就是说 `!important` 相撞时**先比权重，权重相同才比顺序**。

### 6.2 被 grep 的上下文骗了

验证的时候 grep 到这么一条：

```css
.article-content .highlight pre {
  margin-block-end: 0 !important;
  margin-left: calc(var(--card-padding) * -1) !important;
}
```

当时紧张了一下——同样是 `!important`，那不就变成「比权重再看顺序」了吗？赶紧去查它挂在谁身上。用正则按 `{}` 成对解析之后发现，真实选择器是：

```css
.article-content .gitlab-embed-snippets .file-holder.snippet-file-content
```

**GitLab 嵌入片段**，跟代码块半点关系没有。是 `grep -A 1` 抓上下文时，把两条无关规则的首尾连起来读了。

minified CSS 里一条规则体可以很长、选择器分组可以很多，grep 给出的「上下文」基本都是假的。要看清楚只有一条路：**用正则把 `选择器 { 声明体 }` 成对切出来**。

## 七、样式覆盖的两条原则

改主题这一年，实际可用的手段只有两条，它们对应不同的场景：

### 7.1 优先改 SCSS 变量

`assets/scss/custom.scss` 会被主题的 `style.scss` 在最后一行 import，所以在里面覆盖变量最省事。改配色、圆角、阴影、动画时长都属于这一类，**改一处全局生效，不会漏**。

```scss
:root {
  --accent-color: #d49ba8;
  --body-background: #fdf9f2;
  --radius-wobbly: 12px 18px 14px 20px / 18px 12px 20px 14px;
}
```

### 7.2 覆盖 layouts 要整组写

改结构（不只改样式）就得覆盖 `layouts/` 下的同名文件，因为主题是 Hugo Modules 引入的，改不到模块缓存里的源文件。

这里有个陷阱：**主题的高权重选择器会让你只写一条的覆盖静默失效**。比如真图分支那段，主题的规则是：

```css
.article-page .main-article .article-header .article-image img {
  max-height: 50vh;
  object-fit: cover;
}
```

如果只写 `.article-image img { object-fit: contain }`，权重 (0,1,1) 对 (0,4,1)，**会被完全覆盖，且不会报错**。必须写一整组同等或更高权重的选择器，并显式声明 `max-height: none`。

验证方法是下载编译后的 CSS，看 `object-fit` 规则的**出现顺序**——`contain` 必须排在主题所有 `cover` 之后。

## 八、自定义 JS 的唯一入口

这个坑比较隐蔽，记一下：主题通过 `js.Build` 把 `assets/ts/custom.ts` 编译成指纹化的 `/ts/custom.HASH.js` 自动加载。**所有自定义 JS 必须写在那个 .ts 文件里。**

之前我走的另一条路：写 `assets/js/custom.js`，再用 `layouts/partials/footer/custom.html` 静态引用 `/js/custom.js`。这个路径会 **404**，自定义 JS 从来没执行过。改成 `.ts` 单一入口之后才生效。

现在 `custom.ts` 承担的工作大致是：客户端重排（比如把说明区收进卡片）、装饰开关判断、挂件交互。**模板里往 `<script>` 注入值要用 `safeJS`** 防止二次转义。

## 九、这一轮改动的几个判断

把重构期的决策归纳一下，这几条是反复用到的：

1. **装饰不能干扰内容**。背景元素（花瓣、云朵）跟内容分层，可以留；跟随鼠标的元素（花瓣轨迹、光斑）跟内容同层，要谨慎。
2. **容器越多越显空**。内容稀疏时加卡片框是负优化，About 页分块被否掉就是这个原因。
3. **改视觉先改变量**。有设计 token 之后，调整成本从「各处找硬编码」降到「改几个值」。
4. **别猜，去取证**。favicon 白底那件事猜了两轮都错，下载一个文件就查明了。
5. **涉及「页面里已经有什么」的判断，先去构建产物里数一遍**。我曾想当然实现了一个装饰横栏（`branch-garland`），验证时发现构建产物里 0 处、SCSS 里 0 条样式——纯属臆造，全部撤回。

## 参考链接：

1 [Specificity - CSS | MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/Specificity)

2 [!important CSS keyword - CSS | MDN](https://developer.mozilla.org/en-US/docs/Web/CSS/Reference/Values/important)

3 [层叠、优先级与继承 - MDN Web Docs](https://developer.mozilla.org/zh-CN/docs/Learn_web_development/Core/Styling_basics/Handling_conflicts)

4 [模板查找顺序 - Hugo 中文文档](https://hugo.opendocs.io/templates/lookup-order)

5 [Single Content Template - Hugo Docs](http://markdblackwell.gitlab.io/hugo-docsite-blue/templates/content)
