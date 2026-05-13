---
name: fix-category-page-style
overview: 在 list.html 中添加 body-class 定义，使 .template-list CSS 选择器生效，让分类页文章列表和分页器样式与首页一致。
todos:
  - id: add-body-class
    content: 在 list.html 开头添加 body-class 定义使 template-list 样式生效
    status: completed
---

## 用户需求

用户要求从分类页点进去后，文章列表展示和下方分页器的风格与首页保持一致。

## 根因

`list.html` 模板没有定义 `{{ define "body-class" }}` 块，导致 body 元素没有 `template-list` class，`custom.scss` 中已有的 `.template-list` 样式无法生效。分页器样式是全局的，应该已经生效。

## 技术方案

### 修改内容

仅需修改一个文件：`e:/project/SutdownBlog/layouts/_default\list.html`

### 具体改动

在 `list.html` 文件开头添加一行：

```html
{{ define "body-class" }}template-list{{ end }}
```

参考 `archives.html` 第 1 行的写法。

### 已有样式（无需修改）

`custom.scss` 中已有完整的样式定义：

- 第 167-190 行：`.template-list .article-list--compact` 卡片样式（悬浮、边框、阴影）
- 第 441-508 行：`.pagination` 和 `.page-link` 分页器全局样式

### 影响范围

仅影响 `list.html` 模板渲染的页面（分类页、标签页等 taxonomy list 页面），不影响首页、归档页等其他页面。