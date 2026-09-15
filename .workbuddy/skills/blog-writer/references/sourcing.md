# 资料检索与配图规范

## 一、检索：至少交叉 3 类来源

| 层级 | 渠道 | 用来支撑什么 |
| --- | --- | --- |
| 一手 | 论文（arXiv、ACL/NeurIPS、Google Scholar）、官方文档、RFC、源码仓库 | 定义、数字、结论、版本号 |
| 讲解 | 课程视频（Stanford CS146S 等）、技术演讲、官方 tutorial | 直觉解释、图示灵感 |
| 工程 | 高质量博客、知乎专栏、GitHub issue / 源码注释、Stack Overflow | 踩坑细节、代码写法 |

做法：

1. 先用 `WebSearch` 打捞候选（中英文各一轮），再 `WebFetch` 精读 2–4 篇核心来源
2. 每条素材立刻记进主题笔记：`标题 — URL — 它支撑文章哪一段`
3. **数字和结论必须能指回来源**；找不到出处的断言，降级成"个人理解"或删掉
4. 引用他人原文/观点时，在正文里**显式署名**（这是江舟的既有习惯，见 leveldb 系列）
5. 优先近 2 年的来源；经典论文可放宽，但要标明年份

## 二、配图：按内容需要自选来源（用户已确认）

判断顺序：

1. **架构图 / 原理图 / 流程图** → 自己画
   - 简单示意用 `show_widget` 画 SVG 看效果，定稿后导出到 `static/img/<dir>/`
   - 或直接写 SVG 文件存 `static/img/<dir>/xxx.svg`，正文 `![说明](/img/<dir>/xxx.svg)`
   - mermaid 也可以用（确认主题是否支持；不支持就转 SVG）
2. **官方文档里的图、需要真实界面 / 运行结果** → 自己截图
   - 用 `agent-browser` skill 打开页面截图，或跑代码后截终端输出
   - 截图要裁掉无关区域，只留说明问题的那块
3. **现成的示意图（如经典的哈希表结构图）** → 网络搜索下载合规图片
   - 存进 `static/img/<dir>/`，正文引用并在参考链接里注明出处
   - 避开明显有版权的水印图、付费图库图

## 三、配图硬性要求

- **每篇至少 1 张图**，长文 3–6 张，别堆砌
- 图放在"讲到它的那一段"的**紧跟位置**，不要全堆在开头或结尾
- 图的上方或下方**写一句说明**：这张图在讲什么、该看哪里
- 文件命名有意义：`week10_pipeline.png`、`mcp_handshake.png`，不要 `1.png 2.png`（旧文的 `1.jpg` 是历史遗留，新文不学）
- 宽度压到 800–1200px（Pillow 已可用）：
  `C:/Users/xiangfei/.workbuddy/binaries/python/versions/3.13.12/python.exe -c "from PIL import Image; ..."`

## 四、Python 环境提醒

- 直接用 `C:/Users/xiangfei/.workbuddy/binaries/python/versions/3.13.12/python.exe`（自带 Pillow 12）
- `~/.workbuddy/binaries/python/envs/default/Scripts/python.exe` **不存在**，别用
- Git Bash 下 `curl -o` 写文件常报 exit 23 / 假 0 字节；判断下载是否成功请 `curl -s URL > file` 后看文件大小
