---
title:        "Code Agent"
description:  "本文介绍Code Agent项目的架构设计和实现原理，包括client、tools、memory、prompt、mcp和react agent等核心组件。"
date:         2025-12-01
toc: true
categories:
    - AI
    - LangChain
    - Agent
image: /images/92dbc608.jpg
---

项目链接：https://github.com/Sutdown/sutdown.github.io.git

基本分为：client，tools，memory，prompt，mcp，react agent四个部分，最终进行整合。

agent有三种基本架构：ReAct，plan and solve，reflection三种模式。

当前采用的是ReaAct架构，即边思考边行动。首先对于给出的问题利用planner进行分析规划成3-8个具体的步骤（每个步骤中包含具体的推理和行动），在每个步骤中利用现有的tools或者mcp进行执行，同时会memory上一个步骤中的结果，为了节省token，也会对整体内容进行compressor。

- **ReAct (Reasoning and Acting)：** 一种将“思考”和“行动”紧密结合的范式，让智能体边想边做，动态调整。

  核心在于循环 Thought-Action-Observation 这整个过程，在思考当前情况的时，反思上一步结果指定下一步计划，形成一个不断增长的上下文。最终能够达到：**推理让行动更具有目的性，行动为推理提供实时依据，观察则用于不断优化每次的推理**。

- **Plan-and-Solve：** 一种“三思而后行”的范式，智能体首先生成一个完整的行动计划，然后严格执行。

  核心在于 原始问题，完整计划，历史步骤和结果，当前步骤 这整个思路

- **Reflection：** 一种赋予智能体“反思”能力的范式，通过自我批判和修正来优化结果。

  先完成整个问题，再审视前面的结果进行反思，常见会通过 ”事实错误，逻辑漏洞，效率问题，遗漏信息等“ 多个常见的不同角度进行反思，对初稿进行反复修改，形成更完善的修订稿。

每个具体的步骤有一个共同的system prompt和不同的user prompt，user prompt会结合当前任务描述，上个步骤的思考行动输入观察，以及当前的执行计划生成。



## client

```txt
client
	- base_client：    抽象基类
		- send_recv：  向LLM API发送和接收消息
		- extract_txt：从响应消息中提取文本
		- chat：send_recv and extract_txt
```

## tools

核心工具模块，提供了智能体（Agent）可调用的各类工具函数，支持文件操作、代码执行、代码分析等核心功能，是智能体与外部环境交互的主要接口。

```txt
file_tools
	- create_file    创建或者覆盖文本文件
	- read_file      读取文件内容
	- list_directory 列出当前目录内容
	- edit_file      编辑文件的指定行（插入，替换，删除）
	- search_in_file 在文件中搜索文本or正则表达式，支持上下文显示
execution_tools
	- run_python 运行python代码or脚本
	- run_shell  运行shell命令
	- run_tests  运行python测试套件
	- run_linter 运行代码检查/格式化工具
code_analysis_tools
	- parse_ast              解析Python文件的AST抽象语法树提取代码信息
	- get_function_signature 提取指定函数的签名
	- find_dependencies      分析文件的依赖关系
	- get_code_metrics       获取代码度量信息
```

## prompt

提示词的基本要素在于：指令（模型需要执行的任务或命令），上下文（包含的外部信息或者额外的上下文信息），输入数据，输出指示。

另外更加具体的描述：可以在提示词中添加角色，可用的额外工具，少量的样本提示等，以期达到最好的效果。

- 角色定义
- 工具清单（tools）
- 格式规约（thought/action）
- 动态上下文（memory）
- 少样本提示

## memory

```txt
context_compressor
	- should_compress 	当对话轮数大于a时，需要压缩
	- compress 			保留最近b条对话和第1条系统prompt，其余压缩
	- _extract_key_information 提取(a-b-1)条信息的摘要，包括文件路径，执行工具，错误信息，任务完成情况四类
	- get_compression_status   获取压缩信息（原信息，压缩后信息，压缩率，节省的消息数量）
```

## mcp

Model Context Protocol 是一个开放协议，它规范了应用程序如何为 LLMs 提供上下文。可以将 MCP 想象为 AI 应用的 USB-C 端口。就像 USB-C 提供了一种标准方式，让你的设备连接到各种外设和配件，MCP 也提供了一种标准方式，让你的 AI 模型连接到不同的数据源和工具。

```txt
playwright
	- 提供全栈浏览器自动化能力，支持模拟用户在浏览器中的各类操作。
	- 比如网页导航，交互操作，截图，生成pdf等
Context7
	- 增强代码代理对长上下文的理解与处理能力，提供语义搜索和上下文压缩。
Filesystem 
	- 提供高级文件系统交互能力，扩展基础文件操作的边界。
Puppeteer
	- 基于 Chrome 内核的浏览器自动化工具，与 Playwright 功能类似但专注于 Chromium。
SQLite
	- 提供轻量级数据库交互能力，无需额外部署数据库服务。
```

```txt
MCP Client: 和单个服务器通信，建立连接，发送请求和处理响应
MCP Config: MCP服务器的配置结构，从json配置加载，保存和解析
MCP Manager: 多MCP服务器的管理
```

MCP Client：

在MCP客户端采用JSON-RPC格式发送请求，作为标准客户端和服务器之间的通信方式。

JSON-RPC是一种基于JSON的简单远程过程调用协议。MCP客户端可以同时发送多个请求，服务器响应一般为无序返回，但是JSON-RPC的id字段能够确保每个响应准确对应发起的请求。

因此对于每个MCP客户端都会开启一个**线程**，开启线程时通过同步接收来自服务器的信息，筛选其中为该**MCP客户端id**相同的信息，然后将接收到的完整信息放入**输出队列**之中。每个线程会拥有一个**互斥锁**，保证同一时刻只有一个线程能访问共享资源。

初始化时会发送一个连接请求，以此确认连接成功同时获取可用的工具列表。

MCP Manager：

config中存在所有的服务器配置信息，clients中存在所有服务器名称对应的客户端

启用和关闭服务器都是先通过config确认配置信息，之后通过client启用or关闭

_tools_cache中存储着当前可用的工具，工具中的运行函数需要在manager中写，因为是通过client中的通信运行工具的。同时当服务器信息发生变更时，需要同步更新工具信息。

## core

re-act agent 运行流程。ReAct 框架为单步单动作循环逻辑，多动作场景会通过拆分为多个步骤来处理，而非在单个步骤中包含多个 `action`。

1 通过规划器生成流程。planner中的变量包括：step_number, action, reason, completed, result

2 每个步骤都需要调用LLM API，压缩器，prompt，每个步骤具体过程如下：

- system prompt + user prompt（包括planner生成的计划完成情况，ai对上一步生成的step，上一步执行工具后的返回结果）
- 调用API执行当前步骤得到返回结果，返回结果中存在 thought, action, action_input。
- 将返回结果的action input通过action执行得到观察结果observation
- 回调实时输出步骤
- 异常情况：1 调用api错误，直接记录当前error，重新规划。2 action为已完成时，可以直接return。3 检查工具是否正确，不对则重新规划当前步骤



