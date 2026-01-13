---
title:        "Langchain到简单Agent"
description:  "本文介绍从LangChain基础到构建简单Agent的实现过程，包括LLM调用、Prompt模板、工具集成等关键概念。"
date:         2025-10-06
toc: true
categories:
    - AI
    - LangChain
    - LLM
image: /images/d92d9921.jpg
---


## 0 AI相关基础概念

- **LangChain** 是一个用于开发由大型语言模型（LLMs）驱动的应用程序的框架。
- **生成式AI：**使用大模型进行支持，在大量原始未标记的数据基础上对于深度学习模型进行预训练，从而让机器能够理解语言甚至图像，能根据需要自动生成内容。
- **大模型的训练阶段：**预训练（提升本身的知识量），SFT（Supervised Fine-Tuning监督微调，专注于选择某一方面），RLF（Reinforcement Learning with Human Feedback基于人类反馈的强化学习）
- **prompt**：使用大模型时，向模型提供的一些指令或者问题，这些指令作为模型输入，引导模型产生必要的输出。



## 1 基础LLM和聊天模型调用

### Chat API调用

LangChain 将底层 API 调用抽象为统一的接口（如`ChatOpenAI`、`LLM`类），屏蔽了不同模型 API 的差异（如阿里云、百度、OpenAI 的接口格式不同）。

> 当前支持的模型：[聊天模型 | 🦜️🔗 LangChain 框架](https://python.langchain.ac.cn/docs/integrations/chat/)

### APP KEY

以通义千问为例，app key可以从[阿里云百炼大模型服务平台](https://www.aliyun.com/product/bailian)中获取，存在一定的免费额度，app key建议新建一个`.env`文件存储，加上`load_dotenv()`预先加载即可。

### Prompt 模板

`ChatPromptTemplate` 是 LangChain 中用于管理对话提示词的核心工具，专为多角色（如 `system`、`user`、`assistant`）对话场景设计，其优点主要体现在**结构化、灵活性、复用性**和**生态集成能力**上。

它的核心在于增加了提示词的灵活性，将其从”静态字符串“升级成了”可配置，可动态生成，可集成的结构化组件“。在需要处理多角色度化，动态调整提示词，复用模板或者集成langchain的其它功能时，能够显著提升代码质量。

> 官方文档：[提示模板 | 🦜️🔗 LangChain 框架](https://python.langchain.ac.cn/docs/concepts/prompt_templates/)

```python
import os
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
os.environ["DASHSCOPE_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")

def main():
    chat = ChatTongyi(model="qwen3-max")

    prompt_template = ChatPromptTemplate.from_messages([
        ("system", "你是{role}，专业领域为{field}。回答需符合{style}风格，控制在{word_limit}字左右。"),
        ("user", "这个{concept}的作用是什么，担任什么样的角色？")
    ])
    params1 = {
        "role": "科普博主",
        "field": "人工智能",
        "style": "口语化、通俗易懂",
        "word_limit": "30",
        "concept": "通义千问"
    }
    params2 = {
        "role": "AI工程师",
        "field": "大语言模型",
        "style": "专业、技术化",
        "word_limit": "50",
        "concept": "通义千问"
    }

    print("=== 场景1：向普通用户解释 ===")
    messages1 = prompt_template.format_messages(**params1)  # 用参数填充模板
    response1 = chat.invoke(messages1)
    print(response1.content)

    print("\n=== 场景2：向开发者解释 ===")
    messages2 = prompt_template.format_messages(**params2)  # 复用同一模板，仅换参数
    response2 = chat.invoke(messages2)
    print(response2.content)

if __name__ == "__main__":
    main()
    
"""
=== 场景1：向普通用户解释 ===
通义千问是个AI助手，能回答问题、写故事、编程，帮你搞定各种任务！

=== 场景2：向开发者解释 ===
通义千问是大语言模型，用于理解与生成人类语言，支持问答、创作、推理等任务，扮演智能助手角色。
"""
```



## 2 简单Agent

### 基础概念

**AI agents**：基于LLM的能够自主理解，自主规划决策，执行复杂任务的智能体。

**AI agents流程**

- 规划（planing）：将任务分为子任务，对任务进行思考反思  ---  给一个合适的prompt
- 记忆（memory）：记住执行任务的上下文，有助于更好的理解当前任务 --- 短期记忆 长期记忆
- 工具（tools）：为智能体配备工具AI，比如计算器，搜索工具等

### 核心逻辑

`Agent` 的核心逻辑遵循 REACT 框架（Reason→Act→Observe→React），这是一种典型的链式流程。

```txt
用户输入 → 记忆组件（加载历史对话） → Agent（分析是否调用工具） → 工具（执行任务） → Agent（处理结果） → 记忆组件（保存新对话） → 输出回答
```

### LangChain.tools

LangChain 的 Tools 是连接大模型与外部功能的核心组件。工具是一种封装函数及其模式的方式，以便可以将其传递给聊天模型。使用@tool装饰器创建工具，该装饰器简化了工具创建过程，支持以下功能：

- 自动推断工具的**名称**、**描述**和**预期参数**，同时支持自定义。
- 定义返回**工件**（例如图像、数据框等）的工具
- 使用**注入的工具参数**从模式（从而从模型）中隐藏输入参数。

[工具 | 🦜️🔗 LangChain 框架](https://python.langchain.ac.cn/docs/concepts/tools/)

### 旅游助手agent

这段代码实现了一个基于 **LangChain** 和 **通义千问（ChatTongyi）** 的智能旅行助手，具备联网搜索、预算计算和多轮对话记忆功能。

- 谷歌搜索工具（google_search）：通过 Serper API 调用 Google 搜索，获取实时信息如天气、景点、政策等。[Serper API  key](https://serper.dev/logs)，也可以用其余搜索工具，采用相应的api key即可。
- 预算计算工具（calculate_budget）：输入“天数,每日预算”格式，自动计算旅行总费用。
- 对话记忆（ConversationSummaryMemory）：保留历史对话摘要，帮助模型理解上下文（如“刚才的城市”）
- 自定义提示词模板（ChatPromptTemplate）：定义系统行为和工具使用逻辑，让模型决定何时调用工具。
- 智能体（Agent）初始化：通过 `initialize_agent()` 将 LLM、工具和记忆整合为具备推理和工具调用能力的对话体。
- 交互循环（chat_loop）：实现命令行多轮聊天界面，支持实时输入与退出（exit/退出）。

```python
import os
import requests
from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationSummaryMemory
from langchain.agents import initialize_agent, AgentType
from datetime import datetime

load_dotenv()

"""tool"""
@tool("谷歌搜索",
      description="使用 Google 搜索获取实时信息，如‘北京天气’、‘东京景点推荐’、‘签证政策’等")
def google_search(query: str) -> str:
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        return "未设置 SERPER_API_KEY 环境变量"

    url = "https://google.serper.dev/search"
    headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
    payload = {"q": query, "num": 5}

    try:
        res = requests.post(url, headers=headers, json=payload)
        data = res.json()

        if "organic" not in data:
            return "未找到搜索结果。"

        results = [
            f"{r.get('title')}: {r.get('snippet')} ({r.get('link')})"
            for r in data["organic"][:3]
        ]
        return "\n\n".join(results)

    except Exception as e:
        return f"搜索出错: {e}"

@tool("计算旅行预算",
      description="计算旅行总预算，输入格式为“天数,每日预算”（如“3,500”），返回总预算结果",
      return_direct=False)
def calculate_budget(input_str: str) -> str:
    """计算旅行总预算，输入格式为“天数,每日预算”（如“3,500”）"""
    try:
        days, daily_cost = input_str.split(",")
        total = int(days) * int(daily_cost)
        return f"总预算：{days}天×{daily_cost}元/天={total}元"
    except ValueError:
        return "输入格式错误，请使用“天数,每日预算”（如“3,500”）"

tools = [google_search, calculate_budget]

"""llm"""
llm = ChatTongyi(model="qwen3-max", api_key=os.getenv("DASHSCOPE_API_KEY"))

"""memory"""
# 摘要记忆，节省token
memory = ConversationSummaryMemory(
    llm=llm,
    memory_key="chat_history",
    return_messages=True
)

"""prompt"""
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个智能旅行助手，拥有以下能力：
1. 当用户提出涉及实时信息（如天气、景点、签证等）的问题时，请调用【谷歌搜索】工具。
2. 当用户要求预算计算时，调用【计算旅行预算】工具。
3. 其它情况（例如行程建议、交通说明）直接回答。
4. 保持上下文一致性（chat_history），比如“刚才的城市”指之前提到的城市。

工作流程：
- 先分析用户问题是否需要工具：需要则调用，不需要则直接回答
- 调用工具时严格遵循工具的输入格式
- 结合历史对话（chat_history）理解上下文（如“刚才说的城市”指之前提到的城市）
- 用中文简洁回答，避免冗余
"""),
    MessagesPlaceholder(variable_name="chat_history"),  # 插入历史对话
    ("user", "{input}"),                               # 当前用户输入
    ("ai", "{agent_scratchpad}")                       # Agent思考过程（自动填充）
])

"""agent"""
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,  # 带记忆的聊天型Agent
    memory=memory,  # 使用记忆
    # verbose=True,  # 输出详细日志
    agent_kwargs={
        "prompt": prompt,  # 绑定提示词模板
        "system_message": prompt.messages[0].prompt.template,  # 系统提示
        "extra_prompt_messages": [
            MessagesPlaceholder(variable_name="chat_history")
        ]
    },
    handle_parsing_errors=True,  # 忽略解析错误
    return_intermediate_steps=False
)

"""多轮对话"""
def chat_loop():
    print("智能旅行助手")
    print("提示：输入任意问题与我对话，例如：'我打算去东京玩三天' 或 '帮我查一下北京天气'")
    print("输入 'exit' 或 '退出' 可结束对话。\n")

    while True:
        user_input = input("用户：").strip()
        if user_input.lower() in ["exit", "quit", "退出", "bye"]:
            print("助手：好的，下次再见，祝你旅途愉快！👋")
            break

        if not user_input:
            continue  # 忽略空输入

        current_time = datetime.now().strftime("%Y年%m月%d日 %H:%M")
        query = f"{user_input}\n（当前时间：{current_time}）"

        try:
            response = agent.invoke({"input": query})
            result = response["output"] if isinstance(response, dict) and "output" in response else response
            print(f"助手：{result}\n")
        except Exception as e:
            print(f"❌ 出错：{e}\n")


if __name__ == "__main__":
    chat_loop()
```



## 后话

LangChain 实现智能体的推理、工具调用与记忆，但其核心架构仍是「线性链式执行」。在复杂多轮推理、长上下文管理和并发场景下存在一定缺陷。langchain运行过程中会出现官方提示当前功能**仍然可用**，但**推荐迁移到更新的模块或框架**。

**LangGraph** 是 LangChain 团队推出的下一代框架，它基于“有状态计算图（StateGraph）”思想，将每个步骤建模为节点，支持显式状态管理、并发执行、持久记忆与可视化调试，更适合用于创建实际需要的智能体。

[为什么选择 LangGraph？ - LangChain 框架官网](https://github.langchain.ac.cn/langgraph/concepts/high_level/)



## 参考文档

[简介 | 🦜️🔗 LangChain 框架](https://python.langchain.ac.cn/docs/introduction/)

