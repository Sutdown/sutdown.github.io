---
title:        "基于langchain源码剖析常见用法"
description:  "本文通过剖析LangChain源码，深入讲解其常见用法和核心组件的工作原理。"
date:         2025-12-04
toc: true
categories:
    - AI
---

## 声明

有一定的编程基础，该篇属于学习笔记，如有理解不正确的地方欢迎各位指出。

## langchain组件

在正式阅读langchain源码之前，先期望对于langchain有一个初步的理解，langchain往上为agent应用开发的基本框架，往下则是基于LLM，以及一些其它的工具实现。

```text
Models 主要涵盖大预言模型，为不同的基础模型提供统一接口，便于自由切换不同的模型
	- LLMs
	- Chat Models
	- Embeddings 对文档转化成向量，总结等
Prompts 支持各种自定义模板
	- templates
	- few-shot examples
	- examplate selector
	...
Indexs
	- docement loaders 文档加载器（如何从不同的数据源中加载数据，比如email，pdf，markdown，latex等）
	- text splitters 文档拆分器（当输入数据长度过大时的处理）
	- vectorstores 向量存储器（数据的搜索其实就是向量关系的匹配，即向量运算）
	- retrievers 检索器（对接向量存储器）
	...
memory
	- ConversationBufferMemory 所有聊天记录
	- ConversationBufferWindowMemory 最近k轮聊天记录
	- ConversationTokenBufferMemory 最近token条记录
	- ConversationSummaryMemory 只存储一个用户和机器人之间聊天摘要
	- ConversationSummaryBufferMemory 
	- VectorStored-BackedMemory 通过向量的方式存储，匹配最相似k组对话
Chains
	- LLMChain
	- RouterChain
	- SimpleSequentialChain
	- SequentailChain
	- TransformChain
	...
Agents
    action agents
    plan-and-execute agents
	- conversational
	- openAIfunctions
	- self ask with search
	...
```

llangchain处理文本的流程。

embedding之前需要split。原因在于：

1 embedding本身有最大token的限制

2 检索时，一般是能匹配到其中的某个chunk，有助于检索的精确性

```
Document Loader 文档加载
    ↓
Raw Documents
    ↓
TextSplitter 接收Document，输出更细的document
    ↓
Document(chunks)
    ↓
Embeddings 文本转化成向量
    ↓
Vectors（.from_documents)
    ↓
Vectorstore (FAISS/Chroma/Pinecone)
    ↓
Retriever(vectorstore.as_retriever)
```



## Agent

### agent核心

Agent 模块的核心函数用于创建 Agent、配置工具和执行任务。

```
用户输入 → AgentExecutor（执行器） → Agent（决策器） → LLM（推理） + Tools（工具） → 输出结果
```

#### initialize_agent

快速初始化 Agent 的便捷函数，根据指定的工具、LLM 和 Agent 类型（如 `AgentType.ZERO_SHOT_REACT_DESCRIPTION`）创建 `AgentExecutor`（负责运行 Agent 的核心循环，协调 Agent 的决策、工具调用和结果处理。是用户调用 Agent 时的主要入口，封装了 “思考 - 行动” 的迭代过程）。

[langchain源码分析-agent模块整体介绍【12】 - 知乎](https://zhuanlan.zhihu.com/p/656779738)

```python
from langchain_classic.agents import initialize_agent, AgentType
from langchain_openai import OpenAI
from langchain_core.tools import Tool

llm = OpenAI(temperature=0)
tools = [Tool(name="Calculator", func=lambda x: eval(x), description="用于计算数学问题")]
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
)
```

#### create_react_agent/ create_structured_chat_agent

针对特定 Agent 类型（比如reAct）的创建函数，更灵活地配置提示词（Prompt）和解析器。

```python
from langchain_classic.agents import create_react_agent
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(...)  # 自定义 ReAct 提示词
agent = create_react_agent(llm, tools, prompt)
```

**`AgentExecutor.invoke`**

执行 Agent 并获取结果的核心方法，接收用户输入并返回最终答案。支持同步（`invoke`）和异步（`ainvoke`）调用。

```python
result = agent.invoke({"input": "3的平方加上5的立方等于多少？"})
print(result["output"])  # 输出计算结果
```

#### 经典agent体系

```python
# ReAct模式，多步骤推理
AgentType.ZERO_SHOT_REACT_DESCRIPTION
# 基于zere-shot reAct，引入memory，适合聊天型工具调用
AgentType.CONVERSATIONAL_REACT_DESCRIPTION
# 工具描述比ReAct更严格，输出强制JSON
AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION
# OpenAI Function Calling流行后退出的agent，适合生产环境
AgentType.OPENAI_FUNCTIONS
# 长任务拆分agent，适合大任务，多步骤自动化执行
from langchain.agents import PlanAndExecute
# 存在历史基于，有向量库，主要用于知识库，企业文档AI助手之类的RAG系统
ConversationalRetrievalChain  
```



### model

#### 静态模型

常见的聊天模型，封装了对应模型和初始化。

create_agent会创建一个可执行的状态图（StateGraph），核心目标在于构建一个能自主调用工具的智能体执行流程，本质是封装了 “语言模型调用→工具调用→结果处理→循环迭代” 的逻辑，最终返回一个编译后的`StateGraph`（状态图）。

init_chat_model是一个**统一入口**，通过参数动态选择模型提供商和模型，实现 “多对一” 的灵活调用。它可以根据输入的 `model` 或 `model_provider` 参数，自动初始化对应的具体模型类，无需手动显式导入不同的类。通常可以和bind_tools一起使用。

ChatOpenAI是针对特定模型提供商的直接封装（如 OpenAI、Anthropic 等），属于 “一对一” 的绑定关系。

```python
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    # Kwargs passed to the model:
    temperature=0.7,
    timeout=30,
    max_tokens=1000,
)

model = ChatOpenAI(
    model="gpt-5",
    temperature=0.1,
    max_tokens=1000,
    timeout=30
    # ... (other params)
)

# 创建智能体
graph = create_agent(
    model="anthropic:claude-sonnet-4-5-20250929",
    tools=[check_weather],
    system_prompt="You are a helpful assistant",
)

# 调用智能体
inputs = {"messages": [{"role": "user", "content": "what is the weather in sf"}]}
for chunk in graph.stream(inputs, stream_mode="updates"):
    print(chunk)
```

#### 动态模型

 [`@wrap_model_call`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) 装饰器创建中间件，主要作用是拦截模型调用过程进行拦截、修改或增强，例如实现重试逻辑、请求 / 响应改写、错误处理等。它允许开发者在不修改核心模型调用逻辑的前提下，灵活扩展模型交互的行为。

```python
@wrap_model_call
def fallback_model(request, handler):
    try:
        return handler(request)  # 尝试主模型
    except Exception:
        # 切换到备用模型
        request = request.override(model=fallback_model_instance)
        return handler(request)

agent = create_agent(
    model=basic_model,  # Default model
    tools=tools,
    middleware=[fallback_model]
)
```



### invoke

`invoke` 函数是 Agent 与外部交互的主要入口，负责协调工具调用、模型推理和流程控制等核心流程。

```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
response = llm.invoke("介绍一下 LangChain")
```



## tools

Tool 是代理（Agent）可调用的“函数能力单元”，帮助大模型与外部世界交互。

| 模块路径                             | 是否包含 Tool          | 说明                                                         |
| ------------------------------------ | ---------------------- | ------------------------------------------------------------ |
| `langchain.tools`                    | 有基础抽象、少量工具   | 数学&计算类工具，Calculator、Shell、Python REPL 等           |
| `langchain_community.tools`          | ⭐**绝大多数内置工具**  | python执行工具，搜索引擎工具、浏览器、文件类工具、代码、网络请求，文档加载工具等 |
| `langchain_community.agent_toolkits` | 工具包（多 Tool 打包） | SQL、JSON、GSheets、Spark、Vector Store                      |



### 定制Tool

#### 简单场景

```python
from langchain.tools import Tool

def get_user(user_id: str):
    return {"user": user_id, "name": "Zhou Jiang"}

user_tool = Tool(
    name="get_user_info",
    func=get_user,
    description="Get user info by user_id"
)
```

#### 复杂参数

```python
from langchain.tools import StructuredTool
from pydantic import BaseModel

class UserInput(BaseModel):
    user_id: int
    verbose: bool

def fetch_user(user_id: int, verbose: bool):
    return {"id": user_id, "detail": verbose}

user_tool = StructuredTool.from_function(
    func=fetch_user,
    args_schema=UserInput,
    name="fetch_user",
)
```



## prompt

### 基础prompt

基于 PromptTemplate 的基础文本模板生成，属于通用文本提示模板。

```python
# 基础文本生成
prompt = PromptTemplate.from_template("Write a {length} poem about {topic}.")
prompt_text = prompt.format(length="short", topic="spring")
```

### 多角色prompt

基于 ChatPromptTemplate 的**多角色**对话支持，属于对话式消息提示模板。

原生支持**上下文**插入，无需手动拼接。


```python
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

system_template = "You are a helpful assistant that translates {input_language} to {output_language}."
human_template = "Translate: {text}"

# AIMessagePromptTemplate：AI 回复（用于示例）（一般很少使用）
system_prompt = SystemMessagePromptTemplate.from_template(system_template)
human_prompt = HumanMessagePromptTemplate.from_template(human_template)
chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

result = chat_prompt.format_prompt(
    history="", # 历史消息
    input_language="English",
    output_language="French",
    text="Hello world"
).to_messages()
```



### FewShotPromptTemplate

基于 `FewShotPromptTemplate` 的少样本学习能力。当遇到提示词长度过长或者相关性不足时，langchain给出了示例选择器（Example Selector）------- LengthBasedExampleSelector、MaxMarginalRelevanceExampleSelector 和 NGramOverlapExampleSelector。

- **LengthBasedExampleSelector**：根据**提示词长度**动态选择示例，确保最终生成的提示词不超过预设的最大长度（避免超出模型上下文限制）。
- **MaxMarginalRelevanceExampleSelector**：基于**语义相关性**和**多样性**选择示例，优先保留与输入语义相似且彼此差异较大的示例（平衡相关性和多样性）。
- **NGramOverlapExampleSelector**：基于**N-gram 重叠度**（文本表层特征的重合度）选择示例，优先保留与输入共享更多短语或词汇的示例。

```python
from langchain.prompts import PromptTemplate
from langchain.prompts.example_selector import NGramOverlapExampleSelector

examples = [
    {"query": "如何学习Python编程", "response": "建议从基础语法开始..."},
    {"query": "Python入门教程", "response": "推荐使用官方文档..."},
    {"query": "Java学习方法", "response": "先掌握面向对象概念..."}
]

example_prompt = PromptTemplate(
    input_variables=["query", "response"],
    template="Query: {query}\nResponse: {response}"
)

# 初始化N-gram选择器（2-gram，选择1个最佳示例）
selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    k=1,
    ngram_size=2
)

# 输入：与“Python学习”相关的查询
input = {"query": "Python学习路径"}
selected_example = selector.select_examples(input)
print(selected_example[0]["query"])  # 输出："如何学习Python编程"（与输入共享更多2-gram）
```



## chain

> [!CAUTION]
>
> LangChain 的 `Chain` 在 LangGraph 中不再作为核心概念出现，已经被更现代的 “Graph（图）” 和 “Runnable（可运行单元）” 体系替代。
>
> agent内部是依赖chain实现的，也就是封装好的chain，agent相当于在chain外面包装了一层“让模型自动决定下一步”的机制。但是chain太过于限定结构，不太灵活，agent的动态流程也比较难以控制，因此在新版langchain中提出用runnable+function calling替代agent，正式的写法迁移到langgraph中。

chain是核心组件之一，主要用于将多个组件（比如model，prompt，检索器，tools等）按照特定逻辑串联，形成一个可执行的工作流。

在 Chain 被调用前，需完成初始化并配置核心参数，确保运行时所需的组件和上下文就绪。

- `memory`：可选的记忆组件（如 `BaseMemory`），用于保存上下文状态。
- `callbacks`：回调管理器或处理器列表，用于监控运行过程（如日志、追踪）。
- `tags` 和 `metadata`：用于标记和附加链的元数据，便于追踪和分类。

Chain 接收用户输入后，需进行预处理，确保输入格式符合要求，并整合记忆组件的上下文。运行的核心阶段，实际执行链的逻辑（如调用子链、LLM 等），并通过回调机制记录运行状态。



### LLMChain

基础链，一般是将提示词模板和语言模型组合，直接调用模型生成结果。

```python
from langchain_classic.chains import LLMChain
chain = LLMChain(prompt=prompt, llm=llm)
result = chain.invoke({"question": "LangChain 有哪些特点？"})
```



### SequentialChain

顺序链，前一个链的输出作为后一个链的输入，适用于处理多步任务

另还有TransformChain: 对chains之间的输入和输出进行处理，便于chains之间进行数据传输。支持自定义的转换函数。可以理解为顺序链的升级版。

SimpleSequentialChain: 每个步骤都有一个单一的输入/输出，并且一个步骤的输出是下一步的输入

```python
prompt1 = ChatPromptTemplate.from_messages([
    ("system", "将中文翻译成英文。"),
    ("human", "{chinese_text}")
])
chain1 = LLMChain(llm=ChatOpenAI(), prompt=prompt1, output_key="english_text")

prompt2 = ChatPromptTemplate.from_messages([
    ("system", "将英文翻译成法语。"),
    ("human", "{english_text}")
])
chain2 = LLMChain(llm=ChatOpenAI(), prompt=prompt2, output_key="french_text")

sequential_chain = SequentialChain(
    chains=[chain1, chain2],
    input_variables=["chinese_text"],
    output_variables=["french_text"]
)
result = sequential_chain.run(chinese_text="我喜欢编程")
```



### ConversationChain

 LangChain 中用于实现多轮对话功能的链（Chain），其核心功能是结合语言模型（LLM）和记忆组件（Memory），让对话能够保留上下文信息，实现连贯的多轮交互。

```python
class ConversationChain(LLMChain):
    memory: BaseMemory = Field(default_factory=ConversationBufferMemory)
    prompt: BasePromptTemplate = PROMPT
    input_key: str = "input"  # 用户输入的键名
    output_key: str = "response"  # 模型输出的键名
```



### RetrievalQA

结合检索器（Retriever）和问答链，先从知识库中检索相关文档，再基于文档回答问题（RAG 场景）。

```python
# 初始化向量数据库（示例数据）
documents = [
    Document(page_content="LangChain 支持多种链类型，包括 LLMChain、SequentialChain 等。"),
    Document(page_content="RetrievalQA 链用于检索增强问答，结合检索器和语言模型。")
]
embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(documents, embeddings)
retriever = db.as_retriever()

# 创建 RetrievalQA 链
qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(),
    chain_type="stuff",  # 使用 Stuff 策略合并文档
    retriever=retriever,
    return_source_documents=True  # 返回用于回答的源文档
)

# 执行问答
result = qa_chain({"query": "LangChain 有哪些链类型？"})
print(result["result"])  # 输出：LangChain 支持多种链类型，包括 LLMChain、SequentialChain 等。
print("源文档：", [doc.page_content for doc in result["source_documents"]])
```



### RouterChain

根据输入动态选择合适的子链执行，适用于多场景任务分发。

```python
# 定义两个子链和路由规则
math_prompt = ChatPromptTemplate.from_messages([
    ("system", "解决数学问题：{input}")
])
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "生成 Python 代码：{input}")
])
router_template = """
根据输入判断类型：
- 若涉及数学计算，返回 `math`
- 若涉及代码生成，返回 `code`
输入：{input}
"""
router_prompt = PromptTemplate(
    template=router_template,
    input_variables=["input"],
    output_parser=RouterOutputParser()
)

# 初始化路由链和子链
llm = ChatOpenAI()
router_chain = LLMRouterChain.from_llm(llm, router_prompt)
math_chain = LLMChain(llm=llm, prompt=math_prompt)
code_chain = LLMChain(llm=llm, prompt=code_prompt)
chain_map = {"math": math_chain, "code": code_chain}
multi_prompt_chain = MultiPromptChain(
    router_chain=router_chain,
    destination_chains=chain_map,
    default_chain=LLMChain(llm=llm, prompt=ChatPromptTemplate.from_messages([("human", "{input}")]))
)
```



### 其余

另外还有文档处理链：

- `StuffDocumentsChain`：将所有文档一次性传入模型（适合文档量少的场景）
- `MapReduceDocumentsChain`：先单独处理每篇文档（Map），再合并结果（Reduce）（适合大量文档）
- `RefineDocumentsChain`：逐步迭代优化结果（适合需要精确结果的场景）。



## memory

LangChain 中的 `memory` 模块是实现对话状态管理的核心组件，用于存储和管理对话历史，使模型能够理解上下文并生成连贯的多轮对话。

基类chain中，输入前会执行load_memory_variables，输出后会执行save_context。数据最终存储在chat_memory之中。

### 基础逻辑

一般，所有记忆类均通过 `chat_memory`（默认 `InMemoryChatMessageHistory`）存储原始对话消息（`HumanMessage`/`AIMessage`），或通过自定义存储（如数据库）持久化。

对话前，调用 `load_memory_variables` 时，记忆类将原始消息转换为模型可理解的格式（字符串、摘要或键值对），并通过 `memory_key` 暴露给链（如 `ConversationChain`）。

对话后，`save_context` 会提取用户输入和模型输出，更新 `chat_memory`，并触发特殊处理（如摘要生成、窗口截断）。

另外：可通过自定义 `BaseChatMessageHistory` 实现持久化存储（如 `RedisChatMessageHistory`、`MongoDBChatMessageHistory`），或继承 `BaseMemory` 实现特定记忆逻辑。



### 分类

当前的会话信息保存方案为两种：

1. **内存（Memory）**：把对话历史直接保存在程序内（或轻量持久化），在每次 prompt 中把“历史”按某种策略拼接进 prompt 里（完整/滑动窗口/按 token 限制/摘要化等）。适合短期上下文、低延迟、开发快速迭代。

2. **向量检索（RAG, Retriever / Index）**：把对话（或更广义的知识）拆成文档块，定期把块做 embedding 存入向量数据库（FAISS/Chroma/Pinecone/Milvus 等）。每一轮对话用当前用户 query 去检索最相关的 k 条，再把这些检索到的“补充上下文”连同当前 query 一并送给 LLM（常见于长期记忆、知识库问答或大规模应用）。

内存型的延迟比较低，实现简单，延迟低，适合短对话或者多轮对话中的短期记忆场景。但对话轮次变多时，消耗的token越来越多，模型也容易丢失早期的重要信息。对话量大时，比较建议采取**滑动窗口 + 摘要** 的混合：近期用窗口保留详细对话，早期历史做周期性摘要并把摘要保留为长期记忆。

向量检索在其它模块会再次详细说明。



### Usage

现实系统常把两者结合起来：

- **短期 memory（window）**：把最近 2–5 轮的对话直接放进 prompt，保证对话连贯性与即时上下文。
- **长期 retriever（RAG）**：针对需要回溯或查询的请求（“告诉我上个月的报表结论”），从向量库检索历史对话或知识并作为补充。
- 合并策略示例：`[SYSTEM] + {recent_window_history} + {retrieved_docs} + user_question`。
  这种组合既保留低延迟的短期记忆，又能应对长期查询与知识库问答。

```python
# 每轮：
recent = window_memory.format()
retrieved = retriever.get_relevant_docs(query, k=4)
prompt = system + recent + format_retrieved(retrieved) + user_question
answer = llm(prompt)
# 保存新消息到 memory 和向量库（upsert）
memory.save(user_message, assistant_message)
vectorstore.add_texts([user_message, assistant_message], metadata=...)
```



### 内存型常见实现策略

#### ConversationBufferMemory

逐句记录所有对话内容（用户输入 + 模型回复），不做任何截断或总结。适合短对话需要完整保存上下文的场景。

#### ConversationBufferWindowMemory

仅保留最近 `k` 轮对话，避免历史过长导致的冗余。中等长度对话，需要控制上下文长度。

#### ConversationSummaryMemory

通过 LLM 动态总结对话历史，用摘要代替完整历史，减少上下文长度。长对话场景，需要压缩历史信息。

#### ConversationSummaryBufferMemory

结合摘要和窗口记忆的优点：用摘要保存早期对话，用窗口保留最近 `k` 轮对话，平衡信息完整性和长度。

#### CombinedMemory

组合多个记忆组件

```python
from langchain_classic.memory import CombinedMemory, SimpleMemory

# 组合对话记忆和简单键值对记忆
conv_memory = ConversationBufferMemory(memory_key="history")
simple_memory = SimpleMemory(memories={"user_name": "小明"})
combined = CombinedMemory(memories=[conv_memory, simple_memory])

combined.save_context({"input": "你好"}, {"output": "您好！"})
print(combined.load_memory_variables({}))
# 输出：{"history": "Human: 你好\nAI: 您好！", "user_name": "小明"}
```

#### SimpleMemory

存储固定的键值对（如用户信息），不随对话更新，适用于保存静态上下文。



## Document Loaders

**Document Loaders**（文档加载器）是处理数据输入的核心模块，负责从各种数据源（如文件、数据库、API 等）加载加载数据并转换为统一的 `Document` 格式，为后续的处理（如分割、嵌入、检索）提供基础。

[ langchain源码分析-文档加载【9】 - 知乎](https://zhuanlan.zhihu.com/p/652628605)

### 核心逻辑

针对不同类型的数据源实现特定的加载逻辑，会通过懒加载（`lazy_load`）或异步加载（`alazy_load`）方式高效处理大规模数据，避免内存占用过高。将原始数据统一转换为 `Document` 结构，方便下游组件（如文本分割器、向量存储）处理。

- 文件加载：支持csv文件，pdf文件，markdown文件，notebook文件等
- 结构数据源加载：比如xml文件，git数据源，pandas数据源，pyspark.dataframe数据源加载等
- 其它数据源：比如email内容，html内容，云服务数据源（cos），视频加载等

#### 懒加载

懒加载是一种 "按需加载" 机制，通过迭代器（`Iterator`）逐逐批返回文档，而非一次性一次性将所有文档一次性加载到内存中，从而有效减少内存占用，尤其适合处理大文件或海量数据。

- **内存高效**：避免一次性加载全部数据到内存，尤其适合 GB 级文件或大量小文件。
- **流式处理**：支持边加载边处理（如即时分割、嵌入），减少等待时间。
- **兼容性**：`load()` 方法默认基于 `lazy_load` 实现（通过 `list(iterator)` 转换），兼顾便捷性。

#### 异步加载

异步加载是懒加载的异步版本，通过异步迭代器（`AsyncIterator`）实现非阻塞加载，适合需要异步操作的场景（如异步 Web 框架、并发 API 调用）。

- 加载网络资源（如异步网页爬取 `AsyncHtmlLoader`）。
- 异步框架中处理文档（如 FastAPI 接口内加载数据）。
- 需要并发加载多个数据源的场景（如同时请求多个 API）。



## embedding

embedding主要用于将文本转换成稠密向量（dense vector），便于之后的查找。

Embedding 是一个抽象层，底层可以是任何模型（OpenAI、HuggingFace、本地模型、自定义模型）。用户在上层统一用：embed_query / embed_documents。

### Embedding模型

#### 官方内置

| Embedding 类型            | 最适合的应用场景                                 | 优点                                            | 缺点                        | 成本     | 性能（效果） |
| ------------------------- | ------------------------------------------------ | ----------------------------------------------- | --------------------------- | -------- | ------------ |
| **OpenAIEmbeddings**      | 商业级 RAG、FAQ、搜索、推荐、多语言检索          | 语义效果最强、稳定、无需运维                    | 需联网、数据外发、成本较高  | 中等偏高 | ⭐⭐⭐⭐⭐        |
| **AzureOpenAIEmbeddings** | 企业内网、金融/政府行业、合规要求高的知识库      | 与 OpenAI 效果相同；支持私有网络/VNet；合规性强 | 只适合 Azure 生态；成本略高 | 中高     | ⭐⭐⭐⭐⭐        |
| **HuggingFaceEmbeddings** | 私有化部署、大规模向量生成、多语言检索、中文 RAG | 多模型可选、可自部署、低成本、高灵活性          | 需要硬件资源（GPU 推荐）    | 低       | ⭐⭐⭐⭐         |
| **GPT4AllEmbeddings**     | 轻量级本地 demo、离线应用、资源受限设备          | 完全本地、CPU 可跑、隐私安全、免费/低成本       | 效果弱于前面三种            | 极低     | ⭐⭐           |

```python
from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.embeddings import GPT4AllEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings
```

Community内置

```python
from langchain_community.embeddings import HuggingFaceHubEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.embeddings import CohereEmbeddings
from langchain_community.embeddings import GooglePalmEmbeddings
from langchain_community.embeddings import BedrockEmbeddings
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.embeddings import MistralAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.embeddings import VoyageEmbeddings
from langchain_community.embeddings import JinaEmbeddings
```

#### 调用机制

```python
embedding = embeddings.embed_query("hello world")
vectors = embeddings.embed_documents(["a", "b", "c"])

# langchain新标准
embeddings.invoke("text")
embeddings.batch(["t1", "t2"])
```



## retrieves

检索模块更多的是将索引和具体的检索方法作为一个整体，对外提供服务。核心方法是 `get_relevant_documents(query)`（同步）或 `aget_relevant_documents(query)`（异步），直接返回与查询相关的文档列表。

检索器的建立依赖索引，一般和文档加载，embedding，索引，检索链共同出现，构成RAG（检索生成增强的核心组件，可以让LLM基于外部文档回答问题。

#### as_retriever()

> VectorStore.as_retriever() → 将向量库包装成一个标准检索器（Retriever）对象

vectorStore本身负责存储向量，为了能让vectorStore能被RAG，Chain，Agent统一调用，as_retriever将向量库封装成retrieve最为对外服务的搜索接口。

#### 整体逻辑

```python
# 文档加载，加载csv格式的数据
from langchain.document_loaders import CSVLoader
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
docs = loader.load()

# OpenAIEmbeddings对文本进行向量化
# 调用 OpenAI 的嵌入 API（默认使用text-embedding-ada-002模型），将文本转换为 1536 维向量。
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

# 创建向量索引
# 将向量库转换为检索器，提供统一的检索接口，用于根据用户查询获取相关文档。
from langchain.vectorstores import DocArrayInMemorySearch
db = DocArrayInMemorySearch.from_documents(
    docs, 
    embeddings
)

# 将索引转换为检索器，其实是将索引作为检索器的一个变量。检索器提供了不同相关性计算的方法
retriever = db.as_retriever()

# 创建检索式问答链
from langchain.chat_models import ChatOpenAI
llm_model = 'gpt-3.5-turbo-0301' # 后续该模型会下线，替换成其他模型即可
llm = ChatOpenAI(temperature = 0.0, model=llm_model)

from langchain.chains import RetrievalQA
qa_stuff = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=retriever, 
    verbose=True
)
qa_stuff.query("List all your shirts with sun protection in a table")
```

`qa_stuff` 是通过 `RetrievalQA.from_chain_type` 创建的问答链实例，`query` 方法是其对外提供的接口，接收用户输入的自然语言问题（如示例中的 “列出所有带防晒功能的衬衫，用表格展示”）。

问答链首先会利用初始化传入的retriever从向量库中获取和用户查询相关的文档，检索过程中，会先将用户问题通过embedding向量化，然后默认用余弦相似度和向量库中的数据进行匹配，返回相关的文档作为后续回答生成的上下文。

然后文档链会将检索到的文档和用户查询组合为一个完整的提示词，最后将构建的提示词传入初始化好的模型生成最终结果。



## callbacks（可以总结一下分类）

LangChain 的 `callbacks` 模块（现核心接口在 `langchain_core.callbacks`）是用于**监控、记录和干预 LangChain 组件运行过程**的核心工具。它基于**观察者模式**设计，允许开发者在 LLM 调用、链执行、Agent 决策、工具调用等关键节点插入自定义逻辑，实现日志记录、性能监控、数据持久化、流式输出等功能。

Callback 可在**任意 LangChain 组件**中注册，包括 LLM、Chain、Agent、Retriever 等

### 常用callbacks

#### `StdOutCallbackHandler`

将组件运行的关键事件（如 LLM 调用、Agent 决策、工具执行）打印到控制台，适用于调试。

```python
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_openai import ChatOpenAI

handler = StdOutCallbackHandler()
llm = ChatOpenAI(
    model="deepseek-chat",  
    temperature=0,
    callbacks=[handler],  # 直接传入 Handler 列表（自动创建 CallbackManager）
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

llm.invoke("Hello World")
```

#### `FileCallbackHandler`

将事件日志写入指定文件，适用于生产环境的日志持久化。

```python
from langchain_core.callbacks import FileCallbackHandler
import sys

# 写入到标准输出（等同于 StdOutCallbackHandler）
# 或写入到文件：handler = FileCallbackHandler("logs.txt")
handler = FileCallbackHandler(sys.stdout)
```

#### `StreamingStdOutCallbackHandler`

实现 LLM 输出的**流式打印**，适用于需要实时展示生成过程的场景（如聊天机器人）。

```python
from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI

# 配置 DeepSeek 模型参数
DEEPSEEK_API_KEY = "sk-65da967e427c4f86ae4749129ba48166"  # 替换为你的 DeepSeek API Key
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # DeepSeek 的 API 基础地址

# 初始化 DeepSeek 模型，启用流式输出
llm = ChatOpenAI(
    model="deepseek-chat",  # 指定 DeepSeek 模型，可选 deepseek-chat/deepseek-coder
    temperature=0.7,  # 控制生成随机性，0 为确定性输出，1 为最大随机性
    streaming=True,  # 必须开启，否则无法流式输出
    callbacks=[StreamingStdOutCallbackHandler()],  # 绑定流式回调处理器，控制台会实时打印内容
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL
)

# 执行生成任务，控制台会实时打印内容
llm.invoke("介绍你的功能")
```



## 输出解析器Output-parses

`output_parser` 模块用于用于将语言模型（LLM）的原始输出转换为结构化数据（如 JSON、Pydantic 模型、列表等），方便后续处理和使用。[langchain源码剖析-output_parses各模块介绍【6】](https://zhuanlan.zhihu.com/p/649185942)

- **最基础 / 常见**：`StrOutputParser`，适用于绝大多数不需要结构化的场景，是默认首选。
- **结构化需求**：`JsonOutputParser`（简单键值对）和 `PydanticOutputParser`（带校验的复杂结构）。
- **列表类输出**：`CommaSeparatedListOutputParser` (将文本串通过’, ‘分隔，转为list格式返回)简单高效。



## 装饰器

**节点式**（在特定执行点运行）

- `@before_agent` - 代理启动前（每次调用一次）
- [`@before_model`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.before_model) - 每次模型调用前
- [`@after_model`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.after_model) - 每次模型响应后
- `@after_agent` - 代理完成时（每次调用一次）

**包装式**（拦截和控制执行）

- [`@wrap_model_call`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call) - 每次模型调用前后
- [`@wrap_tool_call`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.wrap_tool_call) - 每次工具调用前后

**便利装饰器**:

- [`@dynamic_prompt`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.dynamic_prompt) - 生成动态系统提示（相当于修改提示的 [`@wrap_model_call`](https://reference.langchain.org.cn/python/langchain/middleware/#langchain.agents.middleware.wrap_model_call)）



# 参考文章：

1 [LangChain源码学习 | 李乾坤的博客](https://qiankunli.github.io/2023/08/29/langchain_source.html)

2 [langchain源码剖析-模块整体介绍 - 知乎](https://zhuanlan.zhihu.com/p/640848809)

3 [LangChain 文档](https://docs.langchain.org.cn/oss/python/releases/langchain-v1)

4 [github- langchain](https://github.com/langchain-ai/langchain.git)

5 [LangChain 源码 深度历险：基于GOF的设计模式，穿透 LangChain 源码 - 技术自由圈 - 博客园](https://www.cnblogs.com/crazymakercircle/p/19087400)