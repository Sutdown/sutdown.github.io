---
title:        "langgraph概述"
description:  "LangGraph是一个用于构建状态ful多智能体应用的框架，基于图论的方式组织智能体的交互流程。"
date:         2025-12-26
toc: true
categories:
    - AI
---

## python基础

### 基础语法

```python
## list 管理对话历史
messages = [{"role": "user", "content": "你好"},
           {"role": "assistant", "content": "你好！有什么可以帮你的吗？"}]
## dict agent状态管理
agent_state = {
    "user_id": "user_12345",
    "session_id": "sess_abc",
    "messages": [
        {"role": "user", "content": "帮我查天气"},
        {"role": "assistant", "content": "好的，请问是哪个城市？"}
    ],
    "context": {
        "intent": "weather_query",
        "location": None,
        "confidence": 0.92
    },
    "tool_calls": [],
    "metadata": {
        "start_time": "2024-10-27 10:00:00",
        "turn_count": 1
    }
}
## 元组 不可变数据
message_format = ("role", "content", "timestamp")
## set 去重和快速查找
user_interests = {"科技", "旅游", "美食", "科技", "运动"}
## 【嵌套TypedDict】LangGraph状态的标准定义方式
## 定义清晰的状态结构,便于团队协作和维护
class AgentState(TypedDict):
    messages: List[Message]  # 消息列表,每个元素都是Message类型
    intent: str
    next_node: str
```

注意异常处理。

- 方式1 try ...raise... except ，
- 方式2 自定义异常，
- 方式3 with进行上下文管理，保证资源用完一定能够释放

一般需要建立配置文件或者.env文件管理配置。

添加适当的日志管理。

 装饰器是 Python 的"语法糖"，用于修改函数行为

### AI开发生态

> **Streamlit**：快速构建Agent交互界面，展示对话流、状态可视化、调试工具
>
> **ChromaDB** ：最简单的向量数据库，适合本地开发和原型。

- langchain：链式调用基础，构建LLM应用的基础框架
- langgraph：利用状态图构建复杂的多智能体系统
  - 需要精确控制执行流程，复杂的条件分支和循环
  - 生产环境的可靠性要求，可观测性和调试能力
  - 需要human-in-the-loop
- CrewAI：专注于多智能体角色分工和任务编排，一般为顺序或者层级流程，角色分工清晰，适合写作，研究
- AutoGen：多智能体自动对话和协作，支持代码执行，自动错误修复，支持人类参与，适合编程，数据分析等任务
- Swarm：实验性项目，最小化实现

**Gradio = 模型 Demo / Agent 快速展示**
**Streamlit = 完整应用 / 数据产品 / 控制台级 UI**

| 维度     | Gradio               | Streamlit                  |
| -------- | -------------------- | -------------------------- |
| 设计初衷 | **让模型“马上能用”** | **把 Python 变成 Web App** |
| 关注点   | 模型输入 → 输出      | 应用流程、状态、页面       |
| 思维方式 | ML / Agent 导向      | App / Data App 导向        |
| 学习成本 | 极低                 | 中等                       |

```text
LangGraph / LangChain   ← 核心逻辑
        ↓
      Gradio            ← Demo / 对话
        ↓
   Streamlit / FastAPI  ← 产品化
```



## 基础

### Checker Pointer

Checkpointer 是 LangGraph 的 自动快照系统，在每个节点执行后保存状态，实现多轮对话和时间旅行。

跨会话对话记忆通过 **Checkpointer + Thread ID** 实现状态持久化和恢复。

- Checkpointer = 自动快照系统
- 保存时机：每个节点执行后
- Thread = 隔离不同会话的状态容器
- 用途：多轮对话、状态持久化、时间旅行
- 生产环境：使用 PostgresSaver 或 RedisSaver

链式架构，路由架构（分类器），agent（reAct架构，存在多次循环）



### 生成环境部署

- 开发环境：可直接用InMemoryStore存储，单线程，本地访问，崩溃则停止
- 生产环境：用PostgreSQL/Redis存储，多进程/多线程并发，结构化日志和监控系统实现监控，错误时能够自动重启或者采用合适的降级策略，公网访问



## state reducer memory

记忆起到的作用在于，让Agent从无状态服务变成有记忆的助手

- LangGraph 提供 **Checkpointer（检查点器）** 机制，在每个步骤后自动保存状态。
- 为了管理多个独立的对话，LangGraph 使用 **Thread（线程）** 概念。

### checker pointer

短期记忆一般针对单个对话线程，可以在该线程内的任何时间被回忆。

state会通过**checkpointer**持久化数据库中，哪怕会话中断也可以随时回复。记忆在图被调用时或者某个步骤完成时更新，并在某个步骤开始时读取。记忆中为了避免过长的消息列表，通常存在几种解决方案：1 固定窗口 - trim裁剪，2 总结摘要 - summarization，3 利用相关性进行语义过滤

```python
class ConversationState(BaseModel):
    messages: Annotated[List[str], trim_messages(20)] = []
def trim_messages(max_len: int):
    def reducer(old: List[str], new: List[str]):
        merged = old + new
        return merged[-max_len:]
    return reducer

def summarizer_node(state: ConversationState):
    if len(state.messages) < 15:
        return {}

    prompt = f"""
    当前摘要：
    {state.summary or "无"}

    请将以下对话合并成一个更精炼的摘要：
    {state.messages}
    """

    new_summary = llm(prompt)

    return {
        "summary": new_summary,
        "messages": []  # 清空窗口
    }


def semantic_filter(messages, query_embedding, k=5):
    scored = [
        (cosine_sim(m.embedding, query_embedding), m)
        for m in messages
    ]
    return [m.content for _, m in sorted(scored, reverse=True)[:k]]
```



### 消息处理方式

- 消息删除，基于meessage
- 消息过滤，节点返回
- 消息裁剪，基于token



### langgraph与langchain memory

LangChain 和 LangGraph 解决的是“不同层级的记忆问题”

- LangChain：偏“认知层长期记忆”（semantic memory）
- LangGraph：偏“系统层执行记忆”（execution state）

LangChain Memory 体系本质在于**如何把过去的信息提供给模型**。langchain短期记忆对应memory，比如最近k条，对于会话的总结，限制token等方案；长期记忆则对应向量存储和各种数据库集成，通过retrievers，文档加载器，embedding，文档拆分等工具实现。

| 对比维度        | LangChain Vector Memory | LangGraph Checkpointer |
| --------------- | ----------------------- | ---------------------- |
| 关注点          | 语义                    | 执行                   |
| 是否 embedding  | ✅                       | ❌                      |
| 是否参与 prompt | ✅                       | ❌                      |
| 是否可检索      | ✅                       | ❌                      |
| 是否能恢复执行  | ❌                       | ✅                      |
| 是否长期运行    | ⚠️（有限）               | ✅                      |



## langgraph 图

> 1 LangGraph 将 Agent 建模为**状态机**,执行过程是**状态的连续转换**。
>
> 2 **State = 数据 + Channels + Reducers (技术核心)** 
>
> 3 **Node = State → State (函数式思维)**
>
> 4 **Conditional Edge = 动态路由 (控制流的灵魂)** 
>
> 5 **Human-in-the-Loop = 可控性 (生产环境的必需品)** 

### state

LangGraph 的状态管理基于 **TypedDict + Annotated + Reducer** 的三层架构。

#### state 定义

state是一种信息共享和上下文管理的工具，起源于状态机理论，广泛用于对话系统，任务自动化和复杂逻辑处理。

1 存储上下文信息，实现短期记忆

2 动态逻辑控制，决定系统的下一步行为

3 任务管理，存储中间的结果

4 通过存储状态，管理对话的分支逻辑

#### State Schema

在定义state的过程中，存在三种方式：

- TypedDict：适用于简单快速开发，无运行时验证（性能最优）
- Dataclass：需要默认值/方法，无运行时验证
- Pydantic：会进行严格的运行时验证（安全性最好）

```python
from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import List

class BaseState(BaseModel):
    """所有 State 的统一基类"""

    model_config = ConfigDict(
        validate_assignment=True,   # 赋值即校验
        extra="forbid",              # 禁止野字段
        frozen=False                # LangGraph 需要可变
    )

    trace: List[str] = []
    updated_at: datetime = datetime.utcnow()

class InputState(BaseState):
    query: str = Field(..., description="User original input")
    user_id: str | None = None

class WorkingState(InputState):
    # Planner
    plan: Optional[str] = Field(
        None, description="High level plan generated by planner"
    )

    # Retriever
    documents: List[str] = Field(
        default_factory=list,
        description="Retrieved documents"
    )

    # Memory / Reasoning
    thoughts: List[str] = Field(
        default_factory=list,
        description="Internal reasoning trace"
    )

    # Control
    next_action: Optional[str] = None

class OutputState(WorkingState):
    answer: str = Field(..., description="Final user-facing answer")

    @model_validator(mode="after") # 在所有字段都准备好之后，对整个 State 的‘存在合法性’做最终审判。
    def check_ready(self):
        if not self.plan:
            raise ValueError("Output generated without plan")
        return self
```

#### state reducers

Reducer 决定：当 state 的同一个字段被多次写入时，最终 state 应该长什么样。

在 LangGraph 里：

- **Node**：负责“产生增量状态（partial state）”
- **Reducer**：负责“把多个增量状态合并成一个最终 state”

常见reducer类型有：（也可以选择自定义，比如条件型，组合型，带日志等）

1 累加型，比如add_messages, 2 覆盖型，3 合并型，4 去重型，5 决策型

`add_messages` 解决了对话系统中 **状态覆盖导致历史丢失** 的核心问题。

```python
from pydantic import BaseModel
from typing import List, Annotated
from langgraph.graph import add_messages

class AgentState(BaseModel):
    messages: Annotated[List[str], add_messages] = []
    current_step: str | None = None
```

#### reducer  ---  add_messages

add meaaages主要用于合并消息。但同时能实现

- 消息重写： 如果新消息的 `id` 与现有消息相同，会**覆盖**旧消息
- 消息删除：采用**RemoveMessage**，可以实现滑动窗口，删除敏感信息，清理无关对话历史，优化上下文长度

```python
## 手动方式
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

## 快捷方式（推荐）
class State(MessagesState):
    pass  # 自动包含 messages 字段

result = add_messages(initial_messages, new_message)
delete_messages = [RemoveMessage(id=m.id) for m in messages[:-2]]
```

#### Multiple Schemas

Multiple Schemas = 在同一个 LangGraph 中，为不同阶段 / 不同节点 / 不同角色使用不同的 State Schema。Schema 是“视图（View）”，不是“存储（Storage）”。**只能看到自己需要的字段，只被允许返回自己负责的字段。**

1. **Private State（私有状态）**：节点间可以传递私有数据，不暴露给外部
2. **Input Schema（输入模式）**：限定用户输入的字段
3. **Output Schema（输出模式）**：限定返回给用户的字段
4. **Internal State（内部状态）**：图内部使用的完整状态

```python
## 在node中
graph.add_node("planner", planner_node, input_schema=PlannerState)
graph.add_node("executor", executor_node, input_schema=ExecutorState)
graph.add_node("reviewer", reviewer_node, input_schema=ReviewerState)

## 在state中
class ConversationState(BaseModel):
    messages: Annotated[List[str], add_messages] = []

class PlanningState(BaseModel):
    plan: str | None = None

class ExecutionState(BaseModel):
    observations: Annotated[List[str], add_messages] = []

class ReviewState(BaseModel):
    final_answer: str | None = None

## State Nesting（嵌套 State），本质是 组合（composition）而不是继承
class GlobalState(BaseModel): 
    conversation: ConversationState = ConversationState()
    planning: PlanningState = PlanningState()
    execution: ExecutionState = ExecutionState()
    review: ReviewState = ReviewState() 
```

#### external memory --- SQLite Checkpointer 

**SQLite** 是一个轻量级的嵌入式数据库：

- 不需要独立的数据库服务器
- 整个数据库存储在单个文件中
- 性能优秀，广泛应用（被 Andrej Karpathy 称为"超级流行"）
- Python 内置支持

**SqliteSaver 的作用：**

- 自动创建必要的数据库表
- 保存图的每一步状态（checkpoint）
- 支持状态查询和恢复
- 管理多个对话线程（thread）

```python
import sqlite3

## 创建内存数据库（程序结束后消失）
conn = sqlite3.connect(":memory:", check_same_thread=False)

## 下载示例数据库（如果不存在）
!mkdir -p state_db && [ ! -f state_db/example.db ] && wget -P state_db https://github.com/langchain-ai/langchain-academy/raw/main/module-2/state_db/example.db

## 连接到本地数据库文件
db_path = "state_db/example.db"
conn = sqlite3.connect(db_path, check_same_thread=False)

from langgraph.checkpoint.sqlite import SqliteSaver

## 创建 checkpointer
memory = SqliteSaver(conn)

CREATE TABLE checkpoints (
    thread_id TEXT,      -- 对话线程 ID
    checkpoint_id TEXT,  -- 检查点 ID
    state BLOB,          -- 序列化的状态数据
    timestamp DATETIME   -- 时间戳
);
```



## Human-in-the-loop

人类协作目前存在两种方式：

一是在图的编译时手动添加断点，常见的是interrupt+update state+invoke

二是利用stream输出，在输出中的过程中捕捉特定的状态输出

三是时间旅行，回到特定的状态，修改状态，创建新的分支执行

前两种都是在于一次的运行中存在的中断，第三种侧重于分支。

### breakpoints - 断点

在图的执行过程中设置断点，主动暂停等待人工干预。

三大应用场景：审批，调试，编辑

```python
## interrupt_before 审批，在某个节点或工具真正执行之前，强制暂停整个 Graph 的运行。
## interrupt_after 调试，某个节点执行完成之后，立刻中断流程
## update_state 在前两者到达的中断状态下，你可以人为修改 State，然后让 Graph 继续跑。

## interrupt 负责“停”，update_state 负责“改”，invoke 负责“继续”
graph = builder.compile(
    interrupt_before=["payment_tool"],  # 在支付前暂停
    checkpointer=memory # breakpoint依赖状态持久化，会将当前状态保存倒checkpoint
)

graph.update_state(
    thread_id,
    {"approved": True}
)

graph.invoke(None, thread_id=thread_id)
```

### Node Interrupt

当 Graph 运行到某个 Node 时，主动暂停，让外部（人类 / 系统）介入，然后再决定是否继续。

interrupt() (推荐)：可在代码任意位置调用

```python
from langgraph.errors import Interrupt

def node(state):
    if not state.get("approved"):
        raise Interrupt("waiting for approval")
    return state
```

### Stream

对于langgraph：

- 默认为执行事件试图，包括节点前，节点结束，中断，工具使用前，工具使用后状态；
- updates为state增量视图，在每个node执行完返回当前node修改的字段
- values为完整state试图，在每个node执行完返回完整的字段

```python
for event in graph.stream(inputs):
    ...

for update in graph.stream(inputs, stream_mode="updates"):
    print(update)
```

### Time travel

LangGraph 的 Time Travel = 对 Graph 执行过程中每一次 State 的快照做版本化存储，并允许你回到任意历史点继续执行。

前提：checkerpointer， thread ID

能力：状态快照，回到过去，改写未来

```python
current = graph.get_state(thread_id) # 可以查看当前value和历史history state
print(current.values) # 当前状态
for snapshot in current.history:
    print(snapshot.values, snapshot.branch_id) # 历史状态
    
graph.set_pointer(thread_id, snapshot=current.history[1]) # 回到step之后的快照
## TODO 这里可以用update state更新状态
graph.update_state(thread_id, {"count": 100, "note": "corrected after step1"}) # 修改state
graph.invoke(None, thread_id=thread_id)
```



## Memory

[记忆系统](https://learngraph.online/LearnGraph%201.X/module-6-memory-system/6.5%20Conclusion.html)

**短期记忆 + 长期记忆** 的架构

- Checkpointer：保存对话历史（短期）
- Store：保存用户信息（长期）
- 两者配合实现完整的记忆系统

```python
┌─────────────────────────────────────-────┐
│           Memory Agent                   │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │   短期记忆 (MemorySaver)            │  │
│  │   - 对话历史                        │  │
│  │   - 当前会话状态                     │  │
│  └────────────────────────────────────┘  │
│                                          │
│  ┌────────────────────────────────────┐  │
│  │   长期记忆 (InMemoryStore)          │  │
│  │   ┌──────────────────────────────┐ │  │
│  │   │ Profile (语义记忆) - profile，collection
│  │   └──────────────────────────────┘ │  │
│  │   ┌──────────────────────────────┐ │  │
│  │   │ ToDo Collection (语义记忆)    │ │  │
│  │   └──────────────────────────────┘ │  │
│  │   ┌──────────────────────────────┐ │  │
│  │   │ Instructions (程序性记忆)     │  │  │
│  │   └──────────────────────────────┘ │  │
│  └────────────────────────────────────┘  │
└────────────────────────────────────────-─┘
```

### Store

in memory store

- 在内存中保存数据
- 适合开发和测试
- 生产环境通常使用持久化存储（如数据库

```python
## 创建内存存储实例
in_memory_store = InMemoryStore()

## put - 保存数据
def put(
    namespace: tuple,      # 命名空间（元组）
    key: str,              # 键（字符串）
    value: dict            # 值（字典）
) -> None:
    
user_id = "1"
namespace_for_memory = (user_id, "memories")

key = str(uuid.uuid4())
value = {"food_preference": "I like pizza"}

in_memory_store.put(namespace_for_memory, key, value)

## search - 获取用户所有记忆
memories = in_memory_store.search(namespace_for_memory)
print(memories[0].dict())

## get - 获取特定的一条记忆
memory = in_memory_store.get(namespace_for_memory, key)
print(memory.dict())
```

### 记忆的存储模式

热路径：在用户等待响应的过程中执行的操作。

1. 实时记忆，每次对话都会提取记忆，增加延迟，适用于对话频率低的情况
2. 实现简单，无需考虑并发

冷路径：用消息队列， LangGraph 的 [Background Execution](https://langchain-ai.github.io/langgraph/how-tos/background-execution/)，定义处理记忆更新等方式实现

1. 后台异步处理，响应速度快，适用于大规模or响应速度快的场景
2. 实现复杂，有延迟

### Schema Profile

1 简单模式：typedict + with_structured_output()

2 复杂模式：[**trustcall**](https://learngraph.online/LearnGraph%201.X/module-6-memory-system/6.3%20Memory%20Schema%20Profile.html#%E5%AE%8C%E6%95%B4%E6%A1%88%E4%BE%8B%E4%BB%A3%E7%A0%81-%E5%8F%AF%E7%9B%B4%E6%8E%A5%E8%BF%90%E8%A1%8C)（[Trustcall](https://github.com/hinthornw/trustcall) 是一个专门用于**创建和更新 JSON Schema** 的开源库，由 LangChain 团队的 [Will Fu-Hinthorn](https://github.com/hinthornw) 开发。）

```text
┌─────────────────────────────────────────────┐
│           Trustcall Workflow                │
│                                             │
│  1. 接收输入                                │
│     - 新的对话消息                          │
│     - 现有的 Schema（如果有）               │
│                                             │
│  2. 分析变化                                │
│     - 识别新信息                            │
│     - 识别需要更新的字段                    │
│                                             │
│  3. 生成 JSON Patch                         │
│     - 创建精确的更新操作                    │
│     - 只修改变化的部分                      │
│                                             │
│  4. 应用 Patch                              │
│     - 更新现有 Schema                       │
│     - 保留未变化的信息                      │
│                                             │
│  5. 验证结果                                │
│     - 确保符合 Schema 定义                  │
│     - 如果失败，自动修正                    │
└─────────────────────────────────────────────┘
```

#### trustcall

> Trustcall 是一个让 LLM 在“结构化信息抽取 / 状态写入”场景中，输出“可验证、可约束、可合并”的结果的工具库。定位在于解决 LLM 在“写结构化状态”这件事上的不可信问题。

当agent需要稳定profile，为长期记忆时，可以采用trustcall

Profile 模式（enable_inserts=False）和 Collection 模式（enable_inserts=True）

```python
## 定义shcema
class UserProfile(BaseModel):
    name: Optional[str]
    interests: List[str] = Field(default_factory=list)
    expertise_level: Optional[str]

## 创建extractor，输出一定符合Pydantic/JSON Schema
profile_extractor = create_extractor(
    UserProfile,
    description="""
    Extract stable user profile information from conversation.

    Rules:
    - Only extract explicit facts
    - Do NOT guess or infer
    - Ignore temporary tasks or moods
    """
)

## 在agent中调用
def extract_profile(llm, user_input: str, existing_profile: UserProfile):
    profile = profile_extractor.invoke(
        llm,
        user_input,
        existing=existing_profile
    )
    return profile
```



### Profile vs Collection

#### 对比

- Profile = 稳定的“用户画像 / Agent 自我认知”
- Collection = 可增长、可回放、可组合的“事实条目集合”

```text
Profile 模式（5.3）              Collection 模式（5.4）
    ↓                               ↓
┌────────────────┐            ┌────────────────┐
│  用户资料      │            │  记忆 1        │
│  ┌──────────┐  │            │  content: ...  │
│  │ name     │  │            └────────────────┘
│  │ location │  │            ┌────────────────┐
│  │ interests│  │            │  记忆 2        │
│  └──────────┘  │            │  content: ...  │
│  单一对象      │            └────────────────┘
└────────────────┘            ┌────────────────┐
                              │  记忆 3        │
更新 = 修改字段                │  content: ...  │
                              └────────────────┘
                              多个独立对象

                              更新 = 修改现有项
                              插入 = 添加新项
```

#### Collection

```python
from pydantic import BaseModel, Field

class Memory(BaseModel):
    content: str
    category: Optional[str] = None      # 可选的类别
    importance: Optional[int] = None    # 可选的重要性评分
    timestamp: datetime = Field(default_factory=datetime.now)
    
class MemoryCollection(BaseModel):
    memories: list[Memory] = Field(
        description="A list of memories about the user."
    )

model = ChatOpenAI(model="gpt-5-nano", temperature=0)
model_with_structure = model.with_structured_output(MemoryCollection)
memory_collection = model_with_structure.invoke([
    HumanMessage("My name is Lance. I like to bike.")
])    
for mem in memory_collection.memories:
    print(mem.model_dump())

## 保存到store
key = str(uuid.uuid4())  # 生成唯一 ID
value = memory_collection.memories[0].model_dump()
in_memory_store.put(namespace_for_memory, key, value)
```



## 高级图模式

### Parallelization

核心在于让多个节点在同一时间步内并行执行,充分利用计算资源。

在state上可以采取Reducer机制解决并行冲突，比如覆盖，拼接，去重，合并，**自定义**

###  Sub  Graph

**子图就是在一个大 Graph 内部的“可重用小 Graph”**，它本身也有节点、状态和执行逻辑，可以像普通 Node 一样被调用。

特点：可重用，可嵌套，可调试

适用场景：

- 独立的业务逻辑
- 需要状态隔离
- 可复用的流程
- 并行执行

#### 节点和子图

> 节点是执行单元，主要做“具体操作”；子图是逻辑单元，管理“执行流程和状态”，支持嵌套、复用和局部 Time Travel。

| 特性                     | Node                      | 子图 (Subgraph)                                 |
| ------------------------ | ------------------------- | ----------------------------------------------- |
| 定义粒度                 | 单个执行单元              | 一段**可重用**、独立的 Graph                    |
| State 管理               | 只能访问父 Graph 的 State | 有自己的内部 State，可隔离或映射到父 Graph      |
| 可组合性                 | 父 Graph 执行它 → 完      | 可以嵌套调用，组合成更大 Graph                  |
| 可复用性                 | 复用性低，逻辑固定        | 可以在多个父 Graph 中复用子图                   |
| Time Travel / checkpoint | 依赖父 Graph，粒度粗      | 内部可独立 **checkpoint**，局部 **Time Travel** |
| 调试可观测性             | 只能 stream 单节点变化    | 内部也可 stream / updates / events              |
| 嵌套深度                 | 通常是“叶子节点”          | 可以包含多个节点，甚至子图**嵌套**子图          |

### Map Reduce

#### 定义

**Map-Reduce 是一种将任务拆分成“多个小任务（Map）”，并最终汇总结果（Reduce）的方法**。
 在 LangGraph 中，它被用来处理**批量输入、并行子任务**，最终合并到父 Graph 的状态。

#### 组成

1 有Map function接受输入列表，执行节点或者子图；Reduce Function收集map阶段所有结果聚合

2 每个map任务有独立的state

#### 适用场景

- 需要对多个数据项执行相同操作（如批量翻译、批量摘要）
- 任务可以自然分解为独立子任务（如分段处理文档）
- 需要从多个候选结果中筛选（如生成多个答案选最佳）
- 数据量大，需要并行加速（如分析多个数据源）

```python
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict

## 父 Graph 的 State
class ParentState(TypedDict):
    inputs: list
    results: list

## 定义 Map 子图：对每个元素做操作
def map_task(item: int):
    return item * 2  # 简单倍数处理

## 定义 Reduce 节点
def reduce_task(state: ParentState, map_outputs):
    # 汇总所有 Map 输出
    state["results"] = map_outputs
    return state

## 构建父 Graph
builder = StateGraph(ParentState)

## Map-Reduce 节点
builder.add_map_reduce(
    name="map_reduce_example",
    items_key="inputs",          # 输入列表
    map_func=map_task,           # Map
    reduce_func=reduce_task      # Reduce
)

builder.set_entry_point("map_reduce_example")
graph = builder.compile(checkpointer=MemorySaver())

## 执行
res = graph.invoke({"inputs": [1,2,3,4], "results": []})
print(res)  # {'inputs': [1,2,3,4], 'results': [2,4,6,8]}
```

```python
graph.add_conditional_edges(
    "source_node",           # 源节点
    condition_function,      # 返回 Send 列表的函数
    ["target_node"]         # 目标节点（Send 指向的节点）
)
```

### Map-reduce VS 并行

类比 send vs add edge

动态处理任务 vs 静态处理任务



## Other

### 可视化图

```python
display(Image(graph.get_graph().draw_mermaid_png()))
```

### create_agent和手动构建stategraph

`create_agent` 是“行为封装”，`StateGraph` 是“流程编排”
 两者不是同一层级的东西，而是 可以叠加使用 的。

| 维度     | `create_agent`    | 手动 `StateGraph`   |
| -------- | ----------------- | ------------------- |
| 抽象层级 | 高（Agent 行为）  | 低（执行流程）      |
| 关注点   | 怎么思考 / 用工具 | 怎么走流程 / 管状态 |
| 控制权   | 交给 LLM          | 交给开发者          |
| 可预测性 | 低                | 高                  |
| 适合角色 | 单点智能体        | 系统级编排          |



## 参考资料

1 [learnGraph.online](https://learngraph.online/LearnGraph%201.X/module-0-python-fundamentals/0.1%20Python%20basics.html)

2 [AI-agents](https://github.com/Sutdown/AI-agents)



