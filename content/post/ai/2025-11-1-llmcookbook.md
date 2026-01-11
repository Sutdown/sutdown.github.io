---
title:        "LLM_cookbook 面向开发者的大模型入门教程"
description:  "本文是基于吴恩达大模型系列课程的学习笔记，介绍面向开发者的提示工程和大模型应用开发基础知识。"
date:         2025-11-01
toc: true
categories:
    - AI
image: /images/f874fc13.jpg
---

## 0 前言

主要参考这份资料[datawhalechina/llm-cookbook: 面向开发者的 LLM 入门教程，吴恩达大模型系列课程中文版](https://github.com/datawhalechina/llm-cookbook)。类似于学习笔记，文字笔记部分摘自原文，代码的部分进行了修改，一个是重新修正了代码逻辑，关于一些库的更新也用了新的函数运行；另一个在于将openai的app key改成了阿里的通义千问，有部分免费额度，国内运行也比较稳定。



## 1 面向开发者的提示工程

> **Prompt Engineering**，即是针对特定任务构造能充分发挥大模型能力的 Prompt 的技巧。
>
> 本部分内容基于**吴恩达老师的《Prompt Engineering for Developer》课程**进行编写。

### 简介Introduction

对于开发人员，**大语言模型（LLM） 的更强大功能是能通过 API 接口调用，从而快速构建软件应用程序**。

随着 LLM 的发展，其大致可以分为两种类型，后续称为**基础 LLM** 和**指令微调（Instruction Tuned）LLM**。

- **基础LLM**是基于文本训练数据，训练出预测下一个单词能力的模型。其通常通过在互联网和其他来源的大量数据上训练，来确定紧接着出现的最可能的词。

- **指令微调 LLM** 通过专门的训练，可以更好地理解并遵循指令。

  指令微调 LLM 的训练通常基于预训练语言模型，先在大规模文本数据上进行**预训练**，掌握语言的基本规律。在此基础上进行进一步的训练与**微调（finetune）**，输入是指令，输出是对这些指令的正确回复。有时还会采用**RLHF（reinforcement learning from human feedback，人类反馈强化学习）**技术，根据人类对模型输出的反馈进一步增强模型遵循指令的能力。通过这种受控的训练过程。指令微调 LLM 可以生成对指令高度敏感、更安全可靠的输出，较少无关和损害性内容。

### 1.1 提示原则Guidelines

本章讨论了设计高效 Prompt 的两个关键原则：**编写清晰、具体的指令**和**给予模型充足思考时间**。

#### 原则一：编写清晰具体的指令

下面的几点要求，对于指令清晰则是从输入表示，输出结构，输出检查和简单示例四部分构成，比较经典的过程流。

1 使用分隔符清晰的表示输入的不同部分。有利于防止提示词注入。

2 寻求结构化输出，比如JSON，HTML等格式。

3 要求模型检查是否满足条件，检查不满足则不输出。

4 提供少量示例。即"Few-shot" prompting，在要求模型执行实际任务之前，给模型一两个已完成的样例，让模型了解我们的要求和期望的输出样式。

#### 原则二：给模型时间去思考

1 制定完成任务所需的步骤。而不是直接让他盲目的得到最终结果。

2 指导模型在下结论之前找出自己的解法。因为即使你给的是错误解法，大模型也很容易认为这是正确的，从而导致误判。

#### 局限性

模型偶尔会生成一些看似真实实则编造的知识。

语言模型生成虚假信息的“幻觉”问题，是使用与开发语言模型时需要高度关注的风险。由于幻觉信息往往令人无法辨别真伪，开发者必须警惕并尽量避免它的产生。

### 1.2 迭代优化Iterative

在开发大语言模型应用时，很难通过第一次尝试就得到完美适用的 Prompt。但关键是要有一个**良好的迭代优化过程**，以不断改进 Prompt。相比训练机器学习模型，Prompt 的一次成功率可能更高，但仍需要通过多次迭代找到最适合应用的形式。

### 1.3 文本概括Summarizing

- 单一文本概括
- 多文本概括

### 1.4 推断Inferring

- 情感推断
- 信息提取
- 主题推断

### 1.5 文本转换Transforming

- 文本翻译
- 预期和写作风格调整
- 文件格式转换
- 拼写以及语法纠正

### 1.6 文本扩展Expanding

- 温度系数：一般来说，如果需要可预测、可靠的输出，则将 temperature 设置为0，如果需要更具创造性的多样文本，那么适当提高 temperature 则很有帮助。调整这个参数可以灵活地控制语言模型的输出特性。

### 1.7 聊天机器人Chatbot



## 2 搭建基于ChatGPT的问答系统

### 2.1 基础概念

**大型语言模型**主要可以分为两类:基础语言模型和指令调优语言模型。

- **基础语言模型**（Base LLM）通过反复预测下一个词来训练的方式进行训练，没有明确的目标导向。
- **指令微调的语言模型**（Instruction Tuned LLM）则进行了专门的训练，以便更好地理解问题并给出符合指令的回答。

**LLM 实际上并不是重复预测下一个单词，而是重复预测下一个 token** 。

这种提问格式，我们可以明确地角色扮演，让语言模型理解自己就是助手这个角色，需要回答问题。这可以减少无效输出，帮助其生成针对性强的回复。



### 2.2 电商客服 AI 系统简单框架

> 这个搭建问答系统更偏向于应用层面，可以大体理解一下整体逻辑，从数据集准备（评估输入的分类），到输入检查（检查输入数据是否合法），再到ai的思考（其中涉及的思维链和prompt链），最后输出（检查输出信息是否有效，是否有害）。只是去简单的实现一个调用，缺少的像这篇文章里面的[Langchain到简单Agent - 江舟的博客 | Sutdown Blog](https://sutdown.github.io/2025/10/06/langchain/)比如RAG，Memory之类的并没有详细阐述。建议大致看看这部分即可，用于基本了解。

#### 评估输入 - 分类

类似描述的数据集，将各种信息分类完成，比如如果只是一个简单的商品检索助手，那么分类有助于在ai查找信息时更加快速查找到需要的数据。

#### 检查输入 - 监督

- 审核
- prompt注入

审核的在于审核两点：1 是否输入一些不安全信息，比如违法信息

#### 思维链推理 

- 思维链提示设计
- 内心独白

#### Prompt链 

- 提取产品和类别
- 检索详细信息
- 生成查询答案
- 总结

#### 检查结果

- 检查有害内容
- 检查是否符合产品信息

#### 搭建一个带评估的端到端系统

- 端到端实现问答系统
  1. 对用户的输入进行检验，验证其是否可以通过审核 API 的标准。
  2. 若输入顺利通过审核，我们将进一步对产品目录进行搜索。
  3. 若产品搜索成功，我们将继续寻找相关的产品信息。
  4. 我们使用模型针对用户的问题进行回答。
  5. 最后，我们会使用审核 API 对生成的回答进行再次的检验。
- 持续收集用户和助手消息

#### 评估

- 针对不同的测试用例进行测试，比较理想答案和输出答案比较评估xia0guo
- 注意回归测试：验证模型在以前的代码上的效果
- 运用gpt进行自行评估

#### 完整代码

这个主要实现一个完整的电商客服AI框架，基于[datawhalechina/llm-cookbook](https://github.com/datawhalechina/llm-cookbook)的原代码，进行了适当变化，精简逻辑，添加注释，将openAI appkey改成了阿里的通义千问，openAI国内不太好访问。

```python
import json
import os
from langchain_community.chat_models import ChatTongyi
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv()

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)

# 商品和目录的数据文件
PRODUCTS_FILE = "products.json"
CATEGORIES_FILE = "categories.json"
DELIMITER = "####"

# -------------------------- 系统提示词（修正原重复类别问题）--------------------------
# 第二步（抽取商品）系统信息文本，校验不同类别，并返回一个列表，其中包含所有类别。
# 提取问题分布解决，此为思维链
step_2_system_message_content = f"""
您将获得一次客户服务对话。最近的用户查询将使用{DELIMITER}字符进行分隔。

输出一个Python对象列表，其中每个对象具有以下格式：
'category': <包括以下几个类别：Computers and Laptops、Smartphones and Accessories、Televisions and Home Theater Systems、Gaming Consoles and Accessories、Audio Equipment、Cameras and Camcorders
'products': <必须是下面的允许产品列表中找到的产品>

类别和产品必须在客户服务查询中找到。
如果提到了产品，它必须与下面的允许产品列表中的正确类别相关联。
如果未找到任何产品或类别，请输出一个空列表。
只列出之前对话的早期部分未提及和讨论的产品和类别。

允许的产品：

Computers and Laptops类别：
TechPro Ultrabook
BlueWave Gaming Laptop
PowerLite Convertible
TechPro Desktop
BlueWave Chromebook

Smartphones and Accessories类别：
SmartX ProPhone
MobiTech PowerCase
SmartX MiniPhone
MobiTech Wireless Charger
SmartX EarBuds

Televisions and Home Theater Systems类别：
CineView 4K TV
SoundMax Home Theater
CineView 8K TV
SoundMax Soundbar
CineView OLED TV

Gaming Consoles and Accessories类别：
GameSphere X
ProGamer Controller
GameSphere Y
ProGamer Racing Wheel
GameSphere VR Headset

Audio Equipment类别：
AudioPhonic Noise-Canceling Headphones
WaveSound Bluetooth Speaker
AudioPhonic True Wireless Earbuds
WaveSound Soundbar
AudioPhonic Turntable

Cameras and Camcorders类别：
FotoSnap DSLR Camera
ActionCam 4K
FotoSnap Mirrorless Camera
ZoomMaster Camcorder
FotoSnap Instant Camera

只输出对象列表，不包含其他内容。
"""

# 第四步（生成用户回答）的系统信息，添加身份，进一步区分，可以理解成prompt chain
step_4_system_message_content = f"""
    你是一家大型电子商店的客户服务助理。
    以友好和乐于助人的语气回答，回答保持简洁明了。
    确保让用户提出相关的后续问题。
"""

# 第六步（验证模型回答）的系统信息，重新根据数据校验结果
# 思维链的一部分，检查结果
step_6_system_message_content = f"""
你是一个助手，评估客户服务代理的回答是否足够回答客户的问题，并验证回答中所有产品信息是否与提供的商品数据一致。
请基于以下三部分内容进行判断：
1. 用户的问题
2. 客服的回答
3. 商品数据集（包含所有产品的真实信息）

输出格式：
Y - 回答足够且所有产品信息与数据集一致
N - 回答不足够，或存在与数据集不符的信息

只输出一个字母。
"""

# -------------------------- 模型调用函数（基于 ChatTongyi）--------------------------
def call_llm(messages):
    """统一的模型调用接口"""
    try:
        response = llm.invoke(messages)
        return response.content.strip()
    except Exception as e:
        print(f"[模型错误] {e}")
        return ""

# --------------------------基础数据加载--------------------------
def load_json_file(path):
    """从文件读取 JSON 数据"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"未找到文件：{path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_products():
    return load_json_file(PRODUCTS_FILE)

def load_categories():
    return load_json_file(CATEGORIES_FILE)

# ---------------------- 功能函数 ----------------------
def extract_products_and_categories(user_msg):
    """调用模型识别用户提到的产品和类别"""
    messages = [
        SystemMessage(content=step_2_system_message_content),
        HumanMessage(content=f"{DELIMITER}{user_msg}{DELIMITER}")
    ]
    return call_llm(messages)

def read_string_to_list(json_like_str):
    """将模型输出转为 Python 对象"""
    if not json_like_str:
        return []
    try:
        fixed = json_like_str.replace("'", '"')
        return json.loads(fixed)
    except json.JSONDecodeError:
        print("[警告] 无法解析模型输出：", json_like_str)
        return []

def generate_product_info(data_list, products):
    """根据识别结果提取产品详情"""
    info_text = ""
    for item in data_list:
        for pname in item.get("products", []):
            if pname in products:
                info_text += json.dumps(products[pname], ensure_ascii=False, indent=2) + "\n"
    return info_text.strip()

def answer_user_question(user_msg, product_info):
    """生成客服回答"""
    messages = [
        SystemMessage(content=step_4_system_message_content),
        HumanMessage(content=f"用户问题：{user_msg}\n\n相关产品信息：\n{product_info}")
    ]
    return call_llm(messages)

def validate_answer(user_msg, answer, products):
    """验证客服回答是否正确（传入商品数据作为参考）"""
    # 将商品数据转为字符串，作为参考信息传入
    products_str = json.dumps(products, ensure_ascii=False, indent=2)
    messages = [
        SystemMessage(content=step_6_system_message_content),
        HumanMessage(content=f"""
		用户问题：{user_msg}
		客服回答：{answer}
		商品数据集：{products_str}
        """.strip())
    ]
    return call_llm(messages)

def main():
    """
    电商客服 AI 系统主流程
    """
    print("=== Step 1: 初始化商品与分类数据 ===")
    products = load_products()

    print("=== Step 2: 模型识别用户提到的商品和类别 ===")
    user_msg = "你好，我想了解一下 SmartX ProPhone 的电池续航，以及 CineView 8K TV 有没有HDR功能？"
    print(f"用户消息：{user_msg}")
    response = extract_products_and_categories(user_msg)
    print(f"模型识别结果（原始文本）：\n{response}")

    data_list = read_string_to_list(response)
    print(f"解析后结构：{data_list}")
    product_info_str = generate_product_info(data_list, products)
    print(f"生成的产品信息：\n{product_info_str}")

    print("\n=== Step 3: 生成客服回答 ===")
    answer = answer_user_question(user_msg, product_info_str)
    print(f"客服回答：\n{answer}")

    print("\n=== Step 4: 检查回答质量 ===")
    validation = validate_answer(user_msg, answer, product_info_str)
    print(f"验证结果（Y=合格，N=不合格）：{validation}")

if __name__ == "__main__":
    main()

```



> 基于之前有篇文章写过langchain[Langchain到简单Agent - 江舟的博客 | Sutdown Blog](https://sutdown.github.io/2025/10/06/langchain/)，都是些很基础的用法，这里偏向深层次的语法，可以看看，后续再出个具体的应用代码写的agent，用langgraph尽量。

## 3 基于LangChain开发应用程序

### 3.1 基础介绍

LangChain 是用于构建大模型应用程序的开源框架，有Python和JavaScript两个不同版本的包。LangChain 也是一个开源项目，社区活跃，新增功能快速迭代。LangChain基于模块化组合，有许多单独的组件，可以一起使用或单独使用。

 LangChain 的常用组件：

- 模型(Models)：集成各种语言模型与向量模型。
- 提示(Prompts)：向模型提供指令的途径。
- 索引(Indexes)：提供数据检索功能。
- 链(Chains)：将组件组合实现端到端应用。
- 代理(Agents)：扩展模型的推理能力



### 3.2 组件

#### 模板

在应用于比较复杂的场景时，提示可能会非常长并且包含涉及许多细节。**使用提示模版，可以让我们更为方便地重复使用设计好的提示**。

当然对于特定的条件，也可以采用输出解释器提取用户评价中的信息。

```python
import os

from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.chat_models import ChatTongyi
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)

"""
    prompt template
    默认进行了embedding
"""
customer_style_res = """正式普通话 \
用一个平静、尊敬、有礼貌的语调
"""
customer_res = """
嗯呐，我现在可是火冒三丈，我那个搅拌机盖子竟然飞了出去，把我厨房的墙壁都溅上了果汁！
更糟糕的是，保修条款可不包括清理我厨房的费用。
伙计，赶紧给我过来！
"""

service_reply = """嘿，顾客， \
保修不包括厨房的清洁费用， \
因为您在启动搅拌机之前 \
忘记盖上盖子而误用搅拌机, \
这是您的错。 \
倒霉！ 再见！
"""
service_style_pirate = """\
一个有礼貌的语气 \
使用海盗风格\
"""

customer_review = """\
这款吹叶机非常神奇。 它有四个设置：\
吹蜡烛、微风、风城、龙卷风。 \
两天后就到了，正好赶上我妻子的\
周年纪念礼物。 \
我想我的妻子会喜欢它到说不出话来。 \
到目前为止，我是唯一一个使用它的人，而且我一直\
每隔一天早上用它来清理草坪上的叶子。 \
它比其他吹叶机稍微贵一点，\
但我认为它的额外功能是值得的。
"""
review_template = """\
对于以下文本，请从中提取以下信息：

礼物：该商品是作为礼物送给别人的吗？ \
如果是，则回答 是的；如果否或未知，则回答 不是。

交货天数：产品需要多少天\
到达？ 如果没有找到该信息，则输出-1。

价钱：提取有关价值或价格的任何句子，\
并将它们输出为逗号分隔的 Python 列表。

使用以下键将输出格式化为 JSON：
礼物
交货天数
价钱

文本: {text}
"""

review_template_2 = """\
对于以下文本，请从中提取以下信息：：

礼物：该商品是作为礼物送给别人的吗？
如果是，则回答 是的；如果否或未知，则回答 不是。

交货天数：产品到达需要多少天？ 如果没有找到该信息，则输出-1。

价钱：提取有关价值或价格的任何句子，并将它们输出为逗号分隔的 Python 列表。

文本: {text}

{format_instructions}
"""
"""
    功能函数
"""
def translate_email(style, customer_email):
    human_prompt = HumanMessagePromptTemplate.from_template("""
    请把由三个反引号分隔的文本翻译成一种{style}风格。
    文本: ```{customer_email}```
    """)
    system_prompt = SystemMessagePromptTemplate.from_template("你是一个ai助手。")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    messages = prompt.format_messages(style=style, customer_email=customer_email)
    response = llm.invoke(messages)
    print(" - messages\n", response.content)

def translate_review(text):
    human_prompt = HumanMessagePromptTemplate.from_template(review_template)
    system_prompt = SystemMessagePromptTemplate.from_template("你是一个ai助手。")

    prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])
    messages = prompt.format_messages(text=text)
    response = llm.invoke(messages)
    print(" - messages\n", response.content)

def translate_review_2(text):
    prompt = ChatPromptTemplate.from_template(template=review_template_2)
    gift_schema = ResponseSchema(name="礼物",
                                description="这件物品是作为礼物送给别人的吗？\
                                如果是，则回答 是的，\
                                如果否或未知，则回答 不是。")
    delivery_days_schema = ResponseSchema(name="交货天数",
                                          description="产品需要多少天才能到达？\
                                          如果没有找到该信息，则输出-1。")
    price_value_schema = ResponseSchema(name="价钱",
                                        description="提取有关价值或价格的任何句子，\
                                        并将它们输出为逗号分隔的 Python 列表")
    response_schemas = [gift_schema,
                        delivery_days_schema,
                        price_value_schema]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    messages = prompt.format_messages(text=text, format_instructions=format_instructions)
    response = llm.invoke(messages)
    print(" - messages\n", response.content)

if __name__ == "__main__":
    # translate_email(customer_style_res, customer_res)
    # translate_email(service_style_pirate, service_reply)
    # translate_review(customer_review)
    translate_review_2(customer_review)
```



#### Memory

在 LangChain 中，储存指的是大语言模型（LLM）的短期记忆。当使用 LangChain 中的储存(Memory)模块时，它旨在保存、组织和跟踪整个对话的历史，从而为用户和模型之间的交互提供连续的上下文。

LangChain 提供了多种储存类型。这些记忆组件都是模块化的，可与其他组件组合使用，从而增强机器人的对话管理能力。储存模块可以通过简单的 API 调用来访问和更新，允许开发人员更轻松地实现对话历史记录的管理和维护。

- 缓冲区储存允许保留最近的聊天消息，
- 摘要储存则提供了对整个对话的摘要。
- 实体储存则允许在多轮对话中保留有关特定实体的信息。

| 类型                         | 保存方式       | 优点                       | 缺点                |
| ---------------------------- | -------------- | -------------------------- | ------------------- |
| `InMemoryChatMessageHistory` | 内存           | 简单，快速                 | 会话结束丢失数据    |
| `ConversationBufferMemory`   | 内存，链友好   | 可以直接在 chain 中使用    | 长对话 token 会增大 |
| `ConversationSummaryMemory`  | 内存 + LLM摘要 | 节省 token，保持核心上下文 | 需要额外调用 LLM    |
| 自定义 Memory                | 任意           | 灵活                       | 需要自己管理逻辑    |

单对话单用户，下面代码尝试了一下简单的角色扮演，能够记住上下文，角色扮演的语气场景都不错，看来千问的效果很好。

```python
llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)

# 存储对话的地方（取代 ConversationBufferMemory）
store = {}  # 模拟 session 存储

def get_session_history(session_id: str):
    """根据 session_id 获取或创建消息历史"""
    if session_id not in store:
        # InMemoryChatMessageHistory() 是一个用来保存对话消息历史的容器。
        # 可以保存，用户说的话 和 模型回答
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def runnable_func(inputs):
    history = inputs.get("chat_history", [])
    prompt = ""
    for msg in history:
        prompt += f"{msg.type}: {msg.content}\n"
    prompt += f"user: {inputs['input']}\n"
    return llm.invoke(prompt)

# runablewithmeaasgehistory是一个通用组件，用于处理会话历史
runnable = RunnableLambda(runnable_func)
chain = RunnableWithMessageHistory(
    runnable=runnable,
    get_session_history=get_session_history,
    input_messages_key="input",          # 输入字段名
    history_messages_key="chat_history"  # 历史字段名
)

# 指定一个 session_id（比如不同用户或会话）
session_id = "user1"

print("🧠 开始多轮对话（输入 'exit' 退出）")
while True:
    user_input = input("👤 你：").strip()
    if user_input.lower() in {"exit", "quit"}:
        print("🚪 结束对话")
        break

    response = chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": session_id}}
    )

    print(f"🤖 AI：{response.content}\n")
```



#### 模型链 - Chain

链（Chains）通常将大语言模型（LLM）与提示（Prompt）结合在一起，基于此，我们可以对文本或数据进行一系列操作。链（Chains）可以一次性接受多个输入。

##### 简单顺序链

```python
llm = ChatTongyi(model="qwen3-max", api_key=DASHSCOPE_API_KEY)

first_prompt = ChatPromptTemplate.from_template(
    "把下面的评论review翻译成英文:"
    "\n\n{Review}"
)

# prompt模板 2: 用一句话总结下面的 review
second_prompt = ChatPromptTemplate.from_template(
    "请你用一句话来总结下面的评论review:"
    "\n\n{English_Review}"
)

# prompt模板 3: 下面review使用的什么语言
third_prompt = ChatPromptTemplate.from_template(
    "下面的评论review使用的什么语言:\n\n{Review}"
)

# prompt模板 4: 使用特定的语言对下面的总结写一个后续回复
fourth_prompt = ChatPromptTemplate.from_template(
    "使用特定的语言对下面的总结写一个后续回复:"
    "\n\n总结: {summary}\n\n语言: {language}"
)

# 将 overall_chain 修改为直接串联各个步骤
# 使用 RunnableMap 一次性处理所有步骤
overall_chain = RunnableMap({
    "English_Review": lambda x: (first_prompt | llm).invoke({"Review": x["Review"]}),
    "summary": lambda x: (second_prompt | llm).invoke({
        "English_Review": (first_prompt | llm).invoke({"Review": x["Review"]})
    }),
    "language": lambda x: (third_prompt | llm).invoke({"Review": x["Review"]}),
    "followup_message": lambda x: (fourth_prompt | llm).invoke({
        "summary": (second_prompt | llm).invoke({
            "English_Review": (first_prompt | llm).invoke({"Review": x["Review"]})
        }),
        "language": (third_prompt | llm).invoke({"Review": x["Review"]})
    })
})


review_text = "部员们都很有个性——真田那家伙，严肃得像块石头，但其实比谁都可靠；柳生总是冷静分析，却会在切原胡闹时默默帮他收拾烂摊子；仁王那狐狸，总爱捉弄人，可关键时刻从不含糊；还有丸井和桑原，一个爱吹泡泡糖，一个沉默却温柔，他们之间的默契谁都比不上。"
result = overall_chain.invoke({"Review": review_text})
print("英文评论:", result["English_Review"].content)
print("评论总结:", result["summary"].content)
print("评论语言:", result["language"].content)
print("后续回复:", result["followup_message"].content)
```

##### 路由链

如果你有多个子链，每个子链都专门用于特定类型的输入，那么可以组成一个路由链，它首先决定将它传递给哪个子链，然后将它传递给那个链。

路由器由两个组件组成：

- 路由链（Router Chain）：路由器链本身，负责选择要调用的下一个链
- destination_chains：路由器链可以路由到的链



#### 基于文档的问答



#### 评估 - Evaluation



#### 代理 - Agent



## 4 使用Langchain访问个人数据

### 4.1 基础介绍

### 4.2 组件

#### 文档加载

#### 文档分割

#### 向量数据库与词向量

#### 检索

#### 问答

#### 聊天

### 4.3 总结  

##### 



