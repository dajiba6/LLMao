from openai import OpenAI
import yaml
import os
from duckduckgo_search import DDGS
import json

"""
llm api接口实现调用本地工具
用搜索api搜索网络资料并总结
"""

# --------------------------------------------------------------
# 读取配置
# --------------------------------------------------------------
config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
    api_key = config.get("DEEPSEEK_API_KEY")
    base_url = config.get("DEEPSEEK_BASE_URL")

client = OpenAI(api_key=api_key, base_url=base_url)


# --------------------------------------------------------------
# 定义函数:网络搜索
# --------------------------------------------------------------
def duckducknews(query: str) -> list:
    with DDGS() as ddgs:
        return list(ddgs.news(keywords=query, region="cn-zh", max_results=10))


def duckducktext(query: str) -> list:
    with DDGS() as ddgs:
        return list(ddgs.text(keywords=query, region="cn-zh", max_results=10))


# --------------------------------------------------------------
# 定义工具
# --------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "duckducknews",
            "description": "search infomation on internet through duckduckgo.news()",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "the keywords that user want to search",
                    }
                },
                "required": ["query"],
            },
        },
    },
    # 测试例子
    #! deepseek无脑调用工具,没有自我判别能力?
    {
        "type": "function",
        "function": {
            "name": "add",
            "description": "add two number and return sum",
            "parameters": {
                "type": "object",
                "properties": {
                    "num1": {"type": "number"},
                    "num2": {"type": "number"},
                },
                "required": ["num1", "num2"],
            },
        },
    },
]

system_prompt = "你是一个人工智能助手"
user_prompt = "帮我搜集一些关于中国ai产品manus的信息,并总结向我汇报"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]

# --------------------------------------------------------------
# 加tools请求api
# --------------------------------------------------------------
compeletioin = client.chat.completions.create(
    model="deepseek-chat", messages=messages, tools=tools
)
print(compeletioin.model_dump())


# --------------------------------------------------------------
# 调用工具
# --------------------------------------------------------------
def call_function(name, args):
    if name == "duckducknews":
        return duckducknews(**args)


for tool_call in compeletioin.choices[0].message.tool_calls:
    name = tool_call.function.name
    args = json.loads(tool_call.function.arguments)
    messages.append(compeletioin.choices[0].message)

    result = call_function(name, args)
    messages.append(
        {"role": "tool", "tool_call_id": tool_call.id, "content": json.dumps(result)}
    )
print(f"result: \n{json.dumps(result)}")
# --------------------------------------------------------------
# 返回工具调用结果给api,第二次请求
# --------------------------------------------------------------
# deepseek官方文档没有openai的completions.parse
# todo: deepseek只要加了tools,就一直tool call,不产生content内容
completion2 = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    #!20250307:deepseek无脑调用工具
    # tools=tools
)

print(f"final_response: \n{completion2.choices[0].message.content}")
