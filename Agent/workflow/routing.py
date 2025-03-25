"""
routing:检测到二次元浓度过低时推荐一部动漫, 过高则分享一个相关梗
"""

from openai import OpenAI
import json
import yaml
import os
import sys
import logging

# --------------------------------------------------------------
# 读取工具
# --------------------------------------------------------------
# todo:统一工具的调用
# 读取工具
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.search import duckducktext, duckducknews

# 读取工具描述
tools_path = os.path.join(os.path.dirname(__file__), "..", "utils", "tool.json")
with open(tools_path, "r", encoding="utf-8") as file:
    tools = json.load(file)
# 创建工具字典
tools_dict = {tool["function"]["name"]: tool for tool in tools}
search_tool_info = tools_dict.get("duckducktext")
tools = [search_tool_info]


# --------------------------------------------------------------
# 读取配置
# --------------------------------------------------------------
# Set up logging configuration
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - Line %(lineno)d - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
    api_key = config.get("DEEPSEEK_API_KEY")
    base_url = config.get("DEEPSEEK_BASE_URL")
client = OpenAI(api_key=api_key, base_url=base_url)

# --------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------
structure1 = """

请以以下json结构输出内容
example output
{
  dencity_weeb:
  anime:
  original_content:
}
output_description:
dencity_weeb:0~1的数值,判断此人对二次元动漫文化爱好程度
anime:string,可能喜欢的动漫
original_content:原始输入信息

"""


# --------------------------------------------------------------
# 定义函数
# --------------------------------------------------------------
# 检测二次元含量
def weeb_detection(input_prompt: str):
    system_prompt = "你是一个ai智能助手"
    user_prompt = "帮我分析这个人对二次元动漫的喜好程度,并找出他可能喜欢的动画"
    user_prompt = user_prompt + input_prompt + structure1
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    compeletion = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        response_format={"type": "json_object"},
    )
    return compeletion.choices[0].message.content


# 调用搜索工具
def search_meme(prompt: str):
    system_prompt = "你是一个ai智能助手"
    user_prompt = "根据以下分析,搜索和该动漫相关的梗"
    user_prompt = user_prompt + prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    compeletion = client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=tools
    )
    result = compeletion.choices[0].message.tool_calls
    for tool_call in result:
        name = tool_call.function.name
        args = json.loads(tool_call.function.arguments)
        #! 必须要在messages中添加这个，不然后面调用llm接口会报错
        messages.append(compeletion.choices[0].message)
        result = call_function(name, args)
        logger.debug(f"function_result:\n{result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            }
        )
    return messages


# 工具
def call_function(name, args):
    if name == "duckducktext":
        #! duckducktext接口使用一直报错
        return duckducknews(**args)


# 生成梗
def generate_meme(pervious_message: list):
    system_prompt = "你是一个ai智能助手"
    user_prompt = (
        "根据以下分析和搜索内容,生成一个和该动漫相关的梗,分享给二次元动漫爱好者"
    )
    user_prompt = user_prompt
    pervious_message.extend(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    logger.debug(f"message:\n {pervious_message}")
    compeletion = client.chat.completions.create(
        model="deepseek-chat",
        messages=pervious_message,
    )
    return compeletion.choices[0].message.content


# 动漫推荐
def anime_recommend(prompt: str):
    search_response = duckducknews("当前最热门的动漫")
    system_prompt = "你是一个ai智能助手"
    user_prompt = "根据以下搜索结果以及用户分析，为用户推荐动漫"
    user_prompt = user_prompt + prompt + str(search_response)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    compeletion = client.chat.completions.create(
        model="deepseek-chat", messages=messages
    )
    return compeletion.choices[0].message.content


# --------------------------------------------------------------
# chain 在一起
# --------------------------------------------------------------
def weeb_helper(prompt):
    logger.info(f"input prompt:\n{prompt}")
    detect_result = weeb_detection(prompt)
    logger.info(f"weeb detect result:\n{detect_result}")
    detect_result = json.loads(detect_result)
    if detect_result["dencity_weeb"] < 0.5:
        final_result = anime_recommend(detect_result["anime"])
        logger.info(final_result)
    else:
        search_result = search_meme(detect_result["anime"])
        logger.debug(f"search_result:\n{search_result}")
        final_result = generate_meme(search_result)
        logger.info(final_result)


test_prompt = "我是成年人，小孩才看动漫！"
weeb_helper(test_prompt)
