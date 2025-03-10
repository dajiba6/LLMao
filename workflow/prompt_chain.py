from openai import OpenAI
import json
import yaml
import os
from utils.search import duckducktext

"""
prompt_chain:检测到二次元浓度过低时不给于回复, 过高则分享一个相关梗
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
# 定义函数:二次原程度判断
# --------------------------------------------------------------
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
    )
    return compeletion.choices[0].message.content


def anime_meme(prompt: str):
    system_prompt = "你是一个ai智能助手"
    user_prompt = "根据以下分析,生成一个和该动漫相关的梗,分享给二次元动漫爱好者"
    user_prompt = user_prompt + prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    compeletion = client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
    )
    return compeletion.choices[0].message.content


# --------------------------------------------------------------
# chain 在一起
# --------------------------------------------------------------

test_prompt = "我要给木叶村最强忍者宇智波佐助生猴子"

result1 = weeb_detection(test_prompt)
print(f"weeb detect result:\n{result1}")
print(f"type of result1: {type(result1)}")
