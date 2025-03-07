from openai import OpenAI
import json
import yaml
import os

"""
暂无openai response_format的完善参数功能, 须手动构建json sturcture prompt
"""
# --------------------------------------------------------------
# 读取配置
# --------------------------------------------------------------
config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
    api_key = config.get("DEEPSEEK_API_KEY")
    base_url = config.get("DEEPSEEK_BASE_URL")

# --------------------------------------------------------------
# 构建structure prompt
# --------------------------------------------------------------
structure1 = """
用户将提供一些信息,请以下面的json格式输出分析信息. 
anger:[0~1]

example input:
你别给我们村丢脸,你个王八羔子.
example output:
{
    "place": "village",
    "anger": "1",
}
"""
# --------------------------------------------------------------
# 构建用户prompt
# --------------------------------------------------------------
user_prompt = "站住,不准在这美丽的国度乱跑,老子不会打死你"
messages = [
    {"role": "system", "content": structure1},
    {"role": "user", "content": user_prompt},
]
# --------------------------------------------------------------
# 请求api
# --------------------------------------------------------------
client = OpenAI(api_key=api_key, base_url=base_url)
compeletion = client.chat.completions.create(
    model="deepseek-chat", messages=messages, response_format={"type": "json_object"}
)
ori_reponse = compeletion.choices[0].message.content
print(ori_reponse)

json_response = json.loads(compeletion.choices[0].message.content)
print(json_response)
