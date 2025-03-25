from openai import OpenAI
import yaml
import os

"""
llm api请求
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
"""
Presence Penalty: 控制模型是否会尝试引入新的单词。它影响的是单词是否被使用过, 而不关注单词的使用频率. 提高该值会鼓励模型生成新的词汇,而不是重复已经出现的词.
Frequency Penalty: 控制模型生成重复单词的频率。它基于每个单词在生成过程中的频率来施加惩罚。提高该值会让模型避免频繁生成相同的单词或短语。
"""
# --------------------------------------------------------------
# 请求api
# --------------------------------------------------------------
completion = client.chat.completions.create(
    model="deepseek-chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "给我写一首中国风的古诗,主题是在办公室辛勤工作的人,注意押韵。",
        },
    ],
    temperature=1.5,
    max_tokens=100,
    frequency_penalty=2.0,
)
response1 = completion.choices[0].message.content
print(response1)

response2 = completion.choices[1].message.content
print(response2)
