from openai import OpenAI
import json
import yaml
import os

"""
结构化输出llm
输出愤怒程度0~1

deepseek暂无openai response_format的完善参数功能, 须手动构建json sturcture prompt
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


# --------------------------------------------------------------
# 用instructor库实现llm结构化输出
# --------------------------------------------------------------
"""
# 添加llm结构化输出
import instructor
from pydantic import BaseModel

class EvaluationOutput(BaseModel):
    name: str
    age: int

def generate_evaluation_structured(_model: Llama, _messages: str) -> str:
    # add structured output
    create = instructor.patch(
        create=_model.create_chat_completion_openai_v1,
        mode=instructor.Mode.JSON_SCHEMA,
    )

    response = create(
        messages = _messages,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        max_tokens=4096,
        temperature=0,
        response_model=EvaluationOutput,
    )
    return response


## 结构化输出测试
test_messages =[
    {"role":"system","content":"你是一个人工智能助手"},
    {"role":"user","content":"老王是一个爱捡垃圾的老头. 分析这个人的年纪和姓名"}
] 
test_res = generate_evaluation_structured(myModel,test_messages)
print(test_res)



"""
