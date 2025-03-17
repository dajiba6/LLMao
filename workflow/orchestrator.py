"""
orchestrator:分模块协作实现博客编写
"""

from openai import OpenAI
import yaml
import os


# --------------------------------------------------------------
# 读取配置
# --------------------------------------------------------------
config_file_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
with open(config_file_path, "r") as config_file:
    config = yaml.safe_load(config_file)
    api_key = config.get("DEEPSEEK_API_KEY")
    base_url = config.get("DEEPSEEK_BASE_URL")

client = OpenAI(api_key=api_key, base_url=base_url)
model = "deepseek-chat"

# --------------------------------------------------------------
# 数据结构
# --------------------------------------------------------------
plan_struct = """
用以下json格式进行输出:
topic:str,文章主题
target:str,目标读者
sections:list,包含所有章节的list
{
    "topic": "如何使用 JSON 存储文章",
    "target": "编程新手",
    "sections": [
        {
            "section_type": "introduction",
            "title": "引言",
            "description": "介绍 JSON 的基本概念和用途",
            "style_guide": "简洁易懂，适合初学者",
            "target_length": 200
        },
        {
            "section_type": "syntax",
            "title": "JSON 语法",
            "description": "详细讲解 JSON 的语法规则，包括数据类型、键值对、数组和嵌套对象",
            "style_guide": "提供代码示例，逐步解析",
            "target_length": 500
        },
        {
            "section_type": "application",
            "title": "JSON 在 Web 开发中的应用",
            "description": "讲解 JSON 在前后端交互、API 数据传输等方面的应用",
            "style_guide": "结合实际案例，展示 JSON 在 API 调用中的使用",
            "target_length": 600
        },
        {
            "section_type": "conclusion",
            "title": "总结",
            "description": "总结 JSON 的重要性，并提供学习建议",
            "style_guide": "简明扼要，总结关键点",
            "target_length": 150
        }
    ]
}
"""

content_struct = """
用以下json格式进行输出:
content:str,文章内容
{
 content: 文章内容.....
}
"""

review_struct = """
用以下json格式进行输出:
cohesion_score:float,0~1的数字,代表文章的连贯程度
review:list,包含不同章节的修改减
section:str,当前review的章节
suggest:str,具体修改建议
{
  "cohesion_score":0.5
  "reviews": [
    {
      "section": "introduction",
      "suggest": "引言部分可以增加 JSON 的定义，并举一个简单的 JSON 示例，以便读者更好理解。"
    },
    {
      "section": "syntax",
      "suggest": "建议在 JSON 语法章节中添加 JSON 的数据类型表格，并提供错误 JSON 的示例，帮助理解常见错误。"
    },
    {
      "section": "application",
      "suggest": "可以加入 JSON 在 Web 开发中的代码示例，比如如何解析 JSON 数据，以及 JSON 与 XML 的对比。"
    },
    {
      "section": "conclusion",
      "suggest": "总结部分建议增加 JSON 未来发展趋势的展望，比如 JSON Schema 的使用。"
    }
  ]
}
"""

# --------------------------------------------------------------
# prompt
# --------------------------------------------------------------
planner_prompt = """
Analyze this blog topic and break it down into logical sections. Consider the narrative flow and how sections will work together.

Topic: {topic}
Target Length: {target_length} words
Style: {style}
"""

worker_prompt = """
Write a blog section based on:
Topic: {topic}
Section Type: {section_type}
Section Goal: {description}
Style Guide: {style_guide}
"""

reviewer_prompt = """
Review this blog post for cohesion and flow:

Topic: {topic}
Target Audience: {audience}
"""


# --------------------------------------------------------------
# 构建Orchestrator
# --------------------------------------------------------------
class BlogOrchestrator:
    def __init__(self) -> None:
        self.sections_content = {}

    def get_plan(self, topic: str, target_length: int, sytle: str):
        compeletion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                }
            ],
        )
