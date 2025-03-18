"""
orchestrator:分模块协作实现博客编写
"""

from openai import OpenAI
import yaml
import json
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
example output:
{
    "topic": "如何使用 JSON 存储文章",
    "target": "编程新手",
    "sections": [
        {
            "section_type": "简介",
            "title": "引言",
            "description": "介绍 JSON 的基本概念和用途",
            "style_guide": "简洁易懂，适合初学者",
            "target_length": 200
        },
        {
            "section_type": "语法",
            "title": "JSON 语法",
            "description": "详细讲解 JSON 的语法规则，包括数据类型、键值对、数组和嵌套对象",
            "style_guide": "提供代码示例，逐步解析",
            "target_length": 500
        },
        {
            "section_type": "应用",
            "title": "JSON 在 Web 开发中的应用",
            "description": "讲解 JSON 在前后端交互、API 数据传输等方面的应用",
            "style_guide": "结合实际案例，展示 JSON 在 API 调用中的使用",
            "target_length": 600
        },
        {
            "section_type": "总结",
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
example output:
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
example output:
{
  "cohesion_score":0.5
  "reviews": [
    {
      "section": "简介",
      "suggest": "引言部分可以增加 JSON 的定义，并举一个简单的 JSON 示例，以便读者更好理解。"
    },
    {
      "section": "语法",
      "suggest": "建议在 JSON 语法章节中添加 JSON 的数据类型表格，并提供错误 JSON 的示例，帮助理解常见错误。"
    },
    {
      "section": "应用",
      "suggest": "可以加入 JSON 在 Web 开发中的代码示例，比如如何解析 JSON 数据，以及 JSON 与 XML 的对比。"
    },
    {
      "section": "总结",
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
Target Length: {target_length}
Pervious Sections: {pervious_sections} 
"""

reviewer_prompt = """
Review this blog post for cohesion and flow:

Topic: {topic}
Target Audience: {target_audience}
Pervious Sections: {pervious_sections}
"""


# --------------------------------------------------------------
# 构建Orchestrator
# --------------------------------------------------------------
class BlogOrchestrator:
    def __init__(self) -> None:
        self.sections_content = {}

    def get_plan(self, topic: str, target_length: int, style: str):
        print("Generating plan...")
        messages = [
            {
                "role": "system",
                "content": planner_prompt.format(
                    topic=topic, target_length=target_length, style=style
                )
                + (f"\n{plan_struct}"),
            }
        ]
        compeletion = client.chat.completions.create(
            model=model,
            messages=messages,
            response_format={"type": "json_object"},
        )
        return compeletion.choices[0].message.content

    def write_section(
        self,
        topic: str,
        section_type: str,
        description: str,
        style_guide: str,
        target_length: int,
    ):
        pervious_sections = "\n\n".join(
            f"section_type:{section_type},section_content:{section_content}"
            for section_type, section_content in self.sections_content.items()
        )
        compeletion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": worker_prompt.format(
                        topic=topic,
                        section_type=section_type,
                        description=description,
                        style_guide=style_guide,
                        target_length=target_length,
                        pervious_sections=(
                            pervious_sections
                            if pervious_sections
                            else "this is the first section."
                        ),
                    )
                    + (f"\n{ content_struct}"),
                }
            ],
            response_format={"type": "json_object"},
        )
        return compeletion.choices[0].message.content

    def review_sections(self, topic: str, target_audience: str):
        pervious_sections = "\n\n".join(
            f"section_type:{section_type},section_content:{section_content}"
            for section_type, section_content in self.sections_content.items()
        )
        compeletion = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": reviewer_prompt.format(
                        topic=topic,
                        target_audience=target_audience,
                        pervious_sections=pervious_sections,
                    )
                    + (f"\n{review_struct}"),
                }
            ],
            response_format={"type": "json_object"},
        )
        return compeletion.choices[0].message.content

    def write_blog(
        self, topic: str, target_length: int, style: str, target_audience: str
    ):
        plan = json.loads(self.get_plan(topic, target_length, style))
        total_sections = len(plan["sections"])  # 获取总章节数
        for index, section in enumerate(plan["sections"], start=1):
            print(
                f"Working on section: {index}/{total_sections}: {section['section_type']}"
            )  # 显示当前进度
            res = json.loads(
                self.write_section(
                    topic,
                    section["section_type"],
                    section["description"],
                    section["style_guide"],
                    section["target_length"],
                )
            )
            content = res["content"]
            self.sections_content[section["section_type"]] = (
                content  # 修正为使用section["section_type"]
            )

        comment = json.loads(self.review_sections(topic, target_audience))
        return {"plan": plan, "content": self.sections_content, "review": comment}


if __name__ == "__main__":
    orchestrator = BlogOrchestrator()
    topic = "AI对会计行业的冲击"
    length = 1500
    style = "通俗易懂但不要有太多重复的废话,让人读了感觉这篇文章很有价值"
    audience = "会计从业人员"
    result = orchestrator.write_blog(topic, length, style, audience)
    print(f"Final Result:\n")
    print(f"cohesion_score:{result['review']['cohesion_score']}")
    print("Reviews:")
    # 打印每个章节的审查建议
    for review in result["review"]["reviews"]:
        print(f"sectioin: {review['section']}")
        print(f"suggestion: {review['suggest']}\n")

    print("Saving bolg...")
    # 将sections_content保存到data文件夹的txt文件中
    output_file_path = os.path.join("..", "data", f"{topic}_sections_content.txt")
    with open(output_file_path, "w", encoding="utf-8") as file:
        for section_type, section_content in orchestrator.sections_content.items():
            file.write(f"{section_type}:\n{section_content}\n\n")
    print(f"Blog save to {output_file_path}.")
