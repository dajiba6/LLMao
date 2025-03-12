from openai import OpenAI
import yaml
import os

"""
!deepseek没提供异步接口
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

system_prompt = "you are an AI assistant"
user_prompt = "告诉我一个爆笑级笑话"
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": user_prompt},
]


# --------------------------------------------------------------
# 异步请求函数
# --------------------------------------------------------------
async def fetch_chat_completion(client, messages):
    return await client.chat.completions.create(
        model="deepseek-chat", messages=messages
    )


# --------------------------------------------------------------
# 主函数
# --------------------------------------------------------------
async def main():
    # 创建消息
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 串行请求
    start_time_serial = time.time()
    for _ in range(5):  # 假设我们发送5个请求
        completion_serial = await fetch_chat_completion(client, messages)
    end_time_serial = time.time()
    print(f"串行请求耗时: {end_time_serial - start_time_serial}秒")

    # 并行请求
    start_time_parallel = time.time()
    tasks = [fetch_chat_completion(client, messages) for _ in range(5)]
    await asyncio.gather(*tasks)
    end_time_parallel = time.time()
    print(f"并行请求耗时: {end_time_parallel - start_time_parallel}秒")


# --------------------------------------------------------------
# 运行主函数
# --------------------------------------------------------------
if __name__ == "__main__":
    asyncio.run(main())
