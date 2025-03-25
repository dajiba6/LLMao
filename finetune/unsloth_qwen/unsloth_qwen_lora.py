"""
Unsloth fine-tune
Unsloth llama3 fine-tune example: https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing#scrollTo=pCqnaKmlO1U9
Unsloth fine-tune tutorial doc: https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama
"""

from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer

max_seq_length = 2048
dtype = None
load_in_4bit = False

model_name = "Qwen/Qwen2-0.5B-Instruct"
local_model_path = "/home/cyn/.cache/huggingface/hub/models--unsloth--Qwen2-0.5B-Instruct/snapshots/878a7a23faba6f1ba34cca84ea9349d63fa82e5b"

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# --------------------------------------------------------------
# unsloth inference 测试
# --------------------------------------------------------------
# def UnslothInference(model, tokenizer):
print("Inference test...")
text_streamer = TextStreamer(tokenizer)

# 这部分代码有什么用?
# 解：对格式进行映射
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },
)
FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

messages = [
    {"from": "human", "value": "杭州的省会在哪里？"},
]
messages = [
    {"role": "user", "content": "杭州的省会在哪里？"},
]
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,  # Must add for generation
    return_tensors="pt",
).to("cuda")

from transformers import TextStreamer

text_streamer = TextStreamer(tokenizer, skip_prompt=True)
_ = model.generate(
    input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True
)


# --------------------------------------------------------------
# unsloth fine-tune参数
# --------------------------------------------------------------

# # ? uncloth具体来说用了什么方法?
print("Loading unsloth fine-tuning method...")
model = FastLanguageModel.get_peft_model(
    model,
    r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    lora_alpha=16,
    lora_dropout=0,  # Supports any, but = 0 is optimized
    bias="none",  # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
    random_state=3407,
    use_rslora=False,  # We support rank stabilized LoRA
    loftq_config=None,  # And LoftQ
)

# --------------------------------------------------------------
# prompt模板和数据集
# --------------------------------------------------------------
from unsloth.chat_templates import get_chat_template

tokenizer = get_chat_template(
    tokenizer,
    chat_template="chatml",
    mapping={
        "role": "from",
        "content": "value",
        "user": "human",
        "assistant": "gpt",
    },
)


# ? apply_chat_template里面参数的作用?
# add_generation_prompt:　添加生成提示,确保模型生成文本时只会给出答复，而不会做出意外的行为，比如继续用户的消息。 记住，聊天模型只是语言模型，它们被训练来继续文本，而聊天对它们来说只是一种特殊的文本！ 你需要用适当的控制标记来引导它们，让它们知道自己应该做什么。
def formatting_prompts_func(examples):
    convos = examples["conversations"]
    texts = [
        tokenizer.apply_chat_template(
            convo, tokenize=False, add_generation_prompt=False
        )
        for convo in convos
    ]
    return {
        "text": texts,
    }


pass

from modelscope.msdatasets import MsDataset
