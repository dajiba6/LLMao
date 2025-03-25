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
model_qwen, tokenizer = FastLanguageModel.from_pretrained(
    model_name=local_model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)


# --------------------------------------------------------------
# unsloth fine-tune参数
# --------------------------------------------------------------

# # ? uncloth具体来说用了什么方法?
print("Loading unsloth fine-tuning method...")
model = FastLanguageModel.get_peft_model(
    model_qwen,
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

#! modelscope依赖库有冲突,且poetry没识别出来
# from modelscope import MsDataset

# dataset = MsDataset.load("OminiData/guanaco-sharegpt-style", split="train")
# dataset = dataset.map(formatting_prompts_func)

from datasets import load_dataset

dataset = load_dataset(
    "/home/cyn/.cache/huggingface/hub/datasets--philschmid--guanaco-sharegpt-style/snapshots/69bfddb3a5d32897ee6b224e1c9397857616f556",
    split="train",
)
dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
)

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False,  # Can make training 5x faster for short sequences.
    args=TrainingArguments(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=60,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        output_dir="outputs",
    ),
)

trainer_stats = trainer.train()

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
_ = model_qwen.generate(
    input_ids=inputs, streamer=text_streamer, max_new_tokens=128, use_cache=True
)

# --------------------------------------------------------------
# 保存model
# --------------------------------------------------------------

model.save_pretrained("lora_model_cache")  # Local saving
tokenizer.save_pretrained("lora_model_cache")
# 导出GGUF,量化方式为q4_k_m
model.save_pretrained_gguf("model_cache", tokenizer, quantization_method="q4_k_m")

print(tokenizer._ollama_modelfile)

# --------------------------------------------------------------
# 模型ollama部署
# --------------------------------------------------------------
"""
ollama create unsloth_qwen2 -f *Modelfile 文件地址*
ollama run unsloth_qwen2
"""
