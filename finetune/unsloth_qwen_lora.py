"""
Unsloth fine-tune
Unsloth llama3 fine-tune example: https://colab.research.google.com/drive/1VYkncZMfGFkeCEgN2IzbZIKEDkyQuJAS?usp=sharing#scrollTo=pCqnaKmlO1U9
Unsloth fine-tune tutorial doc: https://docs.unsloth.ai/basics/tutorial-how-to-finetune-llama-3-and-use-in-ollama
"""

from unsloth import FastLanguageModel
import torch

max_seq_length = 2048
dtype = None
load_in_4bit = False

model_name = "Qwen/Qwen2-0.5B-Instruct"
local_model_path = (
    "/home/cyn/.cache/huggingface/hub/models--unsloth--Qwen2-0.5B-Instruct"
)

# Load model
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# test
inputs = tokenizer("Your input text", return_tensors="pt").to(device)
outputs = model(**inputs)
logits = outputs.logits
predictions = logits.argmax(dim=-1)
print(predictions)

# # ? uncloth具体来说用了什么方法?
# print("Loading unsloth fine-tuning method...")
# model = FastLanguageModel.get_peft_model(
#     model,
#     r=16,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
#     target_modules=[
#         "q_proj",
#         "k_proj",
#         "v_proj",
#         "o_proj",
#         "gate_proj",
#         "up_proj",
#         "down_proj",
#     ],
#     lora_alpha=16,
#     lora_dropout=0,  # Supports any, but = 0 is optimized
#     bias="none",  # Supports any, but = "none" is optimized
#     # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
#     use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
#     random_state=3407,
#     use_rslora=False,  # We support rank stabilized LoRA
#     loftq_config=None,  # And LoftQ
# )
