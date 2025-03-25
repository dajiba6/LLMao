from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cuda"  # the device to load the model onto
local_model_path = "/home/cyn/.cache/huggingface/hub/models--Qwen--Qwen2-0.5B-Instruct/snapshots/c540970f9e29518b1d8f06ab8b24cba66ad77b6d"
model_name = "Qwen/Qwen2-0.5B-Instruct"

model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype="auto", device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "杭州的省会在哪里？."
print("Prompt:", prompt)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": prompt},
]
text = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(device)

generated_ids = model.generate(model_inputs.input_ids, max_new_tokens=512)
generated_ids = [
    output_ids[len(input_ids) :]
    for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print("Response:", response)
"""
Sliding Window Attention is enabled but not implemented for `sdpa`; unexpected results may be encountered.
Prompt: 杭州的省会在哪里？.
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Response: 浙江省位于中国东南部，长江三角洲南端。杭州是浙江省的省会和最大城市，也是全国重要的经济、金融中心之一。此外，杭州市还有许多其他重要的城市，如西湖、古港等，都是浙江省的重要组成部分。
"""
