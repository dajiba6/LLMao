from unsloth import FastLanguageModel
import torch
from datasets import load_dataset
from trl import SFTTrainer
from transformers import (
    TrainingArguments,
    TextStreamer,
    AutoModelForCausalLM,
    set_seed,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
import gc

set_seed(42)


def train_unsloth(
    dtype,
    max_seq_length,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    rank,
    lora_alpha=16,
    lora_dropout=0,
    max_steps=50,
    save_steps=50,
    seed=42,
    warmup_steps=5,
    learning_rate=2e-4,
    logging_steps=5,
):
    print(
        f"dtype:{dtype}, max_seq_length:{max_seq_length}, per_device_train_batch_size:{per_device_train_batch_size}, gradient_accumulation_steps:{gradient_accumulation_steps}, rank:{rank}, lora_dropout:{lora_dropout}"
    )
    load_in_4bit = True

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="pretrain_models/Qwen/Qwen1.5-32B-Chat/",
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=seed,
        use_rslora=False,
        max_seq_length=max_seq_length,
    )

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{instruction}. {input}"},
                    {"role": "assistant", "content": f"{output}"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    pass

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(formatting_prompts_func, batched=True)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir="output/llame3-8b-instruct-unsloth",
            save_steps=save_steps,
            max_steps=max_steps,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained("output/llame3-8b-instruct-unsloth-lora")  # Local saving
    tokenizer.save_pretrained("output/llame3-8b-instruct-unsloth-lora")

    # model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)  # Merge to 16bit
    # model.save_pretrained_merged("model", tokenizer, save_method = "merged_4bit",) # Merge to 4bit
    # model.save_pretrained_merged("model", tokenizer, save_method = "lora",) # Just LoRA adapters
    # model.save_pretrained_gguf("model", tokenizer,)   # Save to 8bit Q8_0
    # model.save_pretrained_gguf("model", tokenizer, quantization_method = "f16")   # Save to 16bit GGUF
    # model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")    # Save to q4_k_m GGUF
    del model
    del tokenizer

    torch.cuda.empty_cache()
    for _ in range(3):
        gc.collect()


def train_trans(
    dtype,
    max_seq_length,
    per_device_train_batch_size,
    gradient_accumulation_steps,
    rank,
    lora_alpha=16,
    lora_dropout=0,
    max_steps=50,
    save_steps=50,
    seed=42,
    warmup_steps=5,
    learning_rate=2e-4,
    logging_steps=5,
):
    print(
        f"dtype:{dtype}, max_seq_length:{max_seq_length}, per_device_train_batch_size:{per_device_train_batch_size}, gradient_accumulation_steps:{gradient_accumulation_steps}, rank:{rank}, lora_dropout:{lora_dropout}"
    )

    model_path = "pretrain_models/Qwen/Qwen1.5-32B-Chat/"

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, padding_side="right", model_max_length=8192
    )

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dtype,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        quantization_config=quantization_config,
    )

    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model.enable_input_require_grads()

    config = LoraConfig(
        r=rank,
        lora_alpha=lora_alpha,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        use_rslora=False,
    )

    model = get_peft_model(model, peft_config=config)
    model.gradient_checkpointing_enable()

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": f"{instruction}. {input}"},
                    {"role": "assistant", "content": f"{output}"},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            texts.append(text)
        return {"text": texts}

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=logging_steps,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=seed,
            output_dir="output/llame3-8b-instruct-unsloth",
            save_steps=save_steps,
            max_steps=max_steps,
        ),
    )

    gpu_stats = torch.cuda.get_device_properties(0)
    start_gpu_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
    print(f"GPU = {gpu_stats.name}. Max memory = {max_memory} GB.")
    print(f"{start_gpu_memory} GB of memory reserved.")

    trainer_stats = trainer.train()

    used_memory = round(torch.cuda.max_memory_reserved() / 1024 / 1024 / 1024, 3)
    used_memory_for_lora = round(used_memory - start_gpu_memory)
    used_percentage = round(used_memory / max_memory * 100, 3)
    lora_percentage = round(used_memory_for_lora / max_memory * 100, 3)
    print(f"{trainer_stats.metrics['train_runtime']} seconds used for training.")
    print(
        f"{round(trainer_stats.metrics['train_runtime']/60, 2)} minutes used for training."
    )
    print(f"Peak reserved memory = {used_memory} GB.")
    print(f"Peak reserved memory for training = {used_memory_for_lora} GB.")
    print(f"Peak reserved memory % of max memory = {used_percentage} %.")
    print(f"Peak reserved memory for training % of max memory = {lora_percentage} %.")

    model.save_pretrained("output/llame3-8b-instruct-unsloth-lora")  # Local saving
    tokenizer.save_pretrained("output/llame3-8b-instruct-unsloth-lora")

    del model
    del tokenizer

    torch.cuda.empty_cache()
    for _ in range(3):
        gc.collect()


def infer():

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="output/llame3-8b-instruct-unsloth-lora",
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=True,
    )

    # 2x的速率进行推理
    FastLanguageModel.for_inference(model)

    inputs = tokenizer(
        [
            alpaca_prompt.format(
                "Continue the fibonnaci sequence.", "1, 1, 2, 3, 5, 8", ""
            )
        ],
        return_tensors="pt",
    ).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
    print(tokenizer.batch_decode(outputs))

    text_streamer = TextStreamer(tokenizer)
    outputs = model.generate(**inputs, max_new_tokens=1024, streamer=text_streamer)
    print(tokenizer.batch_decode(outputs))


if __name__ == "__main__":

    train_unsloth(
        dtype=torch.bfloat16,
        max_seq_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        rank=8,
        lora_dropout=0,
    )
    train_unsloth(
        dtype=torch.bfloat16,
        max_seq_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        rank=64,
        lora_dropout=0,
    )
    train_unsloth(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        rank=64,
        lora_dropout=0,
    )
    train_unsloth(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        rank=64,
        lora_dropout=0,
    )
    train_unsloth(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        rank=64,
        lora_dropout=0.05,
    )
    train_unsloth(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        rank=64,
        lora_dropout=0.05,
    )

    train_trans(
        dtype=torch.bfloat16,
        max_seq_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        rank=8,
        lora_dropout=0,
    )
    train_trans(
        dtype=torch.bfloat16,
        max_seq_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        rank=64,
        lora_dropout=0,
    )
    train_trans(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        rank=64,
        lora_dropout=0,
    )
    train_trans(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        rank=64,
        lora_dropout=0,
    )
    train_trans(
        dtype=torch.bfloat16,
        max_seq_length=2048,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        rank=64,
        lora_dropout=0.05,
    )
