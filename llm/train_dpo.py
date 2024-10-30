from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse
from typing import Dict

PatchDPOTrainer()
import torch
from trl import DPOTrainer

argments = argparse.ArgumentParser()
argments.add_argument("--train_dataset", type=str, default=None)
argments.add_argument("--model_path", type=str, default=None)
argments.add_argument("--output_dir", type=str, default="./output_dir")
argments.add_argument("--r", type=int, default=16)
argments.add_argument("--lora_alpha", type=int, default=16)
argments.add_argument("--max_seq_length", type=int, default=1024)
argments.add_argument("--num_train_epochs", type=int, default=5)
argments.add_argument("--save_steps", type=int, default=50)


alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""


# add valid arguments type
def llama3_sft(
    train_dataset: str,
    model_path: str,
    output_dir: str,
    r: int,
    lora_alpha: int,
    max_seq_length: int,
    num_train_epochs: int,
    save_steps: int,
):

    if train_dataset is None:
        raise ValueError("train_dataset is required. train_dataset Cannot be None.")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    EOS_TOKEN = tokenizer.eos_token
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
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
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    train_dpo_dataset = load_dataset("json", data_files=train_dataset, split="train")
    train_dpo_dataset.map(batched=True)
    dpo_trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            num_train_epochs=num_train_epochs,
            learning_rate=5e-6,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.0,
            lr_scheduler_type="linear",
            seed=42,
            output_dir=output_dir,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=5,
        ),
        beta=0.1,
        train_dataset=train_dpo_dataset,
        tokenizer=tokenizer,
        max_length=2048,
        max_prompt_length=2048,
    )

    dpo_trainer.train()


if __name__ == "__main__":
    args = argments.parse_args()

    train_dataset = args.train_dataset
    model_path = args.model_path
    output_dir = args.output_dir
    r = args.r
    lora_alpha = args.lora_alpha
    max_seq_length = args.max_seq_length
    num_train_epochs = args.num_train_epochs
    save_steps = args.save_steps

    llama3_sft(
        train_dataset=train_dataset,
        model_path=model_path,
        output_dir=output_dir,
        r=r,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
    )
