from unsloth import FastLanguageModel
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import load_dataset
import argparse

argments = argparse.ArgumentParser()
argments.add_argument("--train_dataset", type=str, default=None)
argments.add_argument("--eval_dataset", type=str, default=None)
argments.add_argument("--output_dir", type=str, default="./output_dir")
argments.add_argument("--r", type=int, default=16)
argments.add_argument("--lora_alpha", type=int, default=16)
argments.add_argument("--max_seq_length", type=int, default=1024)
argments.add_argument("--num_train_epochs", type=int, default=3)
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
    eval_dataset: str,
    output_dir: str,
    r: int,
    lora_alpha: int,
    max_seq_length: int,
    num_train_epochs: int,
    save_steps: int,
):
    """
    llama3_sft is a function that performs training using the SFTTrainer class from the trl library. It takes the following parameters:

    Parameters
    ----------
    train_dataset : str
        The path to the training dataset file.
    eval_dataset : str, optional
        The path to the evaluation dataset file. Defaults to None.
    output_dir : str, optional
        The directory where the output files will be saved. Defaults to "./output_dir".
    r : int, optional
        The value of the 'r' parameter. Defaults to 16.
    lora_alpha : int, optional
        The value of the 'lora_alpha' parameter. Defaults to 16.
    max_seq_length : int, optional
        The maximum sequence length. Defaults to 1024.
    num_train_epochs : int, optional
        The maximum number of training epochs. Defaults to 3.
    save_steps : int, optional
        The number of steps between saving checkpoints. Defaults to 10.

    Returns
    -------
    None
    """

    if train_dataset is None:
        raise ValueError("train_dataset is required. train_dataset Cannot be None.")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/llama-3-8b-bnb-4bit",
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )
    EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)

        return {
            "text": texts,
        }

    dataset = load_dataset("json", data_files=train_dataset, split="train")
    dataset = dataset.map(
        formatting_prompts_func,
        batched=True,
    )

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
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing="unsloth",  # True or "unsloth" for very long context
        random_state=3407,
        max_seq_length=max_seq_length,
        use_rslora=False,  # We support rank stabilized LoRA
        loftq_config=None,  # And LoftQ
    )

    if eval_dataset is None:
        print("Not using eval_dataset")
        args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            weight_decay=0.01,
            warmup_steps=10,
            # max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
            save_strategy="steps",
            save_steps=save_steps,
            save_total_limit=5,
        )

    else:
        print("Using eval_dataset")
        eval_dataset = load_dataset("json", data_files=eval_dataset, split="train")
        eval_dataset = eval_dataset.map(
            formatting_prompts_func,
            batched=True,
        )

        args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            weight_decay=0.01,
            warmup_steps=10,
            # max_steps=max_steps,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=1,
            output_dir=output_dir,
            optim="adamw_8bit",
            seed=3407,
            save_strategy="steps",
            save_steps=save_steps,
            fp16_full_eval=True,  # eval
            per_device_eval_batch_size=4,  # eval
            eval_accumulation_steps=4,  # eval
            evaluation_strategy="steps",  # eval
            eval_steps=1000,  # eval
            save_total_limit=5,
        )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # Can make training 5x faster for short sequences.
        args=args,
    )

    trainer.train()


if __name__ == "__main__":
    args = argments.parse_args()

    train_dataset = args.train_dataset
    eval_dataset = args.eval_dataset
    output_dir = args.output_dir
    r = args.r
    lora_alpha = args.lora_alpha
    max_seq_length = args.max_seq_length
    num_train_epochs = args.num_train_epochs
    save_steps = args.save_steps

    llama3_sft(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=output_dir,
        r=r,
        lora_alpha=lora_alpha,
        max_seq_length=max_seq_length,
        num_train_epochs=num_train_epochs,
        save_steps=save_steps,
    )
