import json
import traceback
import warnings

import torch
from huggingface_hub import login
from peft import LoraConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig, Qwen2VLProcessor
from trl import SFTConfig, SFTTrainer

from src.load_dataset import load_dataset

warnings.filterwarnings("ignore")


class LlamaVision:
    def __init__(self, token: str, pretrain_model_dir: str | None = None):
        """
        Llama 3.2 Vision 모델을 initialize합니다.
        한 번만 실행하면 됩니다.
        """
        self.token = token
        self.model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"
        login(
            token=token,
            # add_to_git_credential=True,
        )  # ADD YOUR TOKEN HERE
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        try:
            if pretrain_model_dir:
                print("=========================")
                print("사전 학습된 모델을 사용합니다.")
                print("=========================")
                model = AutoModelForVision2Seq.from_pretrained(
                    pretrain_model_dir,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                )

            else:
                print("=========================")
                print("기본 모델을 사용합니다.")
                print("=========================")
                model = AutoModelForVision2Seq.from_pretrained(
                    self.model_id,
                    device_map="auto",
                    torch_dtype=torch.bfloat16,
                    quantization_config=bnb_config,
                )
        except torch.OutOfMemoryError:
            print("GPU 메모리가 부족하여 모델을 불러올 수 없습니다. gpu를 사용하는 다른 작업을 종료해보세요.")
        except Exception as e:
            print(f"모델을 불러오는 중 오류가 발생했습니다. {e}, {traceback.format_exc()}")
        self.model = model
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    def train(
        self,
        dataset_path: str,
        batch_size: int = 2,
        num_train_epochs: int = 3,
        lora_alpha: int = 16,
        r: int = 8,
        save_dir: str = "fine-tuned-visionllama",
    ):
        """
        Llama 3.2 Vision 모델을 학습합니다.
        lora_alpha와 rank를 조절하여 학습할 수 있습니다.
        만약 OOM(Out of Memory)이 발생한다면 batch_size를 줄이거나 lora_alpha와 rank를 조절해보세요.
        """
        dataset = load_dataset(dataset_path)  # ./data/dataset.json
        peft_config = LoraConfig(
            lora_alpha=lora_alpha,
            lora_dropout=0.05,
            r=r,
            bias="none",
            target_modules=["q_proj", "v_proj"],
            task_type="CAUSAL_LM",
        )

        args = SFTConfig(
            output_dir=save_dir,  # directory to save and repository id
            num_train_epochs=num_train_epochs,  # number of training epochs
            per_device_train_batch_size=batch_size,  # batch size per device during training
            gradient_accumulation_steps=8,  # number of steps before performing a backward/update pass
            gradient_checkpointing=True,  # use gradient checkpointing to save memory
            optim="adamw_torch_fused",  # use fused adamw optimizer
            logging_steps=5,  # log every 10 steps
            save_strategy="epoch",  # save checkpoint every epoch
            learning_rate=2e-4,  # learning rate, based on QLoRA paper
            bf16=True,  # use bfloat16 precision
            tf32=True,  # use tf32 precision
            max_grad_norm=0.3,  # max gradient norm based on QLoRA paper
            warmup_ratio=0.03,  # warmup ratio based on QLoRA paper
            lr_scheduler_type="constant",  # use constant learning rate scheduler
            push_to_hub=True,  # push model to hub
            report_to="tensorboard",  # report metrics to tensorboard
            gradient_checkpointing_kwargs={"use_reentrant": False},  # use reentrant checkpointing
            dataset_text_field="",  # need a dummy field for collator
            dataset_kwargs={"skip_prepare_dataset": True},  # important for collator
        )
        args.remove_unused_columns = False

        trainer = SFTTrainer(
            model=self.model,
            args=args,
            train_dataset=dataset,
            data_collator=self._collate_fn,
            dataset_text_field="",  # needs dummy value
            peft_config=peft_config,
            tokenizer=self.processor.tokenizer,
        )
        try:
            trainer.train()
        except torch.OutOfMemoryError as e:
            print(
                "GPU 메모리가 부족하여 학습을 종료합니다. batch_size를 줄이거나 gpu를 사용하는 다른 작업을 종료해보세요."
            )
            print(f"{e}, {traceback.format_exc()}")
        except Exception as e:
            print(f"학습 중 오류가 발생했습니다. {e}, {traceback.format_exc()}")

    def inference(
        self,
        image: str | Image.Image,
        text: str,
        max_new_tokens: int = 100,
        num_return_sequences: int = 1,
    ) -> str:
        """
        image와 text를 입력받아 모델을 통해 결과를 반환합니다.
        max_new_tokens는 생성할 최대 토큰의 개수입니다.
        """
        if isinstance(image, str):
            try:
                image = Image.open(image)
            except FileNotFoundError:
                print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
        messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": text}]}]
        input_text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(image, input_text, add_special_tokens=False, return_tensors="pt").to(self.model.device)

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_return_sequences=num_return_sequences,
        )
        return self.processor.decode(output[0])

    def _collate_fn(self, examples):
        # Get the texts and images, and apply the chat template
        texts = [self.processor.apply_chat_template(example["messages"], tokenize=False) for example in examples]
        image_inputs = [process_vision_info(example["messages"])[0] for example in examples]

        # Tokenize the texts and process the images
        batch = self.processor(text=texts, images=image_inputs, return_tensors="pt", padding=True)

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100  #
        # Ignore the image token index in the loss computation (model specific)
        if isinstance(self.processor, Qwen2VLProcessor):
            image_tokens = [151652, 151653, 151655]
        else:
            image_tokens = [self.processor.tokenizer.convert_tokens_to_ids(self.processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch
