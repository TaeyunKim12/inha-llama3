import pandas as pd
import textwrap
from unsloth import FastLanguageModel
import argparse


class LlamaInha:
    def __init__(
        self,
        model_path: str,
    ) -> None:
        self.max_seq_length = 1024
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,  # YOUR MODEL YOU USED FOR TRAINING
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        self.alpaca_prompt = textwrap.dedent(
            """\
                Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

                ### Instruction:
                {}
                breakpoint()

                ### Input:
                {}

                ### Response:
                {}"""
        )

    def run(self, input: str, system_prompt: str, logging: bool = False) -> str:
        prompt = self.alpaca_prompt.format(
            f"{system_prompt}",
            f"{input}",  # input
            "",  # output - leave this blank for generation!
        )
        if logging:
            print("=====================")
            print(f"Input: {prompt}")
            print("=====================")
        input = self.tokenizer([prompt], return_tensors="pt").to("cuda")

        output_tokens = self.model.generate(
            **input, max_new_tokens=self.max_seq_length, use_cache=True, num_return_sequences=1
        )

        output = str(self.tokenizer.batch_decode(output_tokens)[0]).replace("<|end_of_text|>", "")

        return output
