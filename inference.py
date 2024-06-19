import pandas as pd
import textwrap
from unsloth import FastLanguageModel
import argparse

argments = argparse.ArgumentParser()
argments.add_argument("--inference_data", type=str, default=None)
argments.add_argument("--model_path", type=str, default=None)
argments.add_argument("--save_name", type=str, default=None)


def inference(csv_path: str, model_path: str, save_name: str):
    rectum_val = pd.read_excel(csv_path)
    rectum_val.dropna(inplace=True)

    result_df = []

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=1024,
        dtype=None,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}

    ### Input:
    {}

    ### Response:
    {}"""

    for _, row in rectum_val.iterrows():
        rectum = rectum_val["Conclusion"]
        system_prompt = rectum_val["SystemPrompt"]
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    f"{system_prompt}",
                    f"{rectum}",  # input
                    "",  # output - leave this blank for generation!
                )
            ],
            return_tensors="pt",
        ).to("cuda")
        outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
        # print(tokenizer.batch_decode(outputs)[0].split("### Response:")[1])
        result_df.append(
            {
                "Conclusion": rectum,
                "Findings": str(
                    tokenizer.batch_decode(outputs)[0].split("### Response:")[1].replace("<|end_of_text|>", "")
                ),
            }
        )

    if save_name.endswith(".xlsx"):
        pd.DataFrame(result_df).to_excel(save_name, index=False)
    else:
        pd.DataFrame(result_df).to_excel(f"{save_name}.xlsx", index=False)


if __name__ == "__main__":
    args = argments.parse_args()
    inference_data = args.inference_data
    model_path = args.model_path
    save_name = args.save_name

    inference(inference_data, model_path, save_name)
