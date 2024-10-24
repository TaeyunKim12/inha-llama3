import pandas as pd
import argparse
from model.model import LlamaInha

argments = argparse.ArgumentParser()
argments.add_argument("--inference_data", type=str, default=None)
argments.add_argument("--model_path", type=str, default=None)
argments.add_argument("--save_name", type=str, default="result")


def inference(csv_path: str, model_path: str, save_name: str):
    rectum_val = pd.read_excel(csv_path)
    rectum_val.dropna(inplace=True)

    result_df = []

    llama = LlamaInha(model_path=model_path)
    for _, row in rectum_val.iterrows():
        input = row["Conclusion"]
        system_prompt = row["SystemPrompt"]
        output = llama.run(input, system_prompt, logging=False)
        result_df.append(
            {
                "Conclusion": input,
                "Findings": output,
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
