from src.load_token import load_token
from src.model import LlamaVision

if __name__ == "__main__":
    OUTPUT_PARSING = True
    TOKEN_PATH = "./huggingface_token.json"  # token 파일 경로
    PRETRAIN_MODEL_DIR = "./fine-tuned-visionllama/checkpoint-1"
    TOKEN = load_token(TOKEN_PATH)
    llama_model = LlamaVision(token=TOKEN, pretrain_model_dir=PRETRAIN_MODEL_DIR)

    result = llama_model.inference(
        image="./image/test.jpg",
        text="Describe image simply.",
        max_new_tokens=400,  #
        num_return_sequences=1,
    )
    # output만 가져옵니다.
    if OUTPUT_PARSING:
        result = result.split("<|end_header_id|>")[-1].strip()
        result = result.replace("<|eot_id|>", "").strip()
    print(result)
