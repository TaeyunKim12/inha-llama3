import json

from src.load_token import load_token
from src.model import LlamaVision

if __name__ == "__main__":
    token_path = "./huggingface_token.json"  # token 파일 경로
    TOKEN = load_token(token_path)
    llama_model = LlamaVision(token=TOKEN)
    llama_model.train(
        dataset_path="./dataset/data.json",
        batch_size=1,  # batch size를 키우면 메모리 사용량이 늘어납니다.
        num_train_epochs=3,
        lora_alpha=16,  # lora_alpha를 키우면 메모리 사용량이 늘어납니다.
        r=8,  # r을 키우면 메모리 사용량이 늘어납니다.
    )
