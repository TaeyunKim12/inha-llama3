import json


def load_token(token_path: str):
    """
    huggingface_token을 불러옵니다.
    """
    try:
        with open(token_path, "r") as f:
            token_metadata: dict = json.load(f)
    except FileNotFoundError:
        print("huggingface_token.json 파일을 찾을 수 없습니다.")
        exit(1)
    TOKEN = token_metadata.get("TOKEN", None)
    if TOKEN == "..." or TOKEN is None:
        print(f"토큰이 비어있습니다. {token_path} 파일을 확인해주세요.")
    return TOKEN
