import json

from PIL import Image


def format_data(sample: dict):
    """
    dataset json에서 이미지 경로를 가져와 이를 Image로 읽고 반환합니다.
    """
    img = Image.open(sample["messages"][0]["content"][1]["image"])
    sample["messages"][0]["content"][1]["image"] = img
    return sample


def load_dataset(dataset_path: str):
    """
    dataset json을 읽어와 학습에 사용할 수 있도록 포맷팅합니다.
    """
    with open(dataset_path, "r") as file:
        dataset = json.load(file)
        dataset = [format_data(sample) for sample in dataset]
        return dataset
