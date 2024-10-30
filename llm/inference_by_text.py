from model.model import LlamaInha
import time

if __name__ == "__main__":
    model_path = "./checkpoint"  # 사용하시는 checkpoint 경로로 수정해주세요
    input = "..."  # 입력 문장을 입력해주세요
    system_prompt = "..."  # 사용하시는 시스템 프롬프트를 입력해주세요
    logging = False  # model input과 inference 결과를 출력하려면 True로 설정해주세요.
    PARSE_OUTPUT = True  # output을 parsing하려면 True로 설정해주세요.
    llama = LlamaInha(model_path=model_path)

    output = llama.run(
        input,
        system_prompt,
        logging=logging,
    )

    if PARSE_OUTPUT:
        output = output.split("### Response:")[-1].strip()
    print(output)
