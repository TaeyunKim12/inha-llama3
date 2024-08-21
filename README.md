# inha-llama3
인하대병원 Llama3 학습 코드입니다.

환경 셋팅은 드린 첨부된 Dockerfile과 requirments.txt 파일을 이용하여 build하시면 될 것 같습니다.

## Train Data prepare
```json
{"output": "...", "input": "...", "instruction": "..."}
{"output": "...", "input": "...", "instruction": "..."}
{"output": "...", "input": "...", "instruction": "..."}
{"output": "...", "input": "...", "instruction": "..."}
```
[train_sample.jsonl](samples/train_sample.jsonl)을 참고하시면 됩니다.
해당 파일의 형식과 동일해야 합니다.

* Output : 생성할 findings

* Input : input으로 들어오는 conclusion

* Instruction : prompt


## How To Train
아래의 명령어를 통해 학습이 가능합니다.

```
python3 train.py --train_data ./samples/train_sample.jsonl
```

#### Train Args 설명
Args로 값을 따로 지정하지 않으면 default 값으로 넘어갑니다.

1. `--train_dataset`(str) : train 데이터셋 파일의 경로입니다. 학습을 위해서는 반드시 채워주셔야 합니다.
2. `--eval_dataset` (str, Optional) : eval 데이터셋 파일의 경로입니다. train data와 동일한 형식입니다.
3. `--output_dir` (str) : weight를 저장할 directory입니다. default는 ./output_dir 입니다.
4. `--r` (int) : LoRA의 rank값입니다. default는 16입니다.
5. `--lora_alpha` (int) : LoRA의 alpha값입니다. default는 16입니다.
6. `--max_seq_length` (int) : model이 생성할 최대 output sequence length입니다. default는 1024입니다. 최대 길이기 때문에 키운다고 해서 output findings이 무조건 길어지지는 않습니다.
7. `--num_train_epocs` (int) : 학습할 epoch의 수 입니다. 1 epoch = train data를 전부 1번 봅니다. default는 3입니다.
8. `--save_steps` (int) : step을 몇 번 돌았을 때 모델을 저장할지에 대한 값입니다. default는 50입니다.


## Inference Data Prepare
| Conclusion | SystemPrompt | 
|----------|----------|
| ...   | ...   | 
| ...   | ...   | 

[test_sample.xlsx](samples/test_sample.xlsx)를 참고하시면 됩니다.

해당 파일의 형식(column)이 동일해야 합니다.

여기서 SystemPrompt는 Train data의 instruction과 동일한 개념입니다.

## How To Inference
아래의 명령어를 통해 추론이 가능합니다.
```
python3 inference.py --inference_data ./samples/test_sample.xlsx --model_path ./output/checkpoint-10 --save_name test
```

#### Inference Args 설명
Args로 값을 따로 지정하지 않으면 default 값으로 넘어갑니다.

1. `--inference_data`(str) : inference 할 데이터셋 파일의 경로입니다. 
2. `--model_path` (str) : 학습한 모델의 경로입니다. 해당 경로에 있는 .pt로 추론이 됩니다.
3. `--save_name` (str) : 결과를 저장할 파일의 이름입니다. xlsx로 저장됩니다.

## DPO Data prepare
```json
{"prompt": "...", "chosen": "...", "rejected" :"..."}
{"prompt": "...", "chosen": "...", "rejected" :"..."}
{"prompt": "...", "chosen": "...", "rejected" :"..."}
{"prompt": "...", "chosen": "...", "rejected" :"..."}
```
[dpo_sample.jsonl](samples/dpo_sample.jsonl)을 참고하시면 됩니다.
해당 파일의 형식과 동일해야 합니다.

* prompt : output을 생성한 prompt와 input

* chosen : 동일한 prompt로 생성한 good output

* rejected : 동일한 prompt로 생성한 bad output


## How To DPO
아래의 명령어를 통해 학습이 가능합니다.

```
python3 train_dpo.py --train_data ./samples/dpo_sample.jsonl --model_path ./outputs/checkpoint/
```

#### Train Args 설명
Args로 값을 따로 지정하지 않으면 default 값으로 넘어갑니다.

1. `--train_dataset`(str) : train 데이터셋 파일의 경로입니다. 학습을 위해서는 반드시 채워주셔야 합니다.
2. `--model_path` (str) : SFT를 완료한 모델 폴더의 경로입니다. 반드시 폴더의 경로를 작성해주셔야 합니다.
3. `--output_dir` (str) : weight를 저장할 directory입니다. default는 ./output_dir 입니다.
4. `--r` (int) : LoRA의 rank값입니다. default는 16입니다.
5. `--lora_alpha` (int) : LoRA의 alpha값입니다. default는 16입니다.
6. `--max_seq_length` (int) : model이 생성할 최대 output sequence length입니다. default는 1024입니다. 최대 길이기 때문에 키운다고 해서 output findings이 무조건 길어지지는 않습니다.
7. `--num_train_epocs` (int) : 학습할 epoch의 수 입니다. 1 epoch = train data를 전부 1번 봅니다. default는 3입니다.
8. `--save_steps` (int) : step을 몇 번 돌았을 때 모델을 저장할지에 대한 값입니다. default는 50입니다.