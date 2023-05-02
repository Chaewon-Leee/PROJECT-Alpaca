# PROJECT-Alpaca

- 기간 : 23.04.06 ~ 23.04.13

### Gererating the data

- 실행

    ```python -m generate_instruction generate_instruction_following_data```

- KoAlpaca
  - 영어로 데이터셋 생성 → Instruction과 Input 번역 → output 생성 ⇒ 52k 한국어 데이터셋
  - 즉, 영어로 데이터셋을 생성하는 과정은 동일

### Model Guidline
  1. [input / output 확인](https://github.com/Chaewon-Leee/PROJECT-Alpaca/blob/main/Check_Input%26Output.ipynb)
  2. 데이터셋 fine-tuning
  - [dataset](https://github.com/jadecxliu/codeqa)

### Reference
- [notion page](https://royal-tiger-88d.notion.site/Alpaca-KoAlpaca-b7584b13b81c45f0bdd2ca1a62d29707)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [KoAlpaca](https://github.com/Beomi/KoAlpaca)