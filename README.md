# PROJECT-Alpaca

- 기간 : 23.04.06 ~ 23.04.13

### Gererating the data

- API 연결

    ```python
    ### .env 파일
    OPENAI_API_KEY = {OPENAI_API_KEY}
    ```

- 실행

    ```python -m generate_instruction generate_instruction_following_data```

- trouble shooting

    `typeerror: 'type' object is not subscriptable`

    ```python
    from typing import Optional, Sequence, Union, Dict

    def openai_completion(
        # prompts: Union[str, Sequence[str], Sequence[dict[str, str]], dict[str, str]],
        ### 타입 어노테이션 오류 !
        prompts: Union[str, Sequence[str], Sequence[Dict[str, str]], Dict[str, str]],
        decoding_args: OpenAIDecodingArguments,
        model_name="text-davinci-003",
        sleep_time=2,
        batch_size=1,
        max_instances=sys.maxsize,
        max_batches=sys.maxsize,
        return_text=False,
        **decoding_kwargs,
    ) -> Union[Union[StrOrOpenAIObject], Sequence[StrOrOpenAIObject], Sequence[Sequence[StrOrOpenAIObject]],]:
    ```

- KoAlpaca
  - 영어로 데이터셋 생성 → Instruction과 Input 번역 → output 생성 ⇒ 52k 한국어 데이터셋
  - 즉, 영어로 데이터셋을 생성하는 과정은 동일

### Model Guidline

1. [input / output 확인](https://github.com/Chaewon-Leee/PROJECT-Alpaca/blob/main/Check_Input%26Output.ipynb)
2. code Q&A 데이터셋에 대해 fine-tuning
  1. generation instruction dataset in specific domain (code Q&A)

### Reference
- [notion page](https://royal-tiger-88d.notion.site/Alpaca-KoAlpaca-b7584b13b81c45f0bdd2ca1a62d29707)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [KoAlpaca](https://github.com/Beomi/KoAlpaca)