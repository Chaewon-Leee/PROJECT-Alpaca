# PROJECT-Alpaca

- 기간 : 23.04.06 ~ 23.04.13

----
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

----

### Code Q&A 데이터셋에 대해 fine-tuning

1. [input / output 확인](https://github.com/Chaewon-Leee/PROJECT-Alpaca/blob/main/Check_Input%26Output.ipynb)
2. generation instruction dataset in specific domain (code Q&A)
  - [dataset](https://github.com/jadecxliu/codeqa)
  - [alpaca format 형식 변경](https://github.com/Chaewon-Leee/PROJECT-Alpaca/blob/main/code_alpaca/make_dataset.py)
  - [prompt.txt 변경](https://github.com/Chaewon-Leee/PROJECT-Alpaca/blob/main/code_alpaca/code_prompt.txt)
  - [생성된 instuction dataset 확인](https://github.com/Chaewon-Leee/PROJECT-Alpaca/blob/main/code_alpaca/code_regen.json)
3. Alpaca model FT
----
### Reference
- [notion page](https://royal-tiger-88d.notion.site/Alpaca-KoAlpaca-b7584b13b81c45f0bdd2ca1a62d29707)
- [code Q&A notion page](https://www.notion.so/Alpaca-model-in-specific-domain-code-Q-A-6f3c8647f79c4c7585c65ad739fa1394?pvs=4)
- [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)