from datasets.utils import concatenate_columns, load_from_path, jsonify_columns
import anthropic
import json
import re
import pandas as pd

from argparse import ArgumentParser
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from langchain_core.prompts import (
    FewShotChatMessagePromptTemplate,
    ChatPromptTemplate,
)
from tqdm import tqdm

load_dotenv()

DELIMITER = "\t"
CHUNK_SIZE = 3

TEMPLATE = """당신은 한국어 번역가로서 영어 문장을 한국어로 번역하고 윤문해야 합니다. 원문은 문장 인덱스별로 질문과 답변 선택지로 구성되어 있습니다.\
질문과 답변 선택지는 \t으로 분리되어 있으며, 첫 value는 질문이고 나머지 value는 답변 선택지입니다. 번역문은 각 문장을 번역한 후 \t로 구분하여 작성하세요.\
다음 <guidelines>을 지켜 번역하세요.

<guidelines>
1. 문맥을 반영하여 번역하세요. 여기서 문맥이란 <input>내 해당 인덱스의 문장 전체를 의미합니다. 예를 들어 'organism'은 문맥에 따라 '유기체, 생명체, 생명, 유기적 조직체' 등 다양하게 번역될 수 있습니다.
2. 번역문은 한국어 원어민이 쉽게 이해할 수 있도록 한국인 입장에서 자연스러운 표현으로 이뤄져야 합니다. 영어 원문의 의미를 정확하게 전달하되, 직역이 어색할 경우 자연스러운 표현으로 번역하세요.
3. 10대 청소년도 이해할 수 있을 정도로 문장을 윤문하세요. 영어 원문의 톤은 유지해야 합니다.
4. 문화적 차이를 고려하여 한국어 표현을 선택하세요. 예를 들어 'kick the bucket'은 '세상을 떠나다'로 번역할 수 있습니다.
5. 사람 이름은 한국식 이름으로 번역합니다. 기업 이름은 유사 업종의 가장 유명한 한국 기업으로 변경합니다. 
6. 한국어의 격식체를 사용합니다. 격식체란 다음과 같이 '~있습니다', '~니다', '~할까요?' 등의 문장 끝말을 사용하는 것을 말합니다.
7. 전문 용어나 어려운 용어는 영어 원문 단어를 중괄호 안에 넣어 번역하세요. 예를 들어 '항정신성 약물(Antisychotics)은 ...'와 같이 표기합니다.
8. 문제 형식의 텍스트를 번역할 때는 정답을 추론하거나 표시하지 마세요. 주어진 텍스트만 번역하고 추가적인 설명이나 해석을 덧붙이지 마세요. 번역 시 의심스러운 부분이 있더라도 추측하지 마세요. 예를 들어 '1/3'을 '1월 3일'로 번역하지 말고 '1/3'으로 그대로 번역하세요.
9. 원문의 형식을 그대로 유지하세요. 선택지, 정답 표시, json 포맷 등 원문의 구조적 요소를 변경하지 마세요. json 양식을 제대로 지켜, \" 연속 사용, 콤마(,) 사용 등을 주의하세요.
10. None 또는 nan, null 은 '없음'으로 번역하세요. 
</guidelines>

<examples>
{examples}
</examples>"""

TEMPLATE_RETRY = """Your answer seems to have some issues with <guidelines> and <examples> above in terms of the format and the structure. Look carefully at the uses of quotes or commas and fix them if they would prevent valid json formatting\
Please check the format and the guidelines and provide again the Korean translations **without any explanations** in the right format."""

client = anthropic.Anthropic()


def _complete_anthropic(messages: list[dict], system_message: str = None):

    try:
        response = client.messages.with_raw_response.create(
            model="claude-3-5-sonnet-20240620",
            max_tokens=4096,
            messages=messages,
            system=system_message,
        )
        messages = (
            response.parse()
        )  # get the object that `messages.create()` would have returned
        print(response.headers)
    except anthropic.RateLimitError as e:
        print("🚨 Rate limit exceeded...")
        print(e)

    return messages


def _format_examples(examples: list[dict]):
    input_examples = json.dumps(
        {idx: f"{ex['user']}" for idx, ex in enumerate(examples)}
    )
    output_examples = json.dumps(
        {idx: f"{ex['assistant']}" for idx, ex in enumerate(examples)}
    )

    return "\nshould be translated and structured into\n".join(
        [f"{input_examples}", f"{output_examples}"]
    )


def _format_chunk(chunk: pd.DataFrame, columns) -> dict[int, str]:
    """
    `jsonify_columns` returns:
    {
        "0": {
            "col0": "In which year was the seminal Human Development Report published?",
            "col1": "It was published in 1990."
        },
        "1": {
            "col0": "Sam bought a car from Tesla. The car dealer offered him a 10% discount, and Sam paid $90,000 for the car. How much was the original price of the car?",
            "col1": "Its original price was $110,000.",
            "col2": "Its original price was $100,000."
        }
    }
    """
    chunk_in_format = {}
    # for idx, row in chunk.iterrows():
    # row_data = concatenate_columns(row, columns)
    for idx in chunk.index:
        row = chunk.loc[idx]
        row_data = jsonify_columns(row, columns)
        chunk_in_format[str(idx)] = row_data
    return json.dumps(chunk_in_format)


def _parse_response(response, columns: list) -> dict[int, dict]:
    """Returns
    {
        0: {"column1": "value1", "column2": "value2", "column3": "value3"},
    }
    """
    parsed = {}
    try:
        response = json.loads(response.content[0].text)
        for idx, row_data in response.items():
            parsed[idx] = {col: str(row_data[col]) for col in columns}
            # values = row_data.split(DELIMITER)
            # parsed[idx] = {col: val for col, val in zip(columns, values)}

    except:
        # can encounter SyntaxError when `eval`ing the response
        return None

    return parsed


def translate_df(df, columns, prompt, path):
    instruct_prompt = prompt.lstrip("Human: ")
    chunks = [df.iloc[i : i + CHUNK_SIZE] for i in range(0, df.shape[0], CHUNK_SIZE)]

    with open(path, "a") as f:
        print(f"Translating {path.name}...")
        if f.tell() == 0:
            f.write(",".join(["id"] + columns) + "\n")

        for chunk in tqdm(chunks):
            formatted_rows = _format_chunk(chunk, columns)
            response = _complete_anthropic(
                messages=[{"role": "user", "content": formatted_rows}],
                system_message=instruct_prompt,
            )
            parsed = _parse_response(response, columns)
            if parsed is None:
                # Try in case of SyntaxError
                response = _complete_anthropic(
                    messages=[
                        {"role": "user", "content": formatted_rows},
                        {"role": "assistant", "content": response.content[0].text},
                        {"role": "user", "content": TEMPLATE_RETRY},
                    ],
                    system_message=instruct_prompt,
                )
                parsed = _parse_response(response, columns)
                if parsed is None:
                    parsed = {
                        idx: {col: response.content[0].text for col in columns}
                        for idx in chunk.index
                    }

            pd.DataFrame(parsed).T.to_csv(f, header=False, index=True)


def main(path, *, columns, headless, with_index, start_index):
    _examples = [
        {
            "user": {
                "col0": "In which year was the seminal Human Development Report published?",
                "col1": "It was published in 1990.",
            },
            "assistant": {
                "col0": "큰 영향력을 갖게 되었던 인간 개발 보고서(Human Development Report)는 몇 년도에 발행되었나요?",
                "col1": "보고서는 1990년에 발행되었습니다.",
            },
        },
        {
            "user": {
                "col0": "Sam bought a car from Tesla. The car dealer offered him a 10% discount, and Sam paid $90,000 for the car. How much was the original price of the car?",
                "col1": "Its original price was $110,000.",
                "col2": "Its original price was $100,000.",
            },
            "assistant": {
                "col0": "민수는 기아의 차를 샀습니다. 차 판매자는 그에게 10% 할인을 제공했고, 민수는 차를 $90,000에 샀습니다. 차의 원래 판매가는 얼마였나요?",
                "col1": "차의 판매가는 $110,000이었습니다.",
                "col2": "차의 판매가는 $100,000이었습니다.",
            },
        },
    ]

    # _examples = [
    #     {
    #         "input": DELIMITER.join(
    #             [
    #                 "In which year was the seminal Human Development Report published?",
    #                 "It was published in 1990.",
    #             ]
    #         ),
    #         "output": DELIMITER.join(
    #             [
    #                 "중요한 인간 개발 보고서(Human Development Report)는 몇 년도에 발행되었나요?",
    #                 "보고서는 1990년에 발행되었습니다.",
    #             ]
    #         ),
    #     },
    #     {
    #         "input": DELIMITER.join(
    #             [
    #                 "Sam wants to go to bed.",
    #                 "Tesla makes the coolest car in the world.",
    #             ]
    #         ),
    #         "output": DELIMITER.join(
    #             [
    #                 "민호는 자려고 합니다.",
    #                 "기아는 세상에서 가장 멋진 차를 만듭니다.",
    #             ]
    #         ),
    #     },
    # ]
    # _example_[pr]ompt = ChatPromptTemplate.from_messages(
    #     [("user", "{input}"), ("assistant", "{output}")]  # 🚨 Anthropic: user/assistant
    # )

    # _few_shot_prompt = FewShotChatMessagePromptTemplate(
    #     examples=_examples,
    #     example_prompt=_example_prompt,
    # )

    _few_shot_prompt = _format_examples(_examples)
    instruct_template = ChatPromptTemplate.from_template(TEMPLATE)
    instruct_prompt = instruct_template.format(examples=_few_shot_prompt)

    for p in tqdm(load_from_path(path)):
        if p.suffix == ".csv":
            df = pd.read_csv(
                p,
                header=None if headless else 0,
                index_col=0 if with_index else None,
            )
        elif p.suffix == ".xlsx":
            df = pd.read_excel(
                p,
                header=None if headless else 0,
                index_col=0 if with_index else None,
            )
        if headless:
            df.columns = [f"col{col}" for col in df.columns]
        if columns:
            df = df[columns]
        if start_index:
            df = df.loc[start_index:]
        columns = df.columns.tolist()  # use all the columns

        translate_df(
            df,
            columns,
            instruct_prompt,
            (
                p.with_suffix(f".from_{start_index}.translated.csv")
                if start_index
                else p.with_suffix(".translated.csv")
            ),
        )

        # # fill in missing columns
        # new_df = new_df.combine_first(df)
        # new_df.to_csv(p.with_suffix(".translated.csv"), index=False)


# chain = final_prompt | llm
# output = chain.invoke(
#     {
#         "input": "/t".join(
#             [
#                 "The quick brown fox jumps over the lazy dog.",
#                 "The passage was coined by Noam Chomsky.",
#             ]
#         ),
#     }
# )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--with-index", action="store_true")
    parser.add_argument("--columns", nargs="+")
    parser.add_argument("--start-index", type=int)
    args = parser.parse_args()
    output = main(
        path=args.path,
        columns=args.columns,
        headless=args.headless,
        with_index=args.with_index,
        start_index=args.start_index,
    )
